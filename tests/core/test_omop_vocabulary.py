"""Tests for pyhealth.medcode.omop on a small synthetic Athena bundle.

The bundle is written to a temporary folder. It mimics Athena's layout
(tab-separated ``.csv`` files) with a dozen concepts arranged to cover the
cases that matter: a depth cutoff, a concept with two parents at different
depths, an orphan whose only parent is in another domain, deprecated rows,
and ICD codes that map to zero, one or two concepts.

Hierarchy ("Is a", child -> parent)::

    700 Clinical finding
     |-- 4274025 Disease (root)            300 Fever -> 700 (not under root)
          |-- 100 Heart disease
          |    |-- 101 Ischemic heart disease
          |    |    `-- 102 Acute MI
          |    |         `-- 103 Acute MI, inferior wall
          |    |              `-- 104 (also directly under 100)
          |    |-- 600 Cardiac observation (Observation domain)
          |    |    `-- 105 Orphan disorder
          |    |-- 400 non-standard concept
          |    `-- 401 deprecated concept
          `-- 200 Lung disease
"""

import shutil
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import polars as pl

from pyhealth.medcode import (
    DISEASE_CONCEPT_ID,
    ConditionHierarchy,
    ICDToSNOMEDMapper,
    OMOPVocabulary,
)
from pyhealth.medcode.omop import normalize_icd_code

ROOT = DISEASE_CONCEPT_ID

# concept_id, name, domain, vocabulary, standard, code, invalid_reason
CONCEPTS = [
    (700, "Clinical finding", "Condition", "SNOMED", "S", "404684003", ""),
    (ROOT, "Disease", "Condition", "SNOMED", "S", "64572001", ""),
    (100, "Heart disease", "Condition", "SNOMED", "S", "56265001", ""),
    (200, "Lung disease", "Condition", "SNOMED", "S", "19829001", ""),
    (101, "Ischemic heart disease", "Condition", "SNOMED", "S", "414545008", ""),
    (102, 'Acute "MI', "Condition", "SNOMED", "S", "57054005", ""),
    (103, "Acute MI, inferior wall", "Condition", "SNOMED", "S", "73795002", ""),
    (104, "Two-parent disorder", "Condition", "SNOMED", "S", "1000004", ""),
    (600, "Cardiac observation", "Observation", "SNOMED", "S", "1000600", ""),
    (105, "Orphan disorder", "Condition", "SNOMED", "S", "1000005", ""),
    (300, "Fever", "Condition", "SNOMED", "S", "386661006", ""),
    (400, "Non-standard disorder", "Condition", "SNOMED", "", "1000400", ""),
    (401, "Deprecated disorder", "Condition", "SNOMED", "", "1000401", "D"),
    (9001, "NSTEMI", "Condition", "ICD10CM", "", "I21.4", ""),
    (9002, "CAD", "Condition", "ICD10CM", "", "I25.10", ""),
    (9003, "Fever, unspecified", "Condition", "ICD10CM", "", "R50.9", ""),
    (9004, "Other dependence", "Observation", "ICD10CM", "", "Z99.89", ""),
    (9005, "Type 2 MI", "Condition", "ICD10CM", "", "I21.A1", ""),
    (9006, "Old MI code", "Condition", "ICD10CM", "", "I21.9", "D"),
    (8001, "AMI inferior", "Condition", "ICD9CM", "", "410.71", ""),
    (8002, "Fall on stairs", "Observation", "ICD9CM", "", "E880.9", ""),
    (8003, "CABG status", "Observation", "ICD9CM", "", "V45.81", ""),
    (8004, "CHF", "Condition", "ICD9CM", "", "428.0", ""),
]

# concept_id_1, concept_id_2, relationship_id, invalid_reason
RELATIONSHIPS = [
    (ROOT, 700, "Is a", ""),
    (100, ROOT, "Is a", ""),
    (200, ROOT, "Is a", ""),
    (101, 100, "Is a", ""),
    (102, 101, "Is a", ""),
    (103, 102, "Is a", ""),
    (104, 103, "Is a", ""),
    (104, 100, "Is a", ""),
    (600, 100, "Is a", ""),
    (105, 600, "Is a", ""),
    (300, 700, "Is a", ""),
    (400, 100, "Is a", ""),
    (401, 100, "Is a", ""),
    (102, 200, "Is a", "D"),  # deprecated link: must be ignored
    (100, 101, "Subsumes", ""),  # other relationship types are ignored
    (9001, 102, "Maps to", ""),
    (9001, 101, "Maps to", "D"),  # deprecated mapping: must be ignored
    (9002, 101, "Maps to", ""),
    (9002, 104, "Maps to", ""),
    (9003, 300, "Maps to", ""),
    (9005, 103, "Maps to", ""),
    (9006, 102, "Maps to", ""),
    (8001, 102, "Maps to", ""),
    (8002, 104, "Maps to", ""),
    (8003, 101, "Maps to", ""),
    (8004, 100, "Maps to", ""),
]


def _concept_ancestor_rows():
    """Derives CONCEPT_ANCESTOR the way Athena does: valid standard concepts,
    valid "Is a" links, shortest and longest path lengths, self rows at 0."""
    standard = {
        row[0] for row in CONCEPTS if row[4] == "S" and row[6] == ""
    }
    parents = defaultdict(set)
    for child, parent, relation, invalid in RELATIONSHIPS:
        if relation == "Is a" and not invalid and {child, parent} <= standard:
            parents[child].add(parent)

    def levels(concept):
        found = defaultdict(list)
        stack = [(concept, 0)]
        while stack:
            node, distance = stack.pop()
            found[node].append(distance)
            for parent in parents[node]:
                stack.append((parent, distance + 1))
        return found

    rows = []
    for concept in sorted(standard):
        for ancestor, distances in levels(concept).items():
            rows.append((ancestor, concept, min(distances), max(distances)))
    return rows


def _write(path, header, rows):
    lines = ["\t".join(header)]
    lines += ["\t".join(str(value) for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_athena_bundle(folder):
    """Writes the synthetic bundle in Athena's tab-separated layout."""
    folder = Path(folder)
    _write(
        folder / "CONCEPT.csv",
        [
            "concept_id", "concept_name", "domain_id", "vocabulary_id",
            "concept_class_id", "standard_concept", "concept_code",
            "valid_start_date", "valid_end_date", "invalid_reason",
        ],
        [
            (cid, name, domain, vocab, "Clinical Finding", std, code,
             "19700101", "20991231", invalid)
            for cid, name, domain, vocab, std, code, invalid in CONCEPTS
        ],
    )
    _write(
        folder / "CONCEPT_RELATIONSHIP.csv",
        [
            "concept_id_1", "concept_id_2", "relationship_id",
            "valid_start_date", "valid_end_date", "invalid_reason",
        ],
        [(a, b, rel, "19700101", "20991231", inv)
         for a, b, rel, inv in RELATIONSHIPS],
    )
    _write(
        folder / "CONCEPT_ANCESTOR.csv",
        [
            "ancestor_concept_id", "descendant_concept_id",
            "min_levels_of_separation", "max_levels_of_separation",
        ],
        _concept_ancestor_rows(),
    )


class _AthenaFixture(unittest.TestCase):
    """Builds the bundle once per test class; holds no tests itself."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        write_athena_bundle(cls.tmp)
        cls.vocab = OMOPVocabulary(cls.tmp)
        cls.hierarchy = cls.vocab.condition_hierarchy(max_depth=3)
        cls.mapper = cls.vocab.icd_mapper()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)


class TestOMOPVocabulary(_AthenaFixture):
    def test_missing_folder(self):
        with self.assertRaises(FileNotFoundError):
            OMOPVocabulary(Path(self.tmp) / "nope")

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            write_athena_bundle(folder)
            (Path(folder) / "CONCEPT_ANCESTOR.csv").unlink()
            with self.assertRaises(FileNotFoundError):
                OMOPVocabulary(folder)

    def test_comma_separated_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            write_athena_bundle(folder)
            path = Path(folder) / "CONCEPT.csv"
            path.write_text(path.read_text().replace("\t", ","))
            with self.assertRaisesRegex(ValueError, "tab-separated"):
                OMOPVocabulary(folder)

    def test_scan_reads_ids_as_integers_and_keeps_quotes(self):
        concept = self.vocab.scan("CONCEPT").collect()
        self.assertEqual(concept.schema["concept_id"], pl.Int64)
        names = dict(concept.select("concept_id", "concept_name").iter_rows())
        self.assertEqual(names[102], 'Acute "MI')

    def test_unknown_table(self):
        with self.assertRaises(ValueError):
            self.vocab.scan("DRUG_STRENGTH")


class TestConditionHierarchy(_AthenaFixture):
    def test_nodes_respect_depth_domain_and_validity(self):
        self.assertEqual(
            set(self.hierarchy.nodes["concept_id"]),
            {ROOT, 100, 200, 101, 102, 104, 105},
        )
        for excluded in (103, 600, 300, 400, 401, 700):
            self.assertNotIn(excluded, self.hierarchy)

    def test_depths(self):
        self.assertEqual(self.hierarchy.depth(ROOT), 0)
        self.assertEqual(self.hierarchy.depth(102), 3)
        # two parents: shortest path (via 100) is 2, longest (via 103) is 5
        row = self.hierarchy.nodes.filter(pl.col("concept_id") == 104).row(
            0, named=True
        )
        self.assertEqual((row["depth"], row["min_depth"], row["max_depth"]),
                         (2, 2, 5))

    def test_max_depth_measure(self):
        deepest = self.vocab.condition_hierarchy(max_depth=3, depth_measure="max")
        self.assertNotIn(104, deepest)
        self.assertIn(102, deepest)

    def test_no_depth_limit(self):
        full = self.vocab.condition_hierarchy(max_depth=None)
        self.assertIn(103, full)
        self.assertEqual(full.parents(104), [100, 103])

    def test_edges_only_between_kept_concepts(self):
        edges = set(
            self.hierarchy.edges.filter(~pl.col("rescued"))
            .select("parent_id", "child_id")
            .iter_rows()
        )
        self.assertEqual(
            edges, {(ROOT, 100), (ROOT, 200), (100, 101), (101, 102), (100, 104)}
        )
        # the deprecated 102 -> 200 link is not an edge
        self.assertEqual(self.hierarchy.parents(102), [101])

    def test_orphan_is_attached_to_nearest_kept_ancestor(self):
        self.assertEqual(self.hierarchy.parents(105), [100])
        rescued = self.hierarchy.edges.filter(pl.col("rescued"))
        self.assertEqual(list(rescued.select("parent_id", "child_id").iter_rows()),
                         [(100, 105)])

    def test_orphans_left_alone_when_asked(self):
        loose = self.vocab.condition_hierarchy(max_depth=3, connect_orphans=False)
        self.assertEqual(loose.parents(105), [])
        self.assertFalse(loose.edges["rescued"].any())

    def test_retained_ancestors_of_deep_and_multi_parent_concepts(self):
        # 103 is below the cutoff; roll-up needs its kept ancestors
        self.assertEqual(
            self.hierarchy.retained_ancestors(103),
            [(102, 1), (101, 2), (100, 3), (ROOT, 4)],
        )
        self.assertEqual(
            self.hierarchy.retained_ancestors(104),
            [(100, 1), (102, 2), (ROOT, 2), (101, 3)],
        )
        self.assertEqual(self.hierarchy.retained_ancestors(300), [])

    def test_lookups(self):
        self.assertEqual(self.hierarchy.name(ROOT), "Disease")
        self.assertEqual(self.hierarchy.children(ROOT), [100, 200])
        with self.assertRaises(KeyError):
            self.hierarchy.depth(103)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            self.vocab.condition_hierarchy(depth_measure="mean")
        with self.assertRaises(ValueError):
            self.vocab.condition_hierarchy(max_depth=-1)
        with self.assertRaises(ValueError):
            self.vocab.condition_hierarchy(root_concept_id=600)

    def test_networkx_and_triples(self):
        graph = self.hierarchy.to_networkx()
        self.assertEqual(graph.number_of_nodes(), len(self.hierarchy))
        self.assertEqual(graph.nodes[102]["depth"], 3)
        self.assertTrue(graph.edges[100, 105]["rescued"])
        triples = self.hierarchy.to_triples()
        self.assertIn(("101", "is_a", "100"), triples)
        self.assertEqual(len(triples), self.hierarchy.edges.height)

    def test_restrict_matches_direct_build(self):
        full = self.vocab.condition_hierarchy(max_depth=None)
        for depth, measure in [(3, "min"), (2, "min"), (3, "max"), (0, "min")]:
            direct = self.vocab.condition_hierarchy(
                max_depth=depth, depth_measure=measure
            )
            cut = full.restrict(depth, measure)
            with self.subTest(depth=depth, measure=measure):
                self.assertTrue(cut.nodes.equals(direct.nodes))
                self.assertTrue(cut.edges.equals(direct.edges))
                self.assertTrue(
                    cut.ancestors.sort(cut.ancestors.columns).equals(
                        direct.ancestors.sort(direct.ancestors.columns)
                    )
                )
                self.assertEqual((cut.max_depth, cut.depth_measure),
                                 (depth, measure))

    def test_restrict_refuses_to_add_concepts(self):
        with self.assertRaises(ValueError):
            self.hierarchy.restrict(4)
        by_max = self.vocab.condition_hierarchy(max_depth=3, depth_measure="max")
        with self.assertRaises(ValueError):
            by_max.restrict(3, "min")
        self.assertEqual(len(self.hierarchy.restrict(3, "max")),
                         len(by_max))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            self.hierarchy.save(folder)
            loaded = ConditionHierarchy.load(folder)
        self.assertEqual(loaded.max_depth, 3)
        self.assertEqual(loaded.depth_measure, "min")
        self.assertTrue(loaded.nodes.equals(self.hierarchy.nodes))
        self.assertEqual(loaded.retained_ancestors(103),
                         self.hierarchy.retained_ancestors(103))


class TestICDToSNOMEDMapper(_AthenaFixture):
    def test_normalize(self):
        self.assertEqual(normalize_icd_code(" i21.4 "), "I214")
        self.assertEqual(normalize_icd_code("E880.9"), "E8809")

    def test_map_mimic_and_dotted_codes(self):
        self.assertEqual(self.mapper.map("I214", 10), [102])
        self.assertEqual(self.mapper.map("I21.4", "10"), [102])
        self.assertEqual(self.mapper.map("41071", 9), [102])
        self.assertEqual(self.mapper.map("E8809", 9), [104])
        self.assertEqual(self.mapper.map("V4581", 9), [101])

    def test_one_code_several_concepts(self):
        self.assertEqual(self.mapper.map("I2510", 10), [101, 104])

    def test_unmapped_unknown_and_deprecated(self):
        self.assertEqual(self.mapper.map("Z9989", 10), [])  # no "Maps to"
        self.assertIn((10, "Z9989"), self.mapper)
        self.assertEqual(self.mapper.map("XYZ", 10), [])  # not in Athena
        self.assertNotIn((10, "XYZ"), self.mapper)
        # deprecated source code still maps through a valid link
        self.assertEqual(self.mapper.map("I219", 10), [102])

    def test_versions_are_separate(self):
        self.assertEqual(self.mapper.map("I214", 9), [])
        with self.assertRaises(ValueError):
            self.mapper.map("I214", 11)
        icd10_only = self.vocab.icd_mapper(versions=[10])
        self.assertEqual(icd10_only.map("41071", 9), [])

    def test_map_frame(self):
        diagnoses = pl.DataFrame(
            {
                "subject_id": [1, 1, 2],
                "icd_code": ["I214", "I2510", "Z9989"],
                "icd_version": [10, 10, 10],
            }
        )
        mapped = self.mapper.map_frame(diagnoses)
        self.assertEqual(mapped.height, 4)  # I2510 maps to two concepts
        self.assertEqual(
            sorted(mapped.filter(pl.col("icd_code") == "I2510")["concept_id"]),
            [101, 104],
        )
        self.assertIsNone(
            mapped.filter(pl.col("icd_code") == "Z9989")["concept_id"][0]
        )
        self.assertEqual(mapped.columns[:3], diagnoses.columns)

    def test_coverage(self):
        diagnoses = pl.DataFrame(
            {
                "icd_code": ["I214", "I214", "I2510", "R509", "Z9989", "XYZ",
                             "41071", "E8809"],
                "icd_version": [10, 10, 10, 10, 10, 10, 9, 9],
            }
        )
        self.assertEqual(
            self.mapper.coverage(diagnoses, hierarchy=self.hierarchy),
            {
                "codes": 7,
                "unknown": 1,
                "no_mapping": 1,
                "mapped": 5,
                "in_hierarchy": 4,
                "outside_hierarchy": 1,  # R50.9 -> Fever, not a disease
            },
        )

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "icd_to_snomed.parquet"
            self.mapper.save(path)
            loaded = ICDToSNOMEDMapper.load(path)
        self.assertEqual(loaded.map("I2510", 10), [101, 104])


if __name__ == "__main__":
    unittest.main()
