"""A small synthetic Athena bundle shared by the OMOP vocabulary tests.

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

import csv
import shutil
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from pyhealth.medcode import DISEASE_CONCEPT_ID, OMOPVocabulary

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


def _write(path, header, rows, separator="\t"):
    if separator == "\t":  # Athena style: no quoting at all
        lines = ["\t".join(header)]
        lines += ["\t".join(str(value) for value in row) for row in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=separator, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_athena_bundle(folder, separator="\t", file_name="{table}.csv"):
    """Writes the synthetic bundle, by default in Athena's layout.

    Args:
        folder: Where to write.
        separator: Column separator; anything but a tab is written with
            standard CSV quoting.
        file_name: File name pattern containing ``{table}``.
    """
    folder = Path(folder)

    def path(table):
        return folder / file_name.format(table=table)

    _write(
        path("CONCEPT"),
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
        separator,
    )
    _write(
        path("CONCEPT_RELATIONSHIP"),
        [
            "concept_id_1", "concept_id_2", "relationship_id",
            "valid_start_date", "valid_end_date", "invalid_reason",
        ],
        [(a, b, rel, "19700101", "20991231", inv)
         for a, b, rel, inv in RELATIONSHIPS],
        separator,
    )
    _write(
        path("CONCEPT_ANCESTOR"),
        [
            "ancestor_concept_id", "descendant_concept_id",
            "min_levels_of_separation", "max_levels_of_separation",
        ],
        _concept_ancestor_rows(),
        separator,
    )


class AthenaFixture(unittest.TestCase):
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
