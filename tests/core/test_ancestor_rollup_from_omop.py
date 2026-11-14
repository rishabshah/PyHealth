"""Tests for AncestorRollupProcessor.from_omop on a synthetic Athena bundle.

The bundle follows the layout of ``test_omop_vocabulary.py`` (tab-separated
``.csv`` files) and adds the row-level signature of the Athena discrepancy
recorded in the standalone rehearsal (C-0008): concept ``1075201`` has a
standard ancestor ``140708`` at closure separation exactly one, while its
only "Is a" route runs through the valid-but-nonstandard ``4105351``,
which no filtered hierarchy can keep. The synthetic ``CONCEPT_ANCESTOR``
therefore contains stitched closure rows that the "Is a" adjacency cannot
reproduce, which is exactly what ``path_source="closure"`` compensates for.

Hierarchy ("Is a", child -> parent)::

    4274025 Disease (root)
     |-- 100 Heart disease
     |    |-- 101 Ischemic heart disease
     |    |    `-- 102 Acute MI
     |    |         `-- 103 Acute MI, inferior wall
     |    |              `-- 104 (also directly under 100)
     |    |-- 600 Cardiac observation (Observation domain)
     |    |    `-- 105 Orphan disorder (depth 3: retained identity)
     |    `-- [610 Deep observation (Observation)] -- 106 Deep orphan (depth 5)
     |-- 111 Spinal disorder
     |    `-- 222 Cord disorder
     |         `-- 333 Cord compression
     |              `-- 140708 Myelopathy due to another disorder
     |                   `-- [4105351 (nonstandard)] -- 1075201 (stitched)
    300 Fever -> 700 Clinical finding (not under the root)

The cutoff used by the processors under test is 3, so 102 is retained,
103 (depth 4) must still roll up through the uncut subtree, and 1075201
(depth 5) has no surviving direct path to a retained target.
"""

import pickle
import shutil
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import torch

from pyhealth.graph import KnowledgeGraph
from pyhealth.medcode import DISEASE_CONCEPT_ID, OMOPVocabulary
from pyhealth.processors import AncestorRollupProcessor

ROOT = DISEASE_CONCEPT_ID

# concept_id, name, domain, vocabulary, standard, code, invalid_reason
CONCEPTS = [
    (ROOT, "Disease", "Condition", "SNOMED", "S", "64572001", ""),
    (100, "Heart disease", "Condition", "SNOMED", "S", "56265001", ""),
    (101, "Ischemic heart disease", "Condition", "SNOMED", "S", "414545008", ""),
    (102, "Acute MI", "Condition", "SNOMED", "S", "57054005", ""),
    (103, "Acute MI, inferior wall", "Condition", "SNOMED", "S", "73795002", ""),
    (104, "Two-parent disorder", "Condition", "SNOMED", "S", "1000004", ""),
    (600, "Cardiac observation", "Observation", "SNOMED", "S", "1000600", ""),
    (105, "Orphan disorder", "Condition", "SNOMED", "S", "1000005", ""),
    (106, "Deep orphan disorder", "Condition", "SNOMED", "S", "1000106", ""),
    (610, "Deep observation", "Observation", "SNOMED", "S", "1000610", ""),
    (300, "Fever", "Condition", "SNOMED", "S", "386661006", ""),
    (700, "Clinical finding", "Condition", "SNOMED", "S", "404684003", ""),
    (111, "Spinal disorder", "Condition", "SNOMED", "S", "1000111", ""),
    (222, "Cord disorder", "Condition", "SNOMED", "S", "1000222", ""),
    (333, "Cord compression", "Condition", "SNOMED", "S", "1000333", ""),
    (140708, "Myelopathy due to another disorder", "Condition", "SNOMED",
     "S", "100140708", ""),
    (4105351, "Myelopathy due to neoplastic disease", "Condition", "SNOMED",
     "", "1004105351", ""),
    (1075201, "Myelopathy, unspecified", "Condition", "SNOMED", "S",
     "1001075201", ""),
    (9002, "CAD", "Condition", "ICD10CM", "", "I25.10", ""),
    (9005, "Type 2 MI", "Condition", "ICD10CM", "", "I21.A1", ""),
    (9007, "Convergent dx", "Condition", "ICD10CM", "", "I24.0", ""),
]

# concept_id_1, concept_id_2, relationship_id, invalid_reason
RELATIONSHIPS = [
    (100, ROOT, "Is a", ""),
    (101, 100, "Is a", ""),
    (102, 101, "Is a", ""),
    (103, 102, "Is a", ""),
    (104, 103, "Is a", ""),
    (104, 100, "Is a", ""),
    (600, 100, "Is a", ""),
    (105, 600, "Is a", ""),
    (106, 610, "Is a", ""),
    (610, 102, "Is a", ""),
    (300, 700, "Is a", ""),
    (111, ROOT, "Is a", ""),
    (222, 111, "Is a", ""),
    (333, 222, "Is a", ""),
    (140708, 333, "Is a", ""),
    (4105351, 140708, "Is a", ""),
    (1075201, 4105351, "Is a", ""),
    (9002, 101, "Maps to", ""),
    (9002, 104, "Maps to", ""),
    (9005, 103, "Maps to", ""),
    (9007, 102, "Maps to", ""),
    (9007, 103, "Maps to", ""),
]

#: Closure rows Athena ships but the standard "Is a" adjacency cannot
#: reproduce (the C-0008 finding): the standard-stitched ancestry of
#: 1075201, including the separation-one pair (140708, 1075201).
STITCHED_ANCESTOR_ROWS = [
    (ROOT, 1075201, 5, 5),
    (111, 1075201, 4, 4),
    (222, 1075201, 3, 3),
    (333, 1075201, 2, 2),
    (140708, 1075201, 1, 1),
]


def _concept_ancestor_rows():
    """Derives CONCEPT_ANCESTOR the way Athena does (standard "Is a" links,
    shortest and longest path lengths, self rows at zero), then appends the
    stitched rows above."""
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
    return rows + STITCHED_ANCESTOR_ROWS


def _write(path, header, rows):
    lines = ["\t".join(header)]
    lines += ["\t".join(str(value) for value in row) for row in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_athena_bundle(folder):
    """Writes the synthetic bundle in Athena's tab-separated layout."""
    folder = Path(folder)
    _write(
        folder / "CONCEPT.csv",
        [
            "concept_id", "concept_name", "domain_id", "vocabulary_id",
            "concept_class_id", "standard_concept", "concept_code",
            "invalid_reason",
        ],
        [
            (cid, name, domain, vocab, "Clinical Finding", std, code, invalid)
            for cid, name, domain, vocab, std, code, invalid in CONCEPTS
        ],
    )
    _write(
        folder / "CONCEPT_RELATIONSHIP.csv",
        [
            "concept_id_1", "concept_id_2", "relationship_id",
            "invalid_reason",
        ],
        RELATIONSHIPS,
    )
    _write(
        folder / "CONCEPT_ANCESTOR.csv",
        [
            "ancestor_concept_id", "descendant_concept_id",
            "min_levels_of_separation", "max_levels_of_separation",
        ],
        _concept_ancestor_rows(),
    )


def _has_pyg():
    try:
        import torch_geometric  # noqa: F401

        return True
    except ImportError:
        return False


class _AthenaRollupFixture(unittest.TestCase):
    """Builds the bundle and the processors once per test class."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        write_athena_bundle(cls.tmp)
        cls.vocab = OMOPVocabulary(cls.tmp)
        cls.direct = AncestorRollupProcessor.from_omop(
            cls.vocab,
            target_policy="first_retained",
            include_root=False,
            max_depth=3,
            on_unmapped="empty",
            path_source="direct",
        )
        cls.closure = AncestorRollupProcessor.from_omop(
            cls.vocab,
            target_policy="first_retained",
            include_root=False,
            max_depth=3,
            on_unmapped="empty",
            path_source="closure",
        )
        cls.all_retained = AncestorRollupProcessor.from_omop(
            cls.vocab,
            target_policy="all_retained",
            include_root=False,
            max_depth=3,
            on_unmapped="empty",
            path_source="closure",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def targets(self, processor, code):
        return list(processor.rollup_codes([code])[0].targets)

    def reason(self, processor, code):
        return processor.rollup_codes([code])[0].reason


class TestFromOmopClassification(_AthenaRollupFixture):
    """The classification universe keeps the four failure reasons distinct."""

    def test_retained_concepts_map_to_themselves(self):
        self.assertEqual(self.targets(self.direct, "102"), ["102"])
        self.assertEqual(self.reason(self.direct, "102"), "retained")
        # 104 has min depth 2 (via 100) and max depth 5 (via 103); the
        # pinned "min" measure keeps it retained by identity.
        self.assertEqual(self.targets(self.direct, "104"), ["104"])
        self.assertEqual(self.reason(self.direct, "104"), "retained")

    def test_deep_concepts_roll_up_through_the_uncut_subtree(self):
        # 103 sits at depth 4, beyond the cutoff of 3, but the hierarchy is
        # built uncut, so it is traversable and rolls up to 102.
        result = self.direct.rollup_codes(["103"])[0]
        self.assertEqual(list(result.targets), ["102"])
        self.assertEqual(result.reason, "rolled_up")
        self.assertEqual(result.source_depth, 4)

    def test_condition_concept_outside_the_subtree_is_unreachable(self):
        self.assertEqual(self.reason(self.direct, "300"), "unreachable")

    def test_snomed_concept_outside_the_domain(self):
        self.assertEqual(self.reason(self.direct, "600"), "outside_domain")

    def test_nonstandard_and_non_snomed_concepts_are_unknown(self):
        # 4105351 is valid but nonstandard; 9002 is an ICD10CM source
        # concept; 999999 is absent altogether.
        for code in ("4105351", "9002", "999999"):
            self.assertEqual(self.reason(self.direct, code), "unknown_code")

    def test_the_boundary_uses_concept_id_strings(self):
        with self.assertRaises(TypeError):
            self.direct.rollup_codes([102])
        with self.assertRaises(ValueError):
            self.direct.rollup_codes([""])

    def test_default_configuration(self):
        processor = AncestorRollupProcessor.from_omop(
            self.vocab,
            target_policy="first_retained",
            include_root=False,
        )
        self.assertEqual(processor._config.max_depth, 5)
        self.assertEqual(processor._config.on_unmapped, "error")
        self.assertEqual(processor._config.path_source, "direct")
        # 1075201's closure depth is five, so the default roll-up retains it
        # by identity rather than rolling it up.
        self.assertEqual(self.reason(processor, "1075201"), "retained")

    def test_knowledge_graph_keeps_deep_and_isolated_entities(self):
        graph = self.direct.knowledge_graph
        self.assertIsInstance(graph, KnowledgeGraph)
        for code in ("102", "103", "105", "1075201"):
            self.assertEqual(self.direct.id2entity[graph.entity2id[code]], code)
        self.assertNotIn("300", graph.entity2id)

    def test_provenance_records_the_omop_facts(self):
        provenance = dict(self.direct.provenance)
        self.assertEqual(provenance["depth_method"], "supplied_root_depths")
        self.assertEqual(provenance["path_source"], "direct")
        self.assertEqual(provenance["hierarchy_root"], str(ROOT))
        self.assertEqual(provenance["depth_measure"], "min")
        self.assertEqual(provenance["vocabulary_source"],
                         "pyhealth.medcode.OMOPVocabulary")
        closure_provenance = dict(self.closure.provenance)
        self.assertGreater(
            int(closure_provenance["closure_only_edge_count"]), 0
        )
        self.assertGreater(int(closure_provenance["closure_pair_count"]), 0)


class TestPathSourceModes(_AthenaRollupFixture):
    """Direct vs closure on the recorded Athena discrepancy."""

    def test_direct_mode_reports_no_target_for_the_stitched_source(self):
        # 1075201's only "Is a" parent is the nonstandard 4105351, which no
        # filtered hierarchy keeps: no direct edge survives, so no target.
        result = self.direct.rollup_codes(["1075201"])[0]
        self.assertEqual(result.reason, "no_target")
        self.assertEqual(result.targets, ())
        self.assertEqual(result.source_depth, 5)

    def test_closure_mode_returns_the_expected_target(self):
        self.assertEqual(
            self.targets(self.closure, "1075201"), ["333"]
        )
        self.assertEqual(
            self.targets(self.all_retained, "1075201"), ["111", "222", "333"]
        )

    def test_both_modes_keep_the_same_closure_derived_depths(self):
        direct_result = self.direct.rollup_codes(["1075201"])[0]
        closure_result = self.closure.rollup_codes(["1075201"])[0]
        self.assertEqual(direct_result.source_depth, closure_result.source_depth)
        self.assertEqual(
            self.direct.provenance["depth_method"],
            self.closure.provenance["depth_method"],
        )
        self.assertNotEqual(self.direct.fingerprint, self.closure.fingerprint)

    def test_closure_mode_reaches_concepts_the_direct_mode_still_maps(self):
        # every concept that direct mode maps, closure mode maps identically
        for code in ("102", "103", "104", "300", "600"):
            self.assertEqual(
                self.direct.rollup_codes([code])[0],
                self.closure.rollup_codes([code])[0],
            )

    @unittest.skipUnless(_has_pyg(), "torch-geometric not installed")
    def test_exact_projected_edges_and_closure_only_mask(self):
        data = self.closure.process(["1075201"])
        self.assertEqual(
            [self.closure.id2entity[node_id] for node_id in data.node_ids.tolist()],
            ["1075201", "140708", "333"],
        )
        pairs = [
            (
                self.closure.id2entity[int(data.node_ids[data.edge_index[0, j]])],
                self.closure.id2entity[int(data.node_ids[data.edge_index[1, j]])],
            )
            for j in range(data.num_edges)
        ]
        self.assertEqual(
            pairs, [("140708", "1075201"), ("333", "140708")]
        )
        # the stitched separation-one pair is the only closure-only edge
        self.assertEqual(data.closure_only.tolist(), [True, False])
        self.assertEqual(data.seed_mask.tolist(), [True, False, False])
        self.assertEqual(data.target_mask.tolist(), [False, False, True])
        # the direct processor has no closure-only edges at all
        direct_data = self.direct.process(["103"])
        self.assertEqual(direct_data.num_edges, 1)
        self.assertEqual(direct_data.closure_only.tolist(), [False])

    @unittest.skipUnless(_has_pyg(), "torch-geometric not installed")
    def test_projected_graph_of_a_retained_identity_has_no_edges(self):
        data = self.direct.process(["102"])
        self.assertEqual(
            data.node_ids.tolist(), [self.direct.knowledge_graph.entity2id["102"]]
        )
        self.assertEqual(data.num_edges, 0)
        self.assertEqual(data.seed_mask.tolist(), [True])
        self.assertEqual(data.target_mask.tolist(), [True])

    @unittest.skipUnless(_has_pyg(), "torch-geometric not installed")
    def test_icd_to_graph_embeddings_keeps_untrained_nodes(self):
        processor = AncestorRollupProcessor.from_omop(
            self.vocab,
            target_policy="first_retained",
            include_root=False,
            max_depth=3,
            node_features_by_code={"102": torch.tensor([1.0, 2.0])},
        )
        concept_ids = self.vocab.icd_mapper().map("I21A1", version=10)
        data = processor.process([str(concept_id) for concept_id in concept_ids])
        codes = [processor.id2entity[node_id] for node_id in data.node_ids.tolist()]
        self.assertEqual(codes, ["102", "103"])
        self.assertEqual(data.x.tolist(), [[1.0, 2.0], [0.0, 0.0]])
        self.assertNotEqual(processor.fingerprint, self.direct.fingerprint)


class TestRetainedAncestorParity(_AthenaRollupFixture):
    """all_retained agrees with the reader's strict closure lookup."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.hierarchy = cls.vocab.condition_hierarchy(
            max_depth=None, depth_measure="min", connect_orphans=False
        )

    def retained_vocabulary(self):
        return {
            str(concept_id)
            for concept_id in self.hierarchy.nodes["concept_id"]
            if self.hierarchy.depth(concept_id) <= 3
            and concept_id != self.hierarchy.root_concept_id
        }

    def closure_targets(self, concept_id):
        retained = self.retained_vocabulary()
        return sorted(
            {
                str(ancestor)
                for ancestor, _ in self.hierarchy.retained_ancestors(concept_id)
                if str(ancestor) in retained
            }
        )

    def test_parity_for_complete_direct_paths(self):
        # parity is required for deep sources; retained sources use identity
        # (104 is retained at depth 2 and covered by the identity tests).
        self.assertEqual(
            self.targets(self.all_retained, "103"), self.closure_targets(103)
        )

    def test_expected_divergence_when_an_intermediate_is_filtered_out(self):
        # 106's separation-one closure ancestor is the Observation concept
        # 610, which no filtered subtree keeps, so no closure edge of the
        # selected set reaches the ancestors the reader's lookup reports.
        self.assertEqual(
            self.targets(self.all_retained, "106"), []
        )
        self.assertEqual(self.closure_targets(106), ["100", "101", "102"])

    def test_parity_for_the_stitched_source_in_closure_mode(self):
        self.assertEqual(
            self.targets(self.all_retained, "1075201"),
            self.closure_targets(1075201),
        )

    def test_divergence_when_no_direct_path_survives(self):
        # direct mode cannot reach the closure's ancestors at all
        self.assertEqual(
            self.targets(self.direct, "1075201"), []
        )
        self.assertEqual(self.closure_targets(1075201), ["111", "222", "333"])

    def test_eligible_retained_sources_use_identity(self):
        self.assertEqual(
            self.targets(self.all_retained, "102"), ["102"]
        )


class TestOrphanRescueDeferral(_AthenaRollupFixture):
    """Rescue shortcuts are deferred: parentless deep concepts do not map."""

    def test_shallow_orphan_is_retained_by_identity(self):
        # 105's only "Is a" parent is the Observation concept 600, so it has
        # no hierarchy edge, but its closure depth (3) is within the cutoff.
        self.assertEqual(self.reason(self.direct, "105"), "retained")
        self.assertEqual(self.targets(self.direct, "105"), ["105"])

    def test_deep_orphan_has_no_target_in_either_mode(self):
        # 106 (depth 5) is parentless after filtering: its only parent 610
        # is an Observation concept, and its separation-one closure ancestor
        # is not in the subtree either. The reader's rescue shortcut (which
        # would attach it to 102) is deliberately not used by the processor.
        self.assertEqual(self.reason(self.direct, "106"), "no_target")
        self.assertEqual(self.reason(self.closure, "106"), "no_target")

    def test_rescued_edges_are_not_in_the_processor_graph(self):
        self.assertEqual(self.direct.provenance["rescued_edges"],
                         "0 (orphan rescue deferred)")


class TestICDEventMapping(_AthenaRollupFixture):
    """One result per concept occurrence; the task owns event counting."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mapper = cls.vocab.icd_mapper()

    def rollup_event(self, processor, icd_code, version=10):
        results = [
            processor.rollup_codes([str(concept_id)])[0]
            for concept_id in self.mapper.map(icd_code, version=version)
        ]
        event_targets = sorted({t for result in results for t in result.targets})
        return results, event_targets

    def test_one_to_many_mapping_keeps_one_result_per_concept(self):
        results, event_targets = self.rollup_event(self.direct, "I2510")
        self.assertEqual(
            self.mapper.map("I2510", 10), [101, 104]
        )
        self.assertEqual(len(results), 2)
        self.assertEqual([r.source for r in results], ["101", "104"])
        self.assertEqual(event_targets, ["101", "104"])

    def test_converging_targets_count_once_within_an_event(self):
        # I24.0 maps to both 102 (retained identity) and 103 (which rolls up
        # to 102): the event's target set counts 102 once, while the mapping
        # keeps two results so per-event attribution stays possible.
        results, event_targets = self.rollup_event(self.direct, "I240")
        self.assertEqual(self.mapper.map("I240", 10), [102, 103])
        self.assertEqual(len(results), 2)
        self.assertEqual([r.reason for r in results], ["retained", "rolled_up"])
        self.assertEqual(event_targets, ["102"])

    def test_distinct_events_remain_distinct(self):
        events = {
            "event-1": self.rollup_event(self.direct, "I2510")[1],
            "event-2": self.rollup_event(self.direct, "I21A1")[1],
            "event-3": self.rollup_event(self.direct, "I240")[1],
        }
        self.assertEqual(
            events,
            {"event-1": ["101", "104"], "event-2": ["102"], "event-3": ["102"]},
        )

    def test_icd_conversion_stays_explicit(self):
        # the processor consumes concept_id strings; it never sees ICD codes
        result = self.direct.rollup_codes(["I2510"])[0]
        self.assertEqual(result.reason, "unknown_code")
        self.assertEqual(result.targets, ())


class TestFromOmopValidation(_AthenaRollupFixture):
    def test_rejects_non_vocabulary_input(self):
        with self.assertRaises(TypeError):
            AncestorRollupProcessor.from_omop(
                "not-a-vocabulary",
                target_policy="first_retained",
                include_root=False,
            )

    def test_configuration_validated(self):
        with self.assertRaises(ValueError):
            AncestorRollupProcessor.from_omop(
                self.vocab,
                target_policy="nearest",
                include_root=False,
            )
        with self.assertRaises(ValueError):
            AncestorRollupProcessor.from_omop(
                self.vocab,
                target_policy="first_retained",
                include_root=False,
                path_source="rescue",
            )

    def test_pickle_round_trip(self):
        clone = pickle.loads(pickle.dumps(self.closure))
        self.assertEqual(clone.fingerprint, self.closure.fingerprint)
        self.assertEqual(
            clone.rollup_codes(["1075201"]), self.closure.rollup_codes(["1075201"])
        )

    def test_fingerprint_distinguishes_configurations(self):
        shallower = AncestorRollupProcessor.from_omop(
            self.vocab,
            target_policy="first_retained",
            include_root=False,
            max_depth=2,
            on_unmapped="empty",
        )
        self.assertNotEqual(self.direct.fingerprint, shallower.fingerprint)
        self.assertNotEqual(self.direct.fingerprint, self.closure.fingerprint)
        twin = AncestorRollupProcessor.from_omop(
            self.vocab,
            target_policy="first_retained",
            include_root=False,
            max_depth=3,
            on_unmapped="empty",
            path_source="direct",
        )
        self.assertEqual(self.direct.fingerprint, twin.fingerprint)


if __name__ == "__main__":
    unittest.main()
