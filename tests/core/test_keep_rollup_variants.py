"""Oracles for the agreed baseline and isolated ancestor-selection variants."""

import copy
import itertools
import math
import unittest

from examples.keep_ancestor_rollup_ablation_utils import (
    validate_matrix,
)
from examples.keep_ancestor_rollup_variants import (
    ARMS,
    ROOT,
    SingleParentRollup,
    compare_variants,
    npmi,
    pair_statistics,
    run_study,
)
from pyhealth.graph import KnowledgeGraph
from pyhealth.processors import AncestorRollupProcessor


class TestRollupVariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_study()

    def test_public_counts_match_independent_oracle_and_expected_effects(self):
        report = self.report
        self.assertTrue(report["public_counter_agreement"])
        self.assertTrue(report["hand_oracle_passed"])
        self.assertEqual(report["settings"]["max_depth"], 5)
        self.assertEqual(report["settings"]["path_source"], "direct")
        expected = {
            "dense": (45, 136, 5),
            "single_parent": (1, 2, 5),
            "root_inclusive": (55, 166, 6),
        }
        for arm in ARMS:
            row = report["aggregates"][arm]
            self.assertEqual(
                (
                    row["observed_pairs"],
                    row["pair_mass"],
                    row["patients_with_present_targets"],
                ),
                expected[arm],
            )
            self.assertEqual(row["patients"], 10)
            self.assertEqual(row["comparison_axes"], 12)
            self.assertEqual(row["possible_pairs"], 66)
        self.assertEqual(report["aggregates"]["dense"]["eligible_targets"], 11)
        self.assertEqual(report["aggregates"]["root_inclusive"]["eligible_targets"], 12)

    def test_root_inclusion_preserves_nonroot_counts_and_retained_identity(self):
        matrices = self.report["matrices"]
        before, after = (
            validate_matrix(matrices[a]) for a in ("dense", "root_inclusive")
        )
        nonroot = [c for c in matrices["dense"]["concept_ids"] if c != ROOT]
        for pair in itertools.product(nonroot, repeat=2):
            self.assertEqual(before.get(pair, 0), after.get(pair, 0))
        self.assertEqual(after[ROOT, ROOT], 4)
        self.assertEqual(after[ROOT, "105"], 3)
        self.assertFalse(any(ROOT in pair for pair in before))
        self.assertEqual(
            self.report["targets"]["root_inclusive"][2]["targets"], ("105",)
        )
        self.assertEqual(
            self.report["comparisons"]["root_inclusive"]["summary"][
                "nonroot_pair_absolute_delta"
            ],
            0,
        )

    def test_duplicate_single_row_and_subthreshold_root_controls(self):
        for arm, traces in self.report["traces"].items():
            by_patient = {row["patient_id"]: row for row in traces}
            for patient in ("p4", "p6", "p7", "p8"):
                self.assertEqual(by_patient[patient]["present"], [])
            self.assertEqual(by_patient["p10"]["present"], ["105"])
            if arm == "root_inclusive":
                self.assertEqual(by_patient["p10"]["occurrences"][ROOT], 1)

    def test_single_parent_is_exactly_one_per_successful_source(self):
        for result in self.report["targets"]["single_parent"]:
            self.assertLessEqual(len(result["targets"]), 1)
        self.assertEqual(
            self.report["targets"]["single_parent"][0]["targets"], ("105",)
        )
        self.assertEqual(
            self.report["targets"]["single_parent"][1]["targets"], ("105",)
        )
        # Original failures retain their typed reasons.
        self.assertEqual(
            self.report["targets"]["single_parent"][3]["reason"], "no_target"
        )
        self.assertEqual(
            self.report["targets"]["single_parent"][4]["reason"], "unknown_code"
        )

    def test_shortest_distance_then_numeric_tie_break(self):
        def mapper(edges):
            graph = KnowledgeGraph(triples=[(p, "is_a", c) for p, c in edges])
            dense = AncestorRollupProcessor(
                knowledge_graph=graph,
                domain_by_code={c: "Condition" for c in graph.entity2id},
                hierarchy_relation="is_a",
                root="1",
                max_depth=1,
                target_policy="all_retained",
                include_root=False,
                root_depths={
                    c: depth
                    for c, depth in {"1": 0, "9": 1, "10": 1, "98": 2, "99": 3}.items()
                    if c in graph.entity2id
                },
            )
            return SingleParentRollup(dense)

        tied = [("1", "9"), ("1", "10"), ("9", "99"), ("10", "99")]
        self.assertEqual(mapper(tied).rollup_codes(["99"])[0].targets, ("9",))
        self.assertEqual(
            mapper(list(reversed(tied))).rollup_codes(["99"])[0].targets, ("9",)
        )
        unequal = [("1", "9"), ("1", "10"), ("9", "98"), ("98", "99"), ("10", "99")]
        self.assertEqual(mapper(unequal).rollup_codes(["99"])[0].targets, ("10",))


class TestRollupStatistics(unittest.TestCase):
    def test_npmi_and_degenerate_cases(self):
        self.assertAlmostEqual(npmi(3, 4, 5, 10), math.log(1.5) / -math.log(0.3))
        self.assertAlmostEqual(npmi(1, 5, 5, 10), math.log(0.4) / -math.log(0.1))
        self.assertIsNone(npmi(0, 4, 5, 10))
        self.assertIsNone(npmi(10, 10, 10, 10))

    def test_entropy_density_and_global_distribution_jsd(self):
        before = {
            "patients": 10,
            "concept_ids": ["a", "b", "c"],
            "entries": [
                ["a", "a", 4],
                ["b", "b", 2],
                ["c", "c", 2],
                ["a", "b", 2],
                ["b", "a", 2],
                ["a", "c", 2],
                ["c", "a", 2],
            ],
        }
        stats = pair_statistics(before)
        self.assertEqual(stats["entropy_bits"], 1)
        self.assertEqual(stats["normalized_entropy"], 1 / math.log2(3))
        self.assertEqual(stats["pair_density"], 2 / 3)
        self.assertEqual(
            compare_variants(before, before)["summary"]["global_pair_mass_jsd"], 0
        )
        empty = copy.deepcopy(before)
        empty["entries"] = []
        self.assertIsNone(pair_statistics(empty)["entropy_bits"])
        self.assertIsNone(
            compare_variants(empty, empty)["summary"]["changed_pair_fraction"]
        )
        self.assertIsNone(
            compare_variants(empty, before)["summary"]["global_pair_mass_jsd"]
        )


if __name__ == "__main__":
    unittest.main()
