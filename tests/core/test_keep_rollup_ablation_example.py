"""Independent numerical and identity checks for the synthetic KEEP example."""

import copy
import itertools
import math
import unittest

from examples.keep_ancestor_rollup_ablation_utils import (
    ARMS,
    AXES,
    arm_name,
    assert_fixture,
    compare_matrices,
    cosine,
    count_histories,
    jsd,
    make_processors,
    one_factor_contrasts,
    synthetic_histories,
    trace_pair,
    validate_matrix,
)


def small_matrix(pair):
    return {
        "concept_ids": ["a", "b", "z"],
        "patients": 10,
        "entries": [["a", "a", 4], ["b", "b", 5]]
        + ([["a", "b", pair], ["b", "a", pair]] if pair else []),
    }


class TestRelationshipMeasures(unittest.TestCase):
    def test_hand_derived_scores_and_zero_conventions(self):
        report = compare_matrices(small_matrix(0), small_matrix(3), x_max=4, alpha=1)
        row = report["pairs"][0]
        self.assertEqual(row["count_delta"], 3)
        self.assertEqual(row["status"], "gained")
        self.assertEqual(row["after"]["glove_weight"], 0.75)
        self.assertAlmostEqual(row["after"]["log_count"], math.log(3))
        self.assertAlmostEqual(row["after"]["ppmi"], math.log(1.5))
        self.assertEqual(row["after"]["p_j_given_i"], 0.75)
        self.assertEqual(row["after"]["p_i_given_j"], 0.6)
        self.assertIsNone(row["before"]["log_count"])
        self.assertIsNone(row["log_count_delta"])
        self.assertEqual(row["before"]["ppmi"], 0)
        self.assertIsNone(report["profiles"][2]["count_cosine"])
        self.assertIsNone(report["concentration"]["before"]["hhi"])
        self.assertEqual(report["concentration"]["after"]["hhi"], 1)
        self.assertEqual(report["concentration"]["after"]["possible_pairs"], 3)

    def test_similarity_and_concentration_have_distinct_meanings(self):
        self.assertEqual(cosine({"a": 1}, {"a": 10}), 1)
        self.assertEqual(jsd({"a": 1}, {"a": 10}), 0)
        self.assertEqual(jsd({"a": 1}, {"b": 1}), 1)
        self.assertEqual(cosine({"a": 1}, {"b": 1}), 0)
        self.assertIsNone(jsd({}, {"a": 1}))
        report = compare_matrices(small_matrix(1), small_matrix(2))
        self.assertEqual(report["profiles"][0]["count_cosine"], 1)
        self.assertEqual(report["profiles"][0]["count_jsd"], 0)
        self.assertEqual(report["pairs"][0]["count_delta"], 1)

    def test_axis_permutation_and_corruption(self):
        original = small_matrix(2)
        reordered = copy.deepcopy(original)
        reordered["concept_ids"].reverse()
        self.assertEqual(
            compare_matrices(original, reordered)["summary"]["absolute_count_delta"], 0
        )
        reordered["entries"].pop()
        with self.assertRaisesRegex(ValueError, "symmetric"):
            validate_matrix(reordered)
        impossible = {
            "concept_ids": ["a", "b"],
            "patients": 3,
            "entries": [["a", "a", 2], ["b", "b", 2]],
        }
        with self.assertRaisesRegex(ValueError, "intersection lower bound"):
            validate_matrix(impossible)
        for field, value, message in (
            ("patients", 11, "cohort sizes"),
            ("concept_ids", ["a", "b", "q"], "known IDs|universes"),
        ):
            changed = copy.deepcopy(original)
            changed[field] = value
            with self.assertRaisesRegex(ValueError, message):
                compare_matrices(original, changed)
        for value in (0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                compare_matrices(original, original, x_max=value)


class TestPolicyExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processors = make_processors()
        cls.histories = synthetic_histories()
        cls.matrices, cls.traces = {}, {}
        for name, processor in cls.processors.items():
            cls.matrices[name], cls.traces[name] = count_histories(
                processor, cls.histories
            )

    def test_all_eight_arms_against_hand_derived_presence(self):
        assert_fixture(self.processors, self.matrices, self.traces)
        self.assertEqual(len(self.matrices), 8)
        self.assertEqual(len(list(one_factor_contrasts())), 12)
        for _, before, after in one_factor_contrasts():
            a, b = before.split("|"), after.split("|")
            self.assertEqual(sum(x != y for x, y in zip(a, b)), 1)

    def test_trace_reconciles_every_changed_pair_in_every_contrast(self):
        for _, before, after in one_factor_contrasts():
            report = compare_matrices(self.matrices[before], self.matrices[after])
            for row in report["pairs"]:
                witnesses = trace_pair(
                    self.traces[before], self.traces[after], row["i"], row["j"]
                )
                self.assertEqual(
                    sum(w["pair_delta"] for w in witnesses), row["count_delta"]
                )
            for profile in report["profiles"]:
                if profile["concept_id"] == "Z":
                    self.assertIsNone(profile["count_cosine"])
                    self.assertIsNone(profile["ppmi_jsd"])

    def test_permutations_copy_identity_and_conflicts(self):
        for arm in ARMS:
            name = arm_name(arm)
            reordered = copy.deepcopy(self.histories)
            reordered.reverse()
            for history in reordered:
                history["diagnoses"].reverse()
            matrix, _ = count_histories(self.processors[name], reordered)
            self.assertEqual(matrix, self.matrices[name])
        processor = next(iter(self.processors.values()))
        bad = copy.deepcopy(self.histories)
        bad[3]["diagnoses"][1]["codes"] = ["O"]
        with self.assertRaisesRegex(ValueError, "conflicting copies"):
            count_histories(processor, bad)
        with self.assertRaisesRegex(ValueError, "unique"):
            count_histories(processor, self.histories + [self.histories[0]])

    def test_unchanged_axes_and_zero_rows(self):
        baseline = self.matrices[arm_name(ARMS[0])]
        values = validate_matrix(baseline)
        # Two distinct X diagnoses establish A2/B2 for p1 and p2;
        # retained A2/B2 diagnoses also establish that pair for p5.
        self.assertEqual(values["A2", "B2"], 3)
        self.assertEqual(values["A2", "A2"], 4)
        self.assertEqual(values["B2", "B2"], 4)
        self.assertEqual(values["C2", "C2"], 2)
        self.assertEqual(baseline["concept_ids"], list(AXES))
        for i, j in itertools.product(AXES, repeat=2):
            self.assertEqual(values.get((i, j), 0), values.get((j, i), 0))


if __name__ == "__main__":
    unittest.main()
