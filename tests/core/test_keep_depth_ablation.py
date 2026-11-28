"""Tests for the KEEP depth ablation's matrix measures.

Covers examples/keep_depth_ablation_utils.py. Everything is synthetic:
hand-built matrices with hand-computed statistics, fake processors, and the
team's synthetic Athena bundle for one end-to-end check across two depths.
"""

import math
import tempfile
import unittest
from types import SimpleNamespace

import torch

from examples.keep_depth_ablation_utils import (
    RollupRecorder,
    aligned_comparison,
    common_axes,
    entropy_bits,
    gini,
    hhi,
    matrix_arrays,
    matrix_summary,
    normalized_entropy,
    top_share,
)


def _result(concepts, prevalence, pairs, patients):
    """A KeepCooccurrenceResult-shaped object from prevalence and pair counts."""
    index = {c: k for k, c in enumerate(concepts)}
    rows, cols, values = [], [], []
    for concept, count in prevalence.items():
        rows.append(index[concept]), cols.append(index[concept]), values.append(count)
    for (a, b), count in pairs.items():
        for i, j in ((a, b), (b, a)):
            rows.append(index[i]), cols.append(index[j]), values.append(count)
    matrix = torch.sparse_coo_tensor(
        torch.tensor([rows, cols]), torch.tensor(values), (len(concepts),) * 2
    ).coalesce()
    return SimpleNamespace(
        matrix=matrix,
        concept_ids=tuple(concepts),
        patients_processed=patients,
        patients_with_present_targets=patients - 1,
        unique_diagnosis_events=50,
        mapped_events=40,
    )


class TestDistributionMeasures(unittest.TestCase):
    def test_gini(self):
        self.assertEqual(gini([2, 2, 2]), 0.0)
        self.assertAlmostEqual(gini([0, 0, 0, 4]), 0.75)
        self.assertIsNone(gini([]))

    def test_entropy_top_share_hhi(self):
        self.assertAlmostEqual(normalized_entropy([3, 3, 3]), 1.0)
        self.assertLess(normalized_entropy([100, 1, 1]), 0.5)
        self.assertIsNone(normalized_entropy([5]))
        self.assertAlmostEqual(top_share([10] + [1] * 99), 10 / 109)
        self.assertAlmostEqual(hhi([1, 1, 2]), 0.375)


class TestMatrixSummary(unittest.TestCase):
    # N=10 patients; prevalence A=4, B=5, C=2, D=0; pairs AB=2, AC=1
    def setUp(self):
        self.result = _result(
            ["A", "B", "C", "D"],
            {"A": 4, "B": 5, "C": 2},
            {("A", "B"): 2, ("A", "C"): 1},
            10,
        )

    def test_matrix_arrays(self):
        prevalence, i, j, counts = matrix_arrays(self.result.matrix)
        self.assertEqual(prevalence.tolist(), [4, 5, 2, 0])
        self.assertEqual(
            sorted(zip(i.tolist(), j.tolist(), counts.tolist())), [(0, 1, 2), (0, 2, 1)]
        )

    def test_summary_values(self):
        s = matrix_summary(self.result)
        pmi_ab, pmi_ac = math.log(10 * 2 / 20), math.log(10 * 1 / 8)
        self.assertEqual((s["vocabulary"], s["concepts_observed"]), (4, 3))
        self.assertEqual((s["nonzero_pairs"], s["pair_mass"]), (2, 3))
        self.assertAlmostEqual(s["density"], 2 / 3)
        self.assertAlmostEqual(s["mean_prevalence"], 11 / 3)
        self.assertAlmostEqual(s["mean_pmi"], (pmi_ab + pmi_ac) / 2)
        self.assertAlmostEqual(s["mean_ppmi"], (0 + pmi_ac) / 2)
        self.assertAlmostEqual(s["positive_pmi_share"], 0.5)
        npmi_ab, npmi_ac = pmi_ab / -math.log(0.2), pmi_ac / -math.log(0.1)
        self.assertAlmostEqual(s["mean_npmi"], (npmi_ab + npmi_ac) / 2)

    def test_empty_matrix(self):
        empty = _result(["A", "B"], {}, {}, 5)
        s = matrix_summary(empty)
        self.assertEqual((s["concepts_observed"], s["nonzero_pairs"]), (0, 0))
        self.assertIsNone(s["density"])
        self.assertIsNone(s["mean_ppmi"])


def _jsd_reference(left, right):
    """Independent base-2 JSD between two dict profiles (as in #7's helper)."""
    a, b = sum(left.values()), sum(right.values())
    total = 0.0
    for key in left.keys() | right.keys():
        p, q = left.get(key, 0) / a, right.get(key, 0) / b
        m = (p + q) / 2
        total += (
            (p * math.log2(p / m) if p else 0) + (q * math.log2(q / m) if q else 0)
        ) / 2
    return total


def _cosine_reference(left, right):
    dot = sum(v * right.get(k, 0) for k, v in left.items())
    norm = math.sqrt(sum(v * v for v in left.values())) * math.sqrt(
        sum(v * v for v in right.values())
    )
    return dot / norm


def _profiles(pairs, shared):
    """Count partner profiles on the shared axes, diagonal excluded."""
    out = {c: {} for c in shared}
    for (a, b), count in pairs.items():
        if a in out and b in out and count:
            out[a][b] = count
            out[b][a] = count
    return out


class TestAlignedComparison(unittest.TestCase):
    def setUp(self):
        self.shared = ["A", "B", "C", "D"]
        self.prevalence = {"A": 6, "B": 5, "C": 4, "D": 3}
        self.pairs = {("A", "B"): 3, ("A", "C"): 1, ("B", "C"): 2, ("C", "D"): 2}
        # the reference also keeps X, which the other depth does not
        self.reference = _result(
            ["X", "A", "B", "C", "D"], {**self.prevalence, "X": 2}, self.pairs, 10
        )
        # another depth: keeps Y instead of X, A more prevalent, D rarer;
        # A-D gained, B-C and C-D lost, A-Y and B-Y outside the shared axes
        self.here_pairs = {
            ("A", "B"): 4,
            ("A", "C"): 1,
            ("A", "D"): 1,
            ("A", "Y"): 3,
            ("B", "Y"): 2,
        }
        self.here = _result(
            ["A", "B", "C", "D", "Y"],
            {"A": 8, "B": 5, "C": 4, "D": 2, "Y": 5},
            self.here_pairs,
            10,
        )

    def test_common_axes(self):
        self.assertEqual(common_axes([self.here, self.reference]), self.shared)

    def test_identical_matrices(self):
        out = aligned_comparison(self.reference, self.reference, self.shared)
        self.assertEqual(out["concepts_in_both"], 4)
        self.assertEqual((out["pairs_gained"], out["pairs_lost"]), (0, 0))
        self.assertEqual(out["count_abs_change_sum"], 0)
        self.assertAlmostEqual(out["ppmi_spearman"], 1.0)
        self.assertAlmostEqual(out["count_jsd_mean"], 0.0)
        self.assertAlmostEqual(out["count_cosine_mean"], 1.0)
        self.assertAlmostEqual(out["median_prevalence_ratio"], 1.0)
        self.assertEqual(out["excluded_concepts"], 1)  # X has no partners
        self.assertEqual(out["excluded_pair_mass_share"], 0.0)

    def test_coverage_and_magnitude(self):
        out = aligned_comparison(self.here, self.reference, self.shared)
        self.assertEqual(out["axes_shared"], 4)
        self.assertEqual(out["concepts_in_both"], 4)
        self.assertEqual(out["concepts_only_here"], 0)
        self.assertEqual(out["excluded_concepts"], 1)  # Y
        self.assertAlmostEqual(out["excluded_pair_mass_share"], 5 / 11)
        self.assertEqual(out["pairs_matched"], 2)  # A-B, A-C
        self.assertEqual(out["pairs_gained"], 1)  # A-D
        self.assertEqual(out["pairs_lost"], 2)  # B-C, C-D
        self.assertEqual(out["prevalence_abs_change_sum"], 3)  # |8-6| + |2-3|
        self.assertEqual(out["prevalence_change_median"], 0.0)
        self.assertAlmostEqual(out["median_prevalence_ratio"], 1.0)
        self.assertEqual(out["count_abs_change_sum"], 6)  # 1 + 1 + 2 + 2
        self.assertEqual(out["count_changed_pairs"], 4)

    def test_profiles_match_an_independent_calculation(self):
        out = aligned_comparison(self.here, self.reference, self.shared)
        here = _profiles(self.here_pairs, self.shared)
        ref = _profiles(self.pairs, self.shared)
        jsd = sorted(_jsd_reference(here[c], ref[c]) for c in self.shared)
        cos = sorted(_cosine_reference(here[c], ref[c]) for c in self.shared)
        self.assertEqual(out["count_profile_rows_valid"], 4)
        self.assertEqual(out["count_profile_rows_undefined"], 0)
        self.assertAlmostEqual(out["count_jsd_mean"], sum(jsd) / 4)
        self.assertAlmostEqual(out["count_jsd_median"], (jsd[1] + jsd[2]) / 2)
        self.assertAlmostEqual(out["count_cosine_mean"], sum(cos) / 4)
        # every PPMI here is 0, so every PPMI row is empty: undefined, not 0
        self.assertEqual(out["ppmi_profile_rows_valid"], 0)
        self.assertEqual(out["ppmi_profile_rows_undefined"], 4)
        self.assertIsNone(out["ppmi_jsd_mean"])
        self.assertIsNone(out["ppmi_spearman"])

    def test_association_and_conditional_probability(self):
        out = aligned_comparison(self.here, self.reference, self.shared)
        # matched pairs: A-B has PPMI 0 in both; A-C has negative PMI in both
        self.assertEqual(out["ppmi_change_median"], 0.0)
        npmi_ac = math.log(10 / 32) / -math.log(0.1) - math.log(10 / 24) / -math.log(
            0.1
        )
        self.assertAlmostEqual(out["npmi_change_median"], npmi_ac / 2)
        self.assertAlmostEqual(out["npmi_abs_change_mean"], abs(npmi_ac) / 2)
        self.assertEqual(out["npmi_undefined_matched_pairs"], 0)
        # |P(B|A) change| both ways: A-B 4/8 vs 3/6, 4/5 vs 3/5; A-C 1/8 vs 1/6, 1/4
        changes = sorted([0.0, 0.2, 1 / 6 - 1 / 8, 0.0])
        self.assertAlmostEqual(
            out["conditional_prob_abs_change_median"], (changes[1] + changes[2]) / 2
        )
        self.assertAlmostEqual(
            out["conditional_prob_abs_change_mean"], sum(changes) / 4
        )

    def test_unseen_concepts_are_counted_not_scored(self):
        partial = _result(["A", "B", "C", "D"], {"A": 6, "B": 5}, {("A", "B"): 3}, 10)
        out = aligned_comparison(partial, self.reference, self.shared)
        self.assertEqual(out["concepts_in_both"], 2)
        self.assertEqual(out["concepts_only_reference"], 2)
        self.assertEqual(out["count_profile_rows_valid"], 2)  # A, B
        self.assertEqual(out["count_profile_rows_undefined"], 2)  # C, D empty here
        self.assertIsNone(out["ppmi_spearman"])  # PPMI is 0 everywhere here

    def test_needs_same_patients(self):
        other = _result(["A", "B"], {"A": 1}, {}, 11)
        with self.assertRaises(ValueError):
            aligned_comparison(other, self.reference, self.shared)


class TestEntropyDenominators(unittest.TestCase):
    def test_raw_and_both_normalizations(self):
        # pairs A-B 1, A-C 1, A-D 2 on 4 axes: H = 1.5 bits
        result = _result(
            ["A", "B", "C", "D"],
            {"A": 4, "B": 1, "C": 1, "D": 2},
            {("A", "B"): 1, ("A", "C"): 1, ("A", "D"): 2},
            10,
        )
        s = matrix_summary(result)
        self.assertAlmostEqual(entropy_bits([1, 1, 2]), 1.5)
        self.assertAlmostEqual(s["pair_entropy_bits"], 1.5)
        self.assertAlmostEqual(s["pair_entropy_norm_nonzero_pairs"], 1.5 / math.log2(3))
        self.assertEqual(s["possible_pairs_axes"], 6)
        self.assertAlmostEqual(
            s["pair_entropy_norm_possible_pairs"], 1.5 / math.log2(6)
        )
        # only A has two or more partners: B 1, C 1, D 2 -> 1.5 bits
        self.assertAlmostEqual(s["partner_entropy_bits_median"], 1.5)
        self.assertAlmostEqual(s["partner_entropy_norm_median"], 1.5 / math.log2(3))


class FakeProcessor:
    """Roll-up stand-in: X rolls up to {A, B}; A is retained; U fails."""

    retained_codes = frozenset({"A", "B"})

    def __init__(self):
        self.calls = 0

    def rollup_codes(self, codes):
        self.calls += 1
        table = {
            "X": (("A", "B"), "rolled_up"),
            "A": (("A",), "retained"),
            "U": ((), "unknown_code"),
        }
        return [
            SimpleNamespace(source=c, targets=table[c][0], reason=table[c][1])
            for c in codes
        ]


class TestRollupRecorder(unittest.TestCase):
    def test_counts_and_cache(self):
        processor = FakeProcessor()
        recorder = RollupRecorder(processor)
        self.assertEqual(recorder.retained_codes, processor.retained_codes)
        for codes in (["X"], ["A"], ["X", "A"], ["X"]):
            recorder.rollup_codes(codes)
        self.assertEqual(processor.calls, 2)  # one search per distinct code
        summary = recorder.summary()
        self.assertAlmostEqual(summary["targets_per_diagnosis"], (2 + 1 + 2 + 2) / 4)
        self.assertAlmostEqual(summary["multi_target_share"], 3 / 4)
        self.assertAlmostEqual(summary["rolled_up_share"], 3 / 5)


class TestDepthsEndToEnd(unittest.TestCase):
    """Two real processors over the team's synthetic Athena bundle."""

    def test_deeper_graph_has_larger_vocabulary_and_fewer_roll_ups(self):
        from examples.omop_keep_cooccurrence_matrix import write_athena_bundle
        from pyhealth.medcode import OMOPVocabulary
        from pyhealth.tasks import (
            KeepConditionHistoryTask,
            build_keep_omop_ancestor_rollup,
        )

        with tempfile.TemporaryDirectory() as folder:
            write_athena_bundle(folder)
            vocabulary = OMOPVocabulary(folder)
            task = KeepConditionHistoryTask(
                mimic_version=4, vocabulary=vocabulary, vocabulary_release="test"
            )

            def condition(identity, concepts):
                return {
                    "diagnosis_id": identity,
                    "visit_id": "v",
                    "timestamp": None,
                    "source_value": "ICD10CM:X",
                    "concept_ids": concepts,
                    "unmapped_reason": None,
                }

            # concept 110 sits at depth 6 and 104 at depth 2
            histories = [
                {
                    "patient_id": "1",
                    "conditions": [
                        condition("a", ["110"]),
                        condition("b", ["110"]),
                        condition("c", ["104"]),
                        condition("d", ["104"]),
                    ],
                },
                {
                    "patient_id": "2",
                    "conditions": [condition("a", ["110"]), condition("b", ["109"])],
                },
            ]
            summaries = {}
            for depth in (3, 6):
                recorder = RollupRecorder(
                    build_keep_omop_ancestor_rollup(
                        vocabulary,
                        target_policy="first_retained",
                        include_root=False,
                        max_depth=depth,
                        path_source="closure",
                        on_unmapped="empty",
                    )
                )
                result = task.build_cooccurrence(
                    histories,
                    rollup=recorder,
                    concept_ids=sorted(recorder.retained_codes),
                )
                summaries[depth] = matrix_summary(result, recorder)
        shallow, deep = summaries[3], summaries[6]
        self.assertLess(shallow["vocabulary"], deep["vocabulary"])
        self.assertGreater(shallow["rolled_up_share"], deep["rolled_up_share"])
        # at depth 3, 109 and 110 merge into one concept present for patient 2
        self.assertEqual(shallow["patients_with_conditions"], 2)
        self.assertEqual(deep["patients_with_conditions"], 1)


if __name__ == "__main__":
    unittest.main()
