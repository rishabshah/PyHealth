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
    gini,
    hhi,
    matrix_arrays,
    matrix_summary,
    normalized_entropy,
    shared_concept_comparison,
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


class TestSharedConceptComparison(unittest.TestCase):
    def setUp(self):
        self.prevalence = {"A": 6, "B": 5, "C": 4, "D": 3}
        self.pairs = {("A", "B"): 3, ("A", "C"): 1, ("B", "C"): 2, ("C", "D"): 2}
        self.reference = _result(["A", "B", "C", "D"], self.prevalence, self.pairs, 10)

    def test_identical_matrices(self):
        out = shared_concept_comparison(
            self.reference, self.reference, ["A", "B", "C", "D"]
        )
        self.assertEqual(out["shared_concepts"], 4)
        self.assertAlmostEqual(out["ppmi_spearman"], 1.0)
        self.assertAlmostEqual(out["mean_profile_jsd"], 0.0)
        self.assertAlmostEqual(out["median_prevalence_ratio"], 1.0)

    def test_extra_concepts_are_ignored_and_shifts_are_measured(self):
        # a shallower depth: an extra concept X, doubled prevalence, and
        # C's partner mass moved from B to D
        shallow = _result(
            ["X", "A", "B", "C", "D"],
            {"X": 9, **{k: 2 * v for k, v in self.prevalence.items()}},
            {("X", "A"): 5, ("A", "B"): 3, ("A", "C"): 1, ("B", "C"): 1, ("C", "D"): 6},
            10,
        )
        out = shared_concept_comparison(shallow, self.reference, ["A", "B", "C", "D"])
        self.assertEqual(out["shared_concepts"], 4)
        self.assertAlmostEqual(out["median_prevalence_ratio"], 2.0)
        self.assertGreater(out["mean_profile_jsd"], 0.0)
        self.assertLessEqual(out["mean_profile_jsd"], 1.0)

    def test_unseen_concepts_are_not_compared(self):
        partial = _result(["A", "B", "C", "D"], {"A": 6, "B": 5}, {("A", "B"): 3}, 10)
        out = shared_concept_comparison(partial, self.reference, ["A", "B", "C", "D"])
        self.assertEqual(out["shared_concepts"], 2)
        self.assertIsNone(out["ppmi_spearman"])  # one pair cannot be ranked


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
