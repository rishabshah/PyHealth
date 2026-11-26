"""Occurrence interventions, independent patient oracles, and API regressions."""

import copy
import itertools
import unittest

from examples.keep_ancestor_rollup_ablation_utils import trace_pair, validate_matrix
from examples.keep_occurrence_rule_ablation import RULES, run_study
from pyhealth.tasks import KeepConditionHistoryTask, build_keep_cooccurrence
from tests.core.test_keep_cooccurrence import (
    make_mapper,
    make_patient,
    make_processor,
)


class TestOccurrenceStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_study()

    def test_literal_cohort_outcomes_and_fixed_mapping(self):
        self.assertTrue(self.report["public_counter_agreement"])
        self.assertTrue(self.report["hand_oracle_passed"])
        expected = {
            "rows_1": (8, 316),
            "rows_2": (5, 180),
            "rows_3": (2, 90),
            "admissions_2": (3, 90),
        }
        for arm, (patients, mass) in expected.items():
            row = self.report["aggregates"][arm]
            self.assertEqual(row["patients"], 10)
            self.assertEqual(row["patients_with_present_targets"], patients)
            self.assertEqual(row["pair_mass"], mass)
            baseline = self.report["aggregates"]["rows_2"]
            for key in ("unique_diagnoses", "mapped_diagnoses", "unmapped_diagnoses"):
                self.assertEqual(row[key], baseline[key])
        axes = [m["concept_ids"] for m in self.report["matrices"].values()]
        self.assertTrue(all(a == axes[0] for a in axes))

    def test_monotonicity_and_distinct_admission_counterexample(self):
        traces = self.report["traces"]
        for i in range(10):
            present = {a: set(traces[a][i]["present"]) for a in RULES}
            self.assertLessEqual(present["rows_3"], present["rows_2"])
            self.assertLessEqual(present["rows_2"], present["rows_1"])
            self.assertLessEqual(present["admissions_2"], present["rows_2"])
        # p4: three rows in one admission; p3: two rows in two admissions.
        self.assertTrue(traces["rows_3"][3]["present"])
        self.assertFalse(traces["admissions_2"][3]["present"])
        self.assertFalse(traces["rows_3"][2]["present"])
        self.assertTrue(traces["admissions_2"][2]["present"])

    def test_all_pair_deltas_reconcile_and_zero_axis_is_preserved(self):
        report = self.report
        for arm, comparison in report["comparisons"].items():
            for row in comparison["pairs"]:
                witnesses = trace_pair(
                    report["traces"]["rows_2"],
                    report["traces"][arm],
                    row["i"],
                    row["j"],
                )
                self.assertEqual(
                    sum(w["pair_delta"] for w in witnesses), row["count_delta"]
                )
        for matrix in report["matrices"].values():
            values = validate_matrix(matrix)
            self.assertIn("301", matrix["concept_ids"])
            self.assertFalse(any("301" in pair for pair in values))
            for i, j in itertools.product(matrix["concept_ids"], repeat=2):
                self.assertEqual(values.get((i, j), 0), values.get((j, i), 0))


class TestOccurrenceAPI(unittest.TestCase):
    def setUp(self):
        self.rollup = make_processor()
        self.mapper = make_mapper()
        self.axes = ["10", "11", "12"]

    def count(self, rows, **options):
        return build_keep_cooccurrence(
            rows,
            rollup=self.rollup,
            icd_mapper=self.mapper,
            concept_ids=self.axes,
            **options,
        )

    def test_omitted_settings_match_explicit_baseline(self):
        patient = make_patient(
            "p", [{"dx": "1", "concept": "802"}, {"dx": "2", "concept": "802"}]
        )
        a = self.count([patient])
        b = self.count([patient], min_occurrences=2, occurrence_unit="diagnosis")
        self.assertEqual(a.matrix.to_dense().tolist(), b.matrix.to_dense().tolist())
        self.assertEqual(a.patients_with_present_targets, 1)

    def test_multi_mapping_paths_and_exact_copy_supply_one_row(self):
        event = {"dx": "copy", "source": "ICD10CM:K1", "visit": "a"}
        patient = make_patient("p", [event, copy.deepcopy(event)])
        self.assertEqual(self.count([patient]).patients_with_present_targets, 0)
        one = self.count([patient], min_occurrences=1)
        self.assertEqual(one.unique_diagnosis_events, 1)
        self.assertEqual(
            one.matrix.to_dense().tolist(), [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
        )
        admissions = self.count([patient], occurrence_unit="admission")
        self.assertEqual(admissions.patients_with_present_targets, 0)

    def test_admission_support_unions_different_diagnoses(self):
        patient = make_patient(
            "p",
            [
                {"dx": "1", "concept": "801", "visit": "a"},
                {"dx": "2", "concept": "802", "visit": "a"},
                {"dx": "3", "concept": "803", "visit": "b"},
            ],
        )
        result = self.count([patient], occurrence_unit="admission")
        # A has two diagnoses in one admission, B has support in both.
        self.assertEqual(
            result.matrix.to_dense().tolist(), [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        )
        self.assertEqual(result.mapped_events, 3)

    def test_admission_ids_are_patient_scoped_and_order_independent(self):
        rows = [
            {"dx": "1", "concept": "802", "visit": "a"},
            {"dx": "2", "concept": "802", "visit": "b"},
        ]
        patients = [make_patient("p1", rows), make_patient("p2", list(reversed(rows)))]
        result = self.count(patients, occurrence_unit="admission")
        self.assertEqual(
            result.matrix.to_dense().tolist(), [[2, 2, 0], [2, 2, 0], [0, 0, 0]]
        )
        self.assertEqual(
            self.count(reversed(patients), occurrence_unit="admission")
            .matrix.to_dense()
            .tolist(),
            result.matrix.to_dense().tolist(),
        )

    def test_missing_mapped_admission_fails_but_unmapped_has_no_support(self):
        for missing in (None, "", "  ", "None"):
            patient = make_patient(
                "p", [{"dx": "1", "concept": "802", "visit": missing}]
            )
            with self.assertRaisesRegex(ValueError, "nonempty visit ID"):
                self.count([patient], occurrence_unit="admission")
        unmapped = make_patient("p", [{"dx": "1", "concept": "999"}])
        result = self.count([unmapped], occurrence_unit="admission")
        self.assertEqual(result.patients_processed, 1)
        self.assertEqual(result.unmapped_events["empty_rollup"], 1)

    def test_conflicting_copy_with_different_admission_is_rejected(self):
        rows = [
            {"dx": "same", "concept": "802", "visit": "a"},
            {"dx": "same", "concept": "802", "visit": "b"},
        ]
        with self.assertRaisesRegex(ValueError, "conflicting copies"):
            self.count([make_patient("p", rows)], occurrence_unit="admission")

    def test_invalid_settings_fail_before_consuming_cohort(self):
        def unexpected():
            raise AssertionError("invalid settings consumed the cohort")
            yield

        for threshold in (True, False, 1.5, "2", None):
            with self.assertRaises(TypeError):
                self.count(unexpected(), min_occurrences=threshold)
        for threshold in (0, -1):
            with self.assertRaises(ValueError):
                self.count(unexpected(), min_occurrences=threshold)
        for unit in ("rows", "visit", "", None, []):
            with self.assertRaises(ValueError):
                self.count(unexpected(), occurrence_unit=unit)

    def test_mapped_mimic_route_and_missing_admission(self):
        # Counting needs no reader state; exercise the real bound cohort method.
        task = object.__new__(KeepConditionHistoryTask)
        condition = {
            "diagnosis_id": "1",
            "concept_ids": ["801", "802"],
            "unmapped_reason": None,
            "visit_id": "a",
        }
        histories = [
            {"patient_id": "p", "conditions": [condition, copy.deepcopy(condition)]}
        ]

        def count(**options):
            return task.build_cooccurrence(
                histories, rollup=self.rollup, concept_ids=self.axes, **options
            )

        self.assertEqual(count(min_occurrences=1).unique_diagnosis_events, 1)
        self.assertEqual(count().patients_with_present_targets, 0)
        self.assertEqual(
            count(occurrence_unit="admission").patients_with_present_targets, 0
        )
        histories[0]["conditions"][0]["visit_id"] = None
        histories[0]["conditions"].pop()
        with self.assertRaisesRegex(ValueError, "nonempty visit ID"):
            count(occurrence_unit="admission")
        with self.assertRaises(TypeError):
            count(min_occurrences=True)


if __name__ == "__main__":
    unittest.main()
