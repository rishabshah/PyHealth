"""Tests for the KEEP depth ablation's downstream-model helpers.

Covers examples/keep_depth_mortality_utils.py with fake patients, mappers and
processors.
"""

import unittest
from types import SimpleNamespace

from examples.keep_depth_mortality_utils import (
    ConceptFeaturizer,
    ConditionMortalityMIMIC4,
    check_class_coverage,
    featurize_arms,
    fit_on_train,
    fold_counts,
    patient_folds,
    summarize_scores,
)


class FakeMapper:
    def map(self, code, version):
        if str(version) not in ("9", "10"):
            raise ValueError("bad version")
        return {"K1": [1, 2], "I21": [3], "Z99": []}.get(code, [])


class TestFeaturization(unittest.TestCase):
    def test_arms(self):
        class Processor:
            def rollup_codes(self, codes):
                up = {"1": ("X",), "2": ("X",), "3": ()}
                return [SimpleNamespace(targets=up[c]) for c in codes]

        featurizers = {
            "icd": ConceptFeaturizer(),
            "concepts": ConceptFeaturizer(FakeMapper()),
            "shallow": ConceptFeaturizer(FakeMapper(), Processor()),
        }
        self.assertEqual(featurizers["icd"](["10:K1", "10:K1"]), ["10:K1"])
        self.assertEqual(featurizers["concepts"](["10:K1", "10:I21"]), ["1", "2", "3"])
        self.assertEqual(featurizers["shallow"](["10:K1", "10:I21"]), ["X"])
        self.assertEqual(featurizers["concepts"](["11:K1"]), [])
        samples = [
            {"patient_id": "p1", "conditions": ["10:K1"], "mortality": 1},
            {
                "patient_id": "p2",
                "conditions": ["10:I21"],
                "mortality": 0,
            },  # empty roll-up
            {"patient_id": "p3", "conditions": ["10:Z99"], "mortality": 0},  # unmapped
        ]
        arms = featurize_arms(samples, featurizers)
        for name, rows in arms.items():
            self.assertEqual([r["patient_id"] for r in rows], ["p1"], name)
        self.assertEqual(arms["shallow"][0]["conditions"], ["X"])
        self.assertEqual(samples[0]["conditions"], ["10:K1"])  # inputs untouched

    def test_roll_up_needs_mapper(self):
        with self.assertRaises(ValueError):
            ConceptFeaturizer(processor=object())


class FakePatient:
    def __init__(self, age, admissions, diagnoses, patient_id="p"):
        self.patient_id = patient_id
        self._events = {
            "patients": [SimpleNamespace(anchor_age=age)],
            "admissions": [
                SimpleNamespace(hadm_id=h, hospital_expire_flag=f)
                for h, f in admissions
            ],
            "diagnoses_icd": [
                SimpleNamespace(hadm_id=h, icd_code=c, icd_version=v)
                for h, c, v in diagnoses
            ],
        }

    def get_events(self, event_type):
        return self._events[event_type]


class TestConditionMortalityTask(unittest.TestCase):
    def test_samples_accumulate_history_and_take_next_label(self):
        patient = FakePatient(
            70,
            [("a1", "0"), ("a2", "0"), ("a3", "1")],
            [
                ("a1", "K1", "10"),
                ("a2", "0010", "9"),
                ("a2", "K1", "10"),
                ("a3", "I21", "10"),
            ],
        )
        samples = ConditionMortalityMIMIC4()(patient)
        self.assertEqual([s["visit_id"] for s in samples], ["a1", "a2"])
        self.assertEqual([s["mortality"] for s in samples], [0, 1])
        self.assertEqual(samples[0]["conditions"], ["10:K1"])
        self.assertEqual(samples[1]["conditions"], ["10:K1", "9:0010"])

    def test_exclusions(self):
        task = ConditionMortalityMIMIC4()
        child = FakePatient(12, [("a1", "0"), ("a2", "1")], [("a1", "K1", "10")])
        single = FakePatient(50, [("a1", "1")], [("a1", "K1", "10")])
        no_history = FakePatient(50, [("a1", "0"), ("a2", "1")], [("a2", "K1", "10")])
        self.assertEqual(task(child), [])
        self.assertEqual(task(single), [])
        self.assertEqual(task(no_history), [])


def _unique_code_samples(n=40):
    """The review's fixture: every patient has its own code; one positive."""
    return [
        {
            "patient_id": f"p{i}",
            "visit_id": f"v{i}",
            "conditions": [f"unique_{i}"],
            "mortality": int(i == 0),
        }
        for i in range(n)
    ]


class TestTrainFittedVocabulary(unittest.TestCase):
    def test_folds_are_patient_level_and_shared_by_arms(self):
        samples = _unique_code_samples()
        samples.append({**samples[3], "visit_id": "v3b"})  # p3 has two samples
        train, val, test = patient_folds(samples, seed=0)
        self.assertEqual(sorted(train + val + test), list(range(len(samples))))
        owners = [
            {samples[i]["patient_id"] for i in part} for part in (train, val, test)
        ]
        self.assertFalse(
            owners[0] & owners[1] or owners[0] & owners[2] or owners[1] & owners[2]
        )
        # another arm: same samples and order, different tokens -> same folds
        other_arm = [{**s, "conditions": ["X"]} for s in samples]
        self.assertEqual(patient_folds(other_arm, seed=0), [train, val, test])
        self.assertNotEqual(patient_folds(samples, seed=1), [train, val, test])

    def test_held_out_only_code_maps_to_unknown(self):
        samples = _unique_code_samples()
        folds = patient_folds(samples, seed=0)
        train, val, test = fit_on_train(samples, folds, task_name="t")
        processor = train.input_processors["conditions"]
        unknown = processor.code_vocab["<unk>"]
        test_patient = samples[folds[2][0]]
        code = test_patient["conditions"][0]
        self.assertNotIn(code, processor.code_vocab)
        self.assertEqual(processor.process([code]).tolist(), [unknown])
        # validation and test reuse the train-fitted processors unchanged
        self.assertIs(test.input_processors["conditions"], processor)
        self.assertIs(
            val.output_processors["mortality"], train.output_processors["mortality"]
        )
        self.assertEqual(test[0]["conditions"].tolist(), [unknown])
        train_codes = {samples[i]["conditions"][0] for i in folds[0]}
        self.assertTrue(train_codes <= set(processor.code_vocab))


class TestEvaluationGuards(unittest.TestCase):
    def test_one_positive_cohort_fails_clearly_before_training(self):
        samples = _unique_code_samples()  # one positive patient in 40
        for seed in range(3):
            folds = patient_folds(samples, seed)
            with self.assertRaisesRegex(ValueError, rf"Seed {seed}: .*single class"):
                check_class_coverage(samples, folds, seed)

    def test_usable_folds_return_counts(self):
        samples = [
            {**s, "mortality": i % 2} for i, s in enumerate(_unique_code_samples())
        ]
        folds = patient_folds(samples, seed=0)
        counts = check_class_coverage(samples, folds, seed=0)
        self.assertEqual(counts, fold_counts(samples, folds))
        self.assertEqual(sum(c["samples"] for c in counts.values()), 40)
        self.assertEqual(sum(c["positives"] for c in counts.values()), 20)

    def test_scores_keep_every_seed_and_reject_non_finite(self):
        runs = {0: {"roc_auc": 0.6}, 1: {"roc_auc": 0.8}, 2: {"roc_auc": 0.7}}
        summary = summarize_scores(runs, "depth_5", ["roc_auc"])
        self.assertAlmostEqual(summary["roc_auc"], 0.7)
        self.assertAlmostEqual(summary["roc_auc_std"], 0.1)
        self.assertEqual(summary["roc_auc_per_seed"], [0.6, 0.8, 0.7])
        runs[1]["roc_auc"] = float("nan")
        with self.assertRaisesRegex(ValueError, r"'depth_5'.*seed\(s\) \[1\]"):
            summarize_scores(runs, "depth_5", ["roc_auc"])


if __name__ == "__main__":
    unittest.main()
