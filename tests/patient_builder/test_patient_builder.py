from datetime import datetime

import polars as pl
import pytest

from pyhealth.data import Patient
from pyhealth.patient_builder import (
    ConditionRecord,
    PatientBuilder,
    concepts_with_repeated_diagnoses,
    conflicting_diagnosis_ids,
)


class MockCondition:
    def __init__(self, code):
        self.condition_source_value = code
        self.condition_concept_id = "0"


class MockPatient:
    def __init__(self, patient_id, conditions, visits):
        self.patient_id = patient_id
        self._conditions = conditions
        self._visits = visits

    def get_events(self, event_type):
        if event_type == "condition_occurrence":
            return self._conditions
        if event_type == "visit_occurrence":
            return self._visits
        return []


class MockDataset:
    def __init__(self):
        patient1 = MockPatient(
            patient_id="1",
            conditions=[
                MockCondition("E11.9"),
                MockCondition("I10"),
            ],
            visits=[1, 2, 3],
        )
        patient2 = MockPatient(
            patient_id="2",
            conditions=[MockCondition("I10")],
            visits=[1],
        )
        patient3 = MockPatient(
            patient_id="3",
            conditions=[MockCondition("J44.9")],
            visits=[1, 2],
        )
        self.patients = {
            "1": patient1,
            "2": patient2,
            "3": patient3,
        }

    @property
    def unique_patient_ids(self):
        return list(self.patients)

    def get_patient(self, patient_id):
        return self.patients[patient_id]

    def get_visit_count(self, patient):
        return len(patient.get_events(event_type="visit_occurrence"))

    def get_condition_code(self, event):
        concept = event.condition_concept_id
        if concept not in (None, "", 0, "0"):
            return str(concept)
        return event.condition_source_value

    @property
    def global_event_df(self):
        rows = []
        for patient in self.patients.values():
            for _visit in patient._visits:
                rows.append(
                    {
                        "patient_id": patient.patient_id,
                        "event_type": "visit_occurrence",
                        "timestamp": datetime(2020, 1, 1),
                        "condition_occurrence/condition_concept_id": None,
                        "condition_occurrence/condition_source_value": None,
                    }
                )
            for condition in patient._conditions:
                rows.append(
                    {
                        "patient_id": patient.patient_id,
                        "event_type": "condition_occurrence",
                        "timestamp": datetime(2020, 1, 2),
                        "condition_occurrence/condition_concept_id": str(
                            condition.condition_concept_id
                        ),
                        "condition_occurrence/condition_source_value": (
                            condition.condition_source_value
                        ),
                    }
                )
        return pl.DataFrame(rows).lazy()

    def iter_patients(self, df=None):
        frame = self.global_event_df if df is None else df
        patient_ids = (
            frame.select("patient_id")
            .unique(maintain_order=True)
            .collect()
            .to_series()
        )
        for patient_id in patient_ids:
            patient_df = frame.filter(pl.col("patient_id") == patient_id).collect()
            yield Patient(patient_id=str(patient_id), data_source=patient_df)


def test_builder_creation():
    builder = PatientBuilder(MockDataset())
    assert builder.patient_ids == ["1", "2", "3"]


def test_build_without_filters():
    patients = PatientBuilder(MockDataset()).build()
    assert len(patients) == 3
    assert [patient.patient_id for patient in patients] == ["1", "2", "3"]


def test_with_min_visits():
    patients = PatientBuilder(MockDataset()).with_min_visits(2).build()
    assert {patient.patient_id for patient in patients} == {"1", "3"}


def test_with_condition():
    patients = PatientBuilder(MockDataset()).with_condition("E11.9").build()
    assert len(patients) == 1
    assert patients[0].patient_id == "1"


def test_chained_filters():
    patients = (
        PatientBuilder(MockDataset())
        .with_condition("E11.9")
        .with_min_visits(2)
        .build()
    )
    assert len(patients) == 1
    assert patients[0].patient_id == "1"


def test_no_matching_condition():
    patients = PatientBuilder(MockDataset()).with_condition("UNKNOWN").build()
    assert patients == []


def test_with_min_visits_no_match():
    patients = PatientBuilder(MockDataset()).with_min_visits(10).build()
    assert patients == []


def test_filters_stay_lazy_until_patients_are_read():
    """Chained filters do not load patients. build() loads each survivor once."""

    class CountingDataset(MockDataset):
        def __init__(self):
            super().__init__()
            self.lookups = 0
            self.materialized = 0

        def get_patient(self, patient_id):
            self.lookups += 1
            return super().get_patient(patient_id)

        def iter_patients(self, df=None):
            for patient in super().iter_patients(df):
                self.materialized += 1
                yield patient

    dataset = CountingDataset()
    builder = PatientBuilder(dataset).with_condition("E11.9").with_min_visits(2)
    assert dataset.lookups == 0
    assert dataset.materialized == 0
    patients = builder.build()
    assert [patient.patient_id for patient in patients] == ["1"]
    assert dataset.lookups == 0
    assert dataset.materialized == 1
    visits = patients[0].get_events(event_type="visit_occurrence")
    assert len(visits) == 3


def test_two_diagnoses_qualify_one_ancestor_once_per_row():
    """Two paths inside one diagnosis count once. Two diagnoses count twice."""
    one_diagnosis_two_paths = [
        ConditionRecord("1", "10", "435459", None, None),
    ]
    two_diagnoses = [
        ConditionRecord("1", "10", "435459", None, None),
        ConditionRecord("2", None, None, "04119", 9),
    ]

    def rollup(record):
        if record.diagnosis_id == "1":
            return ["A", "A", "parent"]
        return ["A"]

    assert concepts_with_repeated_diagnoses(one_diagnosis_two_paths, rollup) == set()
    assert concepts_with_repeated_diagnoses(two_diagnoses, rollup) == {"A"}


def test_conflicting_copies_are_flagged():
    """Same diagnosis id with two source codes is flagged."""
    records = [
        ConditionRecord("1", "10", None, "04119", 9),
        ConditionRecord("1", "10", None, "04119", 9),
        ConditionRecord("1", "10", None, "I214", 10),
        ConditionRecord("2", "11", None, "I214", 10),
    ]

    def rollup(record):
        if record.icd_code == "04119":
            return ["100", "100"]
        return ["102", "100"]

    assert conflicting_diagnosis_ids(records, rollup) == {"1"}


def test_repeated_record_stays_below_the_threshold():
    """Two copies of one diagnosis count once, before mapping."""
    record = ConditionRecord("1", "10", "101", None, None)
    calls = []

    def rollup(item):
        calls.append(item)
        return ["101", "101"]

    repeated = concepts_with_repeated_diagnoses([record, record], rollup)
    assert repeated == set()
    assert calls == [record]


def test_conflicting_diagnosis_raises_before_mapping():
    """A diagnosis id with two concepts is rejected before rollup."""
    first = ConditionRecord("1", "10", "101", None, None)
    conflict = ConditionRecord("1", "10", "102", None, None)
    second = ConditionRecord("2", "11", "102", None, None)

    def rollup(item):
        raise AssertionError(item.diagnosis_id)

    with pytest.raises(ValueError, match="Conflicting diagnosis 1"):
        concepts_with_repeated_diagnoses([first, conflict, second], rollup)
