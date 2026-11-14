from pyhealth.patient_builder import PatientBuilder


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
