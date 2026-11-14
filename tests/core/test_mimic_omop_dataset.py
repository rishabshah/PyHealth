from pyhealth.data import Event
from pyhealth.datasets import MIMICOMOPDataset, OMOPDataset


def _dataset():
    """Build the helper object without scanning OMOP files."""
    return MIMICOMOPDataset.__new__(MIMICOMOPDataset)


def test_dataset_can_be_imported():
    assert MIMICOMOPDataset is not None


def test_dataset_is_subclass_of_omop():
    assert issubclass(MIMICOMOPDataset, OMOPDataset)


def test_condition_code_prefers_concept_id():
    event = Event(
        "condition_occurrence",
        condition_concept_id="435459",
        condition_source_value="04119",
    )
    assert _dataset().get_condition_code(event) == "435459"


def test_condition_code_uses_source_when_unmapped():
    event = Event(
        "condition_occurrence",
        condition_concept_id=0,
        condition_source_value="ICD10CM:E11.9",
    )
    assert _dataset().get_condition_code(event) == "ICD10CM:E11.9"


def test_condition_code_missing():
    event = Event("condition_occurrence", condition_concept_id="0")
    assert _dataset().get_condition_code(event) is None


def test_visit_count():
    class _Patient:
        def get_events(self, event_type):
            assert event_type == "visit_occurrence"
            return [1, 2]

    assert _dataset().get_visit_count(_Patient()) == 2
