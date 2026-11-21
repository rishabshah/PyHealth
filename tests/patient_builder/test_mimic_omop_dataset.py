import os
import shutil
from pathlib import Path

from pyhealth.data import Event
from pyhealth.datasets import MIMICOMOPDataset, OMOPDataset
from pyhealth.patient_builder import PatientBuilder, condition_records

FIXTURE = Path(__file__).resolve().parents[2] / "test-resources" / "mimic_omop"
DIAGNOSIS_FIELDS = (
    "condition_occurrence_id",
    "condition_concept_id",
    "condition_source_value",
    "condition_start_datetime",
    "condition_end_datetime",
    "visit_occurrence_id",
)


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


def test_fixture_preserves_diagnosis_fields(tmp_path):
    """ETL-shaped rows: versioned ICD-9 and ICD-10, plus a diagnosis id.

    Concept 435459 wins over ``ICD9CM:04119``. Concept 0 falls back to
    ``ICD9CM:04119`` and ``ICD10CM:E119``, and the ICD-9 leading zero stays.
    Two copies of mapped ``ICD10CM:E119`` keep different occurrence ids.
    """
    dataset = MIMICOMOPDataset(root=FIXTURE, cache_dir=tmp_path)
    assert [str(patient_id) for patient_id in dataset.unique_patient_ids] == ["1"]

    patient = dataset.get_patient(dataset.unique_patient_ids[0])
    conditions = patient.get_events(event_type="condition_occurrence")
    assert len(conditions) == 6
    assert dataset.get_visit_count(patient) == 2

    by_id = {}
    for condition in conditions:
        assert condition.timestamp is not None
        for field in DIAGNOSIS_FIELDS:
            value = getattr(condition, field, None)
            if field == "visit_occurrence_id" and str(condition.condition_occurrence_id) == "6":
                continue
            assert value not in (None, "")
        by_id[str(condition.condition_occurrence_id)] = condition
    assert set(by_id) == {"1", "2", "3", "4", "5", "6"}

    mapped_icd9 = by_id["1"]
    assert str(mapped_icd9.condition_source_value) == "ICD9CM:04119"
    assert str(mapped_icd9.visit_occurrence_id) == "10"
    assert dataset.get_condition_code(mapped_icd9) == "435459"

    assert dataset.get_condition_code(by_id["2"]) == "ICD9CM:04119"
    assert dataset.get_condition_code(by_id["3"]) == "ICD10CM:E119"
    assert dataset.get_condition_code(by_id["4"]) == "201826"
    assert dataset.get_condition_code(by_id["5"]) == "201826"
    assert str(by_id["4"].condition_source_value) == "ICD10CM:E119"
    assert str(by_id["5"].condition_source_value) == "ICD10CM:E119"
    assert str(by_id["4"].condition_occurrence_id) != str(by_id["5"].condition_occurrence_id)

    assert PatientBuilder(dataset).with_condition("435459").build()
    assert PatientBuilder(dataset).with_condition("201826").build()
    assert PatientBuilder(dataset).with_condition("ICD9CM:04119").build()
    assert PatientBuilder(dataset).with_condition("ICD10CM:E119").build()
    assert PatientBuilder(dataset).with_condition("04119").build() == []
    assert PatientBuilder(dataset).with_condition("ICD9CM:4119").build() == []

    records = {record.diagnosis_id: record for record in condition_records(patient)}
    assert records["1"].concept_id == "435459"
    assert records["1"].icd_code is None
    assert records["2"].concept_id is None
    assert records["2"].icd_code == "04119"
    assert records["2"].icd_version == 9
    assert records["3"].icd_code == "E119"
    assert records["3"].icd_version == 10
    assert records["6"].visit_id is None
    assert records["6"].icd_code == "E119"
    assert records["6"].icd_version == 10
    cohort = PatientBuilder(dataset).with_min_visits(2).build()
    assert len(cohort) == 1
    assert str(cohort[0].patient_id) == "1"

    def diagnosis_count(patients):
        return sum(len(condition_records(patient)) for patient in patients)

    assert diagnosis_count(cohort) == diagnosis_count(dataset.iter_patients())
    assert diagnosis_count(dataset.iter_patients()) == 6


def _cache(tmp_path: Path, name: str) -> Path:
    """Keep Dask's scratch files under the Windows path limit."""
    if os.name != "nt":
        return tmp_path / name
    short = Path(__file__).resolve().parents[2] / ".pyhealth_cache" / name
    if short.exists():
        shutil.rmtree(short)
    short.mkdir(parents=True)
    return short


def _write_omop_base(root: Path) -> None:
    root.mkdir()
    (root / "person.csv").write_text(
        "person_id,year_of_birth,month_of_birth,day_of_birth,"
        "gender_concept_id,race_concept_id,ethnicity_concept_id\n"
        "1,1980,1,1,8507,8527,38003564\n",
        encoding="utf-8",
    )
    (root / "visit_occurrence.csv").write_text(
        "visit_occurrence_id,person_id,visit_concept_id,visit_start_date,"
        "visit_start_datetime,visit_end_date,visit_end_datetime,"
        "visit_type_concept_id\n"
        "10,1,9201,2180-01-15,2180-01-15 08:00:00,2180-01-18,"
        "2180-01-18 12:00:00,32817\n",
        encoding="utf-8",
    )
    (root / "death.csv").write_text(
        "person_id,death_date,death_datetime,death_type_concept_id\n",
        encoding="utf-8",
    )


def _absent(event, name: str) -> bool:
    try:
        getattr(event, name)
    except AttributeError:
        return True
    return False


def test_legacy_condition_loads_and_history_config_adds_its_columns(tmp_path):
    """Default config reads a table without history columns.

    The opt-in config then reads a table that has the diagnosis id and
    source value and still omits condition_status_concept_id.
    """
    legacy_root = tmp_path / "legacy"
    history_root = tmp_path / "history"
    _write_omop_base(legacy_root)
    _write_omop_base(history_root)
    (legacy_root / "condition_occurrence.csv").write_text(
        "person_id,condition_concept_id,condition_start_date,"
        "condition_start_datetime,condition_end_date,condition_end_datetime,"
        "condition_type_concept_id,visit_occurrence_id\n"
        "1,435459,2180-01-15,2180-01-15 08:00:00,2180-01-18,"
        "2180-01-18 12:00:00,32019,10\n",
        encoding="utf-8",
    )
    (history_root / "condition_occurrence.csv").write_text(
        "person_id,condition_occurrence_id,condition_concept_id,"
        "condition_source_value,condition_start_date,condition_start_datetime,"
        "condition_end_date,condition_end_datetime,condition_type_concept_id,"
        "visit_occurrence_id\n"
        "1,7,0,ICD9CM:04119,2180-01-15,2180-01-15 08:00:00,2180-01-18,"
        "2180-01-18 12:00:00,32019,10\n",
        encoding="utf-8",
    )

    legacy_cache = _cache(tmp_path, "omop_legacy")
    history_cache = _cache(tmp_path, "omop_history")
    try:
        legacy = OMOPDataset(
            root=str(legacy_root),
            tables=["condition_occurrence"],
            cache_dir=str(legacy_cache),
        )
        legacy_event = legacy.get_patient("1").get_events(
            event_type="condition_occurrence"
        )[0]
        assert str(legacy_event.condition_concept_id) == "435459"
        assert str(legacy_event.visit_occurrence_id) == "10"
        assert _absent(legacy_event, "condition_occurrence_id")
        assert _absent(legacy_event, "condition_source_value")
        assert _absent(legacy_event, "condition_status_concept_id")

        history = MIMICOMOPDataset(root=history_root, cache_dir=history_cache)
        history_event = history.get_patient("1").get_events(
            event_type="condition_occurrence"
        )[0]
        assert str(history_event.condition_occurrence_id) == "7"
        assert str(history_event.condition_source_value) == "ICD9CM:04119"
        assert history.get_condition_code(history_event) == "ICD9CM:04119"
        assert _absent(history_event, "condition_status_concept_id")
    finally:
        if os.name == "nt":
            shutil.rmtree(legacy_cache, ignore_errors=True)
            shutil.rmtree(history_cache, ignore_errors=True)
