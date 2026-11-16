"""Build a PyHealth patient cohort from MIMIC-IV OMOP diagnoses.

``PatientBuilder`` filters ``MIMICOMOPDataset`` and returns ``Patient``
objects. This script prints every ``condition_occurrence`` event on the
cohort.

The CSVs come from the conversion command on the ``patient_builder`` branch:

https://github.com/rishabshah/PyHealth/blob/patient_builder/patient_builder/convert.py

    python convert.py --hosp hosp --out omop

``MIMICOMOPDataset`` expects ``person.csv``, ``visit_occurrence.csv``, and
``death.csv``, plus ``condition_occurrence.csv``.

By default this example reads the one-patient fixture in
``test-resources/mimic_omop``. Source values use the ETL encoding
``ICD9CM:04119`` and ``ICD10CM:E119``. Concept 435459 and 201826 take
precedence over those source values. Concept 0 falls back to the source
value, including the ICD-9 leading zero. Rows 4 and 5 are the same
diagnosis twice and stay distinct by ``condition_occurrence_id``. Row 6
has no visit link and is still returned.
"""

import argparse
from pathlib import Path

from pyhealth.datasets import MIMICOMOPDataset
from pyhealth.patient_builder import PatientBuilder, condition_records

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "test-resources" / "mimic_omop"
DIAGNOSIS_FIELDS = (
    "condition_occurrence_id",
    "condition_concept_id",
    "condition_source_value",
    "condition_source_concept_id",
    "condition_status_concept_id",
    "condition_status_source_value",
    "condition_start_datetime",
    "visit_occurrence_id",
)


def read_diagnoses(root: Path, cache_dir: Path):
    """Load ``root`` and return the small selected cohort as a list."""
    dataset = MIMICOMOPDataset(root=root, cache_dir=cache_dir)
    patients = PatientBuilder(dataset).with_min_visits(1).build()
    return dataset, patients


def count_diagnoses(patients) -> tuple[int, int]:
    """Aggregate an iterable one patient at a time.

    ``patients`` may be the list from ``PatientBuilder.build`` or the
    iterator from ``dataset.iter_patients``. Each patient is counted and
    then dropped. Nothing here retains the whole cohort.
    """
    patient_count = 0
    diagnosis_count = 0
    for patient in patients:
        patient_count += 1
        diagnosis_count += len(condition_records(patient))
    return patient_count, diagnosis_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory of OMOP CSV files. Defaults to the synthetic fixture.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".pyhealth_cache",
        help="Directory for the processed event cache.",
    )
    args = parser.parse_args()

    dataset, patients = read_diagnoses(args.root, args.cache_dir)
    patient = patients[0]
    conditions = patient.get_events(event_type="condition_occurrence")
    print(f"patients in the file: {len(dataset.unique_patient_ids)}")
    print(f"selected list keeps {len(patients)} patient(s)")
    print(f"patient_id: {patient.patient_id}")
    print(f"visits: {dataset.get_visit_count(patient)}")
    print(f"diagnosis events: {len(conditions)}")
    for condition in conditions:
        fields = {name: getattr(condition, name, None) for name in DIAGNOSIS_FIELDS}
        print(dataset.get_condition_code(condition), fields)
    print("mapper inputs:")
    for record in condition_records(patient):
        print(record)

    listed_patients, listed_rows = count_diagnoses(patients)
    streamed_patients, streamed_rows = count_diagnoses(dataset.iter_patients())
    print(
        f"list path: {listed_patients} patient(s), {listed_rows} diagnoses"
    )
    print(
        "iter_patients path: "
        f"{streamed_patients} patient(s), {streamed_rows} diagnoses"
    )


if __name__ == "__main__":
    main()
