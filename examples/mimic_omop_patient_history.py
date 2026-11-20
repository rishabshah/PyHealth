"""Build a KEEP patient history from MIMIC-IV, without an OMOP CSV round-trip.

The notebook workflow loads MIMIC-IV with the opt-in EHR config, maps ICD
codes through the shared Athena vocabulary, and asks ``set_task`` for one
history per patient. PyHealth caches the events and the samples. A CDM
export can still be read later with ``MIMICOMOPDataset``; this script does
not write one.

.. code-block:: python

    vocabulary = OMOPVocabulary(athena_root)
    history_task = KeepPatientHistory(vocabulary)
    dataset = MIMIC4Dataset(
        ehr_root=mimic_root,
        ehr_tables=["diagnoses_icd"],
        ehr_config_path=keep_ehr_config,
        cache_dir=cache_dir,
    )
    histories = dataset.set_task(history_task, num_workers=1)
    histories[0]["conditions"]

With no paths, the script writes a small MIMIC tree and a small Athena
bundle under the cache directory and runs that same call. The bundle is
there so the real ``OMOPVocabulary.icd_mapper`` can be exercised without
an Athena download. Pass ``--ehr-root`` and ``--athena`` for a real extract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.medcode import OMOPVocabulary
from pyhealth.tasks import KeepPatientHistory

REPO_ROOT = Path(__file__).resolve().parents[1]
KEEP_EHR_CONFIG = (
    REPO_ROOT / "pyhealth" / "datasets" / "configs" / "mimic4_keep_ehr.yaml"
)


def write_synthetic_mimic(root: Path) -> None:
    """Write one patient: a leading-zero ICD-9, a one-to-many code, two AMIs, an unmatched visit, and a conflict."""
    hosp = root / "hosp"
    icu = root / "icu"
    hosp.mkdir(parents=True, exist_ok=True)
    icu.mkdir(parents=True, exist_ok=True)
    (hosp / "patients.csv").write_text(
        "subject_id,gender,anchor_age,anchor_year,dod\n1,F,40,2150,\n",
        encoding="utf-8",
    )
    (hosp / "admissions.csv").write_text(
        "subject_id,hadm_id,admittime,dischtime\n"
        "1,10,2150-01-01 00:00:00,2150-01-05 00:00:00\n"
        "1,11,2150-06-01 08:00:00,2150-06-04 08:00:00\n",
        encoding="utf-8",
    )
    # Two identical 04119 rows collapse. seq 5 names both 04119 and I214.
    # hadm 99 is not an admission. I214 and 41071 are two diagnoses.
    (hosp / "diagnoses_icd.csv").write_text(
        "subject_id,hadm_id,seq_num,icd_code,icd_version\n"
        "1,10,1,04119,9\n"
        "1,10,1,04119,9\n"
        "1,10,2,I2510,10\n"
        "1,11,1,I214,10\n"
        "1,11,2,41071,9\n"
        "1,99,1,E8809,9\n"
        "1,10,5,04119,9\n"
        "1,10,5,I214,10\n",
        encoding="utf-8",
    )
    (icu / "icustays.csv").write_text(
        "subject_id,hadm_id,stay_id,intime,outtime\n"
        "1,10,500,2150-01-01 01:00:00,2150-01-02 01:00:00\n",
        encoding="utf-8",
    )


def write_synthetic_athena(root: Path) -> None:
    """Write a tiny Athena bundle the real vocabulary reader can open.

    ``041.19`` maps to heart disease and keeps its leading zero.
    ``I25.10`` maps to two concepts. ``I21.4`` and ``410.71`` both reach
    acute MI. ``E880.9`` maps to the two-parent concept.
    """
    root.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        root / "CONCEPT.csv",
        [
            "concept_id",
            "concept_name",
            "domain_id",
            "vocabulary_id",
            "concept_class_id",
            "standard_concept",
            "concept_code",
            "invalid_reason",
        ],
        [
            (4274025, "Disease", "Condition", "SNOMED", "Clinical Finding", "S", "64572001", ""),
            (100, "Heart disease", "Condition", "SNOMED", "Clinical Finding", "S", "56265001", ""),
            (101, "Ischemic heart disease", "Condition", "SNOMED", "Clinical Finding", "S", "414545008", ""),
            (102, "Acute MI", "Condition", "SNOMED", "Clinical Finding", "S", "57054005", ""),
            (104, "Two-parent disorder", "Condition", "SNOMED", "Clinical Finding", "S", "1000004", ""),
            (9001, "NSTEMI", "Condition", "ICD10CM", "ICD10 code", "", "I21.4", ""),
            (9002, "CAD", "Condition", "ICD10CM", "ICD10 code", "", "I25.10", ""),
            (8001, "AMI inferior", "Condition", "ICD9CM", "ICD9 code", "", "410.71", ""),
            (8002, "Fall on stairs", "Condition", "ICD9CM", "ICD9 code", "", "E880.9", ""),
            (8005, "Streptococcus", "Condition", "ICD9CM", "ICD9 code", "", "041.19", ""),
        ],
    )
    _write_tsv(
        root / "CONCEPT_RELATIONSHIP.csv",
        ["concept_id_1", "concept_id_2", "relationship_id", "invalid_reason"],
        [
            (100, 4274025, "Is a", ""),
            (101, 100, "Is a", ""),
            (102, 101, "Is a", ""),
            (104, 100, "Is a", ""),
            (104, 101, "Is a", ""),
            (9001, 102, "Maps to", ""),
            (9002, 101, "Maps to", ""),
            (9002, 104, "Maps to", ""),
            (8001, 102, "Maps to", ""),
            (8002, 104, "Maps to", ""),
            (8005, 100, "Maps to", ""),
        ],
    )
    _write_tsv(
        root / "CONCEPT_ANCESTOR.csv",
        [
            "ancestor_concept_id",
            "descendant_concept_id",
            "min_levels_of_separation",
            "max_levels_of_separation",
        ],
        [
            (4274025, 4274025, 0, 0),
            (4274025, 100, 1, 1),
            (100, 100, 0, 0),
            (4274025, 101, 2, 2),
            (100, 101, 1, 1),
            (101, 101, 0, 0),
            (4274025, 102, 3, 3),
            (100, 102, 2, 2),
            (101, 102, 1, 1),
            (102, 102, 0, 0),
            (4274025, 104, 2, 3),
            (100, 104, 1, 1),
            (101, 104, 1, 1),
            (104, 104, 0, 0),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ehr-root", type=Path, default=None)
    parser.add_argument("--athena", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".pyhealth_cache",
    )
    args = parser.parse_args()
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    mimic_root = args.ehr_root
    athena_root = args.athena
    if mimic_root is None:
        mimic_root = cache_dir / "synthetic_mimic"
        write_synthetic_mimic(mimic_root)
    if athena_root is None:
        athena_root = cache_dir / "synthetic_athena"
        write_synthetic_athena(athena_root)

    vocabulary = OMOPVocabulary(athena_root)
    history_task = KeepPatientHistory(vocabulary)
    dataset = MIMIC4Dataset(
        ehr_root=str(mimic_root),
        ehr_tables=["diagnoses_icd"],
        ehr_config_path=str(KEEP_EHR_CONFIG),
        cache_dir=str(cache_dir),
    )
    histories = dataset.set_task(history_task, num_workers=1)
    conditions = histories[0]["conditions"]
    print(f"patients: {len(dataset.unique_patient_ids)}")
    print(f"histories: {len(histories)}")
    print(f"patient_id: {histories[0]['patient_id']}")
    print(f"repeated concepts: {histories[0]['repeated_concepts']}")
    for condition in conditions:
        print(
            condition["diagnosis_id"],
            condition["icd_code"],
            condition["concept_ids"],
            "conflict" if condition["conflict"] else "ok",
            "no admission" if condition["timestamp"] is None else condition["timestamp"],
        )


def _write_tsv(path: Path, header: list[str], rows: list[tuple]) -> None:
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
