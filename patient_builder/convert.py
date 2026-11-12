"""Convert MIMIC-IV hosp CSVs to OMOP CDM CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mimic_omop.etl import convert


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MIMIC-IV hosp patients, admissions, diagnoses, and procedures to OMOP CDM."
    )
    parser.add_argument(
        "--hosp",
        type=Path,
        default=Path("hosp"),
        help="Directory that contains patients.csv, admissions.csv, diagnoses_icd.csv, and procedures_icd.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("omop"),
        help="Directory for the OMOP CSV files.",
    )
    parser.add_argument(
        "--limit-patients",
        type=int,
        default=None,
        help="Convert only the first N patients, ordered by subject_id.",
    )
    args = parser.parse_args()
    summary = convert(args.hosp, args.out, args.limit_patients)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
