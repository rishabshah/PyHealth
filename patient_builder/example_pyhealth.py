"""Load the converted OMOP tables in PyHealth and build mortality samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from mimic_omop.task import MimicHospMortalityTask, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Load MIMIC-IV hosp OMOP data with PyHealth.")
    parser.add_argument("--root", type=Path, default=Path("omop"))
    parser.add_argument("--dev", action="store_true", help="Use the first 1000 patients PyHealth selects.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".pyhealth_cache"))
    args = parser.parse_args()

    dataset = load_dataset(args.root, dev=args.dev, cache_dir=args.cache_dir)
    dataset.stats()

    patient_id = (
        dataset.global_event_df.filter(pl.col("event_type") == "condition_occurrence")
        .select("patient_id")
        .head(1)
        .collect()
        .item()
    )
    patient = dataset.get_patient(patient_id)
    conditions = patient.get_events(event_type="condition_occurrence")
    print(f"patient {patient_id} condition codes:")
    print([event.condition_source_value for event in conditions[:8]])

    samples = dataset.set_task(MimicHospMortalityTask())
    print(f"mortality samples: {len(samples)}")
    first = samples[0]
    print(f"sample patient_id: {first['patient_id']}")
    print(f"sample visit_id: {first['visit_id']}")
    print(f"sample mortality label: {int(first['mortality'])}")


if __name__ == "__main__":
    main()
