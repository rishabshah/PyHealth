"""Read OMOP CDM tables and build PyHealth Patient objects.

Each patient object holds that person's visits and condition codes. The
condition-history task then keeps codes that occur at least ``min_occurrence``
times and groups the remaining codes by visit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pyhealth.datasets import OMOPDataset, PatientSplit, get_dataloader
from pyhealth.tasks import BaseTask

CONFIG_PATH = Path(__file__).resolve().parent / "mimic_omop" / "omop.yaml"


class OMOPConditionHistoryTask(BaseTask):
    """One sample per patient: condition codes nested by hospital visit.

    A code is kept when it appears at least ``min_occurrence`` times in that
    patient's history. Standard concept ids are used when they are mapped.
    This hosp extract leaves those ids at 0, so the ICD source value is used.
    """

    task_name: str = "OMOPConditionHistory"
    input_schema: dict[str, str] = {"conditions": "nested_sequence"}
    output_schema: dict[str, str] = {}

    def __init__(self, min_occurrence: int = 2):
        super().__init__()
        self.min_occurrence = min_occurrence

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        visit_conditions: list[list[str]] = []

        for visit in patient.get_events(event_type="visit_occurrence"):
            visit_id = str(getattr(visit, "visit_occurrence_id", "") or "")
            if not visit_id or visit_id == "None":
                continue
            codes = [
                code
                for event in patient.get_events(
                    event_type="condition_occurrence",
                    filters=[("visit_occurrence_id", "==", visit_id)],
                )
                if (code := _condition_code(event)) is not None
            ]
            if not codes:
                continue
            visit_conditions.append(codes)
            for code in codes:
                counts[code] = counts.get(code, 0) + 1

        filtered = []
        for codes in visit_conditions:
            kept = [code for code in codes if counts[code] >= self.min_occurrence]
            if kept:
                filtered.append(kept)
        if not filtered:
            return []
        return [{"patient_id": patient.patient_id, "conditions": filtered}]


def load_omop(root: str | Path, dev: bool, cache_dir: str | Path) -> OMOPDataset:
    """Load person, visit, death, and condition tables into an OMOPDataset."""
    return OMOPDataset(
        root=str(root),
        tables=["condition_occurrence"],
        dataset_name="mimic4_hosp_omop",
        config_path=str(CONFIG_PATH),
        dev=dev,
        cache_dir=str(cache_dir),
    )


def show_patient(dataset: OMOPDataset) -> None:
    """Print one PyHealth Patient built from the OMOP tables."""
    import polars as pl

    patient_id = (
        dataset.global_event_df.filter(pl.col("event_type") == "condition_occurrence")
        .select("patient_id")
        .head(1)
        .collect()
        .item()
    )
    patient = dataset.get_patient(patient_id)
    person = patient.get_events(event_type="person")
    visits = patient.get_events(event_type="visit_occurrence")
    conditions = patient.get_events(event_type="condition_occurrence")

    print(f"Patient type: {type(patient).__name__}")
    print(f"patient_id: {patient.patient_id}")
    if person:
        row = person[0]
        print(f"year_of_birth: {row.year_of_birth}")
        print(f"gender_concept_id: {row.gender_concept_id}")
        print(f"race_concept_id: {row.race_concept_id}")
    print(f"visits: {len(visits)}")
    print(f"conditions: {len(conditions)}")
    print("first condition codes:")
    print([_condition_code(event) for event in conditions[:8]])


def _condition_code(event: Any) -> str | None:
    concept = getattr(event, "condition_concept_id", None)
    if concept not in (None, "", "0"):
        return str(concept)
    source = getattr(event, "condition_source_value", None)
    if source not in (None, ""):
        return str(source)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PyHealth Patient objects from the OMOP tables in omop/."
    )
    parser.add_argument("--root", type=Path, default=Path("omop"))
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Load the 1000 patients PyHealth selects, instead of every patient.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".pyhealth_cache"))
    parser.add_argument("--min-occurrence", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("Loading OMOP tables...")
    dataset = load_omop(args.root, dev=args.dev, cache_dir=args.cache_dir)
    dataset.stats()
    show_patient(dataset)

    print("Building condition-history samples and splitting by patient...")
    train_ds, val_ds, test_ds = dataset.set_task(
        task=OMOPConditionHistoryTask(min_occurrence=args.min_occurrence),
        split=PatientSplit(ratios=(0.8, 0.1, 0.1), seed=42),
    )
    print(f"train samples: {len(train_ds)}")
    print(f"validation samples: {len(val_ds)}")
    print(f"test samples: {len(test_ds)}")
    if len(train_ds) == 0:
        print(
            "No training samples. Lower --min-occurrence or load more patients."
        )
        return

    sample = train_ds[0]
    print(f"sample patient_id: {sample['patient_id']}")
    print(f"sample conditions shape: {tuple(sample['conditions'].shape)}")
    processor = train_ds.input_processors["conditions"]
    print(f"vocabulary size: {len(processor.code_vocab)}")

    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(train_loader))
    print(f"batch keys: {list(batch.keys())}")
    print(f"batch conditions shape: {tuple(batch['conditions'].shape)}")


if __name__ == "__main__":
    main()
