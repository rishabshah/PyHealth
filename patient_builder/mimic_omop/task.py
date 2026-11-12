"""PyHealth dataset and mortality task for the converted MIMIC-IV hosp OMOP tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyhealth.tasks import BaseTask

CONFIG_PATH = Path(__file__).with_name("omop.yaml")


def load_dataset(root: str | Path, dev: bool = False, cache_dir: str | Path | None = None):
    """Load converted tables with PyHealth's OMOPDataset.

    Person, visit, and death load by default. Conditions, procedures, and
    observation periods are requested here. The config adds the ICD source
    values that this conversion stores beside concept id 0.
    """
    from pyhealth.datasets import OMOPDataset

    return OMOPDataset(
        root=str(root),
        tables=[
            "condition_occurrence",
            "procedure_occurrence",
            "observation_period",
        ],
        dataset_name="mimic4_hosp_omop",
        config_path=str(CONFIG_PATH),
        dev=dev,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )


class MimicHospMortalityTask(BaseTask):
    """Predict whether the patient has an OMOP death record.

    Samples are one hospital visit. Codes are standard concept ids when the
    ETL has mapped them, and ``VOCAB:code`` source values otherwise. This
    hosp extract has no drug table, so drugs are not part of the input.
    """

    task_name: str = "MimicHospMortality"
    input_schema: dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
    }
    output_schema: dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        died = len(patient.get_events(event_type="death")) > 0
        samples = []
        for visit in patient.get_events(event_type="visit_occurrence"):
            visit_id = str(getattr(visit, "visit_occurrence_id", "") or "")
            if not visit_id or visit_id == "None":
                continue
            conditions = _codes(
                patient,
                "condition_occurrence",
                visit_id,
                "condition_source_value",
                "condition_concept_id",
            )
            procedures = _codes(
                patient,
                "procedure_occurrence",
                visit_id,
                "procedure_source_value",
                "procedure_concept_id",
            )
            if not conditions and not procedures:
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": visit_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "mortality": int(died),
                }
            )
        return samples


def _codes(patient, event_type: str, visit_id: str, source_attr: str, concept_attr: str) -> list[str]:
    events = patient.get_events(
        event_type=event_type,
        filters=[("visit_occurrence_id", "==", visit_id)],
    )
    codes = []
    for event in events:
        code = _event_code(event, source_attr, concept_attr)
        if code is not None:
            codes.append(code)
    return codes


def _event_code(event, source_attr: str, concept_attr: str) -> str | None:
    concept = getattr(event, concept_attr, None)
    if concept not in (None, "", "0"):
        return str(concept)
    source = getattr(event, source_attr, None)
    if source not in (None, ""):
        return str(source)
    return None
