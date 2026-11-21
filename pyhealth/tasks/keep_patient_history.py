"""One KEEP patient history from MIMIC-IV, through the usual task API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from pyhealth.medcode import DISEASE_CONCEPT_ID, OMOPVocabulary

from .base_task import BaseTask

_MISSING = {"", "none", "nan", "<na>", "nat"}


class KeepPatientHistory(BaseTask):
    """One sample per patient: each diagnosis, its source code, and its targets.

    Build it with the shared Athena :class:`~pyhealth.medcode.OMOPVocabulary`.
    ICD codes are mapped with that vocabulary's ``icd_mapper``. Ancestors come
    from ``condition_hierarchy``. A diagnosis is one ``hadm_id`` plus
    ``seq_num``. Identical copies of that pair collapse. Copies that name
    different source codes stay on the same diagnosis, are flagged, and
    contribute their targets once. The two-diagnosis threshold is applied
    after that roll-up.

    The dataset should use ``mimic4_keep_ehr.yaml``. That config keeps a
    diagnosis whose ``hadm_id`` matches no admission. The timestamp is the
    matching admission's ``admittime``, or ``None`` when there is no match.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.medcode import OMOPVocabulary
        >>> from pyhealth.tasks import KeepPatientHistory
        >>> vocabulary = OMOPVocabulary("~/data/athena")  # doctest: +SKIP
        >>> history_task = KeepPatientHistory(vocabulary)  # doctest: +SKIP
        >>> dataset = MIMIC4Dataset(  # doctest: +SKIP
        ...     ehr_root="~/data/mimic-iv/2.2",
        ...     ehr_tables=["diagnoses_icd"],
        ...     ehr_config_path="pyhealth/datasets/configs/mimic4_keep_ehr.yaml",
        ... )
        >>> histories = dataset.set_task(history_task, num_workers=1)  # doctest: +SKIP
        >>> histories[0]["conditions"]  # doctest: +SKIP
    """

    task_name: str = "KeepPatientHistory"
    input_schema: ClassVar[dict[str, str]] = {"conditions": "raw"}
    output_schema: ClassVar[dict[str, str]] = {"repeated_concepts": "raw"}

    def __init__(
        self,
        vocabulary: OMOPVocabulary,
        *,
        min_diagnoses: int = 2,
        max_depth: int = 5,
        include_root: bool = False,
    ) -> None:
        """Map ICD codes with ``vocabulary`` and roll them up to ancestors.

        Args:
            vocabulary: An :class:`~pyhealth.medcode.OMOPVocabulary` pointed
                at an unzipped Athena download.
            min_diagnoses: A concept is repeated when at least this many
                distinct diagnoses reach it. Default 2.
            max_depth: Ancestor cutoff passed to ``condition_hierarchy``.
                Default 5.
            include_root: Whether the Disease root may be a target.
                Default False.
        """
        super().__init__()
        if not isinstance(vocabulary, OMOPVocabulary):
            raise TypeError(
                "vocabulary must be an OMOPVocabulary, got "
                f"{type(vocabulary).__name__}"
            )
        if min_diagnoses < 1:
            raise ValueError("min_diagnoses must be at least 1")
        self.vocabulary = vocabulary
        self.min_diagnoses = min_diagnoses
        self.max_depth = max_depth
        self.include_root = include_root
        self.mapper = vocabulary.icd_mapper()
        self.hierarchy = vocabulary.condition_hierarchy(
            max_depth=max_depth, depth_measure="min"
        )

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Return the patient's history as one sample.

        Args:
            patient: A patient loaded from :class:`~pyhealth.datasets.MIMIC4Dataset`
                with the opt-in EHR config.

        Returns:
            A one-item list. ``conditions`` is one dict per diagnosis.
            ``repeated_concepts`` lists concept ids reached by at least
            ``min_diagnoses`` diagnoses.
        """
        return [patient_history(patient, self)]


def patient_history(patient: Any, task: KeepPatientHistory) -> dict[str, Any]:
    """Build one history dict from ``patient`` using ``task``'s vocabulary.

    Args:
        patient: A PyHealth patient with ``admissions`` and ``diagnoses_icd``.
        task: The history task that holds the mapper and the hierarchy.

    Returns:
        ``patient_id``, ``conditions``, and ``repeated_concepts``.

    Examples:
        >>> history = patient_history(patient, task)  # doctest: +SKIP
        >>> history["conditions"][0]["icd_code"]  # doctest: +SKIP
        '04119'
    """
    admissions = _admission_times(patient)
    grouped: dict[tuple[str | None, str | None], list[tuple[str | None, int | None]]] = {}
    order: list[tuple[str | None, str | None]] = []
    for event in patient.get_events(event_type="diagnoses_icd"):
        hadm_id = _text(getattr(event, "hadm_id", None))
        seq_num = _text(getattr(event, "seq_num", None))
        source = (_text(getattr(event, "icd_code", None)), _version(event))
        key = (hadm_id, seq_num)
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        if source not in grouped[key]:
            grouped[key].append(source)

    conditions = []
    counts: dict[int, int] = {}
    for hadm_id, seq_num in order:
        sources = grouped[(hadm_id, seq_num)]
        concept_ids = _concept_ids(task, sources)
        targets = _targets(task, concept_ids)
        for concept_id in targets:
            counts[concept_id] = counts.get(concept_id, 0) + 1
        conditions.append(
            {
                "diagnosis_id": f"{hadm_id or ''}:{seq_num or ''}",
                "hadm_id": hadm_id,
                "seq_num": seq_num,
                "timestamp": admissions.get(hadm_id) if hadm_id is not None else None,
                "icd_code": sources[0][0] if len(sources) == 1 else None,
                "icd_version": sources[0][1] if len(sources) == 1 else None,
                "sources": [
                    {"icd_code": code, "icd_version": version}
                    for code, version in sources
                ],
                "conflict": len(sources) > 1,
                "concept_ids": concept_ids,
                "targets": targets,
            }
        )
    repeated = sorted(
        concept_id
        for concept_id, count in counts.items()
        if count >= task.min_diagnoses
    )
    return {
        "patient_id": str(patient.patient_id),
        "conditions": conditions,
        "repeated_concepts": repeated,
    }


def _admission_times(patient: Any) -> dict[str, datetime | None]:
    times: dict[str, datetime | None] = {}
    for event in patient.get_events(event_type="admissions"):
        hadm_id = _text(getattr(event, "hadm_id", None))
        if hadm_id is not None and hadm_id not in times:
            times[hadm_id] = event.timestamp
    return times


def _concept_ids(
    task: KeepPatientHistory,
    sources: list[tuple[str | None, int | None]],
) -> list[int]:
    found: set[int] = set()
    for code, version in sources:
        if code is None or version is None:
            continue
        found.update(task.mapper.map(code, version))
    return sorted(found)


def _targets(task: KeepPatientHistory, concept_ids: list[int]) -> list[int]:
    root = DISEASE_CONCEPT_ID
    targets: set[int] = set()
    for concept_id in concept_ids:
        if concept_id in task.hierarchy and (task.include_root or concept_id != root):
            targets.add(concept_id)
        for ancestor_id, _distance in task.hierarchy.retained_ancestors(concept_id):
            if task.include_root or ancestor_id != root:
                targets.add(ancestor_id)
    return sorted(targets)


def _version(event: Any) -> int | None:
    text = _text(getattr(event, "icd_version", None))
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _MISSING:
        return None
    return text
