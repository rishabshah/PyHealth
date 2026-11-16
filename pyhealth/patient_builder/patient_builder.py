"""Filter an OMOP dataset down to a cohort of PyHealth patients.

``PatientBuilder`` is a single exact-code cohort filter. One condition row
whose code equals the requested string selects the patient.
``with_min_visits`` counts every ``visit_occurrence`` event, including visits
that do not carry the selected diagnosis. Neither filter rolls codes up an
ontology or requires a code to occur twice.

KEEP counts a condition over the whole history after ontology roll-up
(Appendix A.4). The prototype history task on the ``patient_builder`` branch
applies its threshold to raw codes before that mapping and returns nested
code strings grouped by visit, without diagnosis row ids. Rows that have no
visit are left out of that task. ``condition_records`` keeps those rows and
``concepts_with_repeated_diagnoses`` applies the row-level contract used by
the downstream KEEP path.
"""

from dataclasses import dataclass

from pyhealth.data import Patient

_MISSING = {"", "none", "nan", "<na>", "nat"}
_ICD_VERSION = {"ICD9CM": 9, "ICD10CM": 10}


class PatientBuilder:
    """Exact-code cohort filter that returns PyHealth :class:`Patient` objects.

    ``with_condition`` keeps a patient when one condition row matches the
    code. It does not roll that code up to ancestors and does not require a
    second occurrence. ``with_min_visits`` counts all ``visit_occurrence``
    events. Each of those filters calls ``dataset.get_patient`` for every
    candidate, and ``build`` loads the survivors again into one list. That
    list suits a small cohort. A whole-cohort scan should pass
    ``dataset.iter_patients()`` to the consumer and aggregate one patient
    at a time.

    Examples:
        >>> from pyhealth.patient_builder import PatientBuilder
        >>> patients = (
        ...     PatientBuilder(dataset)
        ...     .with_condition("201826")
        ...     .with_min_visits(2)
        ...     .build()
        ... )  # doctest: +SKIP
    """

    def __init__(self, dataset) -> None:
        """Start a cohort from every patient in ``dataset``.

        Args:
            dataset: An OMOP-backed dataset with ``unique_patient_ids``,
                ``get_patient``, ``get_condition_code``, and ``get_visit_count``.
                :class:`~pyhealth.datasets.MIMICOMOPDataset` provides these.
        """
        self.dataset = dataset
        self.patient_ids = list(dataset.unique_patient_ids)

    def with_condition(self, condition_code: str) -> "PatientBuilder":
        """Keep patients with one condition row equal to ``condition_code``.

        One matching row is enough. The comparison is the string from
        ``get_condition_code``: the standard concept when it is mapped, and
        the versioned source value when the concept id is 0.

        Args:
            condition_code: Concept id or source code, compared as a string.

        Returns:
            This builder, so another filter can be chained.
        """
        wanted = str(condition_code)
        filtered_ids = []
        for patient_id in self.patient_ids:
            patient = self.dataset.get_patient(patient_id)
            conditions = patient.get_events(event_type="condition_occurrence")
            for condition in conditions:
                if self.dataset.get_condition_code(condition) == wanted:
                    filtered_ids.append(patient_id)
                    break
        self.patient_ids = filtered_ids
        return self

    def with_min_visits(self, min_visits: int) -> "PatientBuilder":
        """Keep patients with at least ``min_visits`` visits.

        The count is every ``visit_occurrence`` event. A diagnosis that is
        not linked to a visit does not change this count.

        Args:
            min_visits: Minimum number of ``visit_occurrence`` events.

        Returns:
            This builder, so another filter can be chained.
        """
        filtered_ids = []
        for patient_id in self.patient_ids:
            patient = self.dataset.get_patient(patient_id)
            if self.dataset.get_visit_count(patient) >= min_visits:
                filtered_ids.append(patient_id)
        self.patient_ids = filtered_ids
        return self

    def build(self) -> list[Patient]:
        """Load the remaining patients and keep them all in one list.

        Each chained filter has already called ``get_patient`` for its
        candidates. This loads each survivor once more. The list is the
        right result for a small selected cohort. For a whole-cohort
        aggregation, pass ``dataset.iter_patients()`` to the consumer
        instead of calling ``build``.

        Returns:
            One :class:`~pyhealth.data.Patient` per remaining patient id.
        """
        return [self.dataset.get_patient(patient_id) for patient_id in self.patient_ids]


@dataclass(frozen=True)
class ConditionRecord:
    """One diagnosis row prepared for the KEEP mapper in PR #2.

    ``concept_id`` is set when the row already has a standard concept. Otherwise
    ``icd_code`` and ``icd_version`` are set for ``ICDToSNOMEDMapper.map(code,
    version=version)``. ``visit_id`` is ``None`` when the row has no visit link.
    ``diagnosis_id`` is ``condition_occurrence_id``, so two copies of the same
    code stay distinct.
    """

    diagnosis_id: str
    visit_id: str | None
    concept_id: str | None
    icd_code: str | None
    icd_version: int | None


def condition_records(patient) -> list[ConditionRecord]:
    """Return every condition event, including diagnoses with no visit link.

    Args:
        patient: A PyHealth patient loaded from an OMOP dataset.

    Returns:
        One record per ``condition_occurrence`` event.
    """
    return [
        condition_record(event)
        for event in patient.get_events(event_type="condition_occurrence")
    ]


def condition_record(event) -> ConditionRecord:
    """Turn one condition event into a mapper input.

    Args:
        event: A ``condition_occurrence`` event.

    Returns:
        The diagnosis id, visit id, and either a standard concept or an ICD
        code plus version.
    """
    diagnosis_id = getattr(event, "condition_occurrence_id", None)
    if _missing(diagnosis_id):
        raise ValueError("condition event is missing condition_occurrence_id")
    concept = getattr(event, "condition_concept_id", None)
    if not _missing(concept) and str(concept) != "0":
        return ConditionRecord(str(diagnosis_id), _visit_id(event), str(concept), None, None)
    code, version = _icd_code_and_version(getattr(event, "condition_source_value", None))
    return ConditionRecord(str(diagnosis_id), _visit_id(event), None, code, version)


def concepts_with_repeated_diagnoses(records, rollup, min_diagnoses: int = 2) -> set[str]:
    """Keep concepts that occur in at least ``min_diagnoses`` distinct rows.

    ``rollup(record)`` yields the concept ids reached from that diagnosis,
    including every ancestor path from PR #2. Those ids are unioned inside
    one ``diagnosis_id`` before they are counted. Two paths from one
    diagnosis to ancestor A contribute one occurrence. Two diagnoses that
    each reach A contribute two, so A meets a threshold of 2.

    Args:
        records: Diagnosis rows from :func:`condition_records`.
        rollup: Callable returning the concept ids for one record.
        min_diagnoses: Minimum number of distinct diagnosis ids.

    Returns:
        Concept ids whose diagnosis count is at least ``min_diagnoses``.
    """
    by_diagnosis: dict[str, set[str]] = {}
    for record in records:
        targets = {str(concept) for concept in rollup(record)}
        by_diagnosis.setdefault(record.diagnosis_id, set()).update(targets)
    counts: dict[str, int] = {}
    for targets in by_diagnosis.values():
        for concept in targets:
            counts[concept] = counts.get(concept, 0) + 1
    return {concept for concept, count in counts.items() if count >= min_diagnoses}


def _visit_id(event) -> str | None:
    visit = getattr(event, "visit_occurrence_id", None)
    if _missing(visit):
        return None
    return str(visit)


def _icd_code_and_version(source) -> tuple[str | None, int | None]:
    if _missing(source) or ":" not in str(source):
        return None, None
    vocabulary, code = str(source).split(":", 1)
    version = _ICD_VERSION.get(vocabulary.strip().upper())
    code = code.strip()
    if version is None or code == "":
        return None, None
    return code, version


def _missing(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in _MISSING
