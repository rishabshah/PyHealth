"""Filter an OMOP dataset down to a cohort of PyHealth patients.

``PatientBuilder`` composes exact-code filters on the dataset event table.
One condition row whose code equals the requested string selects the patient.
``with_min_visits`` counts every ``visit_occurrence`` event, including visits
that do not carry the selected diagnosis. Neither filter rolls codes up an
ontology or requires a code to occur twice. Patients are loaded when the
cohort is iterated or collected into a list.

KEEP counts a condition over the whole history after ontology roll-up
(Appendix A.4). The prototype history task on the ``patient_builder`` branch
applies its threshold to raw codes before that mapping and returns nested
code strings grouped by visit, without diagnosis row ids. Rows that have no
visit are left out of that task. ``condition_records`` keeps those rows and
``concepts_with_repeated_diagnoses`` applies the row-level contract used by
the downstream KEEP path.
"""

from collections import Counter
from dataclasses import dataclass

import polars as pl

from pyhealth.data import Patient

_MISSING = {"", "none", "nan", "<na>", "nat"}
_ICD_VERSION = {"ICD9CM": 9, "ICD10CM": 10}
_CONCEPT = "condition_occurrence/condition_concept_id"
_SOURCE = "condition_occurrence/condition_source_value"


class PatientBuilder:
    """Exact-code cohort filter that returns PyHealth :class:`Patient` objects.

    ``with_condition`` and ``with_min_visits`` add a semi-join to the event
    table and do not load patients. The join keeps every event of a selected
    patient, so the cohort still has that patient's full history.
    ``iter_patients`` loads one patient at a time. ``build`` collects that
    iterator into a list, which is convenient for a small cohort.

    Examples:
        >>> from pyhealth.patient_builder import PatientBuilder
        >>> patients = (
        ...     PatientBuilder(dataset)
        ...     .with_condition("201826")
        ...     .with_min_visits(2)
        ...     .build()
        ... )  # doctest: +SKIP
        >>> for patient in (
        ...     PatientBuilder(dataset).with_min_visits(2).iter_patients()
        ... ):  # doctest: +SKIP
        ...     len(patient.get_events(event_type="visit_occurrence"))
    """

    def __init__(self, dataset) -> None:
        """Start a cohort from every event in ``dataset``.

        Args:
            dataset: A dataset with ``global_event_df`` and
                ``iter_patients``. :class:`~pyhealth.datasets.MIMICOMOPDataset`
                provides these.
        """
        self.dataset = dataset
        self._events = dataset.global_event_df

    @property
    def patient_ids(self) -> list[str]:
        """Return the selected patient ids without building patient objects.

        Returns:
            Patient ids in first-seen order on the filtered event table.
        """
        return (
            self._events.select(pl.col("patient_id").cast(pl.Utf8))
            .unique(maintain_order=True)
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )

    def with_condition(self, condition_code: str) -> "PatientBuilder":
        """Keep patients with one condition row equal to ``condition_code``.

        One matching row is enough. The comparison is the standard concept
        when it is mapped, and the versioned source value when the concept
        id is 0. The plan stays lazy until patients are read.

        Args:
            condition_code: Concept id or source code, compared as a string.

        Returns:
            This builder, so another filter can be chained.
        """
        code = _condition_code(self._events)
        eligible = (
            self._events.filter(pl.col("event_type") == "condition_occurrence")
            .filter(code == str(condition_code))
            .select("patient_id")
            .unique()
        )
        return self._restrict(eligible)

    def with_min_visits(self, min_visits: int) -> "PatientBuilder":
        """Keep patients with at least ``min_visits`` visits.

        The count is every ``visit_occurrence`` event. A diagnosis that is
        not linked to a visit does not change this count. The plan stays
        lazy until patients are read.

        Args:
            min_visits: Minimum number of ``visit_occurrence`` events.

        Returns:
            This builder, so another filter can be chained.
        """
        if min_visits <= 0:
            return self
        eligible = (
            self._events.filter(pl.col("event_type") == "visit_occurrence")
            .group_by("patient_id")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") >= min_visits)
            .select("patient_id")
        )
        return self._restrict(eligible)

    def iter_patients(self):
        """Load each selected patient when the caller asks for the next one.

        ``dataset.iter_patients`` still reads one patient at a time. The
        filters have already been joined onto the event table, so only the
        selected patients are read.

        Returns:
            An iterator of :class:`~pyhealth.data.Patient` objects. Each one
            still contains that patient's full event history.
        """
        return self.dataset.iter_patients(df=self._events)

    def build(self) -> list[Patient]:
        """Collect the selected patients into one list.

        This is the list to inspect for a small cohort. It reads each
        selected patient once, through :meth:`iter_patients`.

        Returns:
            One :class:`~pyhealth.data.Patient` per remaining patient id.
        """
        return list(self.iter_patients())

    def _restrict(self, eligible: pl.LazyFrame) -> "PatientBuilder":
        self._events = self._events.join(eligible, on="patient_id", how="semi")
        return self


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


def collapse_diagnosis_copies(records, rollup) -> list[dict]:
    """Collapse identical copies and flag copies that disagree.

    Copies that share a ``diagnosis_id`` and the same concept, ICD code, and
    version become one diagnosis. Copies of that id with a different source
    stay on the same diagnosis, with ``conflict`` set, and their targets are
    unioned. One diagnosis then contributes each target once.

    Args:
        records: Diagnosis rows from :func:`condition_records`.
        rollup: Callable returning the concept ids for one record.

    Returns:
        One dict per diagnosis id, in first-seen order, with ``targets`` and
        ``conflict``.

    Examples:
        >>> rows = [ConditionRecord("1", "10", None, "04119", 9)]
        >>> collapse_diagnosis_copies(rows, lambda record: ["100"])[0]["conflict"]
        False
    """
    grouped: dict[str, dict[tuple, set[str]]] = {}
    order: list[str] = []
    for record in records:
        signature = (record.concept_id, record.icd_code, record.icd_version)
        targets = {str(concept) for concept in rollup(record)}
        if record.diagnosis_id not in grouped:
            order.append(record.diagnosis_id)
            grouped[record.diagnosis_id] = {}
        copies = grouped[record.diagnosis_id]
        copies[signature] = copies.get(signature, set()) | targets
    collapsed = []
    for diagnosis_id in order:
        copies = grouped[diagnosis_id]
        targets: set[str] = set()
        for copy_targets in copies.values():
            targets |= copy_targets
        collapsed.append(
            {
                "diagnosis_id": diagnosis_id,
                "conflict": len(copies) > 1,
                "targets": targets,
            }
        )
    return collapsed


def conflicting_diagnosis_ids(records, rollup) -> set[str]:
    """Return diagnosis ids whose copies do not carry the same source.

    Args:
        records: Diagnosis rows from :func:`condition_records`.
        rollup: Callable returning the concept ids for one record.

    Returns:
        Diagnosis ids flagged by :func:`collapse_diagnosis_copies`.

    Examples:
        >>> same = ConditionRecord("1", "10", None, "04119", 9)
        >>> other = ConditionRecord("1", "10", None, "I214", 10)
        >>> conflicting_diagnosis_ids([same, other], lambda record: ["100"])
        {'1'}
    """
    return {
        item["diagnosis_id"]
        for item in collapse_diagnosis_copies(records, rollup)
        if item["conflict"]
    }


def concepts_with_repeated_diagnoses(
    records, rollup, min_diagnoses: int = 2
) -> set[str]:
    """Keep concepts that occur in at least ``min_diagnoses`` distinct diagnoses.

    Repeated ids are checked before ``rollup`` runs. An identical copy is
    kept once. A copy that disagrees on any field raises ``ValueError``.
    Each remaining diagnosis contributes each mapped target once, so two
    ancestor paths from one diagnosis count as one occurrence. Two diagnoses
    that each reach ancestor A contribute two.

    Args:
        records: Diagnosis rows from :func:`condition_records`.
        rollup: Callable returning the concept ids for one record.
        min_diagnoses: Minimum number of distinct diagnosis ids.

    Returns:
        Concept ids whose diagnosis count is at least ``min_diagnoses``.

    Raises:
        ValueError: The same diagnosis id appears as two different records.

    Examples:
        >>> one = ConditionRecord("1", "10", "435459", None, None)
        >>> two = ConditionRecord("2", None, None, "04119", 9)
        >>> concepts_with_repeated_diagnoses(
        ...     [one, two], lambda record: ["A"]
        ... )
        {'A'}
    """
    by_id: dict[str, ConditionRecord] = {}
    for record in records:
        previous = by_id.setdefault(record.diagnosis_id, record)
        if previous != record:
            raise ValueError(f"Conflicting diagnosis {record.diagnosis_id}")

    counts: Counter[str] = Counter()
    for record in by_id.values():
        counts.update({str(target) for target in rollup(record)})
    return {concept for concept, count in counts.items() if count >= min_diagnoses}


def _condition_code(events: pl.LazyFrame) -> pl.Expr:
    """Match ``MIMICOMOPDataset.get_condition_code`` on the event table."""
    names = events.collect_schema().names()
    if _CONCEPT not in names:
        return pl.lit(None, dtype=pl.Utf8)
    concept = pl.col(_CONCEPT).cast(pl.Utf8)
    unmapped = concept.is_null() | concept.is_in(["", "0"])
    if _SOURCE not in names:
        return pl.when(unmapped).then(pl.lit(None, dtype=pl.Utf8)).otherwise(concept)
    source = pl.col(_SOURCE).cast(pl.Utf8)
    blank_source = source.is_null() | (source == "")
    return (
        pl.when(~unmapped)
        .then(concept)
        .when(~blank_source)
        .then(source)
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )


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
