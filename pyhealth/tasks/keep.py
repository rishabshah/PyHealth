# Author: lee-leonardo (NetID: ll37)
# Description: KEEP diagnosis histories, ancestor roll-up, and patient presence.
#   KEEP (Elhussein et al., CHIL 2025), https://arxiv.org/abs/2510.05049

"""KEEP cohort workflows for raw OMOP patients and mapped MIMIC histories."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import polars as pl
import torch

from pyhealth.graph import KnowledgeGraph
from pyhealth.graph.cooccurrence import build_cooccurrence
from pyhealth.medcode import DISEASE_CONCEPT_ID, OMOPVocabulary
from pyhealth.medcode.omop import ICDToSNOMEDMapper
from pyhealth.processors import AncestorRollupProcessor

from .base_task import BaseTask

if TYPE_CHECKING:
    from pyhealth.data import Patient


_OMOP_HIERARCHY_RELATION = "is_a"


def build_keep_omop_ancestor_rollup(
    vocabulary: OMOPVocabulary,
    *,
    target_policy: str,
    include_root: bool,
    max_depth: int = 5,
    on_unmapped: str = "error",
    path_source: str = "direct",
    node_features_by_code: Mapping[str, torch.Tensor] | None = None,
) -> AncestorRollupProcessor:
    """Build an ancestor roll-up processor over KEEP's OMOP Disease hierarchy.

    The factory shares the vocabulary reader's queries for the uncut SNOMED
    Condition subtree rooted at Disease (concept 4274025) and its direct
    "Is a" edges. Minimum separations from Disease in ``CONCEPT_ANCESTOR``
    supply the authoritative root depths, following KEEP appendix A.1.1.
    The processor's ``max_depth`` governs *target eligibility* over the
    uncut subtree, so deep sources and intermediates stay traversable.
    For the embedding stage, choose the training vocabulary explicitly
    and supply trained vectors by concept code for projection.

    The classification universe is every valid standard SNOMED concept
    with its domain, so ``unknown_code`` (not a valid standard SNOMED
    concept), ``outside_domain`` (SNOMED concept outside ``Condition``)
    and ``unreachable`` (Condition concept outside the Disease subtree)
    stay distinct. Concept codes are OMOP ``concept_id`` strings at the
    processor boundary. Decode projected ``node_ids`` through the graph's
    ``id2entity`` lookup.

    Direct mode builds paths from the direct ``Is a`` rows. Closure mode
    also loads separation-one ``CONCEPT_ANCESTOR`` pairs within the subtree.
    The public hierarchy export offers orphan-rescue shortcuts through
    ``connect_orphans``. Adding those shortcuts to KEEP traversal remains
    a separate policy decision.

    Args:
        vocabulary: an :class:`~pyhealth.medcode.OMOPVocabulary` pointed
            at an unzipped Athena download.
        target_policy: ``"first_retained"`` or ``"all_retained"``.
        include_root: whether the Disease root may appear as a target.
        max_depth: roll-up cutoff in minimum closure levels from the
            root (eligibility, not hierarchy truncation). Default 5.
        on_unmapped: ``"error"`` or ``"empty"``. Default ``"error"``.
        path_source: ``"direct"`` or ``"closure"``; see the class
            documentation.
        node_features_by_code: Optional pretrained vectors keyed by the
            string OMOP concept ID. Vectors must share a shape, dtype
            and device. Traversal entities without vectors receive zeros.

    Returns:
        A configured :class:`AncestorRollupProcessor` with a
        KnowledgeGraph over the uncut Disease subtree.

    Raises:
        TypeError: ``vocabulary`` is not an
            :class:`~pyhealth.medcode.OMOPVocabulary`.
        ValueError: invalid configuration values, or invalid hierarchy
            structure in the selected edge set.

    Example:
        >>> from pyhealth.medcode import OMOPVocabulary
        >>> from pyhealth.tasks import build_keep_omop_ancestor_rollup
        >>> vocabulary = OMOPVocabulary("~/data/athena")  # doctest: +SKIP
        >>> processor = build_keep_omop_ancestor_rollup(
        ...     vocabulary, target_policy="first_retained",
        ...     include_root=False, max_depth=5,
        ... )  # doctest: +SKIP
        >>> mapper = vocabulary.icd_mapper()  # doctest: +SKIP
        >>> concept_ids = mapper.map("I214", version=10)  # doctest: +SKIP
        >>> processor.rollup_codes([str(concept_ids[0])])  # doctest: +SKIP
        [RollupResult(source='4140231', targets=('4140231',), ...)]
    """
    if not isinstance(vocabulary, OMOPVocabulary):
        raise TypeError(
            "vocabulary must be an OMOPVocabulary, got "
            f"{type(vocabulary).__name__}"
        )

    # The uncut subtree: the processor's max_depth governs target
    # eligibility over it, so no build-time truncation here. KEEP's
    # depth measure ("min") is pinned; "max" stays the reader's option.
    # No orphan rescue: the deferral is documented above.
    root_concept_id = DISEASE_CONCEPT_ID
    nodes = vocabulary._rooted_condition_nodes(root_concept_id)
    edges = vocabulary._direct_is_a_edges(nodes)
    # Classification universe: every valid standard SNOMED concept, all
    # domains (mirrors the reader's own validity/standard filters).
    universe = (
        vocabulary.scan("CONCEPT")
        .filter(
            (pl.col("vocabulary_id") == "SNOMED")
            & (pl.col("standard_concept") == "S")
            & (
                pl.col("invalid_reason").is_null()
                | (pl.col("invalid_reason").str.strip_chars() == "")
            )
        )
        .select("concept_id", "domain_id")
        .collect(engine="streaming")
    )
    domain_by_code = {
        str(concept_id): domain_id
        for concept_id, domain_id in universe.iter_rows()
    }
    root = str(root_concept_id)
    if root not in domain_by_code:
        raise ValueError(
            f"the hierarchy root {root!r} is not a valid standard SNOMED "
            "concept in this vocabulary"
        )
    direct_edges = [
        (str(parent), str(child))
        for parent, child in edges.select(
            "parent_id", "child_id"
        ).iter_rows()
    ]
    closure_pairs = None
    if path_source == "closure":
        node_ids = nodes.select("concept_id").lazy()
        separation_one = (
            vocabulary.scan("CONCEPT_ANCESTOR")
            .filter(pl.col("min_levels_of_separation") == 1)
            .select(
                pl.col("descendant_concept_id").alias("concept_id"),
                pl.col("ancestor_concept_id").alias("ancestor_id"),
            )
            .join(node_ids, on="concept_id", how="semi")
            .join(
                node_ids,
                left_on="ancestor_id",
                right_on="concept_id",
                how="semi",
            )
            .collect(engine="streaming")
        )
        closure_pairs = [
            (str(ancestor), str(descendant))
            for descendant, ancestor in separation_one.iter_rows()
        ]
    root_depths = {
        str(concept_id): int(depth)
        for concept_id, depth in nodes.select("concept_id", "min_depth").iter_rows()
    }
    # Explicit IDs retain concepts whose incoming edges were filtered out.
    entity2id = {code: index for index, code in enumerate(sorted(root_depths))}
    node_features = None
    if node_features_by_code is not None:
        if not isinstance(node_features_by_code, Mapping):
            raise TypeError(
                "node_features_by_code must be a mapping, got "
                f"{type(node_features_by_code).__name__}"
            )
        for code, vector in node_features_by_code.items():
            if code not in entity2id:
                raise ValueError(
                    f"Node features supplied for unknown entity {code!r}"
                )
            if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
                raise ValueError(
                    "Each node feature must be a one-dimensional tensor"
                )
            if node_features is None:
                node_features = vector.new_zeros(len(entity2id), vector.numel())
            if (
                vector.shape != node_features.shape[1:]
                or vector.dtype != node_features.dtype
                or vector.device != node_features.device
            ):
                raise ValueError(
                    "Node features must share a shape, dtype and device"
                )
            node_features[entity2id[code]] = vector.detach()
    relation = _OMOP_HIERARCHY_RELATION
    knowledge_graph = KnowledgeGraph(
        triples=[(parent, relation, child) for parent, child in direct_edges],
        entity2id=entity2id,
        relation2id={relation: 0},
        node_features=node_features,
    )
    return AncestorRollupProcessor(
        knowledge_graph=knowledge_graph,
        domain_by_code=domain_by_code,
        hierarchy_relation=relation,
        root=root,
        target_policy=target_policy,
        include_root=include_root,
        max_depth=max_depth,
        on_unmapped=on_unmapped,
        path_source=path_source,
        root_depths=root_depths,
        closure_pairs=closure_pairs,
        include_external_codes=True,
        loader_provenance={
            "hierarchy_relation": relation,
            "vocabulary_source": "pyhealth.medcode.OMOPVocabulary",
            "hierarchy_root": root,
            "depth_measure": "min",
            "hierarchy_max_depth": "None (uncut subtree)",
            "rescued_edges": "0 (orphan rescue deferred)",
        },
    )


class _MatrixBuilder(Protocol):
    def __call__(
        self, patient_concepts: Iterable[AbstractSet[str]], *,
        concept_ids: Sequence[str],
    ) -> torch.Tensor: ...


_ICD_PREFIX = {"9": "ICD9CM", "10": "ICD10CM"}

#: Event type read once from every patient.
CONDITION_EVENT_TYPE = "condition_occurrence"

#: Event attributes the OMOP config must expose on condition occurrences.
DIAGNOSIS_ID_ATTR = "condition_occurrence_id"
CONCEPT_ID_ATTR = "condition_concept_id"
SOURCE_VALUE_ATTR = "condition_source_value"
VISIT_ID_ATTR = "visit_occurrence_id"

#: Source-value prefixes the peer ETL writes beside concept id 0, mapped to
#: the ICD version ``ICDToSNOMEDMapper.map`` expects.
ICD_PREFIX_TO_VERSION = {"ICD9CM": 9, "ICD10CM": 10}

#: A target counts as present for a patient once this many distinct
#: diagnoses resolve to it (KEEP appendix A.4's two-occurrence rule).
PRESENCE_THRESHOLD = 2

#: Every reason a diagnosis event can be counted as wholly unmapped, in the
#: order ``KeepCooccurrenceResult.unmapped_events`` reports them.
UNMAPPED_EVENT_REASONS = (
    "missing_source",
    "unsupported_source_format",
    "empty_icd_mapping",
    "empty_rollup",
)

#: Attribute names compared when the same diagnosis id repeats.
_DIAGNOSIS_FIELDS = (
    "condition_concept_id",
    "condition_source_value",
    "timestamp",
    "visit_occurrence_id",
)


@dataclass(frozen=True)
class KeepCooccurrenceResult:
    """Outcome of :func:`build_keep_cooccurrence`.

    Example:
        >>> import torch
        >>> from pyhealth.tasks import KeepCooccurrenceResult
        >>> matrix = torch.sparse_coo_tensor(
        ...     torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]),
        ...     torch.ones(4, dtype=torch.int64),
        ...     (2, 2),
        ... ).coalesce()
        >>> result = KeepCooccurrenceResult(
        ...     matrix=matrix,
        ...     concept_ids=("10", "11"),
        ...     patients_processed=1,
        ...     patients_with_present_targets=1,
        ...     unique_diagnosis_events=2,
        ...     mapped_events=2,
        ...     unmapped_events={
        ...         "missing_source": 0,
        ...         "unsupported_source_format": 0,
        ...         "empty_icd_mapping": 0,
        ...         "empty_rollup": 0,
        ...     },
        ... )
        >>> result.matrix.to_dense().tolist()
        [[1, 1], [1, 1]]
        >>> result.concept_ids
        ('10', '11')
    """

    matrix: torch.Tensor
    """Matrix returned by the configured matrix builder.

    The default is a sparse, coalesced CPU COO tensor of ``int64`` patient
    counts. Off-diagonal entries are mirrored, and the diagonal counts
    patient prevalence. Both axes follow ``concept_ids``.
    """

    concept_ids: tuple[str, ...]
    """Axis labels in the exact supplied order, one per matrix row/column."""

    patients_processed: int
    """Number of patients consumed from the iterable."""

    patients_with_present_targets: int
    """Patients with at least one present (two-diagnosis) target."""

    unique_diagnosis_events: int
    """Diagnosis rows after collapsing identical copies of one id."""

    mapped_events: int
    """Unique diagnosis events whose roll-up produced at least one target.

    ``mapped_events`` plus the sum of ``unmapped_events`` equals
    ``unique_diagnosis_events``.
    """

    unmapped_events: Mapping[str, int]
    """Wholly unmapped diagnosis events, one count per reason.

    Always reports every key of :data:`UNMAPPED_EVENT_REASONS`, including
    zero counts: ``missing_source`` (neither a nonzero concept id nor a
    source value), ``unsupported_source_format`` (source value without a
    usable ``ICD9CM:``/``ICD10CM:`` prefix), ``empty_icd_mapping`` (the
    mapper returned no concept for the code), and ``empty_rollup`` (mapped
    concepts whose roll-up produced no target under ``on_unmapped="empty"``).
    """


def build_keep_cooccurrence(
    patients: Iterable[Patient],
    *,
    rollup: AncestorRollupProcessor,
    icd_mapper: ICDToSNOMEDMapper,
    concept_ids: Sequence[str],
    matrix_builder: _MatrixBuilder = build_cooccurrence,
) -> KeepCooccurrenceResult:
    """Builds the sparse patient co-occurrence matrix over rolled-up codes.

    Reads ``patient.get_events(event_type="condition_occurrence")`` once per
    patient. Each diagnosis row is identified by its
    ``condition_occurrence_id`` (scoped to its patient): identical copies of
    one row collapse, copies with conflicting concept id, source value,
    timestamp, or visit id are rejected, and rows without a visit id are
    kept. Distinct rows that share a visit and a code stay distinct.

    Mapping follows the OMOP conventions of the peer's converter: a nonzero
    ``condition_concept_id`` wins; otherwise the source value's
    ``ICD9CM:``/``ICD10CM:`` prefix selects the ICD version and the code
    text (leading zeros preserved) goes to ``icd_mapper.map()``. The roll-up
    targets of all mapped concepts of one diagnosis are unioned, so a
    one-to-many ICD mapping or a multi-parent roll-up contributes each
    target at most once per diagnosis. Under ``on_unmapped="error"`` the
    processor's :class:`~pyhealth.processors.UnmappedCodeError` propagates;
    under ``"empty"``, events with any target count as mapped and wholly
    unmapped events are counted once by reason.

    A target is *present* for a patient after two distinct diagnoses resolve
    to it. Each unordered pair of present targets counts once per patient,
    off-diagonal entries are mirrored, and the diagonal holds the patient
    prevalence of each concept.

    Args:
        patients: single-pass iterable of PyHealth patients, for example
            ``OMOPDataset.iter_patients()``. Each must expose ``patient_id``
            and ``get_events``; repeated patient ids are rejected. The
            condition events must expose :data:`DIAGNOSIS_ID_ATTR`,
            :data:`CONCEPT_ID_ATTR`, :data:`SOURCE_VALUE_ATTR`, and
            :data:`VISIT_ID_ATTR` (the OMOP config controls this).
        rollup: a configured :class:`~pyhealth.processors.AncestorRollupProcessor`
            supplying ``rollup_codes()`` and ``retained_codes``.
        icd_mapper: the peer's :class:`~pyhealth.medcode.ICDToSNOMEDMapper`
            supplying ``map()``.
        concept_ids: ordered axis labels. Must be a sequence containing each
            retained concept id exactly once and nothing else; the retained
            graph, the matrix axes, and later embeddings share this order.
            Isolated concepts and concepts absent from the cohort keep zero
            rows and columns.
        matrix_builder: callable that consumes the patient concept sets once
            before returning the sparse matrix with the supplied axes.
            Use ``functools.partial`` with
            :func:`~pyhealth.graph.cooccurrence.build_cooccurrence` to set a
            device or apply a transform to the raw counts.

    Returns:
        A :class:`KeepCooccurrenceResult` with the matrix of shape ``(V, V)``,
        where ``V`` is the retained vocabulary size, the exact ``concept_ids``
        tuple, and the patient, diagnosis, and mapping counts.

    Raises:
        TypeError: ``patients`` is not an iterable of patients, an element
            lacks ``patient_id``/``get_events``, ``rollup`` or ``icd_mapper``
            lacks its required method, or ``concept_ids`` is not a sequence
            of strings.
        ValueError: a repeated patient id, a missing or empty diagnosis id,
            conflicting copies of one diagnosis id, or invalid axes
            (duplicate, missing, or extra concept ids).
        UnmappedCodeError: a diagnosis cannot be mapped under the
            processor's ``on_unmapped="error"``.

    Example:
        >>> from datetime import datetime
        >>> import polars as pl
        >>> from pyhealth.data import Patient
        >>> from pyhealth.graph import KnowledgeGraph
        >>> from pyhealth.tasks import build_keep_cooccurrence
        >>> from pyhealth.medcode import ICDToSNOMEDMapper
        >>> from pyhealth.processors import AncestorRollupProcessor
        >>> kg = KnowledgeGraph(
        ...     triples=[
        ...         ("1", "is_a", "10"), ("1", "is_a", "11"),
        ...         ("10", "is_a", "801"), ("10", "is_a", "802"),
        ...         ("11", "is_a", "802"), ("11", "is_a", "803"),
        ...     ],
        ...     entity2id={"1": 0, "10": 1, "11": 2, "12": 3,
        ...                "801": 4, "802": 5, "803": 6},
        ...     relation2id={"is_a": 0},
        ... )
        >>> codes = ("1", "10", "11", "12", "801", "802", "803")
        >>> processor = AncestorRollupProcessor(
        ...     knowledge_graph=kg,
        ...     domain_by_code={code: "Condition" for code in codes},
        ...     hierarchy_relation="is_a", root="1",
        ...     target_policy="all_retained", include_root=False,
        ...     max_depth=1, on_unmapped="empty",
        ...     root_depths={"1": 0, "10": 1, "11": 1, "12": 1,
        ...                  "801": 2, "802": 2, "803": 2},
        ... )
        >>> mapper = ICDToSNOMEDMapper(pl.DataFrame({
        ...     "icd_version": [10, 10, 9],
        ...     "icd_code": ["I214", "I25X", "0010"],
        ...     "source_code": ["I21.4", "I25X", "001.0"],
        ...     "source_concept_id": [9001, 9002, 9003],
        ...     "concept_id": [802, 801, 803],
        ...     "concept_name": ["Two-parent", "Under 10", "Under 11"],
        ...     "target_domain_id": ["Condition"] * 3,
        ...     "target_vocabulary_id": ["SNOMED"] * 3,
        ... }))
        >>> events = pl.DataFrame({
        ...     "patient_id": ["1"] * 2,
        ...     "event_type": ["condition_occurrence"] * 2,
        ...     "timestamp": [datetime(2020, 1, 1)] * 2,
        ...     "_source_row": [0, 1],
        ...     "condition_occurrence/condition_occurrence_id": ["301", "302"],
        ...     "condition_occurrence/condition_concept_id": [None, "801"],
        ...     "condition_occurrence/condition_source_value":
        ...         ["ICD10CM:I25X", None],
        ...     "condition_occurrence/visit_occurrence_id": ["7", "8"],
        ... })
        >>> patient = Patient(patient_id="1", data_source=events)
        >>> result = build_keep_cooccurrence(
        ...     [patient], rollup=processor, icd_mapper=mapper,
        ...     concept_ids=["10", "11", "12"],
        ... )
        >>> result.matrix.to_dense().tolist()
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        >>> result.concept_ids
        ('10', '11', '12')
        >>> result.unique_diagnosis_events, result.mapped_events
        (2, 2)
        >>> result.unmapped_events["empty_rollup"]
        0
        >>> result.mapped_events + sum(result.unmapped_events.values()) == (
        ...     result.unique_diagnosis_events
        ... )
        True
    """
    if not callable(getattr(icd_mapper, "map", None)):
        raise TypeError(
            "icd_mapper must be an ICDToSNOMEDMapper exposing map(), got "
            f"{type(icd_mapper).__name__}"
        )
    if isinstance(patients, (str, bytes)):
        raise TypeError("patients must be an iterable of Patient objects")
    return _build_keep_result(
        _omop_diagnoses(patients, icd_mapper),
        rollup=rollup,
        concept_ids=concept_ids,
        matrix_builder=matrix_builder,
    )


def _omop_diagnoses(
    patients: Iterable[Patient], icd_mapper: ICDToSNOMEDMapper
) -> Iterator[tuple[str, Iterable[tuple[Sequence[str], str | None]]]]:
    """Adapt raw OMOP patient events to the KEEP diagnosis input."""
    for position, patient in enumerate(patients):
        patient_id = _omop_patient_id(patient, position)
        yield patient_id, _resolve_omop_diagnoses(patient, patient_id, icd_mapper)


def _resolve_omop_diagnoses(
    patient: Patient, patient_id: str, icd_mapper: ICDToSNOMEDMapper
) -> Iterator[tuple[Sequence[str], str | None]]:
    diagnoses = _collapsed_diagnoses(_condition_events(patient, patient_id), patient_id)
    for concept, source, _, _ in diagnoses.values():
        yield _resolve_canonical_ids(concept, source, icd_mapper)


def _build_keep_result(
    patients: Iterable[tuple[str, Iterable[tuple[Sequence[str], str | None]]]],
    *,
    rollup: AncestorRollupProcessor,
    concept_ids: Sequence[str],
    matrix_builder: _MatrixBuilder,
) -> KeepCooccurrenceResult:
    if not callable(getattr(rollup, "rollup_codes", None)) or not hasattr(
        rollup, "retained_codes"
    ):
        raise TypeError(
            "rollup must be a configured AncestorRollupProcessor exposing "
            "rollup_codes() and retained_codes, got "
            f"{type(rollup).__name__}"
        )
    axes = tuple(_validated_axes(concept_ids, set(rollup.retained_codes)))
    seen_patient_ids: set[str] = set()
    patients_with_present_targets = 0
    unique_diagnosis_events = 0
    mapped_events = 0
    unmapped_events = dict.fromkeys(UNMAPPED_EVENT_REASONS, 0)

    def present_concepts() -> Iterator[set[str]]:
        nonlocal patients_with_present_targets, unique_diagnosis_events, mapped_events
        for position, (patient_id, diagnoses) in enumerate(patients):
            _validate_patient_id(patient_id, position)
            if patient_id in seen_patient_ids:
                raise ValueError(
                    f"patient {patient_id!r} (patients[{position}]) appears more "
                    "than once; supply each patient exactly once"
                )
            seen_patient_ids.add(patient_id)
            target_counts: Counter[str] = Counter()
            for canonical_ids, reason in diagnoses:
                unique_diagnosis_events += 1
                if reason is not None:
                    unmapped_events[reason] += 1
                    continue
                targets = {
                    target
                    for result in rollup.rollup_codes(canonical_ids)
                    for target in result.targets
                }
                if targets:
                    mapped_events += 1
                    target_counts.update(targets)
                else:
                    unmapped_events["empty_rollup"] += 1
            present = {
                code for code, count in target_counts.items()
                if count >= PRESENCE_THRESHOLD
            }
            patients_with_present_targets += bool(present)
            yield present

    matrix = matrix_builder(present_concepts(), concept_ids=axes)
    return KeepCooccurrenceResult(
        matrix=matrix,
        concept_ids=axes,
        patients_processed=len(seen_patient_ids),
        patients_with_present_targets=patients_with_present_targets,
        unique_diagnosis_events=unique_diagnosis_events,
        mapped_events=mapped_events,
        unmapped_events=unmapped_events,
    )


# --- argument validation ------------------------------------------------------


def _validated_axes(concept_ids: Any, retained: set[str]) -> list[str]:
    """Checks the axis sequence against the retained vocabulary."""
    if isinstance(concept_ids, (str, bytes)) or not isinstance(
        concept_ids, Sequence
    ):
        raise TypeError(
            "concept_ids must be a sequence (list or tuple) of concept ID "
            f"strings, got {type(concept_ids).__name__}"
        )
    axes = list(concept_ids)
    for position, code in enumerate(axes):
        if not isinstance(code, str):
            raise TypeError(
                f"concept_ids[{position}] must be a str, got "
                f"{type(code).__name__}"
            )
        if not code:
            raise ValueError(
                f"concept_ids[{position}] must be a nonempty str, got an "
                "empty string"
            )
    repeated = sorted(
        code for code, count in Counter(axes).items() if count > 1
    )
    if repeated:
        raise ValueError(
            "concept_ids contains duplicate concept IDs: "
            + ", ".join(repeated[:5])
        )
    axes_set = set(axes)
    missing = sorted(retained - axes_set)
    if missing:
        raise ValueError(
            "concept_ids is missing retained concept IDs (the matrix "
            "vocabulary must cover the retained hierarchy): "
            + ", ".join(missing[:5])
        )
    extra = sorted(axes_set - retained)
    if extra:
        raise ValueError(
            "concept_ids contains concept IDs outside the processor's "
            "retained vocabulary: " + ", ".join(extra[:5])
        )
    return axes


def _omop_patient_id(patient: Any, position: int) -> str:
    """Checks one element of the patients iterable."""
    if not hasattr(patient, "patient_id") or not callable(
        getattr(patient, "get_events", None)
    ):
        raise TypeError(
            f"patients[{position}] must be a pyhealth Patient exposing "
            f"patient_id and get_events, got {type(patient).__name__}"
        )
    return patient.patient_id


def _validate_patient_id(patient_id: Any, position: int) -> None:
    if not isinstance(patient_id, str):
        raise TypeError(
            f"patients[{position}].patient_id must be a str, got "
            f"{type(patient_id).__name__}"
        )
    if not patient_id:
        raise ValueError(
            f"patients[{position}].patient_id must be a nonempty str, got "
            "an empty string"
        )


def _condition_events(patient: Any, patient_id: str) -> list:
    """Reads the patient's condition events once."""
    events = patient.get_events(event_type=CONDITION_EVENT_TYPE)
    if events is None or isinstance(events, (str, bytes)):
        raise TypeError(
            f"patient {patient_id!r}: get_events(event_type="
            f"{CONDITION_EVENT_TYPE!r}) must return an iterable of events, "
            f"got {type(events).__name__}"
        )
    try:
        return list(events)
    except TypeError:
        raise TypeError(
            f"patient {patient_id!r}: get_events(event_type="
            f"{CONDITION_EVENT_TYPE!r}) must return an iterable of events"
        ) from None


# --- diagnosis identity -------------------------------------------------------


def _diagnosis_id(value: Any) -> str | None:
    """Normalizes a diagnosis id; ``None`` when missing or empty."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None"):
        return None
    return text


def _canonical_concept_id(value: Any) -> str | None:
    """Normalizes a concept id; ``None`` when absent or OMOP's zero."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "0"):
        return None
    return text


def _source_value(value: Any) -> str | None:
    """Normalizes a source value; ``None`` when absent or blank."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _visit_key(value: Any) -> str | None:
    """Normalizes a visit id; ``None`` when the event has no visit."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None"):
        return None
    return text


def _collapsed_diagnoses(
    events: Iterable[Any], patient_id: str
) -> dict[str, tuple]:
    """Collapses each diagnosis id to one record of comparable attributes."""
    diagnoses: dict[str, tuple] = {}
    for position, event in enumerate(events):
        diagnosis_id = _diagnosis_id(getattr(event, DIAGNOSIS_ID_ATTR, None))
        if diagnosis_id is None:
            raise ValueError(
                f"condition event {position} of patient {patient_id!r} is "
                f"missing a usable {DIAGNOSIS_ID_ATTR}; the OMOP config "
                "must expose it so each diagnosis row keeps its identity"
            )
        record = (
            _canonical_concept_id(getattr(event, CONCEPT_ID_ATTR, None)),
            _source_value(getattr(event, SOURCE_VALUE_ATTR, None)),
            getattr(event, "timestamp", None),
            _visit_key(getattr(event, VISIT_ID_ATTR, None)),
        )
        previous = diagnoses.get(diagnosis_id)
        if previous is None:
            diagnoses[diagnosis_id] = record
            continue
        if previous == record:
            continue
        differing = [
            field
            for field, before, after in zip(_DIAGNOSIS_FIELDS, previous, record)
            if before != after
        ]
        raise ValueError(
            f"diagnosis {diagnosis_id!r} of patient {patient_id!r} appears "
            "with conflicting copies (differing " + ", ".join(differing)
            + "); identical copies collapse, conflicting copies are rejected"
        )
    return diagnoses


# --- diagnosis resolution -----------------------------------------------------


def _parse_icd_source_value(source: str) -> tuple[int, str] | None:
    """Splits ``VOCAB:code`` into ``(version, code)``, or ``None``.

    Only the ``ICD9CM:`` and ``ICD10CM:`` prefixes are supported; the code
    text is stripped of surrounding whitespace only, so leading zeros
    survive (the mapper normalizes dots and case itself).
    """
    prefix, separator, code = source.partition(":")
    if not separator:
        return None
    version = ICD_PREFIX_TO_VERSION.get(prefix)
    if version is None:
        return None
    code = code.strip()
    if not code:
        return None
    return version, code


def _resolve_canonical_ids(
    concept: str | None,
    source: str | None,
    icd_mapper: Any,
) -> tuple[list[str], str | None]:
    """Resolves one diagnosis to canonical concept id strings.

    Returns ``(ids, None)`` for a resolvable diagnosis, or
    ``([], reason)`` with the first applicable unmapped reason.
    """
    if concept is not None:
        return [concept], None
    if source is None:
        return [], "missing_source"
    parsed = _parse_icd_source_value(source)
    if parsed is None:
        return [], "unsupported_source_format"
    version, code = parsed
    mapped = icd_mapper.map(code, version)
    if not mapped:
        return [], "empty_icd_mapping"
    return [str(concept_id) for concept_id in mapped], None


def _condition_mapper(vocabulary):
    mapper = vocabulary.icd_mapper()
    # At the runtime revision used here, icd_mapper validates relationships
    # but includes targets from other domains and deprecated concepts.
    valid_targets = (
        vocabulary.scan("CONCEPT")
        .filter(
            (pl.col("domain_id") == "Condition")
            & (pl.col("standard_concept") == "S")
            & (pl.col("concept_id") > 0)
            & (pl.col("invalid_reason").fill_null("").str.strip_chars() == "")
        )
        .select("concept_id")
        .collect(engine="streaming")
    )
    return ICDToSNOMEDMapper(
        mapper.table.join(valid_targets, on="concept_id", how="semi")
    )


def _text(value):
    if value is None:
        return None
    return str(value).strip() or None


class KeepConditionHistoryTask(BaseTask):
    """Produce one raw OMOP-style diagnosis history per MIMIC patient.

    Use the opt-in KEEP diagnosis configs to expose occurrence identity and
    admission time. Each diagnosis keeps its complete list of standard
    Condition concept IDs and its source-level unmapped reason, if any.

    Example:
        >>> from pyhealth.tasks import KeepConditionHistoryTask
        >>> task = KeepConditionHistoryTask(  # doctest: +SKIP
        ...     mimic_version=4, vocabulary=vocabulary, vocabulary_release="release",
        ... )
        >>> histories = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name = "keep_condition_history_v2"
    input_schema: ClassVar = {"conditions": "raw"}
    output_schema: ClassVar = {}

    def __init__(
        self, *, mimic_version: int, vocabulary: OMOPVocabulary,
        vocabulary_release: str,
    ):
        super().__init__()
        if mimic_version not in (3, 4):
            raise ValueError("MIMIC version must be 3 or 4")
        if not isinstance(vocabulary_release, str) or not vocabulary_release.strip():
            raise ValueError("Specify the Athena vocabulary release")
        self.mimic_version = mimic_version
        self.vocabulary_release = vocabulary_release
        self._mapper = _condition_mapper(vocabulary)
        # set_task keys its cache on task attributes. The mapper repr alone
        # omits target IDs, so include a digest of the effective lookup.
        lookup = self._mapper.table.select("icd_version", "icd_code", "concept_id")
        lookup = lookup.unique().sort("icd_version", "icd_code", "concept_id")
        digest = hashlib.sha256()
        for row in lookup.iter_rows():
            digest.update(json.dumps(row, separators=(",", ":")).encode() + b"\n")
        self.mapping_fingerprint = digest.hexdigest()

    def __call__(self, patient):
        admissions = {}
        for admission in patient.get_events(event_type="admissions"):
            visit_id = _text(admission.hadm_id)
            if visit_id is None or admission.timestamp is None:
                continue
            previous = admissions.setdefault(visit_id, admission.timestamp)
            if previous != admission.timestamp:
                raise ValueError("MIMIC contains conflicting admission timestamps")

        conditions = {}
        for event in patient.get_events(event_type="diagnoses_icd"):
            visit_id = _text(event.hadm_id)
            if self.mimic_version == 3:
                diagnosis_id = _text(event.row_id)
                code, version = _text(event.icd9_code), "9"
            else:
                seq_num = _text(event.seq_num)
                code, version = _text(event.icd_code), _text(event.icd_version)
                diagnosis_id = (
                    json.dumps(
                        (visit_id, seq_num, code, version), separators=(",", ":")
                    )
                    if visit_id is not None and seq_num is not None
                    else None
                )
            if diagnosis_id is None:
                raise ValueError("MIMIC diagnosis is missing its occurrence identity")
            prefix = _ICD_PREFIX.get(version)
            source_value = None
            concept_ids = []
            if code is None:
                reason = "missing_source"
            elif prefix is None:
                source_value = f"ICD:{version or ''}:{code}"
                reason = "unsupported_source_format"
            else:
                source_value = f"{prefix}:{code}"
                concept_ids = sorted(
                    str(value) for value in self._mapper.map(code, version)
                )
                reason = None if concept_ids else "empty_icd_mapping"
            record = {
                "diagnosis_id": diagnosis_id,
                "visit_id": visit_id,
                "timestamp": admissions.get(visit_id),
                "source_value": source_value,
                "concept_ids": concept_ids,
                "unmapped_reason": reason,
            }
            if conditions.setdefault(diagnosis_id, record) != record:
                raise ValueError("MIMIC contains conflicting copies of a diagnosis")
        return [
            {"patient_id": patient.patient_id, "conditions": list(conditions.values())}
        ]

    def build_cooccurrence(
        self,
        histories: Iterable[Mapping[str, Any]],
        *,
        rollup: AncestorRollupProcessor,
        concept_ids: Sequence[str],
        matrix_builder: _MatrixBuilder = build_cooccurrence,
    ) -> KeepCooccurrenceResult:
        """Count a cohort of MIMIC histories returned by ``dataset.set_task(self)``.

        The existing mapped IDs and unmapped reasons feed the same roll-up
        and presence rules used for OMOP. The builder consumes the cohort
        stream synchronously using the supplied axis order.

        Example:
            >>> result = task.build_cooccurrence(  # doctest: +SKIP
            ...     histories, rollup=processor, concept_ids=axes,
            ... )
        """
        return _build_keep_result(
            (
                (sample["patient_id"], _mimic_diagnoses(sample["conditions"]))
                for sample in histories
            ),
            rollup=rollup,
            concept_ids=concept_ids,
            matrix_builder=matrix_builder,
        )


def _mimic_diagnoses(
    conditions: Iterable[Mapping[str, Any]],
) -> Iterator[tuple[Sequence[str], str | None]]:
    """Read mapped IDs directly from one sample's distinct diagnoses."""
    seen: dict[str, Mapping[str, Any]] = {}
    for condition in conditions:
        diagnosis_id = _diagnosis_id(condition["diagnosis_id"])
        if diagnosis_id is None:
            raise ValueError("MIMIC diagnosis is missing its occurrence identity")
        if diagnosis_id in seen:
            if seen[diagnosis_id] != condition:
                raise ValueError("MIMIC contains conflicting copies of a diagnosis")
            continue
        seen[diagnosis_id] = condition
        yield condition["concept_ids"], condition["unmapped_reason"]
