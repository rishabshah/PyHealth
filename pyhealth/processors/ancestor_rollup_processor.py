# Author: lee-leonardo (NetID: ll37)
# Description: KEEP depth-limited ancestor roll-up as a PyHealth processor.
#   Maps observed concept codes to retained SNOMED ancestors and projects the
#   successful paths into one PyG graph per sample. The roll-up policies
#   are inspired by KEEP
#   (Elhussein et al., CHIL 2025,
#   https://arxiv.org/abs/2510.05049) and were rehearsed and validated
#   against a standalone behavioural reference before this port.

"""Depth-limited multi-parent ancestor roll-up over a knowledge graph.

This module provides :class:`AncestorRollupProcessor`, registered as
``"ancestor_rollup"`` in the processor registry. It follows eligible
ancestor paths from each observed code up a hierarchy. ``rollup_codes()``
emits one typed result per input occurrence. ``process()`` projects the
successful paths into a PyG :class:`~torch_geometric.data.Data` graph per
sample.

Two entry points build the same internal roll-up index:

- the generic constructor consumes a
  :class:`~pyhealth.graph.KnowledgeGraph` (its ``edge_index``,
  ``edge_type`` and ID maps), a domain mapping, and optional authoritative
  root depths;
- :func:`~pyhealth.tasks.keep.build_keep_omop_ancestor_rollup` builds a
  KnowledgeGraph from :class:`~pyhealth.medcode.OMOPVocabulary`:
  the uncut SNOMED condition subtree supplies the traversal edges and the
  closure-derived root depths, and the classification universe is every
  valid standard SNOMED concept with its domain.

Example:
    >>> from pyhealth.graph import KnowledgeGraph
    >>> from pyhealth.processors import AncestorRollupProcessor
    >>> kg = KnowledgeGraph(triples=[
    ...     ("R", "is_a", "A"), ("R", "is_a", "B"), ("A", "is_a", "C"),
    ...     ("B", "is_a", "D"), ("C", "is_a", "X"), ("D", "is_a", "X"),
    ...     ("X", "is_a", "Y"),
    ... ])
    >>> processor = AncestorRollupProcessor(
    ...     knowledge_graph=kg,
    ...     domain_by_code={code: "Condition" for code in "RABCDXY"},
    ...     hierarchy_relation="is_a", root="R",
    ...     target_policy="first_retained", include_root=False,
    ...     max_depth=2, on_unmapped="empty",
    ... )
    >>> [list(result.targets) for result in processor.rollup_codes(["X", "Y"])]
    [['C', 'D'], ['C', 'D']]
    >>> graph = processor.process(["X", "Y"])
    >>> graph.num_nodes, graph.num_edges
    (4, 3)
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from . import register_processor
from .base_processor import FeatureProcessor

try:  # optional: only ``process`` needs PyG, mapping does not
    from torch_geometric.data import Data

    HAS_PYG = True
except ImportError:
    HAS_PYG = False

#: The single domain eligible as roll-up sources and targets, following the
#: KEEP manuscript's condition concepts. Making it configurable is a future
#: adaptation decision, not a silent default change.
ELIGIBLE_DOMAIN = "Condition"

#: The two named target policies. Neither is certified KEEP author behaviour;
#: callers must choose one explicitly.
TARGET_POLICIES = ("first_retained", "all_retained")

#: Behaviour when a source code cannot be mapped.
ON_UNMAPPED_MODES = ("error", "empty")

#: Where traversal edges come from: ``"direct"`` uses the hierarchy relation's
#: own edges; ``"closure"`` additionally adds closure pairs (ancestor,
#: descendant) at minimum separation one, marked ``closure_only``.
PATH_SOURCES = ("direct", "closure")

#: Every diagnostic reason a result can carry.
REASONS = (
    "retained",
    "rolled_up",
    "unknown_code",
    "outside_domain",
    "unreachable",
    "no_target",
)

#: Reasons that trigger the configured unmapped behaviour.
FAILURE_REASONS = ("unknown_code", "outside_domain", "unreachable", "no_target")

#: The only reasons whose sources count as mapped observed codes.
MAPPED_REASONS = ("retained", "rolled_up")

# --- provenance vocabulary recorded on every processor -----------------------

#: Root distances come from a breadth-first edge count with the root at zero.
_DEPTH_METHOD_BFS = "bfs_edge_distance"

#: Root distances were supplied by the caller (for example OMOP
#: ``CONCEPT_ANCESTOR`` minimum separations) and are authoritative.
_DEPTH_METHOD_SUPPLIED = "supplied_root_depths"

#: Version tag embedded in the fingerprint so scheme changes are visible.
_FINGERPRINT_SCHEMA = "pyhealth.ancestor_rollup_processor/2"


# --- internal validation helpers ---------------------------------------------
#
# Error taxonomy: ``TypeError`` for wrong argument/container types (a
# non-string code identifier, a boolean passed as ``max_depth``); ``ValueError``
# for right-type-but-invalid values (an empty identifier, an unknown policy,
# invalid graph structure such as cycles or self-edges).


def _require_nonempty_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a nonempty str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{name} must be a nonempty str, got an empty string")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return value


def _require_mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _require_choice(value: Any, options: tuple[str, ...], name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be one of {options}, got {type(value).__name__}")
    if value not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}")
    return value


class UnmappedCodeError(ValueError):
    """A source code cannot be mapped under ``on_unmapped="error"``.

    Raised for exactly the four failure reasons documented in the contract:
    ``unknown_code`` (absent from the node universe), ``outside_domain``
    (known concept outside the eligible domain), ``unreachable`` (eligible
    concept with no root depth), and ``no_target`` (root-reachable eligible
    source with no target allowed by the configuration).

    Attributes:
        source: the offending source code.
        reason: one of the failure reasons above.

    Example:
        >>> from pyhealth.processors import UnmappedCodeError
        >>> error = UnmappedCodeError("MISSING", "unknown_code")
        >>> error.source, error.reason
        ('MISSING', 'unknown_code')
    """

    def __init__(self, source: str, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(f"cannot map source code {source!r}: {reason}")


@dataclass(frozen=True)
class RollupResult:
    """One source code's mapping outcome.

    Instances are returned by
    :meth:`AncestorRollupProcessor.rollup_codes` and carry one entry per
    input occurrence, in input order. No clinical priority is implied by
    the target order.

    Example:
        >>> from pyhealth.processors import RollupResult
        >>> RollupResult(source="X", targets=("C", "D"), source_depth=3,
        ...              reason="rolled_up")
        RollupResult(source='X', targets=('C', 'D'), source_depth=3, reason='rolled_up')
    """

    source: str
    """The queried source code."""

    targets: tuple[str, ...]
    """Distinct target codes, sorted lexicographically.

    Multiple targets mean multiple eligible ancestors; repeated hierarchy
    paths to one target never duplicate it.
    """

    source_depth: int | None
    """The source's minimum root distance when available, otherwise ``None``.

    A non-eligible-domain node may still have a depth even though its
    source eligibility fails.
    """

    reason: str
    """``retained`` (identity mapping), ``rolled_up`` (at least one target),
    or one of the failure reasons ``unknown_code``, ``outside_domain``,
    ``unreachable``, ``no_target``.
    """


@dataclass(frozen=True)
class _RollupConfig:
    """Immutable, fully explicit configuration for one roll-up index."""

    root: str
    target_policy: str
    include_root: bool
    max_depth: int = 5
    on_unmapped: str = "error"
    path_source: str = "direct"

    def __post_init__(self) -> None:
        _require_nonempty_str(self.root, "root")
        _require_choice(self.target_policy, TARGET_POLICIES, "target_policy")
        _require_bool(self.include_root, "include_root")
        _require_nonnegative_int(self.max_depth, "max_depth")
        _require_choice(self.on_unmapped, ON_UNMAPPED_MODES, "on_unmapped")
        _require_choice(self.path_source, PATH_SOURCES, "path_source")


# --- graph construction helpers ----------------------------------------------


def _validated_pairs(pairs: Any, name: str) -> frozenset[tuple[str, str]]:
    """Deduplicate edge pairs and raise on the first malformed item.

    This runs when constructing the hierarchy index, before any sample is
    processed. The caller must correct malformed edges before retrying.
    """
    if isinstance(pairs, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of (parent, child) pairs")
    edge_set = set()
    for position, edge in enumerate(pairs):
        if isinstance(edge, (str, bytes)) or not isinstance(
            edge, (tuple, list)
        ):
            raise TypeError(
                f"{name} item {position} must be a (parent, child) pair, "
                f"got {type(edge).__name__}"
            )
        if len(edge) != 2:
            raise TypeError(
                f"{name} item {position} must have exactly two elements "
                f"(parent, child), got {len(edge)}"
            )
        parent, child = edge
        edge_set.add(
            (
                _require_nonempty_str(parent, f"parent of {name} item {position}"),
                _require_nonempty_str(child, f"child of {name} item {position}"),
            )
        )
    return frozenset(edge_set)


def _reject_cycles(
    nodes: Iterable[str], children: dict[str, tuple], parents: dict
) -> None:
    """Kahn's algorithm over every node; reject cycles anywhere in the graph."""
    indegree = {code: len(parents.get(code, ())) for code in nodes}
    queue = deque(sorted(code for code, degree in indegree.items() if degree == 0))
    processed = 0
    while queue:
        code = queue.popleft()
        processed += 1
        for child in children.get(code, ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if processed == len(indegree):
        return
    remaining = {code for code, degree in indegree.items() if degree > 0}
    cycle = _find_cycle(remaining, parents)
    raise ValueError(
        "the hierarchy contains a directed cycle (invalid graph data): "
        + " -> ".join(cycle)
    )


def _find_cycle(remaining: set, parents: Mapping[str, tuple]) -> list:
    """Return one example cycle within ``remaining`` (closed code list)."""
    node = min(remaining)
    seen: dict[str, int] = {}
    path = []
    while node not in seen:
        seen[node] = len(path)
        path.append(node)
        node = next(
            parent for parent in parents.get(node, ()) if parent in remaining
        )
    return path[seen[node]:] + [node]


def _bfs_root_depths(root: str, children: Mapping[str, tuple]) -> dict[str, int]:
    """Breadth-first minimum edge distances from the root.

    The traversal runs over the full selected hierarchy before any depth
    cutoff, and first visit gives the minimum distance even when other paths
    are longer. Children iterate in sorted order, so the result is
    deterministic regardless of input edge order.
    """
    depths = {root: 0}
    queue = deque([root])
    while queue:
        code = queue.popleft()
        next_distance = depths[code] + 1
        for child in children.get(code, ()):
            if child not in depths:
                depths[child] = next_distance
                queue.append(child)
    return depths


def _supplied_root_depths(
    root_depths: Any, entity2id: Mapping[str, int], root: str
) -> dict[str, int]:
    """Validate and adopt caller-supplied authoritative root distances.

    Supplied depths are used as-is (for example OMOP ``CONCEPT_ANCESTOR``
    minimum separations): the root must be present at depth zero, every key
    must be a known code, and codes absent from the mapping are treated as
    unreachable for mapping. Adjacency remains the traversal substrate.
    """
    _require_mapping(root_depths, "root_depths")
    depths: dict[str, int] = {}
    for code, distance in root_depths.items():
        _require_nonempty_str(code, "root_depths key")
        if code not in entity2id:
            raise ValueError(f"root_depths contains unknown code {code!r}")
        depths[code] = _require_nonnegative_int(distance, f"root_depths[{code!r}]")
    if root not in depths:
        raise ValueError(f"root_depths must contain the root {root!r} at depth zero")
    if depths[root] != 0:
        raise ValueError(
            f"root_depths must map the root {root!r} to depth zero, "
            f"got {depths[root]}"
        )
    return depths


def _collect_targets_and_edges(
    source: str,
    *,
    parents: Mapping[str, tuple],
    retained: frozenset[str],
    policy: str,
) -> tuple[set, frozenset[tuple[str, str]]]:
    """Collect targets plus the hierarchy edges on successful paths.

    Same iterative upward traversal, visited-set deduplication, and policy
    semantics as the core roll-up: ``first_retained`` stops each branch at
    its first retained node, ``all_retained`` continues past retained nodes,
    and the source itself is never emitted. Duplicate rows and repeated
    paths collapse by code.

    Additionally returns the direct hierarchy edges ``(parent, child)``
    that lie on at least one successful source-to-target path. A traversed
    edge qualifies exactly when its parent endpoint can reach a target
    through the traversal (a productivity memo over the traversed steps), so
    unproductive dead-end branches are excluded, no complete paths are
    enumerated, and no induced edges are added.

    Args:
        source: the source code. Must not be retained (the public path
            handles identity before calling this helper).
        parents: mapping code -> iterable of direct parent codes.
        retained: the eligible target set.
        policy: ``"first_retained"`` or ``"all_retained"``.

    Returns:
        ``(targets, edges)``: the target set and the frozenset of hierarchy
        edges on successful paths, oriented parent -> child.
    """
    if policy not in TARGET_POLICIES:
        raise ValueError(
            f"policy must be one of {TARGET_POLICIES}, got {policy!r}"
        )
    expansions: list[tuple[str, str]] = []  # (child, parent) upward steps
    visited = set()
    targets = set()
    pending = deque()
    # Record the source's own upward steps so its direct-parent edges can
    # qualify for the projection; the source itself is never a target.
    for parent in parents.get(source, ()):
        expansions.append((source, parent))
        pending.append(parent)
    while pending:
        node = pending.popleft()
        if node in visited:
            continue
        visited.add(node)
        if node in retained:
            targets.add(node)
            if policy == "first_retained":
                continue
        for parent in parents.get(node, ()):
            expansions.append((node, parent))
            pending.append(parent)
    if not targets:
        return set(), frozenset()
    productive = set(targets)
    changed = True
    while changed:
        changed = False
        for child, parent in expansions:
            if parent in productive and child not in productive:
                productive.add(child)
                changed = True
    edges = frozenset(
        (parent, child) for child, parent in expansions if parent in productive
    )
    return targets, edges


@register_processor("ancestor_rollup")
class AncestorRollupProcessor(FeatureProcessor):
    """KEEP depth-limited ancestor roll-up as a graph feature processor.

    Follows eligible ancestor paths from each observed code up a hierarchy,
    using the explicit target policy and depth limit. Already-retained
    ``Condition`` codes map to themselves; deeper codes map to their
    retained ancestors (multiple targets preserved); failures carry one of
    four diagnostic reasons and honour ``on_unmapped``.

    The processor consumes a :class:`~pyhealth.graph.KnowledgeGraph`.
    Hierarchy edges point parent -> child. Triples of the named
    ``hierarchy_relation`` are consumed as ``(parent, relation, child)``.
    The caller supplies this orientation.
    :meth:`pyhealth.medcode.ConditionHierarchy.to_triples` emits
    ``(child, relation, parent)`` triples; reverse them before feeding them
    to a generic processor. The KEEP factory
    :func:`~pyhealth.tasks.keep.build_keep_omop_ancestor_rollup` reads the
    vocabulary's parent -> child edges directly.

    Notes:
        **Depth semantics.** ``max_depth`` is the maximum permitted minimum
        root distance, inclusive, counted from the root at zero. It applies
        to *target eligibility*: deep sources and intermediate concepts stay
        traversable, and already-retained sources map to themselves. Depths
        are minimum directed edge distances from the root, unless
        ``root_depths`` supplies authoritative distances (the OMOP path
        supplies the ``CONCEPT_ANCESTOR`` minimum separations, following
        KEEP appendix A.1.1). The cutoff is computed before any truncation,
        and ``first_retained`` stopping is branch-local.

        **Target policies.** ``"first_retained"`` stops each upward branch
        at its first eligible ancestor; ``"all_retained"`` emits every
        distinct eligible strict ancestor. KEEP's actual choice is
        unresolved; the caller must decide.

        **Failure reasons.** ``unknown_code`` (absent from the node
        universe), ``outside_domain`` (known concept outside ``Condition``),
        ``unreachable`` (eligible concept with no root depth, for example a
        Condition concept outside the OMOP subtree), and ``no_target``
        (reachable source with no eligible target on the selected edges).

        **Path sources.** ``path_source="direct"`` uses the hierarchy
        relation's own edges. ``path_source="closure"`` additionally adds
        closure pairs at minimum separation one (on the OMOP path, from
        ``CONCEPT_ANCESTOR``); pairs supported by both sources stay direct,
        and edges that exist only in the closure are flagged in the
        edge-aligned ``closure_only`` mask of the projected graph. Direct
        provenance wins on overlap. The selected edge set must be acyclic.

        **Occurrence counting.** :meth:`rollup_codes` returns one result per
        input occurrence, in input order; two occurrences produce two
        results even though the projected graph merges them, and multiple
        paths to one target produce one target per result. Event counting
        lives with the task: ICD conversion stays explicit through
        ``vocabulary.icd_mapper()`` so callers keep their diagnosis event
        IDs, and converging targets within one event count once.

    Args:
        knowledge_graph: a :class:`~pyhealth.graph.KnowledgeGraph` (any
            object exposing ``entity2id``, ``relation2id``, ``edge_index``
            and ``edge_type``). Only edges whose relation equals
            ``hierarchy_relation`` become hierarchy edges, consumed as
            (parent, relation, child).
        domain_by_code: mapping of every graph entity to its domain ID.
            Entities missing from it fail construction. Extra codes are
            used for classification when include_external_codes is enabled.
        hierarchy_relation: the named direct relation that defines the
            hierarchy; must exist in the graph.
        root: the structural root code; must be an entity in the eligible
            domain.
        target_policy: ``"first_retained"`` or ``"all_retained"``.
        include_root: whether the root may appear as a target.
        max_depth: maximum permitted minimum root distance, inclusive.
            Default 5, following the KEEP manuscript.
        on_unmapped: ``"error"`` (raise :class:`UnmappedCodeError`) or
            ``"empty"`` (typed empty result) for unmappable codes. Default
            ``"error"``.
        root_depths: optional authoritative mapping of code to minimum root
            distance (for example OMOP ``CONCEPT_ANCESTOR`` separations).
            When supplied, the root must be present at depth zero and codes
            missing from it are unreachable for mapping; adjacency still
            drives the upward traversal. When ``None``, depths are minimum
            directed edge distances from the root.
        path_source: ``"direct"`` (default) or ``"closure"``; see Notes.
        closure_pairs: optional iterable of ``(parent, child)`` pairs from a
            closure/ancestor table, required when ``path_source="closure"``
            on this constructor.
        include_external_codes: keep domain entries for codes outside the
            graph for classification. Defaults to false.
        loader_provenance: optional string metadata from the graph builder.
            Defaults to the graph relation and entity count.

    Raises:
        TypeError: wrong argument types, a knowledge graph missing the
            expected attributes, or non-string identifiers.
        ValueError: unknown relation, missing domain coverage, invalid
            configuration, an unknown or wrong-domain root, invalid supplied
            depths, or invalid hierarchy structure (self-edges, cycles,
            unknown edge endpoints).

    Example:
        >>> from pyhealth.graph import KnowledgeGraph
        >>> from pyhealth.processors import AncestorRollupProcessor
        >>> kg = KnowledgeGraph(triples=[
        ...     ("R", "is_a", "A"), ("R", "is_a", "B"), ("A", "is_a", "C"),
        ...     ("B", "is_a", "D"), ("C", "is_a", "X"), ("D", "is_a", "X"),
        ...     ("X", "is_a", "Y"),
        ... ])
        >>> processor = AncestorRollupProcessor(
        ...     knowledge_graph=kg,
        ...     domain_by_code={code: "Condition" for code in "RABCDXY"},
        ...     hierarchy_relation="is_a", root="R",
        ...     target_policy="first_retained", include_root=False,
        ...     max_depth=2, on_unmapped="empty",
        ... )
        >>> [list(result.targets) for result in processor.rollup_codes(["X", "Y"])]
        [['C', 'D'], ['C', 'D']]
        >>> [result.reason for result in processor.rollup_codes(["C", "MISSING"])]
        ['retained', 'unknown_code']
        >>> graph = processor.process(["X", "Y"])
        >>> graph.num_nodes, graph.num_edges
        (4, 3)

    Example in task schema:
        >>> input_schema = {
        ...     "conditions": processor,  # a configured processor instance
        ... }
    """

    def __init__(
        self,
        knowledge_graph: Any,
        domain_by_code: Mapping[str, str],
        hierarchy_relation: str,
        root: str,
        target_policy: str,
        include_root: bool,
        max_depth: int = 5,
        on_unmapped: str = "error",
        root_depths: Mapping[str, int] | None = None,
        path_source: str = "direct",
        closure_pairs: Iterable[tuple[str, str]] | None = None,
        *,
        include_external_codes: bool = False,
        loader_provenance: Mapping[str, str] | None = None,
    ) -> None:
        for attribute in (
            "entity2id",
            "relation2id",
            "edge_index",
            "edge_type",
        ):
            if not hasattr(knowledge_graph, attribute):
                raise TypeError(
                    f"knowledge_graph must expose {attribute!r}, got "
                    f"{type(knowledge_graph).__name__}"
                )
        _require_mapping(knowledge_graph.entity2id, "knowledge_graph.entity2id")
        _require_mapping(knowledge_graph.relation2id, "knowledge_graph.relation2id")
        self.knowledge_graph = knowledge_graph
        self.hierarchy_relation = _require_nonempty_str(
            hierarchy_relation, "hierarchy_relation"
        )
        if self.hierarchy_relation not in knowledge_graph.relation2id:
            raise ValueError(
                f"hierarchy_relation {self.hierarchy_relation!r} is not a "
                f"relation of the knowledge graph"
            )
        _require_mapping(domain_by_code, "domain_by_code")
        for code, domain in domain_by_code.items():
            _require_nonempty_str(code, "domain_by_code key")
            _require_nonempty_str(domain, f"domain_by_code[{code!r}]")
        missing = sorted(set(knowledge_graph.entity2id) - set(domain_by_code))
        if missing:
            raise ValueError(
                "domain_by_code is missing domains for entities: "
                + ", ".join(missing[:5])
            )
        _require_bool(include_external_codes, "include_external_codes")
        domains = (
            dict(domain_by_code)
            if include_external_codes
            else {code: domain_by_code[code] for code in knowledge_graph.entity2id}
        )
        if loader_provenance is None:
            loader_provenance = {
                "hierarchy_relation": self.hierarchy_relation,
                "knowledge_graph_entities": str(len(knowledge_graph.entity2id)),
            }

        self._initialize(
            domain_by_code=domains,
            closure_pairs=closure_pairs,
            config=_RollupConfig(
                root=root,
                target_policy=target_policy,
                include_root=include_root,
                max_depth=max_depth,
                on_unmapped=on_unmapped,
                path_source=path_source,
            ),
            root_depths=root_depths,
            loader_provenance=loader_provenance,
        )

    def _initialize(
        self,
        *,
        domain_by_code: dict[str, str],
        closure_pairs: Iterable[tuple[str, str]] | None,
        config: _RollupConfig,
        root_depths: Mapping[str, int] | None,
        loader_provenance: Mapping[str, str],
    ) -> None:
        """Validate and index the configured hierarchy."""
        graph = self.knowledge_graph
        entity2id = {
            code: int(entity_id) for code, entity_id in graph.entity2id.items()
        }
        id2entity = {entity_id: code for code, entity_id in entity2id.items()}
        relation_id = int(graph.relation2id[self.hierarchy_relation])
        heads, tails = graph.edge_index.tolist()
        direct = _validated_pairs(
            (
                (id2entity[head], id2entity[tail])
                for head, tail, relation in zip(heads, tails, graph.edge_type.tolist())
                if relation == relation_id
            ),
            "hierarchy edges",
        )
        if config.path_source == "closure":
            if closure_pairs is None:
                raise ValueError(
                    "path_source='closure' requires closure_pairs (pairs "
                    "from a closure/ancestor table)"
                )
            closure = _validated_pairs(closure_pairs, "closure_pairs")
            selected = direct | closure
            closure_only = closure - direct
        else:
            if closure_pairs is not None:
                raise ValueError(
                    "closure_pairs requires path_source='closure'"
                )
            selected = direct
            closure_only = frozenset()

        children_lists: dict[str, list] = {}
        parents_lists: dict[str, list] = {}
        for parent, child in selected:
            if parent not in entity2id:
                raise ValueError(
                    f"hierarchy edge ({parent!r}, {child!r}) references an "
                    "unknown parent code"
                )
            if child not in entity2id:
                raise ValueError(
                    f"hierarchy edge ({parent!r}, {child!r}) references an "
                    "unknown child code"
                )
            if parent == child:
                raise ValueError(
                    f"hierarchy edge ({parent!r}, {child!r}) is a self-edge"
                )
            children_lists.setdefault(parent, []).append(child)
            parents_lists.setdefault(child, []).append(parent)
        children = {
            code: tuple(sorted(neighbours))
            for code, neighbours in children_lists.items()
        }
        parents = {
            code: tuple(sorted(neighbours))
            for code, neighbours in parents_lists.items()
        }
        _reject_cycles(entity2id, children, parents)

        if config.root not in entity2id:
            raise ValueError(
                f"configured root {config.root!r} is not present in nodes"
            )
        if domain_by_code[config.root] != ELIGIBLE_DOMAIN:
            raise ValueError(
                f"configured root {config.root!r} must be in the "
                f"{ELIGIBLE_DOMAIN!r} domain, got "
                f"{domain_by_code[config.root]!r}"
            )

        supplied: dict[str, int] | None = None
        if root_depths is None:
            depths = _bfs_root_depths(config.root, children)
            depth_method = _DEPTH_METHOD_BFS
            root_depths_count = 0
        else:
            supplied = _supplied_root_depths(root_depths, entity2id, config.root)
            depths = supplied
            depth_method = _DEPTH_METHOD_SUPPLIED
            root_depths_count = len(depths)

        retained_graph = {
            code
            for code, distance in depths.items()
            if domain_by_code[code] == ELIGIBLE_DOMAIN
            and distance <= config.max_depth
        }
        retained = (
            retained_graph if config.include_root else retained_graph - {config.root}
        )

        provenance: dict[str, str] = {
            "schema_version": "1",
            "depth_method": depth_method,
            "root_depths_count": str(root_depths_count),
            "edge_orientation": "parent_to_child",
            "domain_eligibility": ELIGIBLE_DOMAIN,
            "identity_policy": "identity",
            "identifier_normalization": (
                "none; loaders normalize identifiers at the boundary"
            ),
            "path_source": config.path_source,
            "root": config.root,
            "max_depth": str(config.max_depth),
            "target_policy": config.target_policy,
            "include_root": str(config.include_root).lower(),
            "on_unmapped": config.on_unmapped,
            "node_count": str(len(domain_by_code)),
            "direct_edge_count": str(len(selected) - len(closure_only)),
            "closure_pair_count": str(
                len(closure) if config.path_source == "closure" else 0
            ),
            "closure_only_edge_count": str(len(closure_only)),
        }
        _require_mapping(loader_provenance, "loader_provenance")
        for key, value in loader_provenance.items():
            _require_nonempty_str(key, "loader_provenance key")
            if not isinstance(value, str):
                raise TypeError(
                    f"loader_provenance[{key!r}] must be a str, "
                    f"got {type(value).__name__}"
                )
            if key in provenance:
                raise ValueError(
                    f"loader_provenance key {key!r} collides with a "
                    "component provenance key"
                )
            provenance[key] = value

        self._config = config
        self._domain_by_code = domain_by_code
        self._children = children
        self._parents = parents
        self._depths = depths
        self._retained = frozenset(retained)
        self._edges = frozenset(selected)
        self._closure_only = frozenset(closure_only)
        self._supplied_depths = supplied
        self._entity2id = entity2id
        self._id2entity = id2entity
        self._relation_id = relation_id
        features = getattr(graph, "node_features", None)
        self._node_features = None if features is None else features.detach().clone()
        self._provenance = provenance
        self._fingerprint = self._compute_fingerprint()

    # -- PyHealth processor hooks --------------------------------------------

    def fit(
        self,
        samples: Iterable[dict[str, Any]] | None = None,
        field: str | None = None,
    ) -> None:
        """No-op: the hierarchy is fully supplied at construction.

        Args:
            samples: sample dictionaries (unused).
            field: field name (unused).
        """

    def is_token(self) -> bool:
        """Graph outputs are not discrete token indices."""
        return False

    def schema(self) -> tuple:
        """Schema of the processed feature: one graph per sample."""
        return ("graph",)

    def dim(self) -> tuple:
        """Graph output has no fixed dimensionality."""
        return (0,)

    def spatial(self) -> tuple:
        """Graph output is not spatial in the grid sense."""
        return (False,)

    # -- mapping and processing ------------------------------------------------

    def rollup_codes(self, codes: Iterable[str]) -> list[RollupResult]:
        """Maps codes to retained ancestor targets.

        One typed result per input element, in input order, duplicates
        preserved: two occurrences produce two results even though the graph
        merges them, and multiple hierarchy paths to one target produce one
        target per result. Already-retained codes map to themselves.
        Failures honour ``on_unmapped``; malformed input is rejected
        independently of that setting. Tasks can call this per visit and
        retain their original event identifiers; the mapping is the
        attribution source for event counting.

        Args:
            codes: iterable of nonempty string codes. A bare string is
                rejected; wrap a single code in a list.

        Returns:
            List of :class:`RollupResult`, same length and order as the
            input.

        Raises:
            TypeError: ``codes`` is not an iterable of codes or an element
                is not a string.
            ValueError: an element is an empty string.
            UnmappedCodeError: a code cannot be mapped under
                ``on_unmapped="error"``.
        """
        if isinstance(codes, (str, bytes)):
            raise TypeError(
                "codes must be an iterable of code strings; wrap a single "
                "code in a list"
            )
        return [self._map_source(code)[0] for code in codes]

    def process(self, value: Any) -> Data:
        """Converts one sample's codes into the roll-up graph.

        Accepts a flat sequence of code strings or a sequence of visits
        (each visit a list or tuple of code strings); visits are flattened
        in order. The graph unions the successful paths across all mapped
        codes: direct hierarchy edges only, no induced edges, identity
        mappings kept as nodes without edges, unmapped codes excluded. Under
        ``on_unmapped="empty"`` a sample with no mapped code yields a
        consistently shaped empty graph.

        The returned :class:`~torch_geometric.data.Data` carries
        ``edge_index`` (shape ``[2, E]``, local node indices, edges pointing
        parent -> child), ``edge_type`` (the hierarchy relation's id),
        ``node_ids`` (global KnowledgeGraph entity IDs, decoded with
        ``id2entity``), ``x`` (aligned graph node features or
        ``torch.zeros(N, 1)``), ``seed_mask`` (mapped observed codes),
        ``target_mask`` (emitted targets, identity mappings included) and
        the edge-aligned ``closure_only`` mask (True for edges that exist
        only in the closure). Use :meth:`rollup_codes` for the ordered
        per-record mapping and per-event attribution.

        Args:
            value: flat sequence of code strings, or sequence of visits.

        Returns:
            One PyG ``Data`` graph per sample.

        Raises:
            ImportError: torch-geometric is not installed.
            TypeError: ``value`` is a bare string or not an iterable of
                codes/visits, or a code is not a string.
            ValueError: a code is an empty string.
            UnmappedCodeError: a code cannot be mapped under
                ``on_unmapped="error"``.
        """
        if not HAS_PYG:
            raise ImportError(
                "torch-geometric is required for AncestorRollupProcessor"
                ".process(). Install with: pip install torch-geometric"
            )
        codes = self._flatten(value)
        mapped = [self._map_source(code) for code in codes]
        seeds = {
            result.source for result, _ in mapped if result.reason in MAPPED_REASONS
        }
        targets: set = set()
        edges: set = set()
        for result, path_edges in mapped:
            targets.update(result.targets)
            edges.update(path_edges)
        node_codes = sorted(
            seeds | targets | {code for edge in edges for code in edge}
        )
        local = {code: position for position, code in enumerate(node_codes)}
        edge_pairs = sorted(edges)
        node_ids = torch.tensor(
            [self._entity2id[code] for code in node_codes], dtype=torch.long
        )
        features = self._node_features
        x = (
            torch.zeros(len(node_codes), 1)
            if features is None
            else features.index_select(0, node_ids.to(features.device))
        )
        return Data(
            x=x,
            edge_index=torch.tensor(
                [
                    [local[parent] for parent, _ in edge_pairs],
                    [local[child] for _, child in edge_pairs],
                ],
                dtype=torch.long,
            ),
            edge_type=torch.full(
                (len(edge_pairs),), self._relation_id, dtype=torch.long
            ),
            node_ids=node_ids,
            seed_mask=torch.tensor(
                [code in seeds for code in node_codes], dtype=torch.bool
            ),
            target_mask=torch.tensor(
                [code in targets for code in node_codes], dtype=torch.bool
            ),
            closure_only=torch.tensor(
                [pair in self._closure_only for pair in edge_pairs],
                dtype=torch.bool,
            ),
        )

    # -- identity and state ----------------------------------------------------

    @property
    def provenance(self) -> Mapping[str, str]:
        """Read-only provenance of the internal roll-up index.

        Records the depth method, configuration, path source, edge
        provenance counts, and loader facts, so adapters expose their
        depth/ancestry source as the data inspection requires.
        The ``schema_version`` entry (``"1"``) versions this processor's
        provenance fields and their meanings. The OMOP vocabulary release
        and the fingerprint format are separate version scopes.
        """
        return MappingProxyType(dict(self._provenance))

    @property
    def fingerprint(self) -> str:
        """Stable hierarchy-and-configuration digest (SHA-256 hex).

        Covers selected hierarchy edges and provenance, entity and relation
        IDs, node features, domains, configuration and supplied depths. Two processors
        built from the same inputs share a fingerprint, so configured task
        schemas can distinguish different graphs of the same size.
        """
        return self._fingerprint

    @property
    def id2entity(self) -> Mapping[int, str]:
        """Read-only entity-id-to-code map used to decode ``node_ids``."""
        return MappingProxyType(dict(self._id2entity))

    @property
    def retained_codes(self) -> frozenset[str]:
        """Return the immutable set of eligible roll-up target codes.

        These are the Condition concepts whose root depth is at most
        ``max_depth``. The root is included when ``include_root`` is true.

        Example:
            >>> from pyhealth.graph import KnowledgeGraph
            >>> from pyhealth.processors import AncestorRollupProcessor
            >>> kg = KnowledgeGraph(
            ...     triples=[("R", "is_a", "A"), ("A", "is_a", "X")]
            ... )
            >>> processor = AncestorRollupProcessor(
            ...     knowledge_graph=kg,
            ...     domain_by_code={code: "Condition" for code in "RAX"},
            ...     hierarchy_relation="is_a", root="R",
            ...     target_policy="first_retained", include_root=False,
            ...     max_depth=1, on_unmapped="empty",
            ... )
            >>> sorted(processor.retained_codes)
            ['A']
            >>> processor.rollup_codes(["X"])[0].targets
            ('A',)
        """
        return self._retained

    def __repr__(self) -> str:
        return (
            f"AncestorRollupProcessor("
            f"hierarchy_relation={self.hierarchy_relation!r}, "
            f"root={self._config.root!r}, "
            f"target_policy={self._config.target_policy!r}, "
            f"include_root={self._config.include_root}, "
            f"max_depth={self._config.max_depth}, "
            f"path_source={self._config.path_source!r}, "
            f"fingerprint={self._fingerprint[:16]})"
        )

    # -- internals ----------------------------------------------------------

    def _map_source(self, code: Any) -> tuple[RollupResult, frozenset[tuple[str, str]]]:
        """Classifies one source and collects its successful-path edges.

        Mirrors the classification order of the behavioural reference:
        unknown code, outside the eligible domain, unreachable, retained
        identity, then strict-ancestor search under the configured policy.
        """
        _require_nonempty_str(code, "code")
        if code not in self._domain_by_code:
            return self._unmapped(code, None, "unknown_code")
        if self._domain_by_code[code] != ELIGIBLE_DOMAIN:
            return self._unmapped(code, self._depths.get(code), "outside_domain")
        if code not in self._depths:
            return self._unmapped(code, None, "unreachable")
        if code in self._retained:
            return (
                RollupResult(
                    source=code,
                    targets=(code,),
                    source_depth=self._depths[code],
                    reason="retained",
                ),
                frozenset(),
            )
        targets, edges = _collect_targets_and_edges(
            code,
            parents=self._parents,
            retained=self._retained,
            policy=self._config.target_policy,
        )
        if not targets:
            return self._unmapped(code, self._depths.get(code), "no_target")
        return (
            RollupResult(
                source=code,
                targets=tuple(sorted(targets)),
                source_depth=self._depths[code],
                reason="rolled_up",
            ),
            edges,
        )

    def _unmapped(
        self, source: str, source_depth: int | None, reason: str
    ) -> tuple[RollupResult, frozenset[tuple[str, str]]]:
        if self._config.on_unmapped == "error":
            raise UnmappedCodeError(source, reason)
        return (
            RollupResult(
                source=source,
                targets=(),
                source_depth=source_depth,
                reason=reason,
            ),
            frozenset(),
        )

    def _flatten(self, value: Any) -> list[str]:
        """Flattens flat codes or visits into one ordered code list."""
        if isinstance(value, (str, bytes)):
            raise TypeError(
                "value must be a sequence of code strings or of visits, "
                "not a single string; wrap one code in a list"
            )
        try:
            records = list(value)
        except TypeError:
            raise TypeError(
                "value must be an iterable of codes or visits, got "
                f"{type(value).__name__}"
            ) from None
        if not records:
            return []
        if isinstance(records[0], (list, tuple)):
            codes: list[str] = []
            for position, visit in enumerate(records):
                if isinstance(visit, (str, bytes)):
                    raise TypeError(
                        f"visit {position} must be a sequence of code strings"
                    )
                try:
                    codes.extend(visit)
                except TypeError:
                    raise TypeError(
                        f"visit {position} must be an iterable of code "
                        f"strings, got {type(visit).__name__}"
                    ) from None
            return codes
        return records

    def _compute_fingerprint(self) -> str:
        """Hashes every semantic input into one stable digest."""
        lines = [
            _FINGERPRINT_SCHEMA,
            f"relation {self.hierarchy_relation} {self._relation_id}",
            f"root {self._config.root}",
            f"policy {self._config.target_policy}",
            f"include_root {self._config.include_root}",
            f"max_depth {self._config.max_depth}",
            f"on_unmapped {self._config.on_unmapped}",
            f"eligible_domain {ELIGIBLE_DOMAIN}",
            f"depth_method {self._provenance['depth_method']}",
            f"path_source {self._config.path_source}",
            f"direct_edges {len(self._edges) - len(self._closure_only)}",
            f"closure_only_edges {len(self._closure_only)}",
        ]
        lines.extend(
            f"node {code} {self._domain_by_code[code]} {self._entity2id.get(code)}"
            for code in sorted(self._domain_by_code)
        )
        lines.extend(
            f"edge {parent} {child} "
            f"{'closure_only' if (parent, child) in self._closure_only else 'direct'}"
            for parent, child in sorted(self._edges)
        )
        if self._supplied_depths is None:
            lines.append("root_depths -")
        else:
            lines.extend(
                f"depth {code} {self._supplied_depths[code]}"
                for code in sorted(self._supplied_depths)
            )
        digest = hashlib.sha256("\n".join(lines).encode("utf-8"))
        features = self._node_features
        if features is not None:
            digest.update(
                f"\nfeatures {features.dtype} {tuple(features.shape)}\n".encode()
            )
            feature_bytes = features.cpu().contiguous().view(torch.uint8).numpy()
            digest.update(feature_bytes)
        return digest.hexdigest()
