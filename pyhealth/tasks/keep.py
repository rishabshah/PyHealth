# Author: lee-leonardo (NetID: ll37)
# Description: Build the OMOP ancestor roll-up for KEEP.
#   KEEP (Elhussein et al., CHIL 2025), https://arxiv.org/abs/2510.05049

"""Helpers for the KEEP workflow."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl
import torch

from pyhealth.graph import KnowledgeGraph
from pyhealth.medcode import DISEASE_CONCEPT_ID, OMOPVocabulary
from pyhealth.processors import AncestorRollupProcessor

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
