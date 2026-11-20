"""SNOMED condition hierarchy built from OMOP vocabularies, as used by KEEP.

Build one with :meth:`pyhealth.medcode.OMOPVocabulary.condition_hierarchy`.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import networkx as nx
import polars as pl

if TYPE_CHECKING:
    from pyhealth.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

#: Relation names for exported edges, by direction: (direct link, rescue edge).
RELATIONS = {
    "child_to_parent": ("is_a", "rescued_is_a"),
    "parent_to_child": ("subsumes", "rescued_subsumes"),
}


def _rescue_orphans(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    ancestors: pl.DataFrame,
    root_concept_id: int,
) -> pl.DataFrame:
    """Attaches each parentless non-root node to its nearest kept ancestor.

    Ties on distance go to the smaller ancestor id, so the result does not
    depend on file order.
    """
    orphans = nodes.filter(pl.col("concept_id") != root_concept_id).join(
        edges, left_on="concept_id", right_on="child_id", how="anti"
    )
    if orphans.height == 0:
        return edges
    rescue = (
        ancestors.join(orphans.select("concept_id"), on="concept_id", how="semi")
        .join(nodes.select("concept_id"), left_on="ancestor_id",
              right_on="concept_id", how="semi")
        .sort("concept_id", "distance", "ancestor_id")
        .group_by("concept_id", maintain_order=True)
        .first()
        .select(
            pl.col("ancestor_id").alias("parent_id"),
            pl.col("concept_id").alias("child_id"),
            pl.lit(True).alias("rescued"),
        )
    )
    return pl.concat([edges, rescue])


class OMOPConditionHierarchy:
    """A depth-limited SNOMED condition graph rooted at one concept.

    Build it with :meth:`OMOPVocabulary.condition_hierarchy`, or reload a saved
    one with :meth:`load`. Concept ids are OMOP ``concept_id`` integers.

    Attributes:
        nodes: One row per kept concept: ``concept_id``, ``concept_name``,
            ``concept_code`` (the SNOMED code), ``min_depth``, ``max_depth``
            and ``depth`` (whichever of the two was used to cut the graph).
        edges: ``parent_id``, ``child_id`` and ``rescued`` for every "Is a"
            link between kept concepts, plus any orphan rescues. Rescue is an
            adapter policy of this module, not part of KEEP, whose paper does
            not say how disconnected concepts are handled.
        ancestors: ``concept_id``, ``ancestor_id``, ``distance`` for every
            concept under the root (kept or deeper) and each of its kept
            ancestors, with ``distance`` the shortest number of levels between
            them. Roll-up uses this to place deep concepts on the graph.
        root_concept_id: The root concept.
        max_depth: The depth limit, or None for no limit.
        depth_measure: ``"min"`` or ``"max"``.
        connect_orphans: Whether orphans were attached to their nearest kept
            ancestor.
    """

    def __init__(
        self,
        nodes: pl.DataFrame,
        edges: pl.DataFrame,
        ancestors: pl.DataFrame,
        root_concept_id: int,
        max_depth: Optional[int],
        depth_measure: str,
        connect_orphans: bool = True,
    ):
        self.nodes = nodes
        self.edges = edges
        self.ancestors = ancestors
        self.root_concept_id = root_concept_id
        self.max_depth = max_depth
        self.depth_measure = depth_measure
        self.connect_orphans = connect_orphans
        self._info = {
            row["concept_id"]: row for row in nodes.iter_rows(named=True)
        }
        self._parents: Dict[int, List[int]] = defaultdict(list)
        self._children: Dict[int, List[int]] = defaultdict(list)
        for parent, child in edges.select("parent_id", "child_id").iter_rows():
            self._parents[child].append(parent)
            self._children[parent].append(child)
        self._ancestor_index: Optional[Dict[int, List[Tuple[int, int]]]] = None

    def __repr__(self) -> str:
        return (
            f"OMOPConditionHierarchy(root={self.root_concept_id}, "
            f"max_depth={self.max_depth}, depth_measure={self.depth_measure!r}, "
            f"nodes={len(self)}, edges={self.edges.height})"
        )

    def __len__(self) -> int:
        return self.nodes.height

    def __contains__(self, concept_id: int) -> bool:
        return concept_id in self._info

    def _require(self, concept_id: int) -> dict:
        try:
            return self._info[concept_id]
        except KeyError:
            raise KeyError(f"Concept {concept_id} is not in the hierarchy.") from None

    def name(self, concept_id: int) -> str:
        """Returns a kept concept's name."""
        return self._require(concept_id)["concept_name"]

    def depth(self, concept_id: int) -> int:
        """Returns a kept concept's depth below the root."""
        return self._require(concept_id)["depth"]

    def parents(self, concept_id: int) -> List[int]:
        """Returns a kept concept's parents in the graph, sorted."""
        self._require(concept_id)
        return sorted(self._parents.get(concept_id, []))

    def children(self, concept_id: int) -> List[int]:
        """Returns a kept concept's children in the graph, sorted."""
        self._require(concept_id)
        return sorted(self._children.get(concept_id, []))

    def retained_ancestors(self, concept_id: int) -> List[Tuple[int, int]]:
        """Lists the kept ancestors of any concept under the root.

        Works for concepts deeper than ``max_depth`` too, which is what
        roll-up needs. A concept with several parents gets ancestors along
        every path.

        Args:
            concept_id: Any concept under the root, kept or not.

        Returns:
            ``(ancestor_id, distance)`` pairs, nearest first. Empty if the
            concept is not under the root.
        """
        if self._ancestor_index is None:
            self._ancestor_index = defaultdict(list)
            for child, ancestor, distance in self.ancestors.iter_rows():
                self._ancestor_index[child].append((ancestor, distance))
        return list(self._ancestor_index.get(concept_id, []))

    def restrict(
        self, max_depth: int, depth_measure: Optional[str] = None
    ) -> "OMOPConditionHierarchy":
        """Cuts this hierarchy to a shallower depth without re-reading Athena.

        Build once with a generous limit (or ``max_depth=None``), then call
        this for each depth in an ablation. The result is the same as
        building at the smaller depth directly.

        Args:
            max_depth: The new, smaller depth limit.
            depth_measure: ``"min"`` or ``"max"``; defaults to this
                hierarchy's measure.

        Returns:
            A new :class:`OMOPConditionHierarchy`.

        Raises:
            ValueError: If the new graph would need concepts this one dropped.
        """
        depth_measure = depth_measure or self.depth_measure
        if depth_measure not in ("min", "max"):
            raise ValueError("depth_measure must be 'min' or 'max'.")
        if max_depth is None or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer.")
        # a concept's max depth is never below its min depth, so a cut on
        # max depth keeps a subset of the same cut on min depth
        covered = self.max_depth is None or (
            max_depth <= self.max_depth
            and (self.depth_measure == "min" or depth_measure == "max")
        )
        if not covered:
            raise ValueError(
                f"Cannot restrict a hierarchy built with max_depth="
                f"{self.max_depth} ({self.depth_measure}) to max_depth="
                f"{max_depth} ({depth_measure}); build a deeper one first."
            )
        nodes = (
            self.nodes.with_columns(pl.col(f"{depth_measure}_depth").alias("depth"))
            .filter(pl.col("depth") <= max_depth)
            .sort("depth", "concept_id")
        )
        kept = nodes.select("concept_id")
        edges = (
            self.edges.filter(~pl.col("rescued"))
            .join(kept, left_on="parent_id", right_on="concept_id", how="semi")
            .join(kept, left_on="child_id", right_on="concept_id", how="semi")
        )
        ancestors = self.ancestors.join(
            kept, left_on="ancestor_id", right_on="concept_id", how="semi"
        )
        if self.connect_orphans:
            edges = _rescue_orphans(nodes, edges, ancestors, self.root_concept_id)
        return OMOPConditionHierarchy(
            nodes=nodes,
            edges=edges.sort("parent_id", "child_id"),
            ancestors=ancestors,
            root_concept_id=self.root_concept_id,
            max_depth=max_depth,
            depth_measure=depth_measure,
            connect_orphans=self.connect_orphans,
        )

    def to_networkx(self) -> nx.DiGraph:
        """Returns the graph as a ``networkx.DiGraph`` with parent -> child edges.

        Node attributes are ``name``, ``code`` and ``depth``; edge attribute
        ``rescued`` marks orphan rescues.
        """
        graph = nx.DiGraph()
        for row in self.nodes.iter_rows(named=True):
            graph.add_node(
                row["concept_id"],
                name=row["concept_name"],
                code=row["concept_code"],
                depth=row["depth"],
            )
        for parent, child, rescued in self.edges.iter_rows():
            graph.add_edge(parent, child, rescued=rescued)
        return graph

    def to_triples(
        self, direction: str = "child_to_parent"
    ) -> List[Tuple[str, str, str]]:
        """Returns the edges as ``(head, relation, tail)`` string triples.

        This is the input format of :class:`pyhealth.graph.KnowledgeGraph`.
        Entities are OMOP ``concept_id`` values as strings. Depth, ancestor
        distances and other metadata stay on this object.

        Args:
            direction: ``"child_to_parent"`` gives ``(child, "is_a", parent)``;
                ``"parent_to_child"`` gives ``(parent, "subsumes", child)``
                (OMOP's name for the reverse of "Is a"). Orphan rescue edges
                get their own relation (``"rescued_is_a"`` or
                ``"rescued_subsumes"``) because they can skip several levels.

        Returns:
            One triple per edge, sorted by head then tail.

        Example:
            >>> hierarchy.to_triples()[:1]  # doctest: +SKIP
            [('100', 'is_a', '4274025')]
        """
        try:
            direct, rescued = RELATIONS[direction]
        except KeyError:
            raise ValueError(
                f"direction must be one of {sorted(RELATIONS)}, got {direction!r}."
            ) from None
        triples = []
        for parent, child, is_rescue in self.edges.iter_rows():
            relation = rescued if is_rescue else direct
            if direction == "child_to_parent":
                triples.append((str(child), relation, str(parent)))
            else:
                triples.append((str(parent), relation, str(child)))
        return sorted(triples, key=lambda triple: (triple[0], triple[2]))

    def to_knowledge_graph(
        self, direction: str = "parent_to_child"
    ) -> "KnowledgeGraph":
        """Returns the hierarchy as a :class:`pyhealth.graph.KnowledgeGraph`.

        Every kept concept is an entity (as a string ``concept_id``), including
        any without edges. The direct and rescue relations are always both
        registered, so ``edge_type`` tells rescue edges apart.

        The default direction matters for
        :class:`pyhealth.processors.GraphProcessor`: its k-hop extraction
        follows edges *into* the seed codes. With ``"parent_to_child"`` edges,
        a patient's codes reach their ancestors; with ``"child_to_parent"``
        they reach only their descendants.

        Args:
            direction: ``"parent_to_child"`` (default) or ``"child_to_parent"``;
                see :meth:`to_triples`.

        Returns:
            A new ``KnowledgeGraph``.

        Raises:
            ValueError: If the hierarchy has no edges (for example
                ``max_depth=0``), which ``KnowledgeGraph`` cannot hold.
        """
        from pyhealth.graph import KnowledgeGraph

        triples = self.to_triples(direction)
        if not triples:
            raise ValueError("The hierarchy has no edges to export.")
        entities = sorted(str(concept_id) for concept_id in self.nodes["concept_id"])
        relations = sorted(RELATIONS[direction])
        return KnowledgeGraph(
            triples=triples,
            entity2id={entity: index for index, entity in enumerate(entities)},
            relation2id={name: index for index, name in enumerate(relations)},
        )

    def save(self, folder: Union[str, Path]) -> None:
        """Saves the hierarchy as parquet files, so later runs skip Athena.

        Args:
            folder: Folder to write to; created if needed.
        """
        folder = Path(folder).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        self.nodes.write_parquet(folder / "nodes.parquet")
        self.edges.write_parquet(folder / "edges.parquet")
        self.ancestors.write_parquet(folder / "ancestors.parquet")
        meta = {
            "root_concept_id": self.root_concept_id,
            "max_depth": self.max_depth,
            "depth_measure": self.depth_measure,
            "connect_orphans": self.connect_orphans,
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, folder: Union[str, Path]) -> "OMOPConditionHierarchy":
        """Loads a hierarchy written by :meth:`save`.

        Args:
            folder: Folder passed to :meth:`save`.
        """
        folder = Path(folder).expanduser()
        meta = json.loads((folder / "meta.json").read_text())
        return cls(
            nodes=pl.read_parquet(folder / "nodes.parquet"),
            edges=pl.read_parquet(folder / "edges.parquet"),
            ancestors=pl.read_parquet(folder / "ancestors.parquet"),
            **meta,
        )
