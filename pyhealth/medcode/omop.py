"""OMOP standardized vocabularies (Athena) for PyHealth.

This module reads a local download of the OMOP standardized vocabularies from
Athena (https://athena.ohdsi.org) and provides two things:

* :class:`ConditionHierarchy`: the SNOMED condition graph rooted at the concept
  "Disease" and cut off at a chosen depth, as used by KEEP (Elhussein et al.,
  CHIL 2025, appendix A.1.1).
* :class:`ICDToSNOMEDMapper`: a lookup from ICD-9-CM and ICD-10-CM diagnosis
  codes, in the dot-free form MIMIC stores them, to standard OMOP concepts.

Athena vocabularies are licensed and cannot be redistributed, so PyHealth does
not download them. Point :class:`OMOPVocabulary` at the folder you unzipped.

Example:
    >>> from pyhealth.medcode import OMOPVocabulary
    >>> vocab = OMOPVocabulary("~/data/athena")  # doctest: +SKIP
    >>> hierarchy = vocab.condition_hierarchy(max_depth=5)  # doctest: +SKIP
    >>> mapper = vocab.icd_mapper()  # doctest: +SKIP
    >>> concept_ids = mapper.map("I214", version=10)  # doctest: +SKIP
    >>> # kept ancestors of each target, nearest first: [(ancestor_id, levels)]
    >>> hierarchy.retained_ancestors(concept_ids[0])  # doctest: +SKIP
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import networkx as nx
import polars as pl

logger = logging.getLogger(__name__)

#: OMOP concept_id of SNOMED 64572001 "Disease", the root KEEP uses.
DISEASE_CONCEPT_ID = 4274025

#: ICD version number (as in MIMIC's ``icd_version`` column) -> OMOP vocabulary.
ICD_VERSION_TO_VOCABULARY = {9: "ICD9CM", 10: "ICD10CM"}

_REQUIRED_COLUMNS = {
    "CONCEPT": [
        "concept_id",
        "concept_name",
        "domain_id",
        "vocabulary_id",
        "concept_class_id",
        "standard_concept",
        "concept_code",
        "invalid_reason",
    ],
    "CONCEPT_RELATIONSHIP": [
        "concept_id_1",
        "concept_id_2",
        "relationship_id",
        "invalid_reason",
    ],
    "CONCEPT_ANCESTOR": [
        "ancestor_concept_id",
        "descendant_concept_id",
        "min_levels_of_separation",
        "max_levels_of_separation",
    ],
}

_ID_COLUMNS = {
    "concept_id",
    "concept_id_1",
    "concept_id_2",
    "ancestor_concept_id",
    "descendant_concept_id",
    "min_levels_of_separation",
    "max_levels_of_separation",
}


def _is_valid() -> pl.Expr:
    """Rows Athena has not marked deprecated ("D") or upgraded ("U")."""
    reason = pl.col("invalid_reason")
    return reason.is_null() | (reason.str.strip_chars() == "")


def _collect(frame: pl.LazyFrame) -> pl.DataFrame:
    """Collects with the streaming engine, so multi-GB files fit in memory."""
    return frame.collect(engine="streaming")


def normalize_icd_code(code: str) -> str:
    """Puts an ICD code in the dot-free, upper-case form MIMIC uses.

    Athena stores ``I21.4`` and MIMIC stores ``I214``. Comparing both in the
    dot-free form avoids having to know where each ICD version puts its dot
    (ICD-9 E-codes put it after the fourth character, everything else after
    the third).

    Args:
        code: An ICD-9-CM or ICD-10-CM code, with or without a dot.

    Returns:
        The code without dots or surrounding whitespace, in upper case.

    Example:
        >>> normalize_icd_code(" i21.4 ")
        'I214'
        >>> normalize_icd_code("E880.9")
        'E8809'
    """
    return str(code).strip().replace(".", "").upper()


def _icd_vocabulary(version: Union[int, str]) -> str:
    try:
        return ICD_VERSION_TO_VOCABULARY[int(version)]
    except (KeyError, ValueError) as err:
        raise ValueError(
            f"Unsupported ICD version {version!r}; expected 9 or 10."
        ) from err


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


class OMOPVocabulary:
    """Reads an Athena download of the OMOP standardized vocabularies.

    Only ``CONCEPT.csv``, ``CONCEPT_RELATIONSHIP.csv`` and
    ``CONCEPT_ANCESTOR.csv`` are read. Athena writes them tab-separated despite
    the ``.csv`` name, and they are several GB, so they are scanned lazily and
    filtered while reading rather than loaded whole.

    Args:
        root: Folder containing the unzipped Athena files.

    Raises:
        FileNotFoundError: If ``root`` or one of the three files is missing.
        ValueError: If a file lacks a column this module needs (for example
            because it was re-saved comma-separated).

    Example:
        >>> vocab = OMOPVocabulary("~/data/athena")  # doctest: +SKIP
        >>> hierarchy = vocab.condition_hierarchy(max_depth=5)  # doctest: +SKIP
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Athena folder not found: {self.root}")
        for table, columns in _REQUIRED_COLUMNS.items():
            self._check_header(table, columns)

    def __repr__(self) -> str:
        return f"OMOPVocabulary(root={str(self.root)!r})"

    def _path(self, table: str) -> Path:
        return self.root / f"{table}.csv"

    def _check_header(self, table: str, columns: Sequence[str]) -> None:
        path = self._path(table)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path.name} not found in {self.root}. Download the "
                "vocabularies from https://athena.ohdsi.org and unzip them there."
            )
        with open(path, encoding="utf-8", errors="replace") as handle:
            header = handle.readline().rstrip("\r\n").split("\t")
        missing = [column for column in columns if column not in header]
        if missing:
            raise ValueError(
                f"{path.name} is missing columns {missing}. Athena files are "
                "tab-separated; use the files exactly as unzipped from Athena."
            )

    def scan(self, table: str) -> pl.LazyFrame:
        """Lazily reads one Athena table.

        Every column is read as text, then id and level columns are cast to
        integers. Quote characters are not treated specially because concept
        names contain unbalanced quotes.

        Args:
            table: ``"CONCEPT"``, ``"CONCEPT_RELATIONSHIP"`` or
                ``"CONCEPT_ANCESTOR"``.

        Returns:
            A polars ``LazyFrame`` over the file.
        """
        if table not in _REQUIRED_COLUMNS:
            raise ValueError(
                f"Unknown table {table!r}; expected one of "
                f"{sorted(_REQUIRED_COLUMNS)}."
            )
        frame = pl.scan_csv(
            self._path(table),
            separator="\t",
            quote_char=None,
            infer_schema=False,
            encoding="utf8-lossy",
        )
        names = frame.collect_schema().names()
        return frame.with_columns(
            pl.col(name).cast(pl.Int64) for name in names if name in _ID_COLUMNS
        )

    def condition_hierarchy(
        self,
        root_concept_id: int = DISEASE_CONCEPT_ID,
        max_depth: Optional[int] = 5,
        depth_measure: str = "min",
        connect_orphans: bool = True,
    ) -> "ConditionHierarchy":
        """Builds the depth-limited SNOMED condition graph KEEP uses.

        Nodes are valid, standard SNOMED concepts in the Condition domain that
        descend from ``root_concept_id`` within ``max_depth`` levels, plus the
        root. Edges are the "Is a" relationships between nodes, so a concept
        with several parents keeps all of them.

        Args:
            root_concept_id: Root of the graph. Defaults to "Disease".
            max_depth: Deepest level to keep, counted from the root (depth 0).
                ``None`` keeps every descendant.
            depth_measure: How a concept's depth is counted, because SNOMED
                concepts can sit at several depths at once. ``"min"`` (the
                KEEP paper) uses the shortest path to the root;
                ``"max"`` uses the longest.
            connect_orphans: Some kept concepts have only parents that were
                filtered out (for example a parent in another domain). If
                True, each such concept is attached to its nearest kept
                ancestor so the graph stays connected; these edges are marked
                ``rescued`` in :attr:`ConditionHierarchy.edges`.

        Returns:
            The :class:`ConditionHierarchy`.
        """
        if depth_measure not in ("min", "max"):
            raise ValueError("depth_measure must be 'min' or 'max'.")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None.")
        logger.info("Building SNOMED condition hierarchy from %s", self.root)

        conditions = (
            self.scan("CONCEPT")
            .filter(
                (pl.col("vocabulary_id") == "SNOMED")
                & (pl.col("domain_id") == "Condition")
                & (pl.col("standard_concept") == "S")
                & _is_valid()
            )
            .select("concept_id", "concept_name", "concept_code")
        )
        levels = (
            self.scan("CONCEPT_ANCESTOR")
            .filter(pl.col("ancestor_concept_id") == root_concept_id)
            .select(
                pl.col("descendant_concept_id").alias("concept_id"),
                pl.col("min_levels_of_separation").alias("min_depth"),
                pl.col("max_levels_of_separation").alias("max_depth"),
            )
        )
        subtree = _collect(levels.join(conditions, on="concept_id", how="inner"))
        root = _collect(
            conditions.filter(pl.col("concept_id") == root_concept_id)
        )
        if root.height == 0:
            raise ValueError(
                f"Root concept {root_concept_id} is not a valid standard SNOMED "
                "Condition concept in this vocabulary."
            )
        if root_concept_id not in subtree["concept_id"]:
            subtree = pl.concat(
                [
                    subtree,
                    root.with_columns(
                        pl.lit(0, pl.Int64).alias("min_depth"),
                        pl.lit(0, pl.Int64).alias("max_depth"),
                    ).select(subtree.columns),
                ]
            )
        subtree = subtree.with_columns(
            pl.col(f"{depth_measure}_depth").alias("depth")
        ).unique("concept_id")

        nodes = subtree
        if max_depth is not None:
            nodes = subtree.filter(pl.col("depth") <= max_depth)
        nodes = nodes.sort("depth", "concept_id")
        node_ids = nodes.select("concept_id").lazy()

        edges = _collect(
            self.scan("CONCEPT_RELATIONSHIP")
            .filter((pl.col("relationship_id") == "Is a") & _is_valid())
            .select(
                pl.col("concept_id_2").alias("parent_id"),
                pl.col("concept_id_1").alias("child_id"),
            )
            .join(node_ids, left_on="parent_id", right_on="concept_id", how="semi")
            .join(node_ids, left_on="child_id", right_on="concept_id", how="semi")
            .filter(pl.col("parent_id") != pl.col("child_id"))
            .unique()
        ).with_columns(pl.lit(False).alias("rescued"))

        ancestors = _collect(
            self.scan("CONCEPT_ANCESTOR")
            .filter(pl.col("min_levels_of_separation") > 0)
            .select(
                pl.col("descendant_concept_id").alias("concept_id"),
                pl.col("ancestor_concept_id").alias("ancestor_id"),
                pl.col("min_levels_of_separation").alias("distance"),
            )
            .join(subtree.select("concept_id").lazy(), on="concept_id", how="semi")
            .join(node_ids, left_on="ancestor_id", right_on="concept_id", how="semi")
        )

        if connect_orphans:
            edges = _rescue_orphans(nodes, edges, ancestors, root_concept_id)
        logger.info(
            "Condition hierarchy: %d concepts, %d edges, %d orphans re-attached",
            nodes.height,
            edges.height,
            int(edges["rescued"].sum()),
        )

        return ConditionHierarchy(
            nodes=nodes,
            edges=edges.sort("parent_id", "child_id"),
            ancestors=ancestors.sort("concept_id", "distance", "ancestor_id"),
            root_concept_id=root_concept_id,
            max_depth=max_depth,
            depth_measure=depth_measure,
            connect_orphans=connect_orphans,
        )

    def icd_mapper(
        self, versions: Iterable[Union[int, str]] = (9, 10)
    ) -> "ICDToSNOMEDMapper":
        """Builds the ICD -> standard concept lookup from "Maps to" links.

        Source ICD concepts are kept even when Athena has deprecated them,
        because older records (such as MIMIC's ICD-9 era) still use them;
        only the "Maps to" links themselves must be valid.

        Args:
            versions: ICD versions to include (9 and/or 10).

        Returns:
            The :class:`ICDToSNOMEDMapper`.
        """
        vocabularies = [_icd_vocabulary(version) for version in versions]
        concept = self.scan("CONCEPT")
        sources = concept.filter(pl.col("vocabulary_id").is_in(vocabularies)).select(
            pl.col("vocabulary_id"),
            pl.col("concept_code").alias("source_code"),
            pl.col("concept_id").alias("source_concept_id"),
        )
        maps_to = (
            self.scan("CONCEPT_RELATIONSHIP")
            .filter((pl.col("relationship_id") == "Maps to") & _is_valid())
            .select(
                pl.col("concept_id_1").alias("source_concept_id"),
                pl.col("concept_id_2").alias("concept_id"),
            )
            .join(sources.select("source_concept_id"), on="source_concept_id",
                  how="semi")
        )
        targets = concept.select(
            "concept_id",
            pl.col("concept_name"),
            pl.col("domain_id").alias("target_domain_id"),
            pl.col("vocabulary_id").alias("target_vocabulary_id"),
        )
        version_of = {name: number for number, name in
                      ICD_VERSION_TO_VOCABULARY.items()}
        table = _collect(
            sources.join(maps_to, on="source_concept_id", how="left")
            .join(targets, on="concept_id", how="left")
            .with_columns(
                pl.col("vocabulary_id")
                .replace_strict(version_of, return_dtype=pl.Int64)
                .alias("icd_version"),
                pl.col("source_code")
                .str.strip_chars()
                .str.replace_all(".", "", literal=True)
                .str.to_uppercase()
                .alias("icd_code"),
            )
            .unique()
        )
        return ICDToSNOMEDMapper(
            table.select(
                "icd_version",
                "icd_code",
                "source_code",
                "source_concept_id",
                "concept_id",
                "concept_name",
                "target_domain_id",
                "target_vocabulary_id",
            ).sort("icd_version", "icd_code", "concept_id", nulls_last=True)
        )


class ConditionHierarchy:
    """A depth-limited SNOMED condition graph rooted at one concept.

    Build it with :meth:`OMOPVocabulary.condition_hierarchy`, or reload a saved
    one with :meth:`load`. Concept ids are OMOP ``concept_id`` integers.

    Attributes:
        nodes: One row per kept concept: ``concept_id``, ``concept_name``,
            ``concept_code`` (the SNOMED code), ``min_depth``, ``max_depth``
            and ``depth`` (whichever of the two was used to cut the graph).
        edges: ``parent_id``, ``child_id`` and ``rescued`` for every "Is a"
            link between kept concepts, plus any orphan rescues.
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
            f"ConditionHierarchy(root={self.root_concept_id}, "
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
    ) -> "ConditionHierarchy":
        """Cuts this hierarchy to a shallower depth without re-reading Athena.

        Build once with a generous limit (or ``max_depth=None``), then call
        this for each depth in an ablation. The result is the same as
        building at the smaller depth directly.

        Args:
            max_depth: The new, smaller depth limit.
            depth_measure: ``"min"`` or ``"max"``; defaults to this
                hierarchy's measure.

        Returns:
            A new :class:`ConditionHierarchy`.

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
        return ConditionHierarchy(
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

    def to_triples(self, relation: str = "is_a") -> List[Tuple[str, str, str]]:
        """Returns ``(child, relation, parent)`` string triples.

        This is the input format of :class:`pyhealth.graph.KnowledgeGraph`.

        Args:
            relation: Relation name to use for every edge.
        """
        return [
            (str(child), relation, str(parent))
            for parent, child in self.edges.select("parent_id", "child_id")
            .iter_rows()
        ]

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
    def load(cls, folder: Union[str, Path]) -> "ConditionHierarchy":
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


class ICDToSNOMEDMapper:
    """Maps ICD-9-CM and ICD-10-CM diagnosis codes to standard OMOP concepts.

    Build it with :meth:`OMOPVocabulary.icd_mapper`, or reload a saved one with
    :meth:`load`. Codes are matched in MIMIC's dot-free form, so ``"I214"``
    and ``"I21.4"`` are the same code. One ICD code can map to several
    concepts, and some map to none.

    Attributes:
        table: One row per (ICD code, target concept): ``icd_version``,
            ``icd_code`` (dot-free), ``source_code`` (as Athena writes it),
            ``source_concept_id``, ``concept_id`` (null when the code has no
            valid "Maps to" link), ``concept_name``, ``target_domain_id`` and
            ``target_vocabulary_id``.
    """

    def __init__(self, table: pl.DataFrame):
        self.table = table
        self._lookup: Dict[Tuple[int, str], List[int]] = defaultdict(list)
        self._known = set()
        for version, code, concept_id in table.select(
            "icd_version", "icd_code", "concept_id"
        ).iter_rows():
            self._known.add((version, code))
            if concept_id is not None:
                self._lookup[(version, code)].append(concept_id)

    def __repr__(self) -> str:
        return f"ICDToSNOMEDMapper(codes={len(self._known)})"

    def __contains__(self, key: Tuple[Union[int, str], str]) -> bool:
        version, code = key
        return (int(version), normalize_icd_code(code)) in self._known

    def map(self, code: str, version: Union[int, str]) -> List[int]:
        """Maps one ICD code to standard concept ids.

        Args:
            code: The ICD code, with or without a dot.
            version: 9 or 10 (MIMIC's ``icd_version``).

        Returns:
            Sorted, de-duplicated target concept ids; empty if the code is
            unknown or has no valid mapping.
        """
        _icd_vocabulary(version)
        key = (int(version), normalize_icd_code(code))
        return sorted(set(self._lookup.get(key, [])))

    def map_frame(
        self,
        frame: pl.DataFrame,
        code_column: str = "icd_code",
        version_column: str = "icd_version",
    ) -> pl.DataFrame:
        """Adds mapped concepts to a table of diagnoses, such as MIMIC's.

        Rows whose code maps to several concepts are repeated, one per
        concept. Rows that do not map keep a null ``concept_id``.

        Args:
            frame: A table with an ICD code column and a version column.
            code_column: Name of the code column.
            version_column: Name of the version column.

        Returns:
            ``frame`` with ``concept_id`` and ``target_domain_id`` added.
        """
        lookup = self.table.select(
            "icd_version", "icd_code", "concept_id", "target_domain_id"
        ).filter(pl.col("concept_id").is_not_null())
        keyed = frame.with_columns(
            pl.col(version_column).cast(pl.Int64).alias("__version"),
            pl.col(code_column)
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.replace_all(".", "", literal=True)
            .str.to_uppercase()
            .alias("__code"),
        )
        return keyed.join(
            lookup,
            left_on=["__version", "__code"],
            right_on=["icd_version", "icd_code"],
            how="left",
        ).drop("__version", "__code")

    def coverage(
        self,
        frame: pl.DataFrame,
        hierarchy: Optional[ConditionHierarchy] = None,
        code_column: str = "icd_code",
        version_column: str = "icd_version",
    ) -> Dict[str, int]:
        """Counts how many distinct codes in a table map, without listing any.

        Args:
            frame: A table with an ICD code column and a version column.
            hierarchy: If given, also counts codes that land under its root
                (kept, or deeper and reachable by roll-up).
            code_column: Name of the code column.
            version_column: Name of the version column.

        Returns:
            Counts of distinct (version, code) pairs: ``codes``,
            ``unknown`` (not in Athena), ``no_mapping`` (in Athena, no valid
            "Maps to"), ``mapped``; with a hierarchy, also ``in_hierarchy``
            (at least one target under the root) and ``outside_hierarchy``.
        """
        codes = frame.select(
            pl.col(version_column).cast(pl.Int64).alias("icd_version"),
            pl.col(code_column)
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.replace_all(".", "", literal=True)
            .str.to_uppercase()
            .alias("icd_code"),
        ).unique()
        known = codes.join(
            self.table.select("icd_version", "icd_code").unique(),
            on=["icd_version", "icd_code"],
            how="semi",
        )
        mapped = self.map_frame(codes).filter(pl.col("concept_id").is_not_null())
        mapped_codes = mapped.select("icd_version", "icd_code").unique()
        result = {
            "codes": codes.height,
            "unknown": codes.height - known.height,
            "no_mapping": known.height - mapped_codes.height,
            "mapped": mapped_codes.height,
        }
        if hierarchy is not None:
            under_root = set(hierarchy.nodes["concept_id"]) | set(
                hierarchy.ancestors["concept_id"]
            )
            inside = (
                mapped.filter(pl.col("concept_id").is_in(list(under_root)))
                .select("icd_version", "icd_code")
                .unique()
                .height
            )
            result["in_hierarchy"] = inside
            result["outside_hierarchy"] = mapped_codes.height - inside
        return result

    def save(self, path: Union[str, Path]) -> None:
        """Saves the mapping table as one parquet file.

        Args:
            path: File to write, for example ``icd_to_snomed.parquet``.
        """
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.write_parquet(path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ICDToSNOMEDMapper":
        """Loads a mapper written by :meth:`save`.

        Args:
            path: File passed to :meth:`save`.
        """
        return cls(pl.read_parquet(Path(path).expanduser()))
