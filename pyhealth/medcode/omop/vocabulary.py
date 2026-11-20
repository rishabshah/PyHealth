"""Reads OMOP standardized vocabularies downloaded from Athena.

Athena vocabularies are licensed and cannot be redistributed, so PyHealth does
not download them. Point :class:`OMOPVocabulary` at the folder you unzipped.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import polars as pl

from .condition_hierarchy import OMOPConditionHierarchy, _rescue_orphans
from .icd_mapper import ICD_VERSION_TO_VOCABULARY, ICDToSNOMEDMapper, _icd_vocabulary

logger = logging.getLogger(__name__)

#: OMOP concept_id of SNOMED 64572001 "Disease", the root KEEP uses.
DISEASE_CONCEPT_ID = 4274025


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


@dataclass(frozen=True)
class OMOPVocabularyFormat:
    """How the vocabulary files are written. The defaults match Athena.

    Change these only for vocabulary tables from another source, such as a
    comma-separated export from an OMOP database.

    Args:
        separator: Column separator. Athena uses a tab despite the ``.csv``
            file names.
        quote_char: Quote character, or None to treat quotes as ordinary
            text. Athena does not quote fields, and some concept names
            contain stray quotes, so the default is None.
        encoding: ``"utf8-lossy"`` (replace invalid bytes) or ``"utf8"``.
        file_name: File name pattern; ``{table}`` becomes ``CONCEPT``,
            ``CONCEPT_RELATIONSHIP`` or ``CONCEPT_ANCESTOR``.

    Example:
        >>> OMOPVocabularyFormat(separator=",", quote_char='"').separator
        ','
    """

    separator: str = "\t"
    quote_char: Optional[str] = None
    encoding: str = "utf8-lossy"
    file_name: str = "{table}.csv"

    def __post_init__(self) -> None:
        if len(self.separator) != 1:
            raise ValueError("separator must be a single character.")
        if self.quote_char is not None and len(self.quote_char) != 1:
            raise ValueError("quote_char must be a single character or None.")
        if self.encoding not in ("utf8", "utf8-lossy"):
            raise ValueError("encoding must be 'utf8' or 'utf8-lossy'.")
        if "{table}" not in self.file_name:
            raise ValueError("file_name must contain '{table}'.")

    def split_header(self, line: str) -> List[str]:
        """Splits a header line into column names."""
        if self.quote_char is None:
            return line.split(self.separator)
        return next(csv.reader([line], delimiter=self.separator,
                               quotechar=self.quote_char))


class OMOPVocabulary:
    """Reads an Athena download of the OMOP standardized vocabularies.

    Only ``CONCEPT.csv``, ``CONCEPT_RELATIONSHIP.csv`` and
    ``CONCEPT_ANCESTOR.csv`` are read. Athena writes them tab-separated despite
    the ``.csv`` name, and they are several GB, so they are scanned lazily and
    filtered while reading rather than loaded whole.

    Args:
        root: Folder containing the unzipped Athena files.
        file_format: How the files are written; defaults to Athena's format.
            See :class:`OMOPVocabularyFormat`.

    Raises:
        FileNotFoundError: If ``root`` or one of the three files is missing.
        ValueError: If a file lacks a column this module needs (for example
            because it was re-saved with a different separator than
            ``file_format`` says).

    Example:
        >>> vocab = OMOPVocabulary("~/data/athena")  # doctest: +SKIP
        >>> hierarchy = vocab.condition_hierarchy(max_depth=5)  # doctest: +SKIP
    """

    def __init__(
        self,
        root: Union[str, Path],
        file_format: Optional[OMOPVocabularyFormat] = None,
    ):
        self.root = Path(root).expanduser()
        self.file_format = file_format or OMOPVocabularyFormat()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Athena folder not found: {self.root}")
        for table, columns in _REQUIRED_COLUMNS.items():
            self._check_header(table, columns)

    def __repr__(self) -> str:
        return f"OMOPVocabulary(root={str(self.root)!r})"

    def _path(self, table: str) -> Path:
        return self.root / self.file_format.file_name.format(table=table)

    def _check_header(self, table: str, columns: Sequence[str]) -> None:
        path = self._path(table)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path.name} not found in {self.root}. Download the "
                "vocabularies from https://athena.ohdsi.org and unzip them there."
            )
        with open(path, encoding="utf-8", errors="replace") as handle:
            line = handle.readline().rstrip("\r\n")
        header = self.file_format.split_header(line)
        missing = [column for column in columns if column not in header]
        if missing:
            raise ValueError(
                f"{path.name} is missing columns {missing} when split on "
                f"{self.file_format.separator!r}. Athena files are tab-separated; "
                "for files from another source, pass a matching "
                "OMOPVocabularyFormat as file_format."
            )

    def scan(self, table: str) -> pl.LazyFrame:
        """Lazily reads one Athena table.

        Every column is read as text, then id and level columns are cast to
        integers. Separator, quoting and encoding come from ``file_format``.

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
            separator=self.file_format.separator,
            quote_char=self.file_format.quote_char,
            infer_schema=False,
            encoding=self.file_format.encoding,
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
    ) -> "OMOPConditionHierarchy":
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
                ``rescued`` in :attr:`OMOPConditionHierarchy.edges`.

        Returns:
            The :class:`OMOPConditionHierarchy`.
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

        return OMOPConditionHierarchy(
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
