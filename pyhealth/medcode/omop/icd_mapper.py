"""Maps ICD-9-CM and ICD-10-CM diagnosis codes to standard OMOP concepts.

Build one with :meth:`pyhealth.medcode.OMOPVocabulary.icd_mapper`.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import polars as pl

if TYPE_CHECKING:
    from .condition_hierarchy import OMOPConditionHierarchy

#: A polars table, in memory or lazy.
Frame = Union[pl.DataFrame, pl.LazyFrame]

#: ICD version number (as in MIMIC's ``icd_version`` column) -> OMOP vocabulary.
ICD_VERSION_TO_VOCABULARY = {9: "ICD9CM", 10: "ICD10CM"}


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

    def map_frame(self, frame: Frame, code_column: str = "icd_code",
                  version_column: str = "icd_version") -> Frame:
        """Adds mapped concepts to a table of diagnoses, such as MIMIC's.

        Rows whose code maps to several concepts are repeated, one per
        concept. Rows that do not map keep a null ``concept_id``. A
        ``LazyFrame`` stays lazy, so this can sit inside a streaming pipeline.

        Args:
            frame: A ``DataFrame`` or ``LazyFrame`` with an ICD code column
                and a version column.
            code_column: Name of the code column.
            version_column: Name of the version column.

        Returns:
            ``frame`` with ``concept_id`` and ``target_domain_id`` added, of
            the same type as ``frame``.
        """
        lookup = self.table.select(
            "icd_version", "icd_code", "concept_id", "target_domain_id"
        ).filter(pl.col("concept_id").is_not_null())
        if isinstance(frame, pl.LazyFrame):
            lookup = lookup.lazy()
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
        frame: Frame,
        hierarchy: Optional[OMOPConditionHierarchy] = None,
        code_column: str = "icd_code",
        version_column: str = "icd_version",
    ) -> Dict[str, int]:
        """Counts how many distinct codes in a table map, without listing any.

        Args:
            frame: A ``DataFrame`` or ``LazyFrame`` with an ICD code column and
                a version column. Only its distinct codes are collected.
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
        if isinstance(codes, pl.LazyFrame):
            codes = codes.collect()
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
