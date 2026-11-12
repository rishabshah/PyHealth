"""Build KEEP's SNOMED condition graph from Athena and check MIMIC-IV coverage.

KEEP (Elhussein et al., CHIL 2025, https://arxiv.org/abs/2510.05049) embeds the
SNOMED "Disease" hierarchy cut at five levels. This script builds that graph
from a local Athena download, reports its size at several depths (the depth
ablation in our proposal), and, given MIMIC-IV, counts how many diagnosis codes
land on it.

It prints counts only: no patient rows and no individual codes, so its output
is safe to share.

Usage:
    python examples/omop_keep_condition_graph_check.py \\
        --athena ~/data/athena \\
        --mimic ~/data/mimiciv/3.1

The first run reads the multi-GB Athena files (several minutes) and caches the
results under --cache; later runs take seconds. Pass --rebuild to ignore the
cache.
"""

import argparse
import time
from pathlib import Path

import polars as pl

from pyhealth.medcode import ConditionHierarchy, ICDToSNOMEDMapper, OMOPVocabulary

BASE_DEPTH = 6  # deepest graph in the ablation; shallower ones are cut from it


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--athena", required=True, help="Unzipped Athena folder")
    parser.add_argument("--mimic", help="MIMIC-IV folder containing hosp/")
    parser.add_argument(
        "--cache",
        default="~/.cache/pyhealth/omop_keep",
        help="Where to save the built graph and mapping",
    )
    parser.add_argument(
        "--depths", type=int, nargs="+", default=[3, 4, 5, 6],
        help=f"Depths to report (at most {BASE_DEPTH})",
    )
    parser.add_argument("--rebuild", action="store_true", help="Ignore the cache")
    return parser.parse_args()


def vocabulary_version(athena: Path) -> str:
    path = athena / "VOCABULARY.csv"
    if not path.is_file():
        return "unknown"
    table = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    row = table.filter(pl.col("vocabulary_id") == "None")
    return row["vocabulary_version"][0] if row.height else "unknown"


def load_or_build(args: argparse.Namespace):
    cache = Path(args.cache).expanduser()
    graph_dir = cache / f"hierarchy_min_d{BASE_DEPTH}"
    mapper_path = cache / "icd_to_snomed.parquet"
    if not args.rebuild and graph_dir.is_dir() and mapper_path.is_file():
        print(f"Using cached graph and mapping in {cache}")
        return ConditionHierarchy.load(graph_dir), ICDToSNOMEDMapper.load(mapper_path)

    vocab = OMOPVocabulary(args.athena)
    start = time.time()
    hierarchy = vocab.condition_hierarchy(max_depth=BASE_DEPTH)
    print(f"Built condition graph in {time.time() - start:.0f}s")
    start = time.time()
    mapper = vocab.icd_mapper()
    print(f"Built ICD mapping in {time.time() - start:.0f}s")
    hierarchy.save(graph_dir)
    mapper.save(mapper_path)
    print(f"Cached in {cache}")
    return hierarchy, mapper


def report_graphs(hierarchy: ConditionHierarchy, depths) -> None:
    print("\nCondition graph by depth limit (root = Disease, depth 0)")
    print(f"{'measure':>8} {'depth':>5} {'nodes':>8} {'edges':>8} {'rescued':>8}")
    for measure in ("min", "max"):
        for depth in sorted(depths):
            graph = hierarchy.restrict(depth, measure)
            rescued = int(graph.edges["rescued"].sum())
            print(
                f"{measure:>8} {depth:>5} {len(graph):>8} "
                f"{graph.edges.height:>8} {rescued:>8}"
            )
    by_depth = hierarchy.nodes.group_by("min_depth").len().sort("min_depth")
    print("\nNodes per level (shortest path):")
    for level, count in by_depth.iter_rows():
        print(f"  level {level}: {count}")


def report_mimic(mimic: Path, hierarchy, mapper: ICDToSNOMEDMapper) -> None:
    path = mimic.expanduser() / "hosp" / "diagnoses_icd.csv.gz"
    diagnoses = pl.read_csv(
        path,
        columns=["icd_code", "icd_version"],
        schema_overrides={"icd_code": pl.Utf8, "icd_version": pl.Int64},
    )
    print(f"\nMIMIC-IV diagnoses: {diagnoses.height} rows")
    for version in (9, 10):
        subset = diagnoses.filter(pl.col("icd_version") == version)
        counts = mapper.coverage(subset, hierarchy=hierarchy)
        print(f"\nICD-{version} distinct codes")
        for key, value in counts.items():
            share = f" ({value / counts['codes']:.1%})" if counts["codes"] else ""
            print(f"  {key:<18} {value:>7}{share}")

    under_root = set(hierarchy.nodes["concept_id"]) | set(
        hierarchy.ancestors["concept_id"]
    )
    rows = (
        mapper.map_frame(diagnoses.with_row_index("row"))
        .group_by("row")
        .agg(
            pl.col("concept_id").is_not_null().any().alias("mapped"),
            pl.col("concept_id").is_in(list(under_root)).any().alias("in_graph"),
        )
    )
    print("\nDiagnosis rows")
    print(f"  mapped to a concept      {rows['mapped'].mean():.1%}")
    print(f"  landing under Disease    {rows['in_graph'].mean():.1%}")


def main() -> None:
    args = parse_args()
    if max(args.depths) > BASE_DEPTH:
        raise SystemExit(f"--depths must be at most {BASE_DEPTH}")
    athena = Path(args.athena).expanduser()
    print(f"Athena vocabulary release: {vocabulary_version(athena)}")
    hierarchy, mapper = load_or_build(args)
    report_graphs(hierarchy, args.depths)
    if args.mimic:
        report_mimic(Path(args.mimic), hierarchy, mapper)


if __name__ == "__main__":
    main()
