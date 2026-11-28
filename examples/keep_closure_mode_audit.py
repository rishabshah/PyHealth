# Author: lee-leonardo (NetID: ll37)
# Description: Analyze the KEEP graph separation-one ancestry overlay.
"""Compare direct/closure traversal without changing production behavior.

Run from the repository root: ``python -m examples.keep_closure_mode_audit``.
Only an explicit ``--athena-dir`` enables real input. Generated data stays local.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import tempfile
from collections import Counter, defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import polars as pl

from pyhealth.medcode import OMOPVocabulary
from pyhealth.tasks import build_keep_omop_ancestor_rollup

ROOT = "4274025"
POLICIES = ("first_retained", "all_retained")
CATEGORIES = (
    "identical_targets",
    "changed_nonempty_targets",
    "recovered_no_target",
    "lost_mapping",
    "other_target_change",
)


def write_synthetic_bundle(
    folder: Path, *, overlay: bool = True, duplicate_pairs: bool = False
) -> None:
    """Write a tiny Athena-format fixture with explicit, invented concepts.

    Args:
        folder: Destination for synthetic tables.
        overlay: Include two separation-one links absent from direct rows.
        duplicate_pairs: Duplicate and reverse ancestor rows to test set semantics.
    """
    folder.mkdir(parents=True, exist_ok=True)
    depths = {
        ROOT: 0,
        **{str(100 + i): i + 1 for i in range(5)},
        "200": 6,
        "201": 6,
        "202": 7,
        "203": 3,
        "204": 6,
        "205": 7,
    }
    direct = [
        (ROOT, "100"),
        ("100", "101"),
        ("101", "102"),
        ("102", "103"),
        ("103", "104"),
        ("104", "200"),
        ("103", "204"),
        ("104", "204"),
        ("204", "205"),
    ]
    concepts = [
        (
            code,
            f"Synthetic {code}",
            "Condition",
            "SNOMED",
            "Clinical Finding",
            "S",
            f"S{code}",
            "",
        )
        for code in depths
    ]
    # Non-Condition intermediate; a standard Condition outside Disease;
    # nonstandard and invalid canaries must not become graph entities.
    concepts += [
        (
            "900",
            "Filtered intermediate",
            "Observation",
            "SNOMED",
            "Finding",
            "S",
            "S900",
            "",
        ),
        ("901", "Outside Disease", "Condition", "SNOMED", "Finding", "S", "S901", ""),
        ("902", "Nonstandard", "Condition", "SNOMED", "Finding", "", "S902", ""),
        ("903", "Invalid", "Condition", "SNOMED", "Finding", "S", "S903", "D"),
    ]
    relationships = [(child, parent, "Is a", "") for parent, child in direct]
    relationships += [
        ("900", "104", "Is a", ""),
        ("202", "900", "Is a", ""),
        ("203", "900", "Is a", ""),
        ("201", "104", "Is a", "D"),
    ]
    pairs = direct + ([("104", "201"), ("102", "204")] if overlay else [])
    # Root depths deliberately authoritative, independent of adjacency distances.
    ancestors = [(ROOT, code, depth, depth) for code, depth in depths.items()]
    ancestors += [(parent, child, 1, 1) for parent, child in pairs if parent != ROOT]
    ancestors += [
        ("104", "202", 2, 2),
        ("100", "203", 2, 2),
        (ROOT, "900", 6, 6),
        (ROOT, "902", 2, 2),
        (ROOT, "903", 2, 2),
    ]
    if duplicate_pairs:
        ancestors = list(reversed(ancestors + ancestors))
    tables = {
        "CONCEPT": (
            [
                "concept_id",
                "concept_name",
                "domain_id",
                "vocabulary_id",
                "concept_class_id",
                "standard_concept",
                "concept_code",
                "invalid_reason",
            ],
            concepts,
        ),
        "CONCEPT_RELATIONSHIP": (
            ["concept_id_1", "concept_id_2", "relationship_id", "invalid_reason"],
            relationships,
        ),
        "CONCEPT_ANCESTOR": (
            [
                "ancestor_concept_id",
                "descendant_concept_id",
                "min_levels_of_separation",
                "max_levels_of_separation",
            ],
            ancestors,
        ),
        "VOCABULARY": (
            ["vocabulary_id", "vocabulary_version"],
            [("SNOMED", "SYNTHETIC")],
        ),
    }
    for name, (header, rows) in tables.items():
        with (folder / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)


def reference_graph(vocabulary: OMOPVocabulary) -> tuple[dict, set, set]:
    """Extract U, D, P independently of the factory via public reader tables.

    Returns:
        String-ID depths and unique parent-to-child direct/separation-one sets.
        This path materializes the full uncut ancestor table, not only P.
    """
    hierarchy = vocabulary.condition_hierarchy(
        max_depth=None, depth_measure="min", connect_orphans=False
    )
    depths = {
        str(code): int(depth)
        for code, depth in hierarchy.nodes.select("concept_id", "depth").iter_rows()
    }
    direct = {
        (str(parent), str(child))
        for parent, child in hierarchy.edges.select("parent_id", "child_id").iter_rows()
    }
    pairs = {
        (str(parent), str(child))
        for child, parent in hierarchy.ancestors.filter(pl.col("distance") == 1)
        .select("concept_id", "ancestor_id")
        .iter_rows()
    }
    return depths, direct, pairs


def structural_gaps(depths: dict, edges: set, eligible: set, max_depth: int) -> dict:
    """Count parentless and root-unreachable nodes, independently of mapping."""
    children = defaultdict(list)
    incoming = set()
    for parent, child in edges:
        children[parent].append(child)
        incoming.add(child)
    reachable, pending = set(), [ROOT]
    while pending:
        node = pending.pop()
        if node not in reachable:
            reachable.add(node)
            pending.extend(children[node])

    def counts(nodes):
        return {
            "total": len(nodes),
            "eligible": len(nodes & eligible),
            "deep": sum(depths[code] > max_depth for code in nodes),
            "excluded_root": int(ROOT in nodes),
        }

    # Eligible excludes the root; deep is a separate fixed-depth population.
    nonroot = set(depths) - {ROOT}
    return {
        "parentless_nonroot": counts(nonroot - incoming),
        "root_unreachable": counts(set(depths) - reachable),
    }


def _category(left, right):
    if set(left.targets) == set(right.targets):
        return "identical_targets"
    if left.reason == "no_target" and right.targets:
        return "recovered_no_target"
    if left.targets and right.targets:
        return "changed_nonempty_targets"
    if left.targets and not right.targets:
        return "lost_mapping"
    return "other_target_change"


def _witness(processor, source: str) -> dict:
    """Decode one small projected graph, including exact overlay edges."""
    try:
        graph = processor.process([source])
    except ImportError:
        return {"projection": "unavailable: torch-geometric is not installed"}
    id_to_code = processor.id2entity
    codes = [id_to_code[node] for node in graph.node_ids.tolist()]
    edges = [
        (codes[parent], codes[child])
        for parent, child in zip(*graph.edge_index.tolist())
    ]
    return {
        "projected_edges": edges,
        "closure_only_edges": [
            edge for edge, added in zip(edges, graph.closure_only.tolist()) if added
        ],
    }


def compare_policy(
    vocabulary: OMOPVocabulary,
    depths: dict,
    direct: set,
    pairs: set,
    *,
    policy: str,
    max_depth: int,
    destination: Path,
    batch_size: int = 512,
) -> dict:
    """Evaluate all U in bounded batches and write a deterministic difference TSV.

    Returns:
        Graph checks, structural gaps, population/reason counts, local witnesses,
        fingerprints, and separate factory/mapping timings for one target policy.
    """
    config = {
        "target_policy": policy,
        "include_root": False,
        "max_depth": max_depth,
        "on_unmapped": "empty",
    }
    processors, timings = {}, {}
    for mode in ("direct", "closure"):
        start = perf_counter()
        processors[mode] = build_keep_omop_ancestor_rollup(
            vocabulary, path_source=mode, **config
        )
        timings[f"{mode}_factory_seconds"] = perf_counter() - start
    left, right = processors["direct"], processors["closure"]
    eligible = {
        code for code, depth in depths.items() if depth <= max_depth and code != ROOT
    }
    assert left.retained_codes == right.retained_codes == eligible
    for mode, processor in processors.items():
        assert set(processor.knowledge_graph.entity2id) == set(depths)
        graph = processor.knowledge_graph
        id_to_code = processor.id2entity
        decoded = set(
            zip(
                (id_to_code[i] for i in graph.edge_index[0].tolist()),
                (id_to_code[i] for i in graph.edge_index[1].tolist()),
            )
        )
        assert decoded == direct
        assert int(processor.provenance["direct_edge_count"]) == len(direct)
        assert int(processor.provenance["closure_only_edge_count"]) == (
            len(pairs - direct) if mode == "closure" else 0
        )
    assert int(right.provenance["closure_pair_count"]) == len(pairs)
    populations = {}
    for name in ("all", "deep"):
        populations[name] = {
            "denominator": 0,
            "categories": dict.fromkeys(CATEGORIES, 0),
            "direct_reasons": Counter(),
            "closure_reasons": Counter(),
            "ordered_targets_differ": 0,
            "reason_changes": 0,
        }
    witnesses = defaultdict(list)
    columns = [
        "concept_id",
        "depth",
        "direct_reason",
        "closure_reason",
        "direct_targets",
        "closure_targets",
        "added_targets",
        "removed_targets",
        "category",
    ]
    sources = sorted(depths)
    mapping_digests = {mode: hashlib.sha256() for mode in processors}
    timings.update(direct_mapping_seconds=0.0, closure_mapping_seconds=0.0)
    with (
        (destination / f"{policy}_differences.tsv").open("w", newline="") as handle,
        (destination / f"{policy}_failures.tsv").open("w", newline="") as failures,
    ):
        writer = csv.DictWriter(
            handle, fieldnames=columns, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        failure_writer = csv.DictWriter(
            failures, fieldnames=columns, delimiter="\t", lineterminator="\n"
        )
        failure_writer.writeheader()
        for offset in range(0, len(sources), batch_size):
            batch = sources[offset : offset + batch_size]
            results = {}
            for mode, processor in processors.items():
                start = perf_counter()
                results[mode] = processor.rollup_codes(batch)
                timings[f"{mode}_mapping_seconds"] += perf_counter() - start
            for source, a, b in zip(batch, results["direct"], results["closure"]):
                for mode, result in (("direct", a), ("closure", b)):
                    mapping_digests[mode].update(
                        (
                            json.dumps(
                                [
                                    source,
                                    result.source_depth,
                                    result.reason,
                                    result.targets,
                                ]
                            )
                            + "\n"
                        ).encode()
                    )
                assert a.source_depth == b.source_depth == depths[source]
                assert set(a.targets) <= set(b.targets), (
                    "Adding edges at fixed eligibility must preserve targets",
                    source,
                )
                category = _category(a, b)
                for name in ("all", "deep"):
                    if name == "deep" and depths[source] <= max_depth:
                        continue
                    stats = populations[name]
                    stats["denominator"] += 1
                    stats["categories"][category] += 1
                    stats["direct_reasons"][a.reason] += 1
                    stats["closure_reasons"][b.reason] += 1
                    stats["ordered_targets_differ"] += a.targets != b.targets
                    stats["reason_changes"] += a.reason != b.reason
                row = dict(
                    zip(
                        columns,
                        [
                            source,
                            depths[source],
                            a.reason,
                            b.reason,
                            a.targets,
                            b.targets,
                            sorted(set(b.targets) - set(a.targets)),
                            sorted(set(a.targets) - set(b.targets)),
                            category,
                        ],
                    )
                )
                if category != "identical_targets" or a.reason != b.reason:
                    writer.writerow(
                        {
                            key: json.dumps(value)
                            if isinstance(value, (tuple, list))
                            else value
                            for key, value in row.items()
                        }
                    )
                if not a.targets or not b.targets:
                    failure_writer.writerow(
                        {
                            key: json.dumps(value)
                            if isinstance(value, (tuple, list))
                            else value
                            for key, value in row.items()
                        }
                    )
                witness_kind = (
                    category
                    if category != "identical_targets"
                    else ("remaining_no_target" if b.reason == "no_target" else None)
                )
                if witness_kind and len(witnesses[witness_kind]) < 3:
                    witnesses[witness_kind].append(
                        {
                            **row,
                            "direct_path": _witness(left, source),
                            "closure_path": _witness(right, source),
                        }
                    )
    for stats in populations.values():
        total = stats["denominator"]
        assert sum(stats["categories"].values()) == total
        for mode in processors:
            reasons = stats[f"{mode}_reasons"]
            assert sum(reasons.values()) == total
            mapped = reasons["retained"] + reasons["rolled_up"]
            stats[f"{mode}_mapped"] = mapped
            stats[f"{mode}_unmapped"] = total - mapped
            stats[f"{mode}_no_target"] = reasons["no_target"]
    result = {
        "config": config,
        "eligible_target_count": len(eligible),
        "populations": populations,
        "witnesses": dict(witnesses),
        "provenance": {mode: dict(p.provenance) for mode, p in processors.items()},
        "fingerprints": {mode: p.fingerprint for mode, p in processors.items()},
        "structural_gaps": {
            "direct": structural_gaps(depths, direct, eligible, max_depth),
            "closure": structural_gaps(depths, direct | pairs, eligible, max_depth),
        },
        "timings": timings,
        "ordered_mapping_sha256": {
            mode: digest.hexdigest() for mode, digest in mapping_digests.items()
        },
    }
    return result


def _input_identity(folder: Path) -> dict:
    tables = {}
    for table in ("CONCEPT", "CONCEPT_RELATIONSHIP", "CONCEPT_ANCESTOR", "VOCABULARY"):
        path = folder / f"{table}.csv"
        if not path.exists():
            tables[table] = {"status": "missing"}
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        tables[table] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    path = folder / "VOCABULARY.csv"
    if path.exists():
        with path.open(encoding="utf-8", errors="replace") as handle:
            release = list(
                csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            )
    else:
        release = "missing: no release identity inferred"
    return {"tables": tables, "vocabulary_rows": release}


def _environment() -> dict:
    def git(*args):
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    packages = {}
    for name in ("torch", "polars", "networkx", "torch-geometric", "pyhealth"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed as distribution"
    repo = Path(__file__).resolve().parents[1]
    source_paths = [
        "examples/keep_closure_mode_audit.py",
        "pyhealth/tasks/keep.py",
        "pyhealth/processors/ancestor_rollup_processor.py",
        "pyhealth/medcode/omop/vocabulary.py",
        "pyhealth/medcode/omop/condition_hierarchy.py",
    ]
    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty_state": git("status", "--short", "--untracked-files=normal"),
        "python": platform.python_version(),
        "packages": packages,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "source_root": str(Path(__file__).resolve().parents[1]),
        "source_sha256": {
            path: hashlib.sha256((repo / path).read_bytes()).hexdigest()
            for path in source_paths
        },
    }


def run_audit(
    folder: Path,
    destination: Path,
    *,
    synthetic: bool,
    max_depth: int = 5,
    batch_size: int = 512,
) -> dict:
    """Run both paired policies; keep all release-specific artifacts local.

    Args:
        folder: Explicit Athena-format input directory.
        destination: Directory for summary JSON and local per-source differences.
        synthetic: Label invented input separately from release observations.
        max_depth: Target eligibility cutoff over authoritative root depths.
        batch_size: Maximum number of source results held at a time.

    Returns:
        JSON-compatible aggregate summary, also written to ``summary.json``.
    """
    if batch_size < 1 or max_depth < 0:
        raise ValueError("batch_size must be positive and max_depth nonnegative")
    folder, destination = Path(folder), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    identity = _input_identity(folder)
    environment = _environment()
    vocabulary = OMOPVocabulary(folder)
    start = perf_counter()
    depths, direct, pairs = reference_graph(vocabulary)
    extraction_seconds = perf_counter() - start
    gc.collect()
    counts = {
        "U": len(depths),
        "D": len(direct),
        "P": len(pairs),
        "intersection": len(direct & pairs),
        "P_minus_D": len(pairs - direct),
        "D_minus_P": len(direct - pairs),
        "union": len(direct | pairs),
    }
    assert counts["union"] == counts["D"] + counts["P_minus_D"]
    policies = {}
    for policy in POLICIES:
        policies[policy] = compare_policy(
            vocabulary,
            depths,
            direct,
            pairs,
            policy=policy,
            max_depth=max_depth,
            destination=destination,
            batch_size=batch_size,
        )
        gc.collect()
    summary = {
        "schema_version": 1,
        "input_kind": "synthetic" if synthetic else "athena",
        "input_identity": identity,
        "environment": environment,
        "config": {
            "root": ROOT,
            "depth_measure": "min",
            "max_depth": max_depth,
            "include_root": False,
            "connect_orphans": False,
            "on_unmapped": "empty",
            "batch_size": batch_size,
            "population": "every concept in uncut standard SNOMED Condition U",
        },
        "graph_counts": counts,
        "policies": policies,
        "resources": {
            "reference_extraction_seconds": extraction_seconds,
            "reference_method": "public uncut reader; full ancestors materialized",
            "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_unit": "bytes" if platform.system() == "Darwin" else "KiB",
            "peak_rss_method": "getrusage RUSAGE_SELF lifetime high-water mark; all phases",
        },
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    """Run the synthetic default or a release supplied explicitly by the user."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--athena-dir", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".artifacts/keep-closure-audit")
    )
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.athena_dir is None:
        with tempfile.TemporaryDirectory() as temp:
            write_synthetic_bundle(Path(temp))
            summary = run_audit(
                Path(temp),
                args.output_dir,
                synthetic=True,
                max_depth=args.max_depth,
                batch_size=args.batch_size,
            )
    else:
        summary = run_audit(
            args.athena_dir,
            args.output_dir,
            synthetic=False,
            max_depth=args.max_depth,
            batch_size=args.batch_size,
        )
    print(json.dumps(summary["graph_counts"], indent=2))
    print(f"Local summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
