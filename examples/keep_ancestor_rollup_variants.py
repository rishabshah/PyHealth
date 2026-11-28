"""KEEP baseline, single-parent and root-inclusive experiments on invented data.

The public OMOP factory and KEEP counter produce the measured matrices. The
separate example counter supplies inspectable diagnosis/presence traces and an
exact cross-check. No clinical data or local configuration is discovered.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import itertools
import json
import math
import tempfile
from collections import defaultdict, deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import median

import polars as pl

from examples.keep_ancestor_rollup_ablation_utils import (
    compare_matrices,
    count_histories,
    jsd,
    trace_pair,
    validate_matrix,
)
from pyhealth.data import Patient
from pyhealth.medcode import DISEASE_CONCEPT_ID, OMOPVocabulary
from pyhealth.tasks import build_keep_cooccurrence, build_keep_omop_ancestor_rollup
from pyhealth.tasks.keep import PRESENCE_THRESHOLD

ROOT = str(DISEASE_CONCEPT_ID)
ARMS = ("dense", "single_parent", "root_inclusive")
CHAIN_A = tuple(str(c) for c in range(101, 106))
CHAIN_B = tuple(str(c) for c in range(201, 206))
DEEP_TARGETS = frozenset(CHAIN_A + CHAIN_B)
# All labels except Disease are invented OMOP-format IDs, not clinical claims.
EDGES = (
    (ROOT, "101"),
    (ROOT, "201"),
    (ROOT, "301"),
    *itertools.pairwise(CHAIN_A),
    *itertools.pairwise(CHAIN_B),
    ("105", "900"),
    ("205", "900"),
    ("900", "901"),
)


def write_fixture(folder):
    """Write a coherent tiny Athena-format DAG, with sources at depths 6/7."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    nodes = sorted({n for edge in EDGES for n in edge}, key=int)
    parents = defaultdict(list)
    for parent, child in EDGES:
        parents[child].append(parent)
    ancestry = []
    for source in nodes:
        distances = defaultdict(list)
        pending = [(source, 0)]
        while pending:
            node, distance = pending.pop()
            distances[node].append(distance)
            pending.extend((p, distance + 1) for p in parents[node])
        ancestry.extend(
            (ancestor, source, min(ds), max(ds))
            for ancestor, ds in sorted(distances.items())
        )
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
            [
                (
                    c,
                    "Disease" if c == ROOT else f"Invented {c}",
                    "Condition",
                    "SNOMED",
                    "Clinical Finding",
                    "S",
                    c,
                    "",
                )
                for c in nodes
            ],
        ),
        "CONCEPT_RELATIONSHIP": (
            ["concept_id_1", "concept_id_2", "relationship_id", "invalid_reason"],
            [(c, p, "Is a", "") for p, c in EDGES],
        ),
        "CONCEPT_ANCESTOR": (
            [
                "ancestor_concept_id",
                "descendant_concept_id",
                "min_levels_of_separation",
                "max_levels_of_separation",
            ],
            ancestry,
        ),
    }
    for name, (header, rows) in tables.items():
        with (folder / f"{name}.csv").open("w", newline="") as out:
            writer = csv.writer(out, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows)
    return folder


def synthetic_histories():
    """Ten fixed patients; diagnosis IDs distinguish rows from copied rows."""

    def patient(identity, rows):
        return {
            "patient_id": identity,
            "diagnoses": [
                {"diagnosis_id": diagnosis, "codes": [code]} for diagnosis, code in rows
            ],
        }

    return [
        patient("p1", [("x1", "900"), ("x2", "900")]),
        patient("p2", [("y1", "901"), ("y2", "901"), ("b1", "205"), ("b2", "205")]),
        patient("p3", [("a1", "105"), ("a2", "105"), ("b1", "205"), ("b2", "205")]),
        patient("p4", [("copy", "900"), ("copy", "900")]),
        patient("p5", [("x", "900"), ("y", "901")]),
        patient("p6", []),
        patient("p7", [("x", "900")]),
        patient("p8", [("u1", "999999"), ("u2", "999999")]),
        patient("p9", [("r1", ROOT), ("r2", ROOT)]),
        patient("p10", [("x", "900"), ("a", "105")]),
    ]


class SingleParentRollup:
    """Example-only selection from the dense mapper's targets per source code.

    Rank candidates by shortest upward direct path, then numeric OMOP ID.
    Retained identities and failures are unchanged. This is not a new library
    target_policy, a first_retained alias, or an ICD one-to-many reduction.
    """

    def __init__(self, dense):
        if dense.provenance["path_source"] != "direct":
            raise ValueError(
                "this study's single-parent adapter requires direct traversal"
            )
        self.dense = dense
        self.retained_codes = dense.retained_codes
        graph = dense.knowledge_graph
        self.parents = defaultdict(set)
        for p, c in graph.edge_index.t().tolist():
            self.parents[graph.id2entity[c]].add(graph.id2entity[p])
        self._distances = {}

    def distances(self, source):
        if source not in self._distances:
            found, pending = {source: 0}, deque([source])
            while pending:
                child = pending.popleft()
                for parent in sorted(self.parents[child], key=int):
                    if parent not in found:
                        found[parent] = found[child] + 1
                        pending.append(parent)
            self._distances[source] = found
        return self._distances[source]

    def rollup_codes(self, codes):
        results = []
        for result in self.dense.rollup_codes(codes):
            if len(result.targets) > 1:
                distances = self.distances(result.source)
                selected = min(result.targets, key=lambda c: (distances[c], int(c)))
                result = replace(result, targets=(selected,))
            results.append(result)
        return results


def make_rollups(vocabulary):
    # max_depth=5 and path_source='direct' deliberately use API defaults.
    dense = build_keep_omop_ancestor_rollup(
        vocabulary,
        target_policy="all_retained",
        include_root=False,
        on_unmapped="empty",
    )
    inclusive = build_keep_omop_ancestor_rollup(
        vocabulary,
        target_policy="all_retained",
        include_root=True,
        on_unmapped="empty",
    )
    assert dense.provenance["max_depth"] == "5"
    assert dense.provenance["path_source"] == "direct"
    return {
        "dense": dense,
        "single_parent": SingleParentRollup(dense),
        "root_inclusive": inclusive,
    }


def as_patients(histories):
    """Bridge invented canonical-concept rows into real PyHealth Patients.

    This fixture bypasses ICD conversion deliberately; a diagnosis contains
    one canonical OMOP concept. Full ingestion is covered by its own tests.
    """
    schema = {
        "patient_id": pl.String,
        "event_type": pl.String,
        "timestamp": pl.Datetime,
        "_source_row": pl.Int64,
        "condition_occurrence/condition_occurrence_id": pl.String,
        "condition_occurrence/condition_concept_id": pl.String,
        "condition_occurrence/condition_source_value": pl.String,
        "condition_occurrence/visit_occurrence_id": pl.String,
    }
    for history in histories:
        rows = []
        for index, event in enumerate(history["diagnoses"]):
            if len(event["codes"]) != 1:
                raise ValueError(
                    "the OMOP fixture bridge requires one canonical concept per row"
                )
            rows.append(
                (
                    history["patient_id"],
                    "condition_occurrence",
                    datetime(2020, 1, 1),
                    index,
                    event["diagnosis_id"],
                    event["codes"][0],
                    None,
                    "synthetic-visit",
                )
            )
        yield Patient(
            patient_id=history["patient_id"],
            data_source=pl.DataFrame(rows, schema=schema, orient="row"),
        )


def expected_presence(arm):
    """Hand-derived patient sets, independent of the mapper and both counters."""
    deep = {"105"} if arm == "single_parent" else set(DEEP_TARGETS)
    if arm == "root_inclusive":
        deep.add(ROOT)
    return [
        deep,
        deep | {"205"},
        {"105", "205"},
        set(),
        deep,
        set(),
        set(),
        set(),
        {ROOT} if arm == "root_inclusive" else set(),
        {"105"},
    ]


def assert_oracle(arm, matrix, traces):
    expected = expected_presence(arm)
    assert matrix["patients"] == len(expected) == 10
    assert [set(row["present"]) for row in traces] == expected
    values = validate_matrix(matrix)
    for i, j in itertools.product(matrix["concept_ids"], repeat=2):
        assert values.get((i, j), 0) == sum(i in row and j in row for row in expected)
    assert all("301" not in pair for pair in values)  # explicit unused eligible axis


def npmi(count, prevalence_i, prevalence_j, patients):
    """Natural-log NPMI, undefined for unobserved pairs and joint prevalence 1."""
    if not count or count == patients:
        return None
    return math.log(patients * count / (prevalence_i * prevalence_j)) / -math.log(
        count / patients
    )


def pair_statistics(matrix):
    values = validate_matrix(matrix)
    pairs = {pair: v for pair, v in sorted(values.items()) if pair[0] < pair[1]}
    total = sum(pairs.values())
    possible = len(matrix["concept_ids"]) * (len(matrix["concept_ids"]) - 1) // 2
    entropy = (
        max(0.0, -math.fsum(v / total * math.log2(v / total) for v in pairs.values()))
        if total
        else None
    )
    return {
        "patients": matrix["patients"],
        "comparison_axes": len(matrix["concept_ids"]),
        "observed_pairs": len(pairs),
        "possible_pairs": possible,
        "pair_density": len(pairs) / possible if possible else None,
        "pair_mass": total,
        "entropy_bits": entropy,
        "normalized_entropy": entropy / math.log2(possible)
        if entropy is not None and possible > 1
        else None,
        "conditions_present": sum(i == j for i, j in values),
    }


def compare_variants(before, after):
    report = compare_matrices(before, after)
    for row in report["pairs"]:
        for side in ("before", "after"):
            scores = row[side]
            scores["npmi"] = npmi(
                scores["count"],
                scores["prevalence_i"],
                scores["prevalence_j"],
                report["patients"],
            )
        a, b = row["before"]["npmi"], row["after"]["npmi"]
        row["npmi_delta"] = b - a if a is not None and b is not None else None
    a, b = validate_matrix(before), validate_matrix(after)
    summary = report["summary"]
    denominator = summary["observed_pair_union"]
    summary["changed_pair_fraction"] = (
        summary["changed_count_pairs"] / denominator if denominator else None
    )
    summary["global_pair_mass_jsd"] = jsd(
        {p: v for p, v in a.items() if p[0] < p[1]},
        {p: v for p, v in b.items() if p[0] < p[1]},
    )
    summary["root_pair_absolute_delta"] = sum(
        abs(r["count_delta"]) for r in report["pairs"] if ROOT in (r["i"], r["j"])
    )
    summary["nonroot_pair_absolute_delta"] = (
        summary["absolute_count_delta"] - summary["root_pair_absolute_delta"]
    )
    common = [r for r in report["pairs"] if r["status"] == "shared"]
    report["common_support"] = {"pairs": len(common)}
    for side in ("before", "after"):
        for metric in ("ppmi", "npmi"):
            scores = [r[side][metric] for r in common if r[side][metric] is not None]
            report["common_support"][f"{side}_{metric}_defined"] = len(scores)
            report["common_support"][f"{side}_{metric}_median"] = (
                median(scores) if scores else None
            )
    return report


def run_study():
    """Run the public factory/counter, independent oracle, and two comparisons."""
    assert PRESENCE_THRESHOLD == 2, (
        "this experiment fixes the public two-diagnosis rule"
    )
    histories = synthetic_histories()
    with tempfile.TemporaryDirectory() as temporary:
        folder = write_fixture(Path(temporary))
        vocabulary = OMOPVocabulary(folder)
        rollups = make_rollups(vocabulary)
        # Canonical OMOP IDs take precedence; no ICD mapping is exercised.
        mapper = vocabulary.icd_mapper()
        axes = sorted(set().union(*(r.retained_codes for r in rollups.values())))
        matrices, traces, aggregates, targets = {}, {}, {}, {}
        for arm, rollup in rollups.items():
            native_axes = sorted(rollup.retained_codes)
            result = build_keep_cooccurrence(
                as_patients(histories),
                rollup=rollup,
                icd_mapper=mapper,
                concept_ids=native_axes,
            )
            sparse = result.matrix.coalesce()
            public_matrix = {
                "patients": result.patients_processed,
                "concept_ids": axes,
                "entries": sorted(
                    [native_axes[i], native_axes[j], int(v)]
                    for (i, j), v in zip(
                        sparse.indices().t().tolist(), sparse.values().tolist()
                    )
                ),
            }
            independent, traces[arm] = count_histories(rollup, histories)
            independent["concept_ids"] = axes
            assert public_matrix == independent, (
                f"public/example counter discrepancy: {arm}"
            )
            assert_oracle(arm, public_matrix, traces[arm])
            matrices[arm] = public_matrix
            targets[arm] = [
                vars(r)
                for r in rollup.rollup_codes(["900", "901", "105", ROOT, "999999"])
            ]
            graph = rollups["dense"].knowledge_graph
            decoded_edges = [
                (graph.id2entity[p], graph.id2entity[c])
                for p, c in graph.edge_index.t().tolist()
            ]
            aggregates[arm] = {
                **pair_statistics(public_matrix),
                "eligible_targets": len(native_axes),
                "traversal_nodes": graph.num_entities,
                "traversal_edges": graph.num_triples,
                "eligible_induced_edges": sum(
                    p in rollup.retained_codes and c in rollup.retained_codes
                    for p, c in decoded_edges
                ),
                "patients_with_present_targets": result.patients_with_present_targets,
                "unique_diagnoses": result.unique_diagnosis_events,
                "mapped_diagnoses": result.mapped_events,
                "unmapped_diagnoses": dict(result.unmapped_events),
            }
        comparisons = {
            arm: compare_variants(matrices["dense"], matrices[arm]) for arm in ARMS[1:]
        }
        for arm, comparison in comparisons.items():
            for row in comparison["pairs"]:
                witnesses = trace_pair(traces["dense"], traces[arm], row["i"], row["j"])
                assert sum(w["pair_delta"] for w in witnesses) == row["count_delta"]
        input_hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.glob("*.csv"))
        }
        source_hashes = {
            name: hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
            for name, obj in (
                ("factory", build_keep_omop_ancestor_rollup),
                ("counter", build_keep_cooccurrence),
                ("processor", type(rollups["dense"])),
                ("study", run_study),
                ("metrics", compare_matrices),
            )
        }
    return {
        "schema": "keep-rollup-variants-v1",
        "synthetic_only": True,
        "settings": {
            "baseline": "dense",
            "root": ROOT,
            "max_depth": 5,
            "target_policy": "all_retained",
            "include_root": False,
            "path_source": "direct",
            "rescue": False,
            "presence_threshold": 2,
            "on_unmapped": "empty",
            "comparison_axes": axes,
            "single_parent_rule": "shortest upward direct path, then numeric concept ID",
            "root_inclusive_rule": "API include_root=True; retained identities preserved",
        },
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "histories_sha256": hashlib.sha256(
            json.dumps(histories, sort_keys=True).encode()
        ).hexdigest(),
        "public_counter_agreement": True,
        "hand_oracle_passed": True,
        "matrices": matrices,
        "traces": traces,
        "targets": targets,
        "aggregates": aggregates,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New synthetic output directory"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must not exist")
    report = run_study()
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        "Passed: public counter agreement, hand oracle, and both roll-up comparisons."
    )


if __name__ == "__main__":
    main()
