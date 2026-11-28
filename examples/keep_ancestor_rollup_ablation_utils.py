"""Synthetic policy experiment and relationship measures for the KEEP notebook.

All histories and graph labels in this module are invented. Counting lives in
this example because the ancestor-rollup branch predates the KEEP counting task.
No patient data, local configuration, or vocabulary directory is discovered.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
from collections import Counter
from pathlib import Path

POLICIES = ("first_retained", "all_retained")
ARMS = tuple(itertools.product(POLICIES, (False, True), (False, True)))
DEPTHS = {
    "R": 0,
    "A1": 1,
    "A2": 2,
    "A3": 3,
    "B1": 1,
    "B2": 2,
    "B3": 3,
    "C1": 1,
    "C2": 2,
    "X": 3,
    "O": 3,
    "Z": 2,
}
DIRECT = (
    ("R", "A1"),
    ("A1", "A2"),
    ("A2", "A3"),
    ("A3", "X"),
    ("R", "B1"),
    ("B1", "B2"),
    ("B2", "B3"),
    ("B3", "X"),
    ("R", "C1"),
    ("C1", "C2"),
)
# A separation-one pair absent from the direct relationship table.
CLOSURE_ONLY = (("C2", "X"),)
# Fixed synthetic shortcut, held identical in direct and closure arms.
# This illustrates adding a rescue edge; it does not implement Athena rescue.
RESCUE = (("A2", "O"),)
AXES = tuple(sorted(c for c, d in DEPTHS.items() if 0 < d <= 2))


def arm_name(arm):
    policy, closure, rescue = arm
    return f"{policy}|closure={int(closure)}|rescue={int(rescue)}"


def synthetic_histories():
    """Return explicit diagnosis identities, including copies and empty patients."""

    def event(identity, *codes):
        return {"diagnosis_id": identity, "codes": list(codes)}

    return [
        {
            "patient_id": "p1",
            "diagnoses": [
                event("x1", "X"),
                event("x2", "X"),
                event("c1", "C2"),
                event("c2", "C2"),
            ],
        },
        {"patient_id": "p2", "diagnoses": [event("x1", "X"), event("x2", "X")]},
        {
            "patient_id": "p3",
            "diagnoses": [
                event("o1", "O"),
                event("o2", "O"),
                event("b1", "B2"),
                event("b2", "B2"),
            ],
        },
        {"patient_id": "p4", "diagnoses": [event("copy", "X"), event("copy", "X")]},
        {
            "patient_id": "p5",
            "diagnoses": [
                event("a1", "A2"),
                event("a2", "A2"),
                event("b1", "B2"),
                event("b2", "B2"),
            ],
        },
        {"patient_id": "p6", "diagnoses": []},
        {
            "patient_id": "p7",
            "diagnoses": [
                event("multi", "X", "O"),
                event("multi", "O", "X"),
                event("unmapped", "UNKNOWN"),
            ],
        },
        {
            "patient_id": "p8",
            "diagnoses": [
                event("c1", "C2"),
                event("c2", "C2"),
                event("a1", "A2"),
                event("a2", "A2"),
            ],
        },
        {"patient_id": "p9", "diagnoses": [event("o1", "O"), event("o2", "O")]},
    ]


def make_processors():
    """Build eight real PyHealth processors with fixed eligibility and graph IDs."""
    from pyhealth.graph import KnowledgeGraph
    from pyhealth.processors import AncestorRollupProcessor

    processors = {}
    for arm in ARMS:
        policy, closure, rescue = arm
        edges = DIRECT + (RESCUE if rescue else ())
        graph = KnowledgeGraph(
            triples=[(p, "is_a", c) for p, c in edges],
            entity2id={code: i for i, code in enumerate(sorted(DEPTHS))},
            relation2id={"is_a": 0},
        )
        processors[arm_name(arm)] = AncestorRollupProcessor(
            knowledge_graph=graph,
            domain_by_code={code: "Condition" for code in DEPTHS},
            hierarchy_relation="is_a",
            root="R",
            root_depths=DEPTHS,
            target_policy=policy,
            max_depth=2,
            include_root=False,
            path_source="closure" if closure else "direct",
            closure_pairs=CLOSURE_ONLY if closure else None,
            on_unmapped="empty",
        )
    return processors


def count_histories(processor, histories):
    """Count whole-history patient presence after two distinct diagnoses.

    Codes are already mapped concept labels. Union all targets within each
    diagnosis, deduplicate identical copies, and reject conflicting copies.
    This example's identity contract consists of diagnosis ID and source codes.
    """
    axes = sorted(processor.retained_codes)
    counts, traces, patients = Counter(), [], set()
    for history in histories:
        patient = history["patient_id"]
        if not isinstance(patient, str) or not patient or patient in patients:
            raise ValueError("patient IDs must be nonempty unique strings")
        patients.add(patient)
        unique = {}
        for event in history["diagnoses"]:
            identity, codes = event["diagnosis_id"], event["codes"]
            if not isinstance(identity, str) or not identity:
                raise ValueError("diagnosis IDs must be nonempty strings")
            if not codes or any(not isinstance(c, str) or not c for c in codes):
                raise ValueError("each diagnosis needs nonempty string codes")
            canonical = tuple(sorted(set(codes)))
            if identity in unique and unique[identity] != canonical:
                raise ValueError("conflicting copies of a diagnosis")
            unique[identity] = canonical
        occurrences, diagnoses = Counter(), []
        for identity, codes in sorted(unique.items()):
            results = processor.rollup_codes(codes)
            targets = sorted({t for result in results for t in result.targets})
            occurrences.update(targets)
            diagnoses.append(
                {
                    "diagnosis_id": identity,
                    "codes": list(codes),
                    "targets": targets,
                    "results": [
                        {
                            "source": r.source,
                            "targets": list(r.targets),
                            "reason": r.reason,
                        }
                        for r in results
                    ],
                }
            )
        present = sorted(c for c, n in occurrences.items() if n >= 2)
        counts.update(itertools.product(present, repeat=2))
        traces.append(
            {
                "patient_id": patient,
                "diagnoses": diagnoses,
                "occurrences": dict(sorted(occurrences.items())),
                "present": present,
            }
        )
    matrix = {
        "concept_ids": axes,
        "patients": len(patients),
        "entries": [[i, j, n] for (i, j), n in sorted(counts.items())],
    }
    validate_matrix(matrix)
    return matrix, traces


def expected_targets(arm):
    """Hand-derived target sets for this fixture, independent of traversal."""
    policy, closure, rescue = arm
    x = {"A2", "B2"}
    o = {"A2"} if rescue else set()
    if closure:
        x.add("C2")
    if policy == "all_retained":
        x |= {"A1", "B1"} | ({"C1"} if closure else set())
        o |= {"A1"} if rescue else set()
    return x, o


def assert_fixture(processors, matrices, traces):
    """Assert targets, literal patient presence, and every matrix coordinate."""
    for arm in ARMS:
        name = arm_name(arm)
        x, o = expected_targets(arm)
        actual = processors[name].rollup_codes(["X", "O", "A2", "Z", "UNKNOWN"])
        assert [set(r.targets) for r in actual] == [x, o, {"A2"}, {"Z"}, set()]
        expected = [
            x | {"C2"},
            x,
            o | {"B2"},
            set(),
            {"A2", "B2"},
            set(),
            set(),
            {"C2", "A2"},
            o,
        ]
        assert [set(row["present"]) for row in traces[name]] == expected, name
        values = validate_matrix(matrices[name])
        assert set(matrices[name]["concept_ids"]) == set(AXES)
        assert matrices[name]["patients"] == 9
        for i, j in itertools.product(AXES, repeat=2):
            assert values.get((i, j), 0) == sum(i in p and j in p for p in expected)
        assert all("Z" not in pair for pair in values)


def validate_matrix(matrix):
    """Check sparse symmetric patient counts, retaining explicit zero axes."""
    ids, n = matrix["concept_ids"], matrix["patients"]
    if any(not isinstance(c, str) or not c for c in ids) or len(set(ids)) != len(ids):
        raise ValueError("concept IDs must be nonempty unique strings")
    if type(n) is not int or n <= 0:
        raise ValueError("patients must be a positive integer")
    values = {}
    for i, j, value in matrix["entries"]:
        if i not in ids or j not in ids or type(value) is not int or value <= 0:
            raise ValueError("entries need known IDs and positive integer counts")
        if (i, j) in values:
            raise ValueError("duplicate coordinate")
        values[i, j] = value
    for (i, j), value in values.items():
        if values.get((j, i)) != value:
            raise ValueError("matrix must be symmetric")
        pi, pj = values.get((i, i), 0), values.get((j, j), 0)
        if value > min(pi, pj, n):
            raise ValueError("count exceeds prevalence or cohort size")
        if value < pi + pj - n:
            raise ValueError("count violates the patient intersection lower bound")
    # Missing entries are zeros and must also obey intersection constraints.
    active = [c for c in ids if values.get((c, c), 0)]
    for i, j in itertools.combinations(active, 2):
        if values.get((i, j), 0) < values[i, i] + values[j, j] - n:
            raise ValueError(
                "missing pair violates the patient intersection lower bound"
            )
    return values


def cosine(left, right):
    a = math.sqrt(sum(v * v for v in left.values()))
    b = math.sqrt(sum(v * v for v in right.values()))
    if not a or not b:
        return None
    result = sum(v * right.get(k, 0) for k, v in left.items()) / (a * b)
    return min(1.0, max(0.0, result))


def jsd(left, right):
    """Base-2 Jensen-Shannon divergence, not its square-root distance."""
    a, b = sum(left.values()), sum(right.values())
    if not a or not b:
        return None
    result = 0.0
    for key in sorted(left.keys() | right.keys()):
        p, q = left.get(key, 0) / a, right.get(key, 0) / b
        m = (p + q) / 2
        result += (
            (p * math.log2(p / m) if p else 0) + (q * math.log2(q / m) if q else 0)
        ) / 2
    return min(1.0, max(0.0, result))


def concentration(values):
    ordered = sorted(values, reverse=True)
    total, pairs = sum(ordered), len(ordered)
    k = math.ceil(pairs * 0.01)
    return {
        "observed_pairs": pairs,
        "possible_pairs": None,
        "pair_mass": total,
        "top_pair_count": k,
        "top_1pct_share": sum(ordered[:k]) / total if total else None,
        "hhi": sum((v / total) ** 2 for v in ordered) if total else None,
    }


def compare_matrices(before, after, *, x_max=50.0, alpha=0.75):
    """Compare fixed-axis counts; producers are responsible for matched patients."""
    if any(not math.isfinite(x) or x <= 0 for x in (x_max, alpha)):
        raise ValueError("x_max and alpha must be finite and positive")
    left, right = validate_matrix(before), validate_matrix(after)
    if set(before["concept_ids"]) != set(after["concept_ids"]):
        raise ValueError("concept universes differ")
    if before["patients"] != after["patients"]:
        raise ValueError("cohort sizes differ")
    ids, n = sorted(before["concept_ids"]), before["patients"]
    pairs = sorted(
        {tuple(sorted((i, j))) for i, j in left.keys() | right.keys() if i != j}
    )

    def scores(values, i, j):
        x, pi, pj = values.get((i, j), 0), values.get((i, i), 0), values.get((j, j), 0)
        return {
            "count": x,
            "prevalence_i": pi,
            "prevalence_j": pj,
            "log_count": math.log(x) if x else None,
            "glove_weight": min((x / x_max) ** alpha, 1.0),
            "ppmi": (max(0.0, math.log(n * x / (pi * pj))) if x else 0.0)
            if pi and pj
            else None,
            "p_j_given_i": x / pi if pi else None,
            "p_i_given_j": x / pj if pj else None,
        }

    rows = []
    profiles = {
        label: {(c, key): {} for c in ids for key in ("count", "ppmi")}
        for label in ("before", "after")
    }
    for i, j in pairs:
        a, b = scores(left, i, j), scores(right, i, j)
        row = {"i": i, "j": j, "before": a, "after": b}
        row["status"] = (
            "gained" if not a["count"] else "lost" if not b["count"] else "shared"
        )
        for key in (
            "count",
            "log_count",
            "glove_weight",
            "ppmi",
            "p_j_given_i",
            "p_i_given_j",
        ):
            row[key + "_delta"] = (
                b[key] - a[key] if a[key] is not None and b[key] is not None else None
            )
        rows.append(row)
        for label, values in (("before", a), ("after", b)):
            for key in ("count", "ppmi"):
                profiles[label][i, key][j] = values[key] or 0.0
                profiles[label][j, key][i] = values[key] or 0.0
    profile_rows = []
    for c in ids:
        row = {"concept_id": c}
        for key in ("count", "ppmi"):
            a, b = profiles["before"][c, key], profiles["after"][c, key]
            row[key + "_cosine"] = cosine(a, b)
            row[key + "_jsd"] = jsd(a, b)
        profile_rows.append(row)
    concentrations = {}
    for label, values in (("before", left), ("after", right)):
        concentrations[label] = concentration(
            [values[p] for p in pairs if values.get(p)]
        )
        concentrations[label]["possible_pairs"] = len(ids) * (len(ids) - 1) // 2
    return {
        "patients": n,
        "x_max": x_max,
        "alpha": alpha,
        "pairs": rows,
        "profiles": profile_rows,
        "prevalence": [
            {
                "concept_id": c,
                "before": left.get((c, c), 0),
                "after": right.get((c, c), 0),
            }
            for c in ids
        ],
        "concentration": concentrations,
        "summary": {
            "observed_pair_union": len(pairs),
            "changed_count_pairs": sum(r["count_delta"] != 0 for r in rows),
            "gained_pairs": sum(r["status"] == "gained" for r in rows),
            "lost_pairs": sum(r["status"] == "lost" for r in rows),
            "absolute_count_delta": sum(abs(r["count_delta"]) for r in rows),
            "changed_ppmi_pairs": sum(
                r["ppmi_delta"] is not None
                and not math.isclose(r["ppmi_delta"], 0, abs_tol=1e-12)
                for r in rows
            ),
            "undefined_ppmi_deltas": sum(r["ppmi_delta"] is None for r in rows),
        },
    }


def one_factor_contrasts():
    """Twelve edges of the 2x2x2 design; exactly one setting changes each time."""
    for index, factor in enumerate(("target_policy", "closure", "rescue")):
        for before in ARMS:
            after = list(before)
            if before[index] != (POLICIES[0] if index == 0 else False):
                continue
            after[index] = POLICIES[1] if index == 0 else True
            yield factor, arm_name(before), arm_name(tuple(after))


def trace_pair(before_trace, after_trace, i, j):
    """Explain one pair delta with synthetic patient and diagnosis evidence."""
    before = {r["patient_id"]: r for r in before_trace}
    after = {r["patient_id"]: r for r in after_trace}
    if before.keys() != after.keys():
        raise ValueError("trace patient membership differs")
    rows = []
    for patient in sorted(before):
        a, b = before[patient], after[patient]
        av, bv = (
            int(i in a["present"] and j in a["present"]),
            int(i in b["present"] and j in b["present"]),
        )
        if av != bv:
            rows.append(
                {"patient_id": patient, "pair_delta": bv - av, "before": a, "after": b}
            )
    return rows


def run_experiment():
    """Run correctness assertions, then all 28 comparisons and 12 controlled ones."""
    processors, histories = make_processors(), synthetic_histories()
    matrices, traces = {}, {}
    for name, processor in processors.items():
        matrices[name], traces[name] = count_histories(processor, histories)
    assert_fixture(processors, matrices, traces)
    comparisons = {}
    for before, after in itertools.combinations(processors, 2):
        comparisons[before + " -> " + after] = compare_matrices(
            matrices[before], matrices[after]
        )
    controlled = []
    for factor, before, after in one_factor_contrasts():
        result = comparisons[before + " -> " + after]
        controlled.append(
            {"factor": factor, "before": before, "after": after, **result["summary"]}
        )
    from pyhealth.processors import AncestorRollupProcessor

    source = Path(inspect.getfile(AncestorRollupProcessor)).read_bytes()
    payload = json.dumps(histories, sort_keys=True).encode()
    return {
        "schema": "keep-expanded-synthetic-v1",
        "synthetic_only": True,
        "correctness_assertions_passed": True,
        "processor_source_sha256": hashlib.sha256(source).hexdigest(),
        "helper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "histories_sha256": hashlib.sha256(payload).hexdigest(),
        "settings": {
            "root": "R",
            "max_depth": 2,
            "include_root": False,
            "presence_threshold": 2,
            "x_max": 50,
            "alpha": 0.75,
            "axes": list(AXES),
            "supplied_root_depths": dict(DEPTHS),
            "direct_edges": list(DIRECT),
            "rescue_edges": list(RESCUE),
            "closure_only_edges": list(CLOSURE_ONLY),
        },
        "matrices": matrices,
        "traces": traces,
        "comparisons": comparisons,
        "controlled_contrasts": controlled,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for generated synthetic evidence",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must not exist")
    result = run_experiment()
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        "Synthetic correctness passed: 8 arms, 28 comparisons, 12 controlled contrasts."
    )


if __name__ == "__main__":
    main()
