"""Occurrence-rule study using the public KEEP counters on invented histories.

Mapping is fixed to the sibling ancestor-rollup example's dense/direct profile.
Only post-rollup presence changes: one/two/three distinct diagnoses or two
admissions. No patient files, vocabulary paths, or local configs are discovered.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import polars as pl

from examples.keep_ancestor_rollup_ablation_utils import trace_pair, validate_matrix
from examples.keep_ancestor_rollup_variants import (
    DEEP_TARGETS,
    compare_variants,
    make_rollups,
    pair_statistics,
    write_fixture,
)
from pyhealth.data import Patient
from pyhealth.medcode import OMOPVocabulary
from pyhealth.tasks import KeepConditionHistoryTask, build_keep_cooccurrence

RULES = {
    "rows_2": {"min_occurrences": 2, "occurrence_unit": "diagnosis"},
    "rows_1": {"min_occurrences": 1, "occurrence_unit": "diagnosis"},
    "rows_3": {"min_occurrences": 3, "occurrence_unit": "diagnosis"},
    "admissions_2": {"min_occurrences": 2, "occurrence_unit": "admission"},
}


def synthetic_histories():
    """Ten fixed patients; visit IDs explicitly represent admissions."""

    def patient(identity, rows):
        return {
            "patient_id": identity,
            "diagnoses": [
                {"diagnosis_id": dx, "codes": [code], "visit_id": admission}
                for dx, code, admission in rows
            ],
        }

    return [
        patient("p1", [("d1", "900", "a")]),
        patient("p2", [("d1", "900", "a"), ("d2", "900", "a")]),
        patient("p3", [("d1", "900", "a"), ("d2", "901", "b")]),
        patient("p4", [("d1", "900", "a"), ("d2", "900", "a"), ("d3", "900", "a")]),
        patient("p5", [("d1", "900", "a"), ("d2", "900", "b"), ("d3", "900", "c")]),
        patient("p6", [("copy", "900", "a"), ("copy", "900", "a")]),
        patient("p7", []),
        patient("p8", [("unknown", "999999", "a")]),
        patient("p9", [("d1", "105", "a"), ("d2", "205", "a")]),
        patient("p10", [("d1", "900", "a"), ("d2", "105", "b")]),
    ]


def as_patients(histories):
    """Bridge canonical OMOP concepts and admission identities to Patient rows."""
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
                raise ValueError("the OMOP fixture bridge needs one canonical concept")
            rows.append(
                (
                    history["patient_id"],
                    "condition_occurrence",
                    datetime(2020, 1, 1),
                    index,
                    event["diagnosis_id"],
                    event["codes"][0],
                    None,
                    event.get("visit_id"),
                )
            )
        yield Patient(
            patient_id=history["patient_id"],
            data_source=pl.DataFrame(rows, schema=schema, orient="row"),
        )


def as_mimic_histories(histories):
    """Supply the documented mapped-history contract without a MIMIC reader."""
    for history in histories:
        yield {
            "patient_id": history["patient_id"],
            "conditions": [
                {
                    "diagnosis_id": d["diagnosis_id"],
                    "concept_ids": d["codes"],
                    "visit_id": d.get("visit_id"),
                    "timestamp": datetime(2020, 1, 1),
                    "unmapped_reason": None,
                }
                for d in history["diagnoses"]
            ],
        }


def count_with_traces(rollup, histories, *, min_occurrences, occurrence_unit):
    """Independent example counter; retain both kinds of target support."""
    if type(min_occurrences) is not int or min_occurrences < 1:
        raise ValueError("min_occurrences must be a positive integer")
    if occurrence_unit not in ("diagnosis", "admission"):
        raise ValueError("unknown occurrence unit")
    pairs, traces, seen_patients = Counter(), [], set()
    for history in histories:
        patient = history["patient_id"]
        if not isinstance(patient, str) or not patient or patient in seen_patients:
            raise ValueError("patient IDs must be nonempty and unique")
        seen_patients.add(patient)
        unique = {}
        for event in history["diagnoses"]:
            identity = event["diagnosis_id"]
            if not isinstance(identity, str) or not identity:
                raise ValueError("diagnosis IDs must be nonempty strings")
            codes = tuple(sorted(set(event["codes"])))
            if not codes or any(not isinstance(c, str) or not c for c in codes):
                raise ValueError("source codes must be nonempty strings")
            visit = event.get("visit_id")
            visit = str(visit).strip() if visit is not None else None
            visit = None if visit in (None, "", "None") else visit
            record = (codes, visit)
            if identity in unique and unique[identity] != record:
                raise ValueError("conflicting diagnosis copies")
            unique[identity] = record
        row_support, admission_support, diagnoses = Counter(), defaultdict(set), []
        for identity, (codes, visit) in sorted(unique.items()):
            results = rollup.rollup_codes(codes)
            targets = sorted({t for r in results for t in r.targets})
            row_support.update(targets)
            if targets and visit is None and occurrence_unit == "admission":
                raise ValueError("mapped diagnosis needs an admission ID")
            if visit is not None:
                for target in targets:
                    admission_support[target].add(visit)
            diagnoses.append(
                {
                    "diagnosis_id": identity,
                    "codes": list(codes),
                    "visit_id": visit,
                    "targets": targets,
                    "reasons": [r.reason for r in results],
                }
            )
        admissions = {c: len(ids) for c, ids in sorted(admission_support.items())}
        support = row_support if occurrence_unit == "diagnosis" else admissions
        present = sorted(c for c, n in support.items() if n >= min_occurrences)
        pairs.update(itertools.product(present, repeat=2))
        traces.append(
            {
                "patient_id": patient,
                "diagnoses": diagnoses,
                "row_support": dict(sorted(row_support.items())),
                "admission_support": admissions,
                "present": present,
            }
        )
    matrix = {
        "patients": len(seen_patients),
        "concept_ids": sorted(rollup.retained_codes),
        "entries": [[i, j, v] for (i, j), v in sorted(pairs.items())],
    }
    validate_matrix(matrix)
    return matrix, traces


def expected_presence():
    """Literal oracle, independent of counters and mapper traversal."""
    d, empty = set(DEEP_TARGETS), set()
    return {
        "rows_1": [d, d, d, d, d, d, empty, empty, {"105", "205"}, d],
        "rows_2": [empty, d, d, d, d, empty, empty, empty, empty, {"105"}],
        "rows_3": [empty, empty, empty, d, d, empty, empty, empty, empty, empty],
        "admissions_2": [
            empty,
            empty,
            d,
            empty,
            d,
            empty,
            empty,
            empty,
            empty,
            {"105"},
        ],
    }


def matrix_payload(result):
    sparse = result.matrix.coalesce()
    return {
        "patients": result.patients_processed,
        "concept_ids": list(result.concept_ids),
        "entries": sorted(
            [result.concept_ids[i], result.concept_ids[j], int(v)]
            for (i, j), v in zip(
                sparse.indices().t().tolist(), sparse.values().tolist()
            )
        ),
    }


def run_study():
    """Compare four presence rules with two public routes and a literal oracle."""
    histories = synthetic_histories()
    matrices, traces, aggregates = {}, {}, {}
    with tempfile.TemporaryDirectory() as temporary:
        folder = write_fixture(Path(temporary))
        vocabulary = OMOPVocabulary(folder)
        rollup = make_rollups(vocabulary)["dense"]
        axes = sorted(rollup.retained_codes)
        mapper = vocabulary.icd_mapper()
        task = KeepConditionHistoryTask(
            mimic_version=4,
            vocabulary=vocabulary,
            vocabulary_release="invented-occurrence-v1",
        )
        for arm, rule in RULES.items():
            result = build_keep_cooccurrence(
                as_patients(histories),
                rollup=rollup,
                icd_mapper=mapper,
                concept_ids=axes,
                **rule,
            )
            matrix = matrix_payload(result)
            mimic = task.build_cooccurrence(
                as_mimic_histories(histories), rollup=rollup, concept_ids=axes, **rule
            )
            independent, trace = count_with_traces(rollup, histories, **rule)
            assert matrix == independent == matrix_payload(mimic), arm
            expected = expected_presence()[arm]
            assert [set(t["present"]) for t in trace] == expected, arm
            values = validate_matrix(matrix)
            for i, j in itertools.product(axes, repeat=2):
                assert values.get((i, j), 0) == sum(i in p and j in p for p in expected)
            assert result.patients_processed == 10
            matrices[arm], traces[arm] = matrix, trace
            aggregates[arm] = {
                **pair_statistics(matrix),
                **rule,
                "patients_with_present_targets": result.patients_with_present_targets,
                "unique_diagnoses": result.unique_diagnosis_events,
                "mapped_diagnoses": result.mapped_events,
                "unmapped_diagnoses": dict(result.unmapped_events),
            }
        comparisons = {
            a: compare_variants(matrices["rows_2"], matrices[a])
            for a in RULES
            if a != "rows_2"
        }
        for arm, comparison in comparisons.items():
            for pair in comparison["pairs"]:
                witnesses = trace_pair(
                    traces["rows_2"], traces[arm], pair["i"], pair["j"]
                )
                assert sum(w["pair_delta"] for w in witnesses) == pair["count_delta"]
        source_hashes = {
            name: hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
            for name, obj in (
                ("counter", build_keep_cooccurrence),
                ("study", run_study),
                ("mapper", type(rollup)),
                ("metrics", compare_variants),
            )
        }
        input_hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.glob("*.csv"))
        }
    return {
        "schema": "keep-occurrence-rule-v1",
        "synthetic_only": True,
        "settings": {
            "baseline": "rows_2",
            "rules": RULES,
            "max_depth": 5,
            "target_policy": "all_retained",
            "path_source": "direct",
            "include_root": False,
            "missing_admission": "error_for_mapped_diagnosis",
            "threshold_stage": "after_mapping_and_rollup",
        },
        "public_counter_agreement": True,
        "hand_oracle_passed": True,
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "histories_sha256": hashlib.sha256(
            json.dumps(histories, sort_keys=True).encode()
        ).hexdigest(),
        "matrices": matrices,
        "traces": traces,
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
        "Passed: four occurrence rules, public OMOP/MIMIC agreement, hand oracle and pair traces."
    )


if __name__ == "__main__":
    main()
