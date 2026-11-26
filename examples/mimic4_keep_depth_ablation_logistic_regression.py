# Author: rishabshah (NetID: rishabs4)
# Description: KEEP roll-up depth ablation on MIMIC-IV: co-occurrence
#   statistics per depth, and next-admission mortality with PyHealth's
#   LogisticRegression on diagnoses rolled up to each depth.

"""Ablation: how deep should KEEP's SNOMED condition graph go?

KEEP (Elhussein et al., CHIL 2025, https://arxiv.org/abs/2510.05049) keeps
standard SNOMED Condition concepts at most **5** levels below "Disease"
(shortest path in ``CONCEPT_ANCESTOR``) and rolls each deeper code up to its
retained ancestors. This script changes only that depth (3, 4, 5, 6) and
measures what it does, holding everything else at the KEEP profile used in
``pyhealth.tasks.keep``: ``target_policy="first_retained"`` (each branch stops
at its nearest retained ancestor; the proposal's "all nearest retained
ancestors"), ``path_source="closure"``, ``include_root=False``, two distinct
diagnoses for a concept to count as present.

Experimental setup
------------------
Part 1, co-occurrence statistics. ``KeepConditionHistoryTask`` builds every
patient's diagnosis history once; for each depth an
``AncestorRollupProcessor`` rolls the histories up and
``task.build_cooccurrence`` counts the KEEP matrix. Per depth we report:

* size: retained ``vocabulary``, ``concepts_observed``, ``patients_with_conditions``
  (patients with a present concept), ``nonzero_pairs`` and ``density``
  (nonzero pairs / possible pairs among observed concepts);
* roll-up: ``targets_per_diagnosis``, ``multi_target_share`` (diagnoses reaching
  several retained ancestors), ``rolled_up_share`` (mapped concepts that had to
  be rolled up rather than kept as-is);
* frequency concentration: ``mean_prevalence``, ``prevalence_gini``, and for
  pair counts ``pair_gini``, ``pair_top1pct_share``, ``pair_hhi`` and
  ``pair_entropy`` (Shannon entropy / its maximum);
* association: ``mean_pmi``, ``positive_pmi_share``, ``mean_ppmi``,
  ``mean_npmi`` over observed pairs (PMI = log(N n_ij / n_i n_j)).

Because each depth has its own vocabulary, it is also compared with depth 5
on the concepts every depth retains: ``ppmi_spearman`` (rank agreement of PPMI
over their pairs), ``mean_profile_jsd`` (base-2 Jensen-Shannon divergence of
each concept's partner distribution) and ``median_prevalence_ratio``.

Part 2, downstream performance (the rubric's "existing PyHealth model").
``ConditionMortalityMIMIC4`` makes one sample per adult admission that has a
next admission; the label is that next admission's ``hospital_expire_flag``
(as in ``MortalityPredictionMIMIC4``) and the input is the patient's
diagnoses up to the current admission. The same samples are featurized as:

* ``icd``: raw ICD codes (no OMOP mapping);
* ``depth_3`` ... ``depth_6``: mapped concepts rolled up to that depth;
* ``uncut``: mapped concepts under "Disease" with no depth limit.

Samples that any arm leaves empty are dropped from every arm, so all arms
use identical patients, labels and patient-level 70/10/20 splits. Each arm
trains ``pyhealth.models.LogisticRegression`` with ``Trainer`` (Adam, model
selection on validation ROC-AUC) and reports test ROC-AUC and PR-AUC, mean
and standard deviation over seeds.

Expected effects (from the proposal): shallower graphs have fewer concepts
with more codes per concept, so prevalence and pair counts concentrate on
broad ancestors and PMI shrinks (a broad concept co-occurs with almost
everything); deeper graphs are larger and sparser. For prediction, shallow
depths may blur distinctions the label depends on, while deep or uncut
vocabularies may spread signal over rare tokens.

Running it
----------
Synthetic (default; no credentials, about a minute)::

    python -m examples.mimic4_keep_depth_ablation_logistic_regression

The synthetic cohort has a planted rule: next-admission death depends on one
ICD-9 code. In the synthetic vocabulary its concept and two other codes'
concepts share an ancestor at depth 5, so depths 3 to 5 merge them and only
depth 6, ``uncut`` and raw ``icd`` keep the risk code apart. Their AUROC
should be clearly higher. This checks the pipeline, not a clinical finding.

Real data (run locally; prints and saves aggregate counts only)::

    python -m examples.mimic4_keep_depth_ablation_logistic_regression \\
        --mimic ~/data/mimiciv/3.1 --athena ~/data/athena \\
        --cache ~/.cache/pyhealth/keep_depth --out keep_depth_ablation.json

Add ``--dev`` for PyHealth's 1,000-patient development subset first.

Findings
--------
Synthetic run (3 seeds): ``icd``, ``depth_6`` and ``uncut`` reach test AUROC
of about 0.73 while depths 3 to 5 stay near 0.48 (chance), as planted. In the
co-occurrence part, depth 6 splits a broad concept, so fewer patients reach
the two-diagnosis presence rule, pair counts spread out (lower top-1% share
and HHI) and mean PPMI rises. Results on MIMIC-IV 3.1 are recorded in the
pull request description.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from examples.keep_depth_ablation_utils import (
    DEPTHS,
    REFERENCE_DEPTH,
    ConceptFeaturizer,
    ConditionMortalityMIMIC4,
    RollupRecorder,
    featurize_arms,
    matrix_summary,
    shared_concept_comparison,
)
from pyhealth.datasets import (
    MIMIC4Dataset,
    create_sample_dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.medcode import OMOPVocabulary
from pyhealth.models import LogisticRegression
from pyhealth.tasks import KeepConditionHistoryTask, build_keep_omop_ancestor_rollup
from pyhealth.trainer import Trainer

#: Depth used for the "uncut" arm: deeper than any SNOMED Disease concept.
UNCUT_DEPTH = 1000

# --- synthetic data -----------------------------------------------------------

# (icd_version, icd_code) pairs that the synthetic Athena bundle in
# examples/omop_keep_cooccurrence_matrix.py maps: K1 -> depth-6 and depth-2
# concepts, I214 -> depth 6, I250 -> depth 6 (closure path only), 0010 ->
# depth 5, Z999 -> no mapping.
SYNTHETIC_CODES = [
    ("10", "K1"),
    ("10", "I214"),
    ("10", "I250"),
    ("10", "Z999"),
    ("9", "0010"),
]
RISK_CODE = ("9", "0010")  # the planted rule: this history raises mortality


def write_synthetic_mimic4(root: Path, patients: int = 240, seed: int = 7) -> None:
    """Writes a small MIMIC-IV-shaped cohort with a planted mortality rule."""
    rng = random.Random(seed)
    tables: Dict[str, List[tuple]] = {
        "hosp/patients.csv.gz": [],
        "hosp/admissions.csv.gz": [],
        "hosp/diagnoses_icd.csv.gz": [],
        "icu/icustays.csv.gz": [],
    }
    hadm = 1000
    for subject in range(1, patients + 1):
        tables["hosp/patients.csv.gz"].append(
            (subject, rng.choice("FM"), rng.randint(20, 90), 2150, "2014 - 2016", "")
        )
        history = set()
        for visit in range(rng.randint(2, 4)):
            hadm += 1
            admit = f"2150-{visit + 1:02d}-0{rng.randint(1, 9)} 08:00:00"
            discharge = admit.replace("08:00:00", "18:00:00")
            died = visit > 0 and rng.random() < (0.55 if RISK_CODE in history else 0.08)
            tables["hosp/admissions.csv.gz"].append(
                (
                    subject,
                    hadm,
                    admit,
                    discharge,
                    discharge if died else "",
                    "URGENT",
                    "EMERGENCY ROOM",
                    "Medicare",
                    "ENGLISH",
                    "SINGLE",
                    "WHITE",
                    "DIED" if died else "HOME",
                    int(died),
                )
            )
            tables["icu/icustays.csv.gz"].append(
                (subject, hadm, hadm * 10, admit, discharge, "MICU", "MICU")
            )
            codes = rng.sample(SYNTHETIC_CODES, rng.randint(2, 3))
            for seq, (version, code) in enumerate(codes, start=1):
                tables["hosp/diagnoses_icd.csv.gz"].append(
                    (subject, hadm, seq, code, version)
                )
            history.update(codes)
            if died:
                break
    headers = {
        "hosp/patients.csv.gz": (
            "subject_id",
            "gender",
            "anchor_age",
            "anchor_year",
            "anchor_year_group",
            "dod",
        ),
        "hosp/admissions.csv.gz": (
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "admission_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "discharge_location",
            "hospital_expire_flag",
        ),
        "hosp/diagnoses_icd.csv.gz": (
            "subject_id",
            "hadm_id",
            "seq_num",
            "icd_code",
            "icd_version",
        ),
        "icu/icustays.csv.gz": (
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "first_careunit",
            "last_careunit",
        ),
    }
    for name, rows in tables.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers[name])
            writer.writerows(rows)


# --- ablation -----------------------------------------------------------------


def build_processors(vocabulary: OMOPVocabulary, depths, args) -> Dict[int, Any]:
    """One KEEP roll-up processor per depth, identical apart from max_depth."""
    processors = {}
    for depth in depths:
        start = time.time()
        processors[depth] = build_keep_omop_ancestor_rollup(
            vocabulary,
            target_policy=args.policy,
            include_root=False,
            max_depth=depth,
            path_source=args.path_source,
            on_unmapped="empty",
        )
        label = "uncut" if depth == UNCUT_DEPTH else f"depth {depth}"
        print(f"  built roll-up for {label} in {time.time() - start:.0f}s")
    return processors


def run_matrix_ablation(dataset, vocabulary, release, processors, depths, args):
    """Part 1: KEEP co-occurrence statistics per depth."""
    task = KeepConditionHistoryTask(
        mimic_version=4, vocabulary=vocabulary, vocabulary_release=release
    )
    histories = dataset.set_task(task, num_workers=args.workers)
    results, summaries = {}, {}
    for depth in depths:
        recorder = RollupRecorder(processors[depth])
        result = task.build_cooccurrence(
            histories, rollup=recorder, concept_ids=sorted(recorder.retained_codes)
        )
        results[depth] = result
        summaries[depth] = matrix_summary(result, recorder)
    shared = sorted(set.intersection(*(set(r.concept_ids) for r in results.values())))
    reference = results[args.reference]
    for depth in depths:
        summaries[depth].update(
            {
                f"vs_depth{args.reference}_{key}": value
                for key, value in shared_concept_comparison(
                    results[depth], reference, shared
                ).items()
            }
        )
    return summaries


def _scores(dataset, seed, args) -> Dict[str, float]:
    """Trains and tests one LogisticRegression on one patient-level split."""
    train, val, test = split_by_patient(dataset, [0.7, 0.1, 0.2], seed=seed)
    torch.manual_seed(seed)
    model = LogisticRegression(dataset=dataset, embedding_dim=args.embedding_dim)
    trainer = Trainer(
        model=model, metrics=["roc_auc", "pr_auc"], device="cpu", enable_logging=False
    )
    trainer.train(
        train_dataloader=get_dataloader(
            train, batch_size=args.batch_size, shuffle=True
        ),
        val_dataloader=get_dataloader(val, batch_size=args.batch_size),
        epochs=args.epochs,
        monitor="roc_auc",
        optimizer_params={"lr": args.lr},
    )
    return trainer.evaluate(get_dataloader(test, batch_size=args.batch_size))


def run_model_ablation(dataset, vocabulary, processors, depths, args):
    """Part 2: next-admission mortality with LogisticRegression per depth."""
    samples = [
        {
            "patient_id": str(s["patient_id"]),
            "visit_id": str(s["visit_id"]),
            "conditions": list(s["conditions"]),
            "mortality": int(float(s["mortality"])),
        }
        for s in dataset.set_task(ConditionMortalityMIMIC4(), num_workers=args.workers)
    ]
    mapper = vocabulary.icd_mapper()
    featurizers = {"icd": ConceptFeaturizer()}
    for depth in depths:
        featurizers[f"depth_{depth}"] = ConceptFeaturizer(mapper, processors[depth])
    featurizers["uncut"] = ConceptFeaturizer(mapper, processors[UNCUT_DEPTH])
    arms = featurize_arms(samples, featurizers)
    report = {}
    for name, arm_samples in arms.items():
        # Built once per arm; seeds change only the split and initialization.
        dataset = create_sample_dataset(
            arm_samples,
            {"conditions": "sequence"},
            {"mortality": "binary"},
            dataset_name="mimic4_keep_depth",
            task_name=f"condition_mortality_{name}",
        )
        runs = [_scores(dataset, seed, args) for seed in range(args.seeds)]
        tokens = [len(s["conditions"]) for s in arm_samples]
        report[name] = {
            "samples": len(arm_samples),
            "positive_rate": sum(s["mortality"] for s in arm_samples)
            / len(arm_samples),
            "vocabulary": len({t for s in arm_samples for t in s["conditions"]}),
            "tokens_per_sample": statistics.fmean(tokens),
        }
        for metric in ("roc_auc", "pr_auc"):
            values = [run[metric] for run in runs]
            report[name][metric] = statistics.fmean(values)
            report[name][f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
    return report


# --- reporting ----------------------------------------------------------------


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 1000 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def print_table(title: str, columns: Dict[str, Dict[str, Any]]) -> None:
    names = list(columns)
    keys = list(next(iter(columns.values())))
    width = max(len(k) for k in keys) + 2
    print(f"\n{title}")
    print(" " * width + "".join(f"{n:>12}" for n in names))
    for key in keys:
        print(
            f"{key:<{width}}" + "".join(f"{_fmt(columns[n][key]):>12}" for n in names)
        )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mimic", help="MIMIC-IV root (contains hosp/ and icu/)")
    parser.add_argument("--athena", help="Unzipped Athena vocabulary folder")
    parser.add_argument(
        "--release",
        default=None,
        help="Athena release label (default: read VOCABULARY.csv)",
    )
    parser.add_argument("--cache", default=None, help="PyHealth cache directory")
    parser.add_argument("--out", default=None, help="Write all results to this JSON")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use PyHealth's 1,000-patient development subset",
    )
    parser.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    parser.add_argument("--reference", type=int, default=REFERENCE_DEPTH)
    parser.add_argument(
        "--policy", default="first_retained", choices=["first_retained", "all_retained"]
    )
    parser.add_argument(
        "--path-source", default="closure", choices=["closure", "direct"]
    )
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if bool(args.mimic) != bool(args.athena):
        parser.error("pass both --mimic and --athena, or neither for synthetic data")
    if args.reference not in args.depths:
        parser.error("--reference must be one of --depths")
    return args


def _release(athena: Path) -> str:
    path = athena / "VOCABULARY.csv"
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split("\t")
            if fields and fields[0] == "None" and len(fields) > 3:
                return fields[3]
    return "unknown"


def main(argv=None) -> Dict[str, Any]:
    args = parse_args(argv)
    logging.getLogger("pyhealth.trainer").setLevel(logging.WARNING)
    with tempfile.TemporaryDirectory(prefix="keep_depth_") as scratch:
        scratch = Path(scratch)
        if args.mimic:
            mimic, athena = (
                Path(args.mimic).expanduser(),
                Path(args.athena).expanduser(),
            )
            release = args.release or _release(athena)
        else:
            from examples.omop_keep_cooccurrence_matrix import write_athena_bundle

            mimic, athena, release = scratch / "mimic", scratch / "athena", "synthetic"
            write_synthetic_mimic4(mimic)
            write_athena_bundle(athena)
            print("Synthetic data: results check the pipeline, not clinical effects.")
        cache = Path(args.cache).expanduser() if args.cache else scratch / "cache"

        print(f"Vocabulary release: {release}")
        vocabulary = OMOPVocabulary(athena)
        dataset = MIMIC4Dataset(
            ehr_root=str(mimic),
            ehr_tables=["diagnoses_icd"],
            cache_dir=str(cache),
            dev=args.dev,
        )
        depths = sorted(set(args.depths))
        print("Building roll-up processors")
        processors = build_processors(vocabulary, depths + [UNCUT_DEPTH], args)

        hidden = ("mimic", "athena", "cache", "out")  # local paths stay out
        config = {k: v for k, v in vars(args).items() if k not in hidden}
        config.update(release=release, synthetic=not args.mimic)
        output: Dict[str, Any] = {"config": config}
        if not args.skip_matrix:
            output["cooccurrence"] = run_matrix_ablation(
                dataset, vocabulary, release, processors, depths, args
            )
            print_table(
                "KEEP co-occurrence by depth",
                {f"depth {d}": s for d, s in output["cooccurrence"].items()},
            )
        if not args.skip_model:
            output["mortality"] = run_model_ablation(
                dataset, vocabulary, processors, depths, args
            )
            print_table(
                "Next-admission mortality, LogisticRegression (test set)",
                output["mortality"],
            )
        if args.out:
            Path(args.out).write_text(json.dumps(output, indent=2, default=str))
            print(f"\nSaved {args.out}")
        return output


if __name__ == "__main__":
    main()
