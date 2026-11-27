# Author: rishabshah (NetID: rishabs4)
# Description: Next-admission mortality task and roll-up featurizer for the
#   KEEP depth ablation's downstream model in
#   examples/mimic4_keep_depth_ablation_logistic_regression.py.

"""Downstream-model helpers for the KEEP roll-up depth ablation.

The model part of the ablation never uses the co-occurrence matrix: it feeds
diagnosis tokens, rolled up to each depth, straight into a PyHealth model.
These helpers build those samples; the matrix measures live separately in
``examples/keep_depth_ablation_utils.py``.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from pyhealth.datasets import PatientSplit, create_sample_dataset
from pyhealth.tasks import BaseTask

#: Patient-level train / validation / test fractions.
SPLIT_RATIOS = (0.7, 0.1, 0.2)
INPUT_SCHEMA = {"conditions": "sequence"}
OUTPUT_SCHEMA = {"mortality": "binary"}

# --- downstream prediction ----------------------------------------------------


class ConditionMortalityMIMIC4(BaseTask):
    """Next-admission mortality from a patient's diagnosis history (MIMIC-IV).

    Like PyHealth's ``MortalityPredictionMIMIC4`` (adults, one sample per
    admission that has a next admission, label = the next admission's
    ``hospital_expire_flag``), but the input is diagnoses only, accumulated
    over the current and all earlier admissions. Codes stay raw
    (``"<icd_version>:<icd_code>"``) so the ablation can roll the same
    samples up to each depth afterwards.

    Example:
        >>> task = ConditionMortalityMIMIC4()
        >>> task.task_name
        'keep_depth_condition_mortality'
    """

    task_name = "keep_depth_condition_mortality"
    input_schema = {"conditions": "raw"}
    output_schema = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        demographics = patient.get_events(event_type="patients")
        if demographics:
            try:
                if int(float(demographics[0].anchor_age)) < 18:
                    return []
            except (TypeError, ValueError, AttributeError):
                pass
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            return []
        by_admission: Dict[str, set] = {}
        for event in patient.get_events(event_type="diagnoses_icd"):
            code, version = event.icd_code, event.icd_version
            if code is None or version is None:
                continue
            code_text, version_text = str(code).strip(), str(version).strip()
            if code_text:
                by_admission.setdefault(str(event.hadm_id), set()).add(
                    f"{version_text}:{code_text}"
                )
        samples, history = [], set()
        for current, following in zip(admissions, admissions[1:]):
            history |= by_admission.get(str(current.hadm_id), set())
            flag = str(following.hospital_expire_flag).strip()
            if not history or flag not in ("0", "1"):
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": str(current.hadm_id),
                    "conditions": sorted(history),
                    "mortality": int(flag),
                }
            )
        return samples


class ConceptFeaturizer:
    """Turns raw ``"<version>:<code>"`` diagnoses into model tokens.

    Args:
        mapper: An ``ICDToSNOMEDMapper``, or None to keep the raw ICD codes.
        processor: An ``AncestorRollupProcessor`` to roll mapped concepts up
            to its depth, or None to use the mapped concepts as they are.

    Results are cached per code, so a cohort's repeated codes cost one
    lookup each.
    """

    def __init__(self, mapper: Any = None, processor: Any = None):
        if processor is not None and mapper is None:
            raise ValueError("rolling up needs a mapper")
        self.mapper = mapper
        self.processor = processor
        self._cache: Dict[str, Tuple[str, ...]] = {}

    def _tokens(self, raw: str) -> Tuple[str, ...]:
        if raw not in self._cache:
            if self.mapper is None:
                tokens: Tuple[str, ...] = (raw,)
            else:
                version, _, code = raw.partition(":")
                try:
                    concepts = [str(c) for c in self.mapper.map(code, version)]
                except ValueError:  # unsupported ICD version
                    concepts = []
                if self.processor is not None and concepts:
                    results = self.processor.rollup_codes(concepts)  # cached here
                    concepts = [t for result in results for t in result.targets]
                tokens = tuple(sorted(set(concepts)))
            self._cache[raw] = tokens
        return self._cache[raw]

    def __call__(self, raw_codes: Iterable[str]) -> List[str]:
        return sorted({token for raw in raw_codes for token in self._tokens(raw)})


def featurize_arms(
    samples: Sequence[Mapping[str, Any]], featurizers: Mapping[str, ConceptFeaturizer]
) -> Dict[str, List[Dict[str, Any]]]:
    """Builds one sample list per arm over exactly the same samples.

    A sample is dropped from every arm if any arm leaves it with no tokens,
    so all arms train and test on the same patients and labels.
    """
    arms: Dict[str, List[Dict[str, Any]]] = {name: [] for name in featurizers}
    for sample in samples:
        features = {name: f(sample["conditions"]) for name, f in featurizers.items()}
        if not all(features.values()):
            continue
        for name, tokens in features.items():
            arms[name].append({**sample, "conditions": tokens})
    return arms


def patient_folds(samples: Sequence[Mapping[str, Any]], seed: int) -> List[List[int]]:
    """Patient-level train / validation / test sample indices for one seed.

    Uses PyHealth's :class:`~pyhealth.datasets.PatientSplit` (patients sorted
    by ID, then shuffled with ``seed``), so every arm built by
    :func:`featurize_arms` (same samples, same order) gets the same folds.

    Args:
        samples: One arm's samples.
        seed: Split seed.

    Returns:
        Three sorted lists of sample indices.
    """
    patient_to_index: Dict[str, List[int]] = {}
    for index, sample in enumerate(samples):
        patient_to_index.setdefault(str(sample["patient_id"]), []).append(index)
    parts = PatientSplit(ratios=SPLIT_RATIOS, seed=seed).split_indices(patient_to_index)
    return [sorted(int(i) for i in part) for part in parts]


def fit_on_train(
    samples: Sequence[Mapping[str, Any]],
    folds: Sequence[Sequence[int]],
    task_name: str,
):
    """Builds train / validation / test datasets with train-fitted processors.

    The code vocabulary and label processor are fitted on the training
    patients only, then reused unchanged for validation and test, so a code
    seen only in held-out patients maps to ``<unk>``.

    Args:
        samples: One arm's samples.
        folds: Output of :func:`patient_folds`.
        task_name: Name recorded on the datasets.

    Returns:
        ``(train, validation, test)`` sample datasets.
    """
    train_part, *held_out = [[dict(samples[i]) for i in fold] for fold in folds]
    train = create_sample_dataset(
        train_part,
        INPUT_SCHEMA,
        OUTPUT_SCHEMA,
        dataset_name="mimic4_keep_depth",
        task_name=task_name,
    )
    fitted = {
        "input_processors": train.input_processors,
        "output_processors": train.output_processors,
    }
    return (train,) + tuple(
        create_sample_dataset(
            part,
            INPUT_SCHEMA,
            OUTPUT_SCHEMA,
            dataset_name="mimic4_keep_depth",
            task_name=task_name,
            **fitted,
        )
        for part in held_out
    )


PART_NAMES = ("train", "validation", "test")


def fold_counts(
    samples: Sequence[Mapping[str, Any]], folds: Sequence[Sequence[int]]
) -> Dict[str, Dict[str, int]]:
    """Sample and positive counts of each part of one seed's folds."""
    return {
        name: {
            "samples": len(fold),
            "positives": sum(int(samples[i]["mortality"]) for i in fold),
        }
        for name, fold in zip(PART_NAMES, folds)
    }


def check_class_coverage(
    samples: Sequence[Mapping[str, Any]], folds: Sequence[Sequence[int]], seed: int
) -> Dict[str, Dict[str, int]]:
    """Fails clearly unless train, validation and test each hold both labels.

    ROC-AUC and PR-AUC are undefined on a one-class part (scikit-learn returns
    NaN or raises), and a one-class validation part gives no basis for picking
    the best epoch. Folds are shared by every arm, so this is checked once per
    seed before any arm trains.

    Returns:
        The :func:`fold_counts` of the seed, when every part is usable.

    Raises:
        ValueError: Naming the seed and each part's sample / positive counts.
    """
    counts = fold_counts(samples, folds)
    bad = [
        name
        for name, part in counts.items()
        if part["positives"] == 0 or part["positives"] == part["samples"]
    ]
    if bad:
        detail = ", ".join(
            f"{name} {part['positives']}/{part['samples']} positive"
            for name, part in counts.items()
        )
        raise ValueError(
            f"Seed {seed}: {', '.join(bad)} would have a single class ({detail}). "
            "Use a larger cohort, or different seeds or split ratios."
        )
    return counts


def summarize_scores(
    runs: Mapping[int, Mapping[str, float]], arm: str, metrics: Sequence[str]
) -> Dict[str, Any]:
    """Per-seed scores plus mean and standard deviation, all finite.

    Raises:
        ValueError: If any seed's score is not finite; seeds are never
            dropped silently.
    """
    summary: Dict[str, Any] = {}
    for metric in metrics:
        values = [float(runs[seed][metric]) for seed in sorted(runs)]
        bad = [
            seed
            for seed, value in zip(sorted(runs), values)
            if not math.isfinite(value)
        ]
        if bad:
            raise ValueError(f"Arm {arm!r}: {metric} is not finite for seed(s) {bad}.")
        summary[metric] = statistics.fmean(values)
        summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[f"{metric}_per_seed"] = values
    return summary
