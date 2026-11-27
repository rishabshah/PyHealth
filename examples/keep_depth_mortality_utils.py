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

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from pyhealth.tasks import BaseTask

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
