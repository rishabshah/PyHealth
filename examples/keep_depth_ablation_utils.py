# Author: rishabshah (NetID: rishabs4)
# Description: Co-occurrence matrix measures for the KEEP roll-up depth
#   ablation in examples/mimic4_keep_depth_ablation_logistic_regression.py.

"""Co-occurrence matrix measures for the KEEP roll-up depth ablation.

KEEP (Elhussein et al., CHIL 2025, https://arxiv.org/abs/2510.05049) keeps
SNOMED condition concepts at most five levels below "Disease" and rolls
deeper codes up to them. Changing that depth changes the vocabulary itself,
so the ablation reports two kinds of evidence:

* Co-occurrence statistics for each depth's KEEP matrix
  (:func:`matrix_summary`), plus a comparison with the reference depth on the
  concepts every depth shares (:func:`shared_concept_comparison`).
* Downstream performance, kept separately in
  ``examples/keep_depth_mortality_utils.py``: the same patients, labels and
  PyHealth model, with diagnosis codes rolled up to each depth.

This module holds only the reusable matrix measures.

Every function returns aggregates only, so results computed on credentialed
data contain no patient rows or concept-level counts.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

#: Depths compared by default; 5 is KEEP's setting and the reference.
DEPTHS = (3, 4, 5, 6)
REFERENCE_DEPTH = 5


# --- roll-up accounting -------------------------------------------------------


class RollupRecorder:
    """Wraps a roll-up processor, caches it per code, and counts its output.

    ``build_keep_cooccurrence`` calls ``rollup_codes`` once per mapped
    diagnosis, so the counts are per diagnosis: how many targets it reached
    and whether its concepts were kept as-is or rolled up. A code's roll-up
    never changes for a given processor, so each distinct code is searched
    once instead of once per occurrence (millions of rows in MIMIC-IV).

    Args:
        processor: A configured ``AncestorRollupProcessor``.
    """

    def __init__(self, processor: Any):
        self.processor = processor
        self._cache: Dict[str, Any] = {}
        self.events = 0
        self.targets = 0
        self.multi_target_events = 0
        self.reasons: Counter = Counter()

    @property
    def retained_codes(self) -> frozenset:
        return self.processor.retained_codes

    def rollup_codes(self, codes: Iterable[str]) -> list:
        results = []
        for code in codes:
            if code not in self._cache:
                self._cache[code] = self.processor.rollup_codes([code])[0]
            results.append(self._cache[code])
        union = {target for result in results for target in result.targets}
        self.events += 1
        self.targets += len(union)
        self.multi_target_events += len(union) > 1
        self.reasons.update(result.reason for result in results)
        return results

    def summary(self) -> Dict[str, Optional[float]]:
        """Targets per diagnosis and the share of concepts that rolled up."""
        mapped = self.reasons["retained"] + self.reasons["rolled_up"]
        return {
            "targets_per_diagnosis": (
                self.targets / self.events if self.events else None
            ),
            "multi_target_share": (
                self.multi_target_events / self.events if self.events else None
            ),
            "rolled_up_share": self.reasons["rolled_up"] / mapped if mapped else None,
        }


# --- distribution measures ----------------------------------------------------


def gini(values: Sequence[float]) -> Optional[float]:
    """Gini coefficient: 0 when all values are equal, near 1 when one dominates.

    Example:
        >>> gini([1, 1, 1, 1])
        0.0
        >>> round(gini([0, 0, 0, 4]), 3)
        0.75
    """
    data = np.sort(np.asarray(values, dtype=float))
    total = data.sum()
    if data.size == 0 or total == 0:
        return None
    ranks = np.arange(1, data.size + 1)
    return float(
        2 * np.sum(ranks * data) / (data.size * total) - (data.size + 1) / data.size
    )


def normalized_entropy(values: Sequence[float]) -> Optional[float]:
    """Shannon entropy of the normalized values divided by its maximum.

    1 means mass is spread evenly; values near 0 mean a few entries hold it.

    Example:
        >>> normalized_entropy([5, 5, 5, 5])
        1.0
    """
    data = np.asarray(values, dtype=float)
    data = data[data > 0]
    if data.size < 2:
        return None
    p = data / data.sum()
    return float(-np.sum(p * np.log(p)) / math.log(data.size))


def top_share(values: Sequence[float], fraction: float = 0.01) -> Optional[float]:
    """Share of total mass held by the largest ``fraction`` of entries.

    Example:
        >>> top_share([10] + [1] * 99)
        0.09174311926605505
    """
    data = np.sort(np.asarray(values, dtype=float))[::-1]
    total = data.sum()
    if data.size == 0 or total == 0:
        return None
    k = max(1, math.ceil(data.size * fraction))
    return float(data[:k].sum() / total)


def hhi(values: Sequence[float]) -> Optional[float]:
    """Herfindahl-Hirschman index: sum of squared shares (1 = one entry)."""
    data = np.asarray(values, dtype=float)
    total = data.sum()
    if data.size == 0 or total == 0:
        return None
    return float(np.sum((data / total) ** 2))


def _mean(values: np.ndarray) -> Optional[float]:
    return float(values.mean()) if values.size else None


# --- co-occurrence matrices ---------------------------------------------------


def matrix_arrays(matrix: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits a KEEP co-occurrence matrix into prevalence and pair arrays.

    Args:
        matrix: The sparse (or dense) ``torch`` tensor from
            ``KeepCooccurrenceResult.matrix``: prevalence on the diagonal,
            mirrored patient pair counts off it.

    Returns:
        ``(prevalence, i, j, counts)``: prevalence per axis position, and each
        unordered pair ``i < j`` with a nonzero count.
    """
    size = matrix.shape[0]
    if matrix.is_sparse:
        coalesced = matrix.coalesce()
        rows, cols = coalesced.indices().cpu().numpy()
        values = coalesced.values().cpu().numpy()
    else:
        dense = matrix.cpu().numpy()
        rows, cols = np.nonzero(dense)
        values = dense[rows, cols]
    prevalence = np.zeros(size, dtype=np.int64)
    diagonal = rows == cols
    prevalence[rows[diagonal]] = values[diagonal]
    upper = (rows < cols) & (values > 0)
    return prevalence, rows[upper], cols[upper], values[upper].astype(np.int64)


def pair_pmi(
    i: np.ndarray,
    j: np.ndarray,
    counts: np.ndarray,
    prevalence: np.ndarray,
    patients: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pointwise mutual information and its normalized form for each pair.

    PMI is ``log(N * n_ij / (n_i * n_j))`` with ``N`` patients; NPMI divides
    by ``-log(n_ij / N)`` and lies in [-1, 1]. Pairs whose count equals the
    cohort size get NaN NPMI.
    """
    joint = counts / patients
    pmi = np.log(patients * counts / (prevalence[i].astype(float) * prevalence[j]))
    with np.errstate(divide="ignore", invalid="ignore"):
        npmi = pmi / -np.log(joint)
    npmi[~np.isfinite(npmi)] = np.nan
    return pmi, npmi


def matrix_summary(
    result: Any, recorder: Optional[RollupRecorder] = None
) -> Dict[str, Any]:
    """Size, density, concentration and association statistics for one matrix.

    Args:
        result: A ``KeepCooccurrenceResult``.
        recorder: The :class:`RollupRecorder` used to build it, if any.

    Returns:
        A flat dictionary of aggregates; see the example script's docstring
        for what each one means.
    """
    prevalence, i, j, counts = matrix_arrays(result.matrix)
    patients = result.patients_processed
    observed = int(np.count_nonzero(prevalence))
    possible = observed * (observed - 1) // 2
    summary: Dict[str, Any] = {
        "vocabulary": len(result.concept_ids),
        "concepts_observed": observed,
        "patients": patients,
        "patients_with_conditions": result.patients_with_present_targets,
        "diagnoses": result.unique_diagnosis_events,
        "diagnoses_mapped": result.mapped_events,
        "nonzero_pairs": int(counts.size),
        "density": counts.size / possible if possible else None,
        "pair_mass": int(counts.sum()),
        "mean_prevalence": _mean(prevalence[prevalence > 0].astype(float)),
        "prevalence_gini": gini(prevalence[prevalence > 0]),
        "pair_gini": gini(counts),
        "pair_top1pct_share": top_share(counts),
        "pair_hhi": hhi(counts),
        "pair_entropy": normalized_entropy(counts),
    }
    if counts.size and patients:
        pmi, npmi = pair_pmi(i, j, counts, prevalence, patients)
        summary.update(
            mean_pmi=_mean(pmi),
            positive_pmi_share=float(np.mean(pmi > 0)),
            mean_ppmi=_mean(np.maximum(pmi, 0)),
            mean_npmi=_mean(npmi[np.isfinite(npmi)]),
        )
    else:
        summary.update(
            mean_pmi=None, positive_pmi_share=None, mean_ppmi=None, mean_npmi=None
        )
    if recorder is not None:
        summary.update(recorder.summary())
    return summary


def _restricted(result: Any, shared: Sequence[str]):
    """Prevalence and pairs of ``result`` on the ``shared`` concepts only."""
    index = {code: k for k, code in enumerate(shared)}
    remap = np.array([index.get(code, -1) for code in result.concept_ids])
    prevalence, i, j, counts = matrix_arrays(result.matrix)
    shared_prevalence = np.zeros(len(shared), dtype=np.int64)
    keep_axis = remap >= 0
    shared_prevalence[remap[keep_axis]] = prevalence[keep_axis]
    a, b = remap[i], remap[j]
    keep = (a >= 0) & (b >= 0)
    a, b, counts = a[keep], b[keep], counts[keep]
    low, high = np.minimum(a, b), np.maximum(a, b)
    return shared_prevalence, low, high, counts


def _jsd_by_row(rows_p, cols_p, p_vals, rows_q, cols_q, q_vals, size):
    """Base-2 Jensen-Shannon divergence between matching sparse rows."""
    width = size
    keys_p = rows_p * width + cols_p
    keys_q = rows_q * width + cols_q
    keys = np.union1d(keys_p, keys_q)
    p = np.zeros(keys.size)
    q = np.zeros(keys.size)
    p[np.searchsorted(keys, keys_p)] = p_vals
    q[np.searchsorted(keys, keys_q)] = q_vals
    rows = keys // width
    mass_p = np.bincount(rows, weights=p, minlength=size)
    mass_q = np.bincount(rows, weights=q, minlength=size)
    valid = (mass_p > 0) & (mass_q > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = p / mass_p[rows]
        q = q / mass_q[rows]
        m = (p + q) / 2
        terms = np.where(p > 0, p * np.log2(p / m), 0) + np.where(
            q > 0, q * np.log2(q / m), 0
        )
    divergence = np.bincount(rows, weights=terms / 2, minlength=size)
    return divergence[valid]


def shared_concept_comparison(
    result: Any, reference: Any, shared: Sequence[str]
) -> Dict[str, Any]:
    """Compares a matrix with the reference matrix on concepts both keep.

    Only concepts retained at every depth (and seen in both cohorts) are
    compared, so differences come from how deeper codes were rolled up, not
    from the vocabularies being different.

    Args:
        result: The ``KeepCooccurrenceResult`` being compared.
        reference: The reference depth's ``KeepCooccurrenceResult``.
        shared: Concept ids retained at every depth.

    Returns:
        ``shared_concepts`` (seen in both), ``ppmi_spearman`` (rank
        correlation of PPMI over pairs seen in either), ``mean_profile_jsd``
        (base-2 Jensen-Shannon divergence of each concept's partner
        distribution, averaged) and ``median_prevalence_ratio``.
    """
    prev_a, ia, ja, ca = _restricted(result, shared)
    prev_b, ib, jb, cb = _restricted(reference, shared)
    both = (prev_a > 0) & (prev_b > 0)
    out: Dict[str, Any] = {
        "shared_concepts": int(both.sum()),
        "median_prevalence_ratio": (
            float(np.median(prev_a[both] / prev_b[both])) if both.any() else None
        ),
        "ppmi_spearman": None,
        "mean_profile_jsd": None,
    }
    size = len(shared)
    in_a = both[ia] & both[ja]
    in_b = both[ib] & both[jb]
    ia, ja, ca = ia[in_a], ja[in_a], ca[in_a]
    ib, jb, cb = ib[in_b], jb[in_b], cb[in_b]

    def ppmi(i, j, c, prev, n):
        return np.maximum(np.log(n * c / (prev[i].astype(float) * prev[j])), 0)

    keys_a, keys_b = ia * size + ja, ib * size + jb
    keys = np.union1d(keys_a, keys_b)
    if keys.size >= 3:
        va, vb = np.zeros(keys.size), np.zeros(keys.size)
        va[np.searchsorted(keys, keys_a)] = ppmi(
            ia, ja, ca, prev_a, result.patients_processed
        )
        vb[np.searchsorted(keys, keys_b)] = ppmi(
            ib, jb, cb, prev_b, reference.patients_processed
        )
        if np.ptp(va) > 0 and np.ptp(vb) > 0:
            out["ppmi_spearman"] = float(spearmanr(va, vb).statistic)
    divergence = _jsd_by_row(
        np.concatenate([ia, ja]),
        np.concatenate([ja, ia]),
        np.concatenate([ca, ca]),
        np.concatenate([ib, jb]),
        np.concatenate([jb, ib]),
        np.concatenate([cb, cb]),
        size,
    )
    if divergence.size:
        out["mean_profile_jsd"] = float(divergence.mean())
    return out
