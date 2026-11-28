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
  concepts every depth retains (:func:`aligned_comparison`).
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


def entropy_bits(values: Sequence[float]) -> Optional[float]:
    """Shannon entropy, in bits, of the distribution proportional to ``values``.

    Example:
        >>> entropy_bits([1, 1, 1, 1])
        2.0
    """
    data = np.asarray(values, dtype=float)
    data = data[data > 0]
    if data.size == 0:
        return None
    p = data / data.sum()
    return float(-np.sum(p * np.log2(p)))


def _normalized(entropy: Optional[float], support: int) -> Optional[float]:
    """Entropy divided by its maximum, log2(support); None if undefined."""
    if entropy is None or support < 2:
        return None
    return entropy / math.log2(support)


def _quantile(values: np.ndarray, q: float) -> Optional[float]:
    return float(np.quantile(values, q)) if values.size else None


def partner_entropies(
    size: int, i: np.ndarray, j: np.ndarray, counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Entropy (bits) of each concept's partner distribution, diagonal excluded.

    Returns:
        ``(raw, normalized)`` for concepts with at least two partners;
        ``normalized`` divides by log2(number of partners).
    """
    rows = np.concatenate([i, j])
    weights = np.concatenate([counts, counts]).astype(float)
    mass = np.bincount(rows, weights=weights, minlength=size)
    partners = np.bincount(rows, minlength=size)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = weights / mass[rows]
        terms = np.where(p > 0, -p * np.log2(p), 0.0)
    raw = np.bincount(rows, weights=terms, minlength=size)
    keep = partners >= 2
    return raw[keep], raw[keep] / np.log2(partners[keep])


def matrix_summary(
    result: Any, recorder: Optional[RollupRecorder] = None
) -> Dict[str, Any]:
    """Size, density, concentration and association statistics for one matrix.

    Pair-level measures are computed over the **nonzero pairs on this depth's
    own axes** (diagonal excluded). Their denominators are explicit:
    ``possible_pairs_observed`` (pairs among observed concepts, the density
    denominator) and ``possible_pairs_axes`` (pairs among all retained
    concepts, the universe #7 uses to normalize entropy).

    Args:
        result: A ``KeepCooccurrenceResult``.
        recorder: The :class:`RollupRecorder` used to build it, if any.

    Returns:
        A flat dictionary of aggregates; see the example script's docstring
        for what each one means.
    """
    prevalence, i, j, counts = matrix_arrays(result.matrix)
    patients = result.patients_processed
    axes = len(result.concept_ids)
    observed = int(np.count_nonzero(prevalence))
    possible_observed = observed * (observed - 1) // 2
    possible_axes = axes * (axes - 1) // 2
    bits = entropy_bits(counts)
    raw_partner, norm_partner = partner_entropies(axes, i, j, counts)
    summary: Dict[str, Any] = {
        "vocabulary": axes,
        "concepts_observed": observed,
        "patients": patients,
        "patients_with_conditions": result.patients_with_present_targets,
        "diagnoses": result.unique_diagnosis_events,
        "diagnoses_mapped": result.mapped_events,
        "nonzero_pairs": int(counts.size),
        "possible_pairs_observed": possible_observed,
        "possible_pairs_axes": possible_axes,
        "density": counts.size / possible_observed if possible_observed else None,
        "pair_mass": int(counts.sum()),
        "mean_prevalence": _mean(prevalence[prevalence > 0].astype(float)),
        "prevalence_gini": gini(prevalence[prevalence > 0]),
        "pair_gini": gini(counts),
        "pair_top1pct_share": top_share(counts),
        "pair_hhi": hhi(counts),
        "pair_entropy_bits": bits,
        "pair_entropy_norm_nonzero_pairs": _normalized(bits, int(counts.size)),
        "pair_entropy_norm_possible_pairs": _normalized(bits, possible_axes),
        "partner_entropy_bits_median": _quantile(raw_partner, 0.5),
        "partner_entropy_norm_median": _quantile(norm_partner, 0.5),
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


def common_axes(results: Iterable[Any]) -> list:
    """Concept ids retained at every depth: the aligned comparison's axes."""
    sets = [set(result.concept_ids) for result in results]
    return sorted(set.intersection(*sets)) if sets else []


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
    excluded_mass = counts[~keep].sum()
    total_mass = counts.sum()
    excluded_concepts = int(np.count_nonzero(prevalence[~keep_axis]))
    a, b, counts = a[keep], b[keep], counts[keep]
    low, high = np.minimum(a, b), np.maximum(a, b)
    coverage = {
        "excluded_concepts": excluded_concepts,
        "excluded_pair_mass_share": (
            float(excluded_mass / total_mass) if total_mass else None
        ),
    }
    return shared_prevalence, low, high, counts, coverage


def _row_similarity(rows, left, right, size, eligible):
    """Base-2 JSD and cosine between each concept's two partner profiles.

    Rows that are empty in either profile are undefined.
    """
    mass_l = np.bincount(rows, weights=left, minlength=size)
    mass_r = np.bincount(rows, weights=right, minlength=size)
    valid = eligible & (mass_l > 0) & (mass_r > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p, q = left / mass_l[rows], right / mass_r[rows]
        m = (p + q) / 2
        terms = np.where(p > 0, p * np.log2(p / m), 0.0) + np.where(
            q > 0, q * np.log2(q / m), 0.0
        )
        jsd = np.bincount(rows, weights=terms / 2, minlength=size)
        dot = np.bincount(rows, weights=left * right, minlength=size)
        norm_l = np.sqrt(np.bincount(rows, weights=left**2, minlength=size))
        norm_r = np.sqrt(np.bincount(rows, weights=right**2, minlength=size))
        cosine = dot / (norm_l * norm_r)
    jsd = np.clip(jsd[valid], 0.0, 1.0)
    cosine = np.clip(cosine[valid], 0.0, 1.0)
    return jsd, cosine, int(valid.sum()), int((eligible & ~valid).sum())


def aligned_comparison(
    result: Any, reference: Any, shared: Sequence[str]
) -> Dict[str, Any]:
    """Compares a matrix with the reference matrix on common retained axes.

    Both matrices are restricted to ``shared`` (concepts retained at every
    depth), aligned by concept ID, with the diagonal excluded from partner
    profiles. Both come from the same patients, so differences come from how
    codes were rolled up. Concepts and pairs outside ``shared`` are excluded,
    and how much of this matrix that leaves out is reported.

    Profile measures cover concepts observed in either matrix; a concept
    whose count (or PPMI) partner profile is empty in either matrix is
    counted as undefined rather than scored. JSD is the base-2 divergence in
    [0, 1] (not its square root); lower means closer to the reference.
    Higher cosine or entropy does not by itself mean clinically better.

    Args:
        result: The ``KeepCooccurrenceResult`` being compared.
        reference: The reference depth's ``KeepCooccurrenceResult``.
        shared: Concept ids retained at every depth (:func:`common_axes`).

    Returns:
        A flat dictionary of coverage, magnitude, profile, association and
        conditional-probability changes; keys are listed in the example
        script's docstring.
    """
    if result.patients_processed != reference.patients_processed:
        raise ValueError("aligned comparison needs the same patients")
    n = result.patients_processed
    prev_a, ia, ja, ca, coverage = _restricted(result, shared)
    prev_b, ib, jb, cb, _ = _restricted(reference, shared)
    size = len(shared)
    in_a, in_b = prev_a > 0, prev_b > 0
    both = in_a & in_b

    keys_a, keys_b = ia * size + ja, ib * size + jb
    keys = np.union1d(keys_a, keys_b)
    count_a, count_b = np.zeros(keys.size), np.zeros(keys.size)
    count_a[np.searchsorted(keys, keys_a)] = ca
    count_b[np.searchsorted(keys, keys_b)] = cb
    low, high = keys // size, keys % size
    matched = (count_a > 0) & (count_b > 0)

    def ppmi_npmi(counts, prev):
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(n * counts / (prev[low].astype(float) * prev[high]))
            npmi = pmi / -np.log(counts / n)
        defined = counts > 0
        ppmi = np.where(defined, np.maximum(pmi, 0.0), 0.0)
        npmi = np.where(defined & np.isfinite(npmi), npmi, np.nan)
        return ppmi, npmi

    ppmi_a, npmi_a = ppmi_npmi(count_a, prev_a)
    ppmi_b, npmi_b = ppmi_npmi(count_b, prev_b)

    out: Dict[str, Any] = {
        "axes_shared": size,
        "concepts_in_both": int(both.sum()),
        "concepts_only_here": int((in_a & ~in_b).sum()),
        "concepts_only_reference": int((in_b & ~in_a).sum()),
        **coverage,
        "pairs_matched": int(matched.sum()),
        "pairs_gained": int(((count_a > 0) & (count_b == 0)).sum()),
        "pairs_lost": int(((count_b > 0) & (count_a == 0)).sum()),
        "prevalence_abs_change_sum": int(np.abs(prev_a - prev_b).sum()),
        "prevalence_change_median": _quantile((prev_a - prev_b)[both], 0.5),
        "median_prevalence_ratio": (
            _quantile(prev_a[both] / prev_b[both], 0.5) if both.any() else None
        ),
        "count_abs_change_sum": int(np.abs(count_a - count_b).sum()),
        "count_changed_pairs": int((count_a != count_b).sum()),
    }

    rows = np.concatenate([low, high])
    eligible = in_a | in_b
    for name, left, right in (
        ("count", count_a, count_b),
        ("ppmi", ppmi_a, ppmi_b),
    ):
        jsd, cosine, valid, undefined = _row_similarity(
            rows,
            np.concatenate([left, left]),
            np.concatenate([right, right]),
            size,
            eligible,
        )
        out.update(
            {
                f"{name}_profile_rows_valid": valid,
                f"{name}_profile_rows_undefined": undefined,
                f"{name}_jsd_mean": _mean(jsd),
                f"{name}_jsd_median": _quantile(jsd, 0.5),
                f"{name}_jsd_p90": _quantile(jsd, 0.9),
                f"{name}_cosine_mean": _mean(cosine),
                f"{name}_cosine_median": _quantile(cosine, 0.5),
                f"{name}_cosine_p10": _quantile(cosine, 0.1),
            }
        )

    ppmi_change = (ppmi_a - ppmi_b)[matched]
    npmi_ok = matched & np.isfinite(npmi_a) & np.isfinite(npmi_b)
    npmi_change = (npmi_a - npmi_b)[npmi_ok]
    out.update(
        {
            "ppmi_spearman": None,
            "ppmi_change_median": _quantile(ppmi_change, 0.5),
            "ppmi_abs_change_mean": _mean(np.abs(ppmi_change)),
            "npmi_change_median": _quantile(npmi_change, 0.5),
            "npmi_abs_change_mean": _mean(np.abs(npmi_change)),
            "npmi_undefined_matched_pairs": int((matched & ~npmi_ok).sum()),
        }
    )
    if keys.size >= 3 and np.ptp(ppmi_a) > 0 and np.ptp(ppmi_b) > 0:
        out["ppmi_spearman"] = float(spearmanr(ppmi_a, ppmi_b).statistic)

    # P(B | A) = n_AB / n_A, both directions of every matched pair
    pa, pb = prev_a.astype(float), prev_b.astype(float)
    lo, hi = low[matched], high[matched]
    cond_a = np.concatenate([count_a[matched] / pa[lo], count_a[matched] / pa[hi]])
    cond_b = np.concatenate([count_b[matched] / pb[lo], count_b[matched] / pb[hi]])
    cond_change = np.abs(cond_a - cond_b)
    out.update(
        {
            "conditional_prob_abs_change_median": _quantile(cond_change, 0.5),
            "conditional_prob_abs_change_mean": _mean(cond_change),
        }
    )
    return out
