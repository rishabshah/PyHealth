"""DeepHit ablations on synthetic competing-risk data.

Contributor: Bin Yang (biny4).
Paper: DeepHit: A Deep Learning Approach to Survival Analysis with Competing
Risks. https://ojs.aaai.org/index.php/AAAI/article/view/11842
Reproduction context: Neural Fine-Gray: Monotonic neural networks for
competing risks. https://proceedings.mlr.press/v209/jeanselme23a.html

Runs two ablations on self-contained synthetic data, with no download and no
real dataset, so the whole script finishes in well under a minute on CPU.

Study 1 - hyperparameter variation. Sweeps time-grid resolution, encoder
width, dropout, learning rate and ranking weight. Grid resolution is the
interesting axis: DeepHit must discretize time, and the bin count trades
temporal precision against the number of observed events supporting each cell.

Study 2 - censoring regime. Varies the independent censoring rate while
holding the event-time generating process fixed, then rescores discrimination
and calibration. This ablation is absent from the Neural Fine-Gray paper,
which varies quadrature points and architecture but not the censoring rate.
It asks whether a discrete-time grid degrades faster than its parameter count
alone would suggest once fewer observed events are available per cell.

Experimental setup. Two competing causes are drawn from covariate-dependent
Weibull hazards over eight standard-normal features, the earlier draw wins,
and independent uniform censoring is applied afterwards. Features and times
pass through CompetingRisksPreprocessor, fitted on the training split only.
Models are scored with the IPCW metrics in pyhealth.metrics.competing_risks at
the median uncensored event time, with the censoring distribution also fitted
on training labels only.

Findings on the seeded data below.

Study 1. Discrimination is insensitive to most hyperparameters, spanning
0.673 to 0.689 C-index across width, dropout and ranking weight. Learning
rate is the one that bites: 1e-2 drops it to 0.625. The ranking term is close
to neutral, moving C-index from 0.686 at weight 0 to 0.689 at weight 1.

The 5-bin row is the interesting one. It scores exactly 0.500, and the bins
column shows why: the evaluation horizon falls at 0.138 in scaled units while
a 5-bin grid places its first edge at 0.200, so no bin has completed, every
cumulative incidence is zero, and concordance is 0.5 by construction rather
than by any property of the fit. A discrete-time model cannot express risk
before its first bin edge. This is a structural limit of the discretization
that Neural Fine-Gray's continuous-time formulation removes, and it is
invisible unless the evaluation horizon is reported alongside the grid.

Study 2. The two censoring tables disagree, and that disagreement is the
result. Recomputing the horizon per run makes the model look fragile: C-index
falls from 0.681 at 10% censoring to 0.602 at 70%, and Brier appears to
improve from 0.134 to 0.055. Holding the horizon fixed removes the effect
entirely, giving 0.677 to 0.718 C-index, flat to slightly rising, with Brier
between 0.132 and 0.117.

The drifting horizon, not the model, produced the apparent collapse. Heavier
censoring truncates follow-up, so a horizon defined as a quantile of
uncensored training times moves earlier, fewer subjects have failed by it,
and both metrics shift for reasons that have nothing to do with fit quality.
Read across the two tables rather than down either one: on this generating
process DeepHit's discrimination does not degrade up to 70% censoring once
the evaluation point is held still. Censoring ablations that do not fix the
horizon risk reporting estimator drift as model degradation.

Usage:
    python examples/synthetic_competing_risk_deephit.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pyhealth.metrics.competing_risks import (
    CensoringDistribution,
    competing_risk_brier,
    competing_risk_concordance,
)
from pyhealth.models import DeepHit
from pyhealth.tasks.competing_risks import CompetingRisksPreprocessor

NUM_RISKS = 2
NUM_FEATURES = 8
NUM_PATIENTS = 1500
EPOCHS = 40
BATCH_SIZE = 256


def make_samples(
    censoring_rate: float,
    seed: int = 0,
    num_patients: int = NUM_PATIENTS,
) -> list[dict[str, Any]]:
    """Simulate competing-risk records at a target independent censoring rate.

    Two covariate-dependent Weibull causes race, the earlier draw is observed,
    and uniform censoring is applied afterwards to reach the requested rate.
    The event-time process does not depend on censoring_rate, so sweeps of that
    argument isolate the censoring regime.

    Args:
        censoring_rate: Target fraction of censored records in [0, 1).
        seed: Seed for the generating process.
        num_patients: Number of simulated records.

    Returns:
        Sample dicts with features, time and event, where event is 0 for
        censored records and 1..NUM_RISKS for the observed first cause.
    """
    generator = np.random.default_rng(seed)
    features = generator.standard_normal((num_patients, NUM_FEATURES))
    first = np.exp(0.8 * features[:, 0] - 0.5 * features[:, 1] ** 2)
    second = np.exp(0.6 * features[:, 2] + 0.4 * features[:, 3] * features[:, 4])
    draws = generator.uniform(size=(num_patients, NUM_RISKS))
    times = np.stack(
        [
            (-np.log(draws[:, 0]) / first) ** 0.9,
            (-np.log(draws[:, 1]) / second) ** 1.1,
        ],
        axis=1,
    )
    event_time = times.min(axis=1)
    cause = times.argmin(axis=1) + 1
    if censoring_rate > 0:
        cutoff = np.quantile(event_time, 1 - censoring_rate)
        censor_time = generator.uniform(0, 2 * cutoff, size=num_patients)
        censored = censor_time < event_time
        event_time = np.where(censored, censor_time, event_time)
        cause = np.where(censored, 0, cause)
    return [
        {"features": row.tolist(), "time": float(t), "event": int(c)}
        for row, t, c in zip(features, event_time, cause)
    ]


def split(
    samples: list[dict[str, Any]],
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shuffle and cut records into a 70/30 train and test split.

    Args:
        samples: Records to divide.
        seed: Seed for the permutation.

    Returns:
        Training and test sample lists.
    """
    order = np.random.default_rng(seed).permutation(len(samples))
    pivot = int(0.7 * len(samples))
    shuffled = [samples[index] for index in order]
    return shuffled[:pivot], shuffled[pivot:]


def tensors(
    samples: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack preprocessed samples into feature, time and event tensors.

    Args:
        samples: Transformed records sharing one feature width.

    Returns:
        Features [n, width], times [n] and integer events [n], all float64
        except events, which are long.
    """
    features = torch.tensor(
        [sample["features"] for sample in samples], dtype=torch.float64
    )
    times = torch.tensor([sample["time"] for sample in samples], dtype=torch.float64)
    events = torch.tensor([sample["event"] for sample in samples], dtype=torch.long)
    return features, times, events


def train(model: DeepHit, data: tuple[torch.Tensor, ...], rate: float) -> DeepHit:
    """Fit a model with Adam over shuffled minibatches.

    PyHealth's Trainer also drives this model directly with monitor="loss" and
    monitor_criterion="min"; a plain loop is used here to keep the example
    self-contained and fast.

    Args:
        model: Model to optimize in place.
        data: Training features, times and events.
        rate: Adam learning rate.

    Returns:
        The fitted model in eval mode.
    """
    features, times, events = data
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(len(features))
        for start in range(0, len(features), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = model(features[batch], times[batch], events[batch])["loss"]
            loss.backward()
            optimizer.step()
    return model.eval()


def score(
    model: DeepHit,
    train_data: tuple[torch.Tensor, ...],
    test_data: tuple[torch.Tensor, ...],
    risk: int = 1,
    horizon: float | None = None,
) -> tuple[float, float, int]:
    """Score IPCW discrimination and calibration at one horizon.

    Args:
        model: Fitted model.
        train_data: Training tensors, used to fit the censoring distribution.
        test_data: Held-out tensors.
        risk: Target cause numbered from 1.
        horizon: Scaled horizon to score at. None takes the median uncensored
            training event time, which moves earlier as censoring rises; pass
            an explicit value when comparing across censoring regimes.

    Returns:
        Concordance, Brier score, and the number of time bins completed at the
        horizon. A completed count of zero means the grid cannot resolve the
        horizon at all: every cumulative incidence is zero and concordance
        degenerates to 0.5. The concordance helper also reports its
        comparable-pair count, which is dropped here.
    """
    train_times, train_events = train_data[1].numpy(), train_data[2].numpy()
    features, times, events = test_data
    if horizon is None:
        observed = train_times[train_events > 0]
        horizon = float(np.quantile(observed, 0.5))
    with torch.no_grad():
        predictions = model.predict_cif(features, [horizon])[:, 0, risk - 1].numpy()
    censoring = CensoringDistribution(train_times, train_events)
    arguments = (times.numpy(), events.numpy(), predictions, horizon, risk, censoring)
    concordance, _pairs = competing_risk_concordance(*arguments)
    completed = int((model.bin_edges <= horizon).sum())
    return concordance, competing_risk_brier(*arguments), completed


def run(censoring_rate: float, seed: int = 0, **overrides: Any) -> tuple[float, float,int]:
    """Build data, fit one configuration and score it.

    Args:
        censoring_rate: Target censoring fraction for the simulation.
        seed: Seed shared by the simulation, split and model initialization.
        **overrides: Model keyword arguments, an optional learning_rate, and an
            optional raw_horizon in untransformed time units. Supplying
            raw_horizon holds the evaluation point fixed across censoring
            regimes, which the preprocessor's per-split time scaling would
            otherwise shift.

    Returns:
        Concordance, Brier score, and completed bins for this configuration.
    """
    torch.manual_seed(seed)
    learning_rate = overrides.pop("learning_rate", 1e-3)
    raw_horizon = overrides.pop("raw_horizon", None)
    raw_train, raw_test = split(make_samples(censoring_rate, seed), seed)
    processor = CompetingRisksPreprocessor().fit(raw_train)
    train_data = tensors(processor.transform(raw_train))
    test_data = tensors(processor.transform(raw_test))
    horizon = (
        None
        if raw_horizon is None
        else float(raw_horizon) / processor.state["time_scale"]
    )
    settings: dict[str, Any] = {
        "num_bins": 20,
        "hidden_dims": (32,),
        "cause_dims": (32,),
        "dropout": 0.0,
        "ranking_weight": 0.1,
    }
    settings.update(overrides)
    model = DeepHit(None, NUM_FEATURES, num_risks=NUM_RISKS, **settings)
    fitted = train(model, train_data, learning_rate)
    return score(fitted, train_data, test_data, horizon=horizon)


def table(title: str, rows: list[tuple[str, float, float, int]]) -> None:
    """Print one ablation block as an aligned table.

    The bins column is the number of time bins completed at the evaluation
    horizon. Zero means the grid cannot resolve that horizon, so every
    cumulative incidence is zero and the C-index is 0.5 by construction
    rather than by any property of the fit.

    Args:
        title: Heading for the block.
        rows: Label, concordance, Brier score and completed bins per run.
    """
    width = max(len(label) for label, _, _, _ in rows)
    print(f"\n{title}")
    header = f"{'configuration'.ljust(width)}  {'C-index':>8}  {'Brier':>8}"
    print(f"{header}  {'bins':>5}")
    print("-" * (width + 27))
    for label, concordance, brier, completed in rows:
        note = "  (unresolved)" if completed == 0 else ""
        print(
            f"{label.ljust(width)}  {concordance:8.4f}  {brier:8.4f}"
            f"  {completed:5d}{note}"
        )


def main() -> None:
    """Run both ablation studies and print their tables."""
    baseline = {"num_bins": 20, "hidden_dims": (32,), "learning_rate": 1e-3}
    variations: list[tuple[str, dict[str, Any]]] = [
        ("baseline (20 bins, 32 hidden)", {}),
        ("num_bins = 5", {"num_bins": 5}),
        ("num_bins = 50", {"num_bins": 50}),
        ("hidden_dims = (64, 64)", {"hidden_dims": (64, 64), "cause_dims": (64,)}),
        ("dropout = 0.25", {"dropout": 0.25}),
        ("learning_rate = 1e-2", {"learning_rate": 1e-2}),
        ("ranking_weight = 0.0", {"ranking_weight": 0.0}),
        ("ranking_weight = 1.0", {"ranking_weight": 1.0}),
    ]
    rows = []
    for label, override in variations:
        settings = {**baseline, **override} if override else dict(baseline)
        rows.append((label, *run(0.3, **settings)))
    table("Study 1 - hyperparameter ablation (30% censoring)", rows)

    # Pin the horizon so every censoring regime is scored at the same point in
    # time rather than at a drifting quantile. The feasible window is narrow:
    # heavy censoring truncates follow-up, so the horizon must stay inside
    # observed training follow-up at 70% censoring, while light censoring
    # stretches the time scale, so the horizon must still complete at least
    # one bin at 10%. The 40th percentile of uncensored times satisfies both.
    uncensored = make_samples(0.0, seed=0)
    raw_horizon = float(np.quantile([sample["time"] for sample in uncensored], 0.40))
    rates = (0.1, 0.3, 0.5, 0.7)
    drifting = [
        (f"censoring = {rate:.0%}", *run(rate, **baseline)) for rate in rates
    ]
    table("Study 2a - censoring ablation, horizon recomputed per run", drifting)

    fixed = [
        (f"censoring = {rate:.0%}", *run(rate, raw_horizon=raw_horizon, **baseline))
        for rate in rates
    ]
    table("Study 2b - censoring ablation, horizon held fixed", fixed)
    print(
        "\nStudy 2a recomputes the horizon from each run's uncensored training\n"
        "times, so the evaluation point drifts earlier as censoring rises and\n"
        "the two columns partly measure that drift. Study 2b scores every\n"
        "regime at one fixed time. Comparing them separates genuine model\n"
        "degradation from movement in the estimator."
    )


if __name__ == "__main__":
    main()