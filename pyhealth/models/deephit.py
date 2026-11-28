"""DeepHit: deep learning for survival analysis with competing risks.

Contributor: Bin Yang (NetID: biny4)

Paper:
    C. Lee, W. R. Zame, J. Yoon, M. van der Schaar.
    "DeepHit: A Deep Learning Approach to Survival Analysis with Competing
    Risks." AAAI 2018, pp. 2314-2321.
    https://doi.org/10.1609/aaai.v32i1.11842

DeepHit predicts a joint probability mass function over the pair (which
competing cause, which discrete time bin), rather than modelling a hazard or
assuming a parametric event-time distribution. A shared encoder feeds one
cause-specific stack per risk, and a single softmax runs across all
num_risks * num_bins cells at once, so the causes share one probability
budget by construction. Cumulative incidence is then just a running total
along the time axis of each cause's row, which makes it nondecreasing
without any constraint being imposed. Training combines a discrete
likelihood, under which observed events contribute their own cell and
censored records contribute event-free probability, with a cause-specific
ranking penalty over comparable pairs, because the likelihood alone is
indifferent to the orderings that concordance metrics measure.

Implementation notes:
    The original finite-grid joint softmax is available with tail_mass=False.
    By default, one additional event-free-at-horizon category permits valid
    likelihoods for patients censored at the final edge. This is an explicit
    adaptation of DeepHit, not an exact reproduction of its output grid.
    Cause-specific networks receive both the shared representation and input.
    Times use the preprocessing units of CompetingRisksPreprocessor. Set the
    horizon from training data only; administratively censor later outcomes
    before calling forward. No out-of-range times are silently clipped.
"""

import math
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from ..datasets import SampleDataset
from ..processors import LabelProcessor
from .base_model import BaseModel

MaskMode = Literal["original", "observed"]


def _mlp(
    dimensions: Sequence[int],
    dropout: float = 0.0,
    activate_last: bool = True,
) -> nn.Sequential:
    """Build a ReLU multilayer perceptron for shared or cause-specific stacks.

    Module activations are used rather than functional calls so that
    gradient-based interpretability methods can traverse the network.

    Args:
        dimensions: Input width followed by output widths for the linear layers.
        dropout: Dropout probability applied after each activated linear layer.
        activate_last: Append activation and dropout after the final layer.

    Returns:
        Sequential linear, activation, and optional dropout layers. An empty
        sequential when fewer than two dimensions are supplied.

    Examples:
        >>> tuple(_mlp([3, 2])(torch.zeros(4, 3)).shape)
        (4, 2)
    """
    modules: list[nn.Module] = []
    last = len(dimensions) - 2
    for position in range(last + 1):
        modules.append(nn.Linear(dimensions[position], dimensions[position + 1]))
        if position < last or activate_last:
            modules.append(nn.ReLU())
            if dropout:
                modules.append(nn.Dropout(dropout))
    return nn.Sequential(*modules)


class DeepHit(BaseModel):
    """Discrete-time competing risks over a shared cause-time softmax grid.

    A shared encoder and residual input feed one cause-specific stack per
    risk. A joint softmax normalizes the cause-time mass, with an optional
    additional category for survival beyond the modeled horizon. CIFs sum the
    event masses at completed bin edges.

    Events in (edge[k], edge[k+1]] are assigned to bin k; time-zero events
    are assigned to the first bin. Censoring uses only completed bins, so
    a patient censored inside a bin is known event-free at its left edge.
    Predictions are right-continuous steps at the bin right edges, with
    zero incidence at time zero. Times outside the horizon are rejected.

    The objective combines discrete negative log likelihood and a mean
    pairwise ranking penalty over comparable subjects in distinct bins.
    The mean reduction is a minibatch normalization of the paper's sum.

    Labels are `0` for censored and `1..num_risks` for the observed first
    event. The model opts out of dataset label processors, so `Trainer`
    monitors its joint first-event loss rather than classification metrics; use
    :mod:`pyhealth.metrics.competing_risks` for discrimination and calibration.

    Args:
        dataset: PyHealth SampleDataset, or None to call with tensors directly.
        input_dim: Number of input features; must be positive.
        num_risks: Number of competing causes; labels run 1..num_risks.
        num_bins: Number of discrete time bins; must be positive.
        hidden_dims: Hidden widths for the shared encoder.
        cause_dims: Hidden widths for each cause-specific stack.
        dropout: Dropout probability in [0, 1).
        mask_mode: Original features or features plus observed missingness mask.
        ranking_weight: Weight on the mean ranking penalty; zero disables it.
        ranking_sigma: Positive temperature scaling the ranking penalty.
        dtype: Floating parameter and likelihood precision.
        time_horizon: Positive final bin edge in preprocessing units.
        tail_mass: Include an event-free-at-horizon category (default True).
            False uses the original finite-grid softmax and rejects censoring
            at the final edge, where survival is necessarily zero.

    Examples:
        >>> import torch
        >>> from pyhealth.models import DeepHit
        >>> model = DeepHit(None, 3, num_risks=2, num_bins=4)
        >>> x = torch.zeros(2, 3)
        >>> out = model(features=x, time=[0.25, 1.0], event=[1, 0])
        >>> tuple(out["cif"].shape)
        (2, 2)
        >>> tuple(model.predict_cif(x, [0.25, 0.5, 0.75, 1.0]).shape)
        (2, 4, 2)
    """

    def __init__(
        self,
        dataset: SampleDataset | None,
        input_dim: int,
        num_risks: int = 2,
        num_bins: int = 50,
        hidden_dims: Sequence[int] = (25,),
        cause_dims: Sequence[int] = (25,),
        dropout: float = 0.0,
        mask_mode: MaskMode = "original",
        ranking_weight: float = 0.1,
        ranking_sigma: float = 0.1,
        dtype: torch.dtype = torch.float64,
        time_horizon: float = 1.0,
        tail_mass: bool = True,
    ) -> None:
        """Initialize the shared encoder, risk heads, and time grid.

        Args:
            dataset: Optional PyHealth sample dataset.
            input_dim: Number of clinical features.
            num_risks: Number of competing events.
            num_bins: Number of equal-width time intervals.
            hidden_dims: Shared encoder hidden widths.
            cause_dims: Cause-specific hidden widths.
            dropout: Dropout probability.
            mask_mode: Feature-only or observed-mask input mode.
            ranking_weight: Weight of the ranking penalty.
            ranking_sigma: Positive ranking temperature.
            dtype: Floating-point parameter precision.
            time_horizon: Last time-bin edge in preprocessing units.
            tail_mass: Whether to include survival beyond the horizon.

        Raises:
            ValueError: A dimension or configuration value is invalid.
            TypeError: tail_mass is not a boolean.
        """
        super().__init__(dataset)
        if any(
            not isinstance(width, int) or isinstance(width, bool) or width < 1
            for width in (input_dim, num_risks, num_bins, *hidden_dims, *cause_dims)
        ):
            raise ValueError("All dimensions must be positive integers")
        if mask_mode not in {"original", "observed"}:
            raise ValueError("Unknown mask_mode")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if (
            not math.isfinite(ranking_weight)
            or ranking_weight < 0
            or not math.isfinite(ranking_sigma)
            or ranking_sigma <= 0
        ):
            raise ValueError("Invalid ranking weight or sigma")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be float32 or float64")
        if not math.isfinite(time_horizon) or time_horizon <= 0:
            raise ValueError("time_horizon must be finite and positive")
        if not isinstance(tail_mass, bool):
            raise TypeError("tail_mass must be boolean")
        self.time_horizon = float(time_horizon)
        self.tail_mass = tail_mass
        self.input_dim, self.num_risks = input_dim, num_risks
        self.num_bins, self.mask_mode = num_bins, mask_mode
        self.ranking_weight, self.ranking_sigma = ranking_weight, ranking_sigma
        encoder_input = input_dim if mask_mode == "original" else 2 * input_dim
        self.shared = _mlp([encoder_input, *hidden_dims, input_dim], dropout)
        self.cause = nn.ModuleList(
            _mlp(
                [encoder_input + input_dim, *cause_dims, num_bins],
                dropout,
                activate_last=False,
            )
            for _ in range(num_risks)
        )
        self.tail = nn.Linear(encoder_input + input_dim, 1) if tail_mass else None
        self.register_buffer(
            "bin_edges", torch.linspace(0, time_horizon, num_bins + 1, dtype=dtype)[1:]
        )
        self.to(dtype=dtype)

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """Use the joint first-event loss rather than classification metrics.

        Returns:
            Empty mapping of label processors for loss-only evaluation.
        """
        return {}

    def _tensor(self, value: Any) -> Tensor:
        """Convert input to the model's device and floating-point precision.

        Args:
            value: Tensor or array-like input.

        Returns:
            Tensor matching the shared encoder's dtype and device.
        """
        return torch.as_tensor(
            value,
            dtype=self.shared[0].weight.dtype,
            device=self.device,
        )

    def _embedding(self, features: Any, missing_mask: Any) -> Tensor:
        """Validate inputs and encode them with the shared stack.

        Args:
            features: Finite imputed/scaled array-like [batch, input_dim].
            missing_mask: Binary indicators, 1=missing, or None.

        Returns:
            Shared representation concatenated with encoder input.

        Raises:
            ValueError: Features or masks are malformed or nonfinite.
        """
        features = self._tensor(features)
        if (
            features.ndim != 2
            or features.shape[0] == 0
            or features.shape[1] != self.input_dim
            or not torch.isfinite(features).all()
        ):
            raise ValueError("features must be finite, nonempty [batch, input_dim]")
        mask = None if missing_mask is None else self._tensor(missing_mask)
        if mask is not None and (
            mask.shape != features.shape or not ((mask == 0) | (mask == 1)).all()
        ):
            raise ValueError("missing_mask must be binary and match features")
        if self.mask_mode == "original":
            inputs = features
        else:
            if mask is None:
                raise ValueError("observed mask_mode requires missing_mask")
            inputs = torch.cat((features, mask), dim=1)
        return torch.cat((inputs, self.shared(inputs)), dim=1)

    def _bin_index(self, times: Tensor) -> Tensor:
        """Assign events to right-closed bins; zero belongs to the first bin.

        Args:
            times: Validated times within the modeled horizon, shape [batch].

        Returns:
            Long tensor of event bin indices, from zero to num_bins - 1.
        """
        return torch.bucketize(times.contiguous(), self.bin_edges, right=False)

    def _completed_bins(self, times: Tensor) -> Tensor:
        """Count bin right edges at or before each time.

        Args:
            times: Validated observation or prediction times.

        Returns:
            Long tensor of completed-bin counts, from zero to num_bins.
        """
        return torch.bucketize(times.contiguous(), self.bin_edges, right=True)

    def _incidence_grid(
        self, representation: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Compute jointly normalized event masses and cumulative incidence.

        Args:
            representation: Concatenated encoder inputs and shared embedding.

        Returns:
            Event log masses and CIFs of shape [batch, risks, bins], plus
            optional log tail mass of shape [batch] (None if disabled).

        Raises:
            FloatingPointError: The joint log-softmax contains nonfinite values.
        """
        logits = torch.stack([stack(representation) for stack in self.cause], dim=1)
        flat = logits.flatten(start_dim=1)
        if self.tail is not None:
            flat = torch.cat((flat, self.tail(representation)), dim=1)
        flat = torch.log_softmax(flat, dim=1)
        log_mass = flat[:, : self.num_risks * self.num_bins].reshape(
            -1, self.num_risks, self.num_bins
        )
        log_tail = flat[:, -1] if self.tail is not None else None
        if not torch.isfinite(flat).all():
            raise FloatingPointError("Nonfinite cause-time log mass")
        return log_mass, log_mass.exp().cumsum(dim=2), log_tail

    def _log_survival(
        self, log_mass: Tensor, completed: Tensor, log_tail: Tensor | None
    ) -> Tensor:
        """Sum remaining event mass and tail in log space, without subtraction.

        Args:
            log_mass: Joint event log probabilities [batch, risks, bins].
            completed: Number of completed bins per subject [batch].
            log_tail: Optional log probability beyond the final edge [batch].

        Returns:
            Log event-free probability [batch]; negative infinity when no
            probability remains in the original finite-grid formulation.
        """
        remaining = torch.arange(self.num_bins, device=log_mass.device)
        remaining = remaining[None, None, :] >= completed[:, None, None]
        terms = log_mass.masked_fill(~remaining, -torch.inf).flatten(1)
        if log_tail is not None:
            return torch.logsumexp(torch.cat((terms, log_tail[:, None]), 1), 1)
        # Avoid differentiating logsumexp over an entirely empty set.
        exhausted = completed == self.num_bins
        terms = torch.where(exhausted[:, None], torch.zeros_like(terms), terms)
        return torch.logsumexp(terms, dim=1).masked_fill(exhausted, -torch.inf)

    def _ranking_penalty(
        self,
        incidence: Tensor,
        times: Tensor,
        events: Tensor,
        bins: Tensor,
    ) -> Tensor:
        """Penalize discordant cumulative incidence among comparable pairs.

        For each cause, a subject observed to fail at their own time should
        carry higher incidence at that time than any subject still at risk
        beyond it. Pairs are weighted by an exponential of the incidence gap.

        Args:
            incidence: Cumulative incidence grid [batch, risks, bins].
            times: Scaled times [batch].
            events: Integer labels [batch], 0=censored, 1..num_risks observed.
            bins: Discrete bin index per subject [batch].

        Returns:
            Scalar mean penalty, or zero when no comparable pair exists.
        """
        later = (times[None, :] > times[:, None]) & (bins[None, :] > bins[:, None])
        total = incidence.new_zeros(())
        pairs = incidence.new_zeros(())
        for risk in range(self.num_risks):
            comparable = later & (events == risk + 1)[:, None]
            if not comparable.any():
                continue
            # curve[j, i] is subject j's incidence at subject i's own time.
            curve = incidence[:, risk, :][:, bins]
            gap = curve.diagonal()[:, None] - curve.transpose(0, 1)
            total = total + torch.exp(-gap[comparable] / self.ranking_sigma).sum()
            pairs = pairs + comparable.sum()
        return total / pairs if bool(pairs > 0) else total

    def forward(
        self,
        features: Any,
        time: Any = None,
        event: Any = None,
        missing_mask: Any = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Compute subject-time incidence and the optional DeepHit objective.

        Args:
            features: Finite imputed/scaled array-like [batch, input_dim].
            time: One finite nonnegative time per subject in preprocessing units.
            event: One integer label per subject, or None for prediction only.
            missing_mask: Binary indicators, 1=missing; used by the observed arm.
            **kwargs: Unused PyHealth sample metadata.

        Returns:
            cif and y_prob [batch, risks], survival [batch]; labeled batches also
            return scalar loss and y_true [batch, 2] containing time/event.

        Raises:
            ValueError: Features, times, events, or masks are invalid.
            FloatingPointError: The grid or likelihood is nonfinite.
        """
        if time is None:
            raise ValueError("forward requires time; use predict_cif for a grid")
        representation = self._embedding(features, missing_mask)
        times = self._tensor(time).reshape(-1)
        if (
            len(times) != len(representation)
            or not torch.isfinite(times).all()
            or (times < 0).any()
            or (times > self.bin_edges[-1]).any()
        ):
            raise ValueError("time must match batch and lie within [0, time_horizon]")
        raw_events = None if event is None else self._tensor(event).reshape(-1)
        if raw_events is not None and (
            len(raw_events) != len(times)
            or not torch.isfinite(raw_events).all()
            or (raw_events != raw_events.round()).any()
            or (raw_events < 0).any()
            or (raw_events > self.num_risks).any()
        ):
            raise ValueError("event must match batch and contain integers 0..risks")
        log_mass, incidence, log_tail = self._incidence_grid(representation)
        bins = self._bin_index(times)
        completed = self._completed_bins(times)
        rows = torch.arange(len(times), device=bins.device)
        padded = torch.cat((torch.zeros_like(incidence[:, :, :1]), incidence), 2)
        cif = padded[rows, :, completed]
        log_survival = self._log_survival(log_mass, completed, log_tail)
        result = {"cif": cif, "y_prob": cif, "survival": log_survival.exp()}
        if raw_events is not None:
            observed = raw_events.long()
            censored = observed == 0
            if log_tail is None and (censored & (completed == self.num_bins)).any():
                raise ValueError("Final-edge censoring requires tail_mass=True")
            selected = log_mass[rows, (observed - 1).clamp_min(0), bins]
            likelihood = torch.where(censored, log_survival, selected)
            if not torch.isfinite(likelihood).all():
                raise FloatingPointError("Nonfinite competing-risk likelihood")
            loss = -likelihood.mean()
            if self.ranking_weight:
                loss = loss + self.ranking_weight * self._ranking_penalty(
                    incidence, times, observed, bins
                )
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Nonfinite DeepHit objective")
            result["loss"] = loss
            result["y_true"] = torch.stack((times, raw_events), dim=1)
        return result

    def predict_cif(
        self,
        features: Any,
        horizons: Any,
        missing_mask: Any = None,
    ) -> Tensor:
        """Predict cumulative incidence on a nonnegative time grid.

        Call ``eval()`` to disable dropout. Gradients follow the caller's
        context; use ``no_grad()`` for numerical inference.

        Args:
            features: Imputed/scaled clinical [batch, input_dim].
            horizons: Finite, nonnegative grid in preprocessing time units.
            missing_mask: Binary indicators matching features.

        Returns:
            CIF tensor [batch, horizons, risks], in the supplied horizon order.
            Empty grids retain a zero-length horizon dimension.

        Raises:
            ValueError: Features, masks, or horizons are invalid.
            FloatingPointError: The joint log-softmax is nonfinite.
        """
        features = self._tensor(features)
        representation = self._embedding(features, missing_mask)
        horizons = self._tensor(horizons).reshape(-1)
        if (
            not torch.isfinite(horizons).all()
            or (horizons < 0).any()
            or (horizons > self.bin_edges[-1]).any()
        ):
            raise ValueError("horizons must lie within [0, time_horizon]")
        if not len(horizons):
            return features.new_empty((len(features), 0, self.num_risks))
        _, incidence, _ = self._incidence_grid(representation)
        padded = torch.cat((torch.zeros_like(incidence[:, :, :1]), incidence), 2)
        return padded[:, :, self._completed_bins(horizons)].transpose(1, 2)