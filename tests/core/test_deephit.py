"""Small synthetic tests for DeepHit; no clinical data required.

Contributor: Bin Yang (NetID: biny4)
Paper: DeepHit: A Deep Learning Approach to Survival Analysis with Competing
Risks. https://doi.org/10.1609/aaai.v32i1.11842

Every test uses hand-written tensors with zeroed or seeded parameters, touches
no real dataset and no network, and runs in milliseconds.
"""

import math

import pytest
import torch

from pyhealth.models import DeepHit


def uniform_model(**kwargs):
    """Build a zero-parameter model whose cause-time softmax is uniform.

    Args:
        **kwargs: Constructor overrides such as tail_mass or ranking_weight.

    Returns:
        A DeepHit instance with every parameter set to zero, so each of the
        num_risks * num_bins (+1) grid cells carries identical mass.
    """
    model = DeepHit(None, 3, num_bins=4, hidden_dims=(4,), cause_dims=(4,), **kwargs)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def test_forward_and_gradients() -> None:
    """Forward returns documented keys and trainable parameters get gradients."""
    torch.manual_seed(7)
    model = DeepHit(None, 3, num_bins=4)
    output = model(
        torch.randn(3, 3),
        [0.25, 0.5, 1.0],
        [1, 2, 0],
    )

    assert set(output) == {
        "cif",
        "y_prob",
        "survival",
        "loss",
        "y_true",
    }
    assert output["cif"].shape == (3, 2)
    assert output["y_prob"].shape == (3, 2)
    assert output["y_true"].shape == (3, 2)
    assert output["survival"].shape == (3,)
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])

    output["loss"].backward()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.numel() > 0:
            assert parameter.grad is not None, f"Missing gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), (
                f"Nonfinite gradient: {name}"
            )


def test_cif_probability_and_boundaries():
    """CIF starts at zero, never decreases, and stays a valid probability."""
    model = uniform_model()
    cif = model.predict_cif(torch.zeros(2, 3), [0, 0.1, 0.25, 0.5, 1.0])
    assert cif.shape == (2, 5, 2)
    assert (cif[:, :2] == 0).all()
    assert (cif.diff(dim=1) >= 0).all()
    assert (cif.sum(dim=-1) <= 1).all()
    torch.testing.assert_close(cif[0, -1], torch.full((2,), 4 / 9, dtype=cif.dtype))


def test_likelihood_hand_calculated():
    """Uniform-grid likelihood and survival match values computed by hand."""
    model = uniform_model(ranking_weight=0)
    out = model(torch.zeros(3, 3), [0.1, 0.5, 1.0], [1, 0, 0])
    expected = -torch.tensor([1 / 9, 5 / 9, 1 / 9], dtype=torch.float64).log().mean()
    torch.testing.assert_close(out['loss'], expected)
    torch.testing.assert_close(
        out['survival'], torch.tensor([1., 5 / 9, 1 / 9], dtype=torch.float64))


def test_final_censor_has_gradient():
    """Censoring at the final edge trains the tail category."""
    model = uniform_model(ranking_weight=0)
    out = model(torch.zeros(1, 3), [1.0], [0])
    out['loss'].backward()
    assert model.tail.bias.grad.abs().item() > 0


def test_original_softmax() -> None:
    """The finite grid gives zero survival at the horizon and finite gradients."""
    model = uniform_model(tail_mass=False, ranking_weight=0)

    output = model(
        torch.zeros(1, 3),
        [1.0],
        [2],
    )

    assert output["survival"].item() == 0
    assert torch.isfinite(output["loss"])

    output["loss"].backward()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.numel() > 0:
            assert parameter.grad is not None, f"Missing gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), (
                f"Nonfinite gradient: {name}"
            )

    with pytest.raises(ValueError, match="Final-edge censoring"):
        model(torch.zeros(1, 3), [1.0], [0])


def test_ranking_hand_calculated():
    """The pairwise penalty matches its closed form and vanishes without pairs."""
    model = uniform_model(ranking_sigma=0.1)
    cif = torch.tensor([[[0.4, 0.5, 0.6, 0.7], [0., 0., 0., 0.]],
                        [[0.2, 0.3, 0.4, 0.5], [0., 0., 0., 0.]]], dtype=torch.float64)
    times = torch.tensor([0.25, 0.75])
    bins = torch.tensor([0, 2])
    penalty = model._ranking_penalty(cif, times, torch.tensor([1, 0]), bins)
    torch.testing.assert_close(penalty, torch.tensor(-2., dtype=torch.float64).exp())
    assert model._ranking_penalty(cif, times, torch.zeros(2), bins).item() == 0
\

def test_all_censored_and_empty_grid():
    """An all-censored batch stays finite and an empty horizon grid is allowed."""
    model = uniform_model()
    x = torch.zeros(2, 3)
    assert torch.isfinite(model(x, [0.5, 1.0], [0, 0])['loss'])
    assert model.predict_cif(x, []).shape == (2, 0, 2)


def test_invalid_inputs_and_masks():
    """Out-of-range times, horizons, labels and missing masks are rejected."""
    model = uniform_model()
    x = torch.zeros(1, 3)
    with pytest.raises(ValueError):
        model(x, [1.1], [1])
    with pytest.raises(ValueError):
        model.predict_cif(x, [-0.1])
    with pytest.raises(ValueError):
        model(x, [0.5], [3])
    masked = uniform_model(mask_mode='observed')
    with pytest.raises(ValueError):
        masked(x, [0.5], [1])
    assert torch.isfinite(masked(x, [0.5], [1], torch.ones_like(x))['loss'])


def test_analytical_uniform_grid():
    """Verify completed-bin CIF, tail survival, and exact uniform-grid NLL."""
    model = uniform_model(ranking_weight=0.0)
    output = model(torch.zeros(3, 3), [0.0, 0.5, 1.0], [1, 0, 2])
    completed = torch.tensor([0., 2., 4.], dtype=torch.float64)
    torch.testing.assert_close(output['cif'], (completed / 9)[:, None].expand(-1, 2))
    torch.testing.assert_close(output['survival'], (2 * (4 - completed) + 1) / 9)
    expected = -torch.tensor([1 / 9, 5 / 9, 1 / 9], dtype=torch.float64).log().mean()
    torch.testing.assert_close(output['loss'], expected)
    torch.testing.assert_close(
        output['cif'].sum(1) + output['survival'], torch.ones(3, dtype=torch.float64))


def test_invalid_tail_mass_type():
    """Reject non-boolean tail configuration with a type error."""
    with pytest.raises(TypeError, match='tail_mass'):
        uniform_model(tail_mass=1)

@pytest.mark.parametrize('override', [
    {'input_dim': 0},
    {'num_risks': 0},
    {'num_bins': 0},
    {'dropout': 1.0},
    {'ranking_sigma': 0.0},
    {'ranking_weight': -1.0},
    {'mask_mode': 'augmented'},
    {'dtype': torch.float16},
    {'time_horizon': 0.0},
])
def test_invalid_constructor_arguments(override):
    """Reject malformed dimensions and configuration values."""
    base = {'input_dim': 3, 'num_bins': 4, 'hidden_dims': (4,), 'cause_dims': (4,)}
    with pytest.raises(ValueError):
        DeepHit(None, **{**base, **override})


def test_single_risk_and_float32():
    """A single risk reduces to discrete-time survival; float32 is supported."""
    single = DeepHit(None, 3, num_risks=1, num_bins=4, hidden_dims=(4,))
    output = single(torch.randn(2, 3), [0.25, 0.75], [0, 1])
    assert output['cif'].shape == (2, 1)
    assert torch.isfinite(output['loss'])
    output['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in single.parameters())
    narrow = DeepHit(None, 3, num_bins=4, dtype=torch.float32)
    assert narrow(torch.randn(2, 3), [0.25, 0.75], [1, 0])['cif'].dtype == torch.float32


def test_state_dict_round_trip(tmp_path):
    """Save and reload parameters and the time-grid buffer."""
    model = DeepHit(None, 3, num_bins=4).eval()
    path = tmp_path / 'deephit.pt'
    torch.save(model.state_dict(), path)
    restored = DeepHit(None, 3, num_bins=4)
    restored.load_state_dict(torch.load(path))
    restored.eval()
    assert 'bin_edges' in model.state_dict()
    features = torch.randn(2, 3)
    with torch.no_grad():
        torch.testing.assert_close(
            model.predict_cif(features, [0.25, 0.75]),
            restored.predict_cif(features, [0.25, 0.75]),
        )


def test_sample_dataset_and_trainer_integration():
    """Drive the model through PyHealth's dataset and trainer pipeline."""
    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.processors import BinaryLabelProcessor
    from pyhealth.trainer import Trainer

    samples = [
        {
            'patient_id': f'patient-{index}',
            'features': [0.1 * index, -0.2 * index, 0.3],
            'time': 0.2 + 0.2 * index,
            'event': index % 3,
            'auxiliary_label': index % 2,
        }
        for index in range(4)
    ]
    dataset = create_sample_dataset(
        samples,
        {'features': 'raw'},
        {'time': 'raw', 'event': 'raw', 'auxiliary_label': BinaryLabelProcessor},
    )
    model = DeepHit(dataset, 3, num_bins=4)
    assert model.label_processors == {}
    assert model.mode is None
    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    scores = Trainer(model, device='cpu', enable_logging=False).evaluate(loader)
    assert set(scores) == {'loss'}
    assert math.isfinite(scores['loss'])