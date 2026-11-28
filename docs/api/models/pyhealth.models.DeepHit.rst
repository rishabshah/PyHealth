DeepHit
=======

.. automodule:: pyhealth.models.deephit
    :members:
    :show-inheritance:

.. math::

    \sum_{k=1}^{K}\sum_{m=1}^{T} o_{k,m}(x) + o_\varnothing(x) = 1,
    \qquad
    \hat{F}_k(t \mid x) = \sum_{m\,:\,e_m \le t} o_{k,m}(x)

Notes
-----

A single softmax over the joint (cause, time-bin) grid gives a probability mass
function over all causes and horizons, with :math:`e_m` the bin right edges and
:math:`o_\varnothing` the event-free-at-horizon category; cumulative incidence
for each cause is its running sum along the time axis over completed bins.
Training combines a discrete likelihood, in which observed events contribute
their own cell and censored records contribute the remaining event-free mass,
with a cause-specific ranking penalty over comparable pairs.

By default the extra event-free-at-horizon category is included, so subjects
censored at the final bin edge have strictly positive survival and a valid
likelihood. Setting ``tail_mass=False`` recovers the paper's finite grid, where
total incidence reaches one at the horizon and final-edge censoring is
rejected. Times are supplied in the preprocessing units of the competing-risks
preprocessor and must lie within ``time_horizon``.

Usage
-----

Train with ``Trainer``, monitoring loss in minimize mode. The model reports no
label processors, so no classification metrics are computed; use
:mod:`pyhealth.metrics.competing_risks` for discrimination and calibration.
Call ``eval()`` before predicting on a time grid.

References
----------

Lee, C., Zame, W. R., Yoon, J., van der Schaar, M. "DeepHit: A Deep Learning
Approach to Survival Analysis with Competing Risks." AAAI 2018, 2314-2321.
https://doi.org/10.1609/aaai.v32i1.11842

See ``examples/synthetic_competing_risk_deephit.py`` for hyperparameter and
censoring ablations on self-contained synthetic data.