"""Break detector recovers break points planted in synthetic piecewise-linear data."""
import numpy as np

from src.breaks import fit_no_break, fit_one_break, fit_two_breaks


def piecewise(n, breaks, slopes, intercepts, noise_sd, seed):
    """Piecewise-linear series; segment k starts at breaks[k-1] and is
    intercepts[k] + slopes[k] * t, plus N(0, noise_sd) noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    edges = [0, *breaks, n]
    y = np.empty(n)
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        y[a:b] = intercepts[k] + slopes[k] * t[a:b]
    return y + rng.normal(0, noise_sd, n)


def test_two_breaks_with_level_shifts_recovered_exactly():
    # Same length as the CDC national series (136 months).
    y = piecewise(136, [40, 95], slopes=[10, 60, -40], intercepts=[1000, 500, 9000],
                  noise_sd=20, seed=0)
    tau1, tau2, fitted, sse = fit_two_breaks(y, min_seg=12)
    assert (tau1, tau2) == (40, 95)


def test_two_kinks_without_level_shift_recovered_within_tolerance():
    # Continuous trend whose slope changes at 50 and 100 -- the harder case and
    # closer to the shape of a smoothed (12-month-ending) series.
    n, b1, b2 = 136, 50, 100
    t = np.arange(n, dtype=float)
    trend = np.where(t < b1, 20 * t,
                     np.where(t < b2, 20 * b1 + 80 * (t - b1),
                              20 * b1 + 80 * (b2 - b1) - 50 * (t - b2)))
    y = trend + np.random.default_rng(1).normal(0, 30, n)
    tau1, tau2, _, _ = fit_two_breaks(y, min_seg=12)
    assert abs(tau1 - b1) <= 3
    assert abs(tau2 - b2) <= 3


def test_one_break_recovered_and_sse_ordering():
    y = piecewise(100, [60], slopes=[5, -15], intercepts=[0, 2000], noise_sd=10, seed=2)
    tau, _, sse1 = fit_one_break(y, min_seg=12)
    assert tau == 60
    _, sse0 = fit_no_break(y)
    *_, sse2 = fit_two_breaks(y, min_seg=12)
    # Nested models: more breaks can never fit worse.
    assert sse2 <= sse1 <= sse0
