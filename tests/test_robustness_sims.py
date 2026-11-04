"""Phase 4a methods validated on SIMULATED data with a planted truth (no real data).

Simulation counts: env ROBUST_SIMS (default 200; CI sets a smaller number)."""
import os

import numpy as np
import pandas as pd
import pytest

from src.survey import contrast, holm, proportion, rao_wu_log_ratio

N_SIMS = int(os.environ.get("ROBUST_SIMS", "200"))
H, M = 30, 40                     # strata (2 PSUs each), respondents per PSU per year
A1, A2, SD_U = np.log(0.04 / 0.96), np.log(0.03 / 0.97), 0.5


def _truth():
    """Population log ratio E[p2]/E[p1] when logit p = a_t + u, u ~ N(0, SD_U^2) (numerical integral)."""
    u = np.linspace(-6 * SD_U, 6 * SD_U, 20001)
    dens = np.exp(-0.5 * (u / SD_U) ** 2)
    e = [np.sum(dens / (1 + np.exp(-(a + u)))) / np.sum(dens) for a in (A1, A2)]
    return np.log(e[1] / e[0])


TRUE_LOG_RATIO = _truth()


def _survey(rng):
    """Two survey years sharing one design: the same PSUs (and PSU effects) are used in both years."""
    u = rng.normal(0, SD_U, 2 * H)
    rows = []
    for t, a in ((1, A1), (2, A2)):
        for k in range(2 * H):
            p = 1 / (1 + np.exp(-(a + u[k])))
            rows.append(pd.DataFrame({"h": k // 2, "j": k % 2 + 1, "year": t,
                                      "y": (rng.random(M) < p).astype(float), "w": 1.0}))
    return pd.concat(rows, ignore_index=True)


def test_delta_method_ci_for_log_ratio_covers_planted_change():
    rng = np.random.default_rng(101)
    cover = 0
    for _ in range(N_SIMS):
        d = _survey(rng)
        c = contrast(d, "y", "w", "h", "j", (d.year == 1).to_numpy(), (d.year == 2).to_numpy(), df_design=H)
        cover += c["log_ratio_ci_low"] <= TRUE_LOG_RATIO <= c["log_ratio_ci_high"]
    rate = cover / N_SIMS
    lo = 0.95 - 3 * np.sqrt(0.95 * 0.05 / N_SIMS)        # ~3 binomial SEs below 95%
    assert lo <= rate <= 0.995, f"coverage {rate:.3f} over {N_SIMS} sims"


def test_rao_wu_bootstrap_se_matches_linearization():
    d = _survey(np.random.default_rng(7))
    a, b = (d.year == 1).to_numpy(), (d.year == 2).to_numpy()
    c = contrast(d, "y", "w", "h", "j", a, b, df_design=H)
    est, reps = rao_wu_log_ratio(d, "y", "w", "h", "j", a, b, B=4000, seed=3)
    assert est == pytest.approx(c["log_ratio"], abs=1e-12)
    assert np.std(reps) == pytest.approx(c["log_ratio_se"], rel=0.10)


def test_rao_wu_bootstrap_interval_covers_planted_change():
    rng = np.random.default_rng(202)
    n = max(N_SIMS // 2, 30)
    cover = 0
    for s in range(n):
        d = _survey(rng)
        est, reps = rao_wu_log_ratio(d, "y", "w", "h", "j", (d.year == 1).to_numpy(), (d.year == 2).to_numpy(),
                                     B=400, seed=s)
        se = np.std(reps)
        cover += abs(est - TRUE_LOG_RATIO) <= 2.042 * se          # t(30) critical value
    rate = cover / n
    assert 0.95 - 3 * np.sqrt(0.95 * 0.05 / n) <= rate <= 1.0


def test_shared_design_and_independent_variances_differ_when_psus_correlate():
    # With PSU effects shared across years, ignoring the covariance should give a LARGER SE.
    d = _survey(np.random.default_rng(11))
    a, b = (d.year == 1).to_numpy(), (d.year == 2).to_numpy()
    shared = contrast(d, "y", "w", "h", "j", a, b, df_design=H)["log_ratio_se"]
    pa, pb = (proportion(d, "y", "w", "h", "j", domain=x, df_design=H) for x in (a, b))
    indep = np.sqrt((pa["se"] / pa["p"]) ** 2 + (pb["se"] / pb["p"]) ** 2)
    assert indep > shared


def test_holm_matches_textbook_example():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.5, 0.9]), [1.0, 1.0])            # adjusted p is capped at 1
    assert np.allclose(holm([0.2, 0.6]), [0.4, 0.6])
