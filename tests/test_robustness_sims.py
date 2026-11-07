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


# ---------------------------------------------------------------- R5: break date in monthly counts
def test_profile_ci_for_break_month_covers_planted_break():
    from src.segmented import design, profile, profile_ci
    rng = np.random.default_rng(303)
    n, tau0 = 85, 40
    months = (np.arange(n) % 12) + 1
    beta = np.r_[np.log(8000), 0.012, -0.032, 0.04 * np.sin(np.arange(11))]
    mu = np.exp(design(n, (tau0,), months) @ beta)
    cover = 0
    for _ in range(N_SIMS):
        y = rng.poisson(mu * rng.gamma(400, 1 / 400, n))           # overdispersed (gamma-Poisson)
        lo, hi = profile_ci(profile(y, 1, months))
        cover += lo <= tau0 <= hi
    rate = cover / N_SIMS
    assert 0.95 - 3 * np.sqrt(0.95 * 0.05 / N_SIMS) <= rate <= 1.0, f"coverage {rate:.3f}"


# ---------------------------------------------------------------- R3: county gap with 1-9 suppression
def _counties(rng, n_per_q=150, gap=0.12):
    from src.step_d import COL
    rows = []
    for q in (1, 2, 3, 4):
        for i in range(n_per_q):
            pop = float(np.exp(rng.normal(10.3, 1.0)))
            for year, post in ((2022, 0), (2025, 1)):
                log_rate = np.log(30e-5) + 0.15 * (q - 1) - 0.5 * post + (gap if (q == 4 and post) else 0.0)
                mu = pop * np.exp(log_rate)
                yv = rng.poisson(mu * rng.gamma(4, 1 / 4))                 # NB-like
                rows.append({"FIPS": f"{q}{i:04d}", "Year": year, COL: float(yv), "quartile": q, "E_TOTPOP": pop})
    m = pd.DataFrame(rows)
    m.loc[m[COL].between(1, 9), COL] = np.nan                               # artificial NCHS suppression
    m["censored"] = m[COL].isna()
    return m


def test_censored_model_recovers_planted_svi_gap_where_zeroing_does_not():
    from src.step_d import fit
    rng = np.random.default_rng(404)
    n = max(N_SIMS // 10, 6)
    est, zero, cover = [], [], 0
    for _ in range(n):
        m = _counties(rng)
        tab, _, _ = fit(m, 2022, 2025, "censored")
        r = tab.set_index("term").loc["year_2025:SVI_Q4"]
        est.append(r.coef)
        cover += r.ci_low <= 0.12 <= r.ci_high
        zero.append(fit(m, 2022, 2025, "zero")[0].set_index("term").loc["year_2025:SVI_Q4"].coef)
    mc_se = np.std(est, ddof=1) / np.sqrt(n)                       # Monte Carlo error of the mean
    assert abs(np.mean(est) - 0.12) < 3 * mc_se + 0.01
    assert cover / n >= 0.8
    assert abs(np.mean(zero) - 0.12) > abs(np.mean(est) - 0.12)


# ---------------------------------------------------------------- TEDS (a): TWFE + wild cluster bootstrap
def test_twfe_point_estimate_matches_statsmodels_with_dummies():
    import statsmodels.api as sm

    from src.panel import twfe_design, wild_cluster_bootstrap
    d = __import__("src.panel", fromlist=["simulate_panel"]).simulate_panel(np.random.default_rng(1))
    X, names = twfe_design(d, ["x"], "unit", "time")
    ref = sm.OLS(d.y, X).fit(cov_type="cluster", cov_kwds={"groups": d.unit})
    r = wild_cluster_bootstrap(d, "y", ["x"], "unit", "time", B=99, seed=1)
    assert r["estimate"] == pytest.approx(ref.params.iloc[0], abs=1e-10)
    assert r["se_cr1"] == pytest.approx(ref.bse.iloc[0], rel=1e-6)


def test_wild_cluster_bootstrap_ci_covers_planted_slope_and_null_rejects_about_5pct():
    from src.panel import simulate_panel, wild_cluster_bootstrap
    rng = np.random.default_rng(505)
    n = max(N_SIMS // 2, 40)
    cover, reject = 0, 0
    for s in range(n):
        d = simulate_panel(rng, beta=0.6)
        r = wild_cluster_bootstrap(d, "y", ["x"], "unit", "time", B=199, seed=s)
        cover += r["ci_low"] <= 0.6 <= r["ci_high"]
        d0 = simulate_panel(rng, beta=0.0)
        reject += wild_cluster_bootstrap(d0, "y", ["x"], "unit", "time", B=199, seed=10_000 + s)["p_wild"] < 0.05
    se_band = 3 * np.sqrt(0.95 * 0.05 / n)
    assert 0.95 - se_band <= cover / n <= 1.0, f"coverage {cover / n:.3f}"
    assert reject / n <= 0.05 + se_band, f"null rejection {reject / n:.3f}"
