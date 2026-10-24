"""Design-based proportion estimator checked against answers worked out by hand."""
import numpy as np
import pandas as pd
import pytest

from src.survey import contrast, proportion


def test_hand_calculated_two_strata_two_psus():
    # Equal weights; p = 4/8 = 0.5. Linearized scores z_i = (y_i - 0.5)/8.
    # PSU totals: s1p1 = 0, s1p2 = 1/8, s2p1 = -1/8, s2p2 = 0.
    # Each stratum: 2/(2-1) * 2*(1/16)^2 = 4/256  ->  Var = 8/256 = 1/32.
    df = pd.DataFrame({
        "h": [1, 1, 1, 1, 2, 2, 2, 2],
        "j": [1, 1, 2, 2, 1, 1, 2, 2],
        "y": [1, 0, 1, 1, 0, 0, 1, 0],
        "w": [1.0] * 8,
    })
    r = proportion(df, "y", "w", "h", "j")
    assert r["p"] == pytest.approx(0.5)
    assert r["se"] == pytest.approx(np.sqrt(1 / 32))
    assert r["df"] == 2  # 4 PSUs - 2 strata


def test_simple_random_sample_reduces_to_s2_over_n():
    # One stratum, every record its own PSU, equal weights: Var(p) = s^2 / n.
    y = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1], float)
    df = pd.DataFrame({"h": 1, "j": np.arange(len(y)), "y": y, "w": 1.0})
    r = proportion(df, "y", "w", "h", "j")
    assert r["p"] == pytest.approx(y.mean())
    assert r["se"] == pytest.approx(np.sqrt(y.var(ddof=1) / len(y)))


def test_domain_point_estimate_is_weighted_subset_mean_and_missing_excluded():
    df = pd.DataFrame({"h": [1, 1, 1, 1, 2, 2, 2, 2], "j": [1, 1, 2, 2, 1, 1, 2, 2],
                       "y": [1, 0, np.nan, 1, 0, 1, 1, 0],
                       "w": [2.0, 1.0, 5.0, 3.0, 1.0, 4.0, 2.0, 2.0],
                       "g": [1, 1, 1, 2, 2, 1, 1, 2]})
    r = proportion(df, "y", "w", "h", "j", domain=(df.g == 1).to_numpy())
    sub = df[(df.g == 1) & df.y.notna()]
    assert r["p"] == pytest.approx(np.average(sub.y, weights=sub.w))
    assert r["n"] == 4  # the missing-y record is excluded, not counted as 0


def test_contrast_of_domains_in_disjoint_strata_adds_variances():
    # Domains that never share a stratum are independent: Var(b - a) = Var(a) + Var(b).
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({"h": np.repeat(np.arange(20), n // 20), "j": np.tile([1, 2], n // 2),
                       "y": rng.integers(0, 2, n).astype(float), "w": rng.uniform(0.5, 2, n)})
    a = (df.h < 10).to_numpy()
    b = ~a
    c = contrast(df, "y", "w", "h", "j", a, b)
    va = proportion(df, "y", "w", "h", "j", domain=a)["se"] ** 2
    vb = proportion(df, "y", "w", "h", "j", domain=b)["se"] ** 2
    assert c["diff"] == pytest.approx(c["p_b"] - c["p_a"])
    assert c["diff_se"] ** 2 == pytest.approx(va + vb)


def test_ci_coverage_on_simulated_clustered_survey():
    # Known truth p = 0.10; 40 strata x 2 PSUs, PSU-level random effect induces clustering.
    # A correct design-based 95% CI should cover the truth roughly 95% of the time.
    rng = np.random.default_rng(42)
    H, m, reps, cover = 40, 30, 300, 0
    logit_true = np.log(0.1 / 0.9)
    h = np.repeat(np.arange(H), 2 * m)
    j = np.tile(np.repeat([1, 2], m), H)
    for _ in range(reps):
        u = rng.normal(0, 0.6, 2 * H)                       # PSU effects
        lin = logit_true + np.repeat(u, m)
        y = (rng.random(2 * H * m) < 1 / (1 + np.exp(-lin))).astype(float)
        df = pd.DataFrame({"h": h, "j": j, "y": y, "w": 1.0})
        r = proportion(df, "y", "w", "h", "j")
        cover += r["ci_low"] <= TRUE_P <= r["ci_high"]
    assert 0.90 <= cover / reps <= 0.99


# Population proportion implied by logit(0.1) + N(0, 0.6^2) PSU effects, by numerical integration.
_u = np.linspace(-6, 6, 20001)
_dens = np.exp(-0.5 * (_u / 0.6) ** 2)
TRUE_P = float(np.sum(_dens / (1 + np.exp(-(np.log(0.1 / 0.9) + _u)))) / np.sum(_dens))
