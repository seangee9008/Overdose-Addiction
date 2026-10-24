"""Survey logistic regression: known answers on synthetic data."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.survey_glm import design_matrix, svy_logit


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=n), "g": rng.integers(0, 3, n),
                       "h": np.repeat(np.arange(30), n // 30), "j": np.tile([1, 2], n // 2),
                       "w": rng.uniform(0.5, 3, n)})
    lin = -1 + 0.8 * df.x + 0.5 * (df.g == 2)
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)
    return df


def test_srs_equal_weights_matches_unweighted_mle_and_robust_se():
    # One stratum, each record its own PSU, equal weights: coefficients = ordinary logistic MLE,
    # and V = HC0 robust variance * n/(n-1).
    df = _data()
    n = len(df)
    X, names = design_matrix(df, numeric=["x"], categorical={"g": 0})
    res, _ = svy_logit(X, df.y, np.ones(n), np.zeros(n), np.arange(n), design_df=n - 1, names=names)
    sm_fit = sm.Logit(df.y, X).fit(disp=0, cov_type="HC0")
    assert np.allclose(res.coef, sm_fit.params, atol=1e-8)
    assert np.allclose(res.se, sm_fit.bse * np.sqrt(n / (n - 1)), rtol=1e-6)


def test_rescaling_weights_changes_nothing():
    df = _data(seed=1)
    X, names = design_matrix(df, numeric=["x"], categorical={"g": 0})
    a, _ = svy_logit(X, df.y, df.w, df.h, df.j, 30, names)
    b, _ = svy_logit(X, df.y, 7.5 * df.w, df.h, df.j, 30, names)
    assert np.allclose(a.coef, b.coef) and np.allclose(a.se, b.se)


def test_single_binary_predictor_equals_logit_difference_hand_calculation():
    # With only a 0/1 group dummy, the slope is logit(p_1) - logit(p_0) of the weighted
    # proportions, and its SE equals the linearized SE of that difference.
    df = _data(seed=2)
    df["b"] = (df.g == 2).astype(float)
    X, names = design_matrix(df, numeric=["b"])
    res, _ = svy_logit(X, df.y, df.w, df.h, df.j, 30, names)
    p1 = np.average(df.y[df.b == 1], weights=df.w[df.b == 1])
    p0 = np.average(df.y[df.b == 0], weights=df.w[df.b == 0])
    hand = np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))
    assert res.coef[1] == pytest.approx(hand, abs=1e-9)
