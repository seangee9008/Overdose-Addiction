"""Censored negative binomial: known answers on synthetic data."""
import numpy as np
import statsmodels.api as sm

from src.count_models import censored_nb


def _sim(n=4000, seed=0, alpha=0.4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    pop = rng.uniform(5_000, 200_000, n)
    mu = np.exp(-9.0 + 0.5 * x + np.log(pop))
    r = 1 / alpha
    y = rng.negative_binomial(r, r / (r + mu))
    X = np.column_stack([np.ones(n), x])
    return X, y.astype(float), np.log(pop)


def test_no_censoring_matches_statsmodels_nb2():
    X, y, off = _sim()
    tab, _, info = censored_nb(X, y, np.zeros(len(y), bool), np.zeros(len(y)), np.zeros(len(y)), offset=off)
    ref = sm.NegativeBinomial(y, X, offset=off, loglike_method="nb2").fit(disp=0, maxiter=500)
    assert np.allclose(tab.coef[:2], ref.params[:2], atol=1e-4)
    assert np.isclose(info["alpha"], ref.params[2], rtol=1e-3)
    # each observation its own cluster: sandwich = HC0 robust * n/(n-1)
    rob = sm.NegativeBinomial(y, X, offset=off, loglike_method="nb2").fit(disp=0, maxiter=500, cov_type="HC0")
    n = len(y)
    assert np.allclose(tab.se[:2], rob.bse[:2] * np.sqrt(n / (n - 1)), rtol=2e-2)


def test_suppression_1_to_9_recovers_truth_where_zeroing_is_biased():
    X, y, off = _sim(n=6000, seed=1)
    cens = (y >= 1) & (y <= 9)
    assert 0.15 < cens.mean() < 0.6          # a realistic share is suppressed
    tab, _, info = censored_nb(X, np.where(cens, np.nan, y), cens, np.full(len(y), 1.0), np.full(len(y), 9.0),
                               offset=off)
    assert abs(tab.coef[1] - 0.5) < 0.05 and abs(tab.coef[0] + 9.0) < 0.1
    assert info["n_censored"] == cens.sum()
    # contrast: treating suppressed cells as 0 distorts the slope
    bad, _, _ = censored_nb(X, np.where(cens, 0.0, y), np.zeros(len(y), bool), np.zeros(len(y)),
                            np.zeros(len(y)), offset=off)
    assert abs(bad.coef[1] - 0.5) > abs(tab.coef[1] - 0.5)


def test_censored_poisson_without_censoring_matches_statsmodels_glm():
    from src.count_models import censored_poisson
    rng = np.random.default_rng(5)
    n = 3000
    x = rng.normal(size=n)
    pop = rng.uniform(5_000, 200_000, n)
    y = rng.poisson(np.exp(-9.0 + 0.4 * x + np.log(pop))).astype(float)
    X = np.column_stack([np.ones(n), x])
    tab, _, _ = censored_poisson(X, y, np.zeros(n, bool), np.zeros(n), np.zeros(n), offset=np.log(pop))
    ref = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(pop)).fit(cov_type="HC0")
    assert np.allclose(tab.coef, ref.params, atol=1e-4)
    assert np.allclose(tab.se, ref.bse * np.sqrt(n / (n - 1)), rtol=2e-2)
