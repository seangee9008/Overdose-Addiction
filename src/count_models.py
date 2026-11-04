"""
Negative binomial (NB2) regression with interval-censored counts, for suppressed CDC county data.

Each observation is either an exact count y, or censored to an interval [lo, hi] (CDC suppresses
county counts of 1-9, so a suppressed cell contributes P(1 <= Y <= 9) to the likelihood instead of
being set to 0 or dropped).

  mean:        mu_i = exp(x_i' beta + offset_i)
  NB2:         Var(Y) = mu + alpha * mu^2,  alpha = exp(log_alpha) > 0
  likelihood:  exact y -> pmf(y);  censored -> cdf(hi) - cdf(lo - 1)
  SEs:         cluster-robust sandwich A^-1 B A^-1 with clusters (e.g. county), B from summed
               per-cluster scores, small-sample factor G/(G-1). Scores and Hessian by finite differences.
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats


def _loglik_obs(params, X, y, lo, hi, cens, offset):
    beta, alpha = params[:-1], np.exp(params[-1])
    mu = np.exp(X @ beta + offset)
    n = 1.0 / alpha
    p = n / (n + mu)
    ll = np.empty(len(y))
    ex = ~cens
    ll[ex] = stats.nbinom.logpmf(y[ex], n, p[ex])
    if cens.any():
        prob = stats.nbinom.cdf(hi[cens], n, p[cens]) - stats.nbinom.cdf(lo[cens] - 1, n, p[cens])
        ll[cens] = np.log(np.clip(prob, 1e-300, None))
    return ll


def _num_grad_obs(f, x, h=1e-5):
    g = []
    for k in range(len(x)):
        e = np.zeros_like(x)
        e[k] = h * max(1.0, abs(x[k]))
        g.append((f(x + e) - f(x - e)) / (2 * e[k]))
    return np.column_stack(g)


def censored_nb(X, y, censored, lo, hi, offset=None, clusters=None, names=None):
    """Fit NB2 by maximum likelihood with interval censoring.
    y: exact counts (ignored where censored); censored: bool; lo/hi: interval bounds where censored.
    Returns (coef table DataFrame incl. log_alpha, dict with loglik, alpha, n, n_censored, converged)."""
    X = np.asarray(X, float)
    y = np.nan_to_num(np.asarray(y, float))
    cens = np.asarray(censored, bool)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)

    def f_obs(p):
        return _loglik_obs(p, X, y, lo, hi, cens, off)

    # start: Poisson on exact obs (censored at interval midpoint), alpha = 0.5
    ys = np.where(cens, (lo + hi) / 2, y)
    b0 = np.linalg.lstsq(X, np.log(ys + 0.5) - off, rcond=None)[0]
    start = np.concatenate([b0, [np.log(0.5)]])
    res = optimize.minimize(lambda p: -f_obs(p).sum(), start, method="BFGS",
                            options=dict(gtol=1e-7, maxiter=5000))
    if not res.success:
        res = optimize.minimize(lambda p: -f_obs(p).sum(), res.x, method="Nelder-Mead",
                                options=dict(maxiter=20000, xatol=1e-8, fatol=1e-10))
    theta = res.x

    S = _num_grad_obs(f_obs, theta)                                  # per-observation scores
    A = -_num_grad_obs(lambda p: _num_grad_obs(f_obs, p).sum(axis=0), theta)  # -Hessian
    A = (A + A.T) / 2
    g = np.arange(len(y)) if clusters is None else np.asarray(clusters)
    Sc = pd.DataFrame(S).groupby(g).sum().to_numpy()
    G = len(Sc)
    B = G / (G - 1) * Sc.T @ Sc
    Ainv = np.linalg.inv(A)
    V = Ainv @ B @ Ainv
    se = np.sqrt(np.diag(V))
    z = theta / se
    names = (names or [f"x{i}" for i in range(X.shape[1])]) + ["log_alpha"]
    tab = pd.DataFrame({"term": names, "coef": theta, "se": se, "z": z, "p": 2 * stats.norm.sf(np.abs(z)),
                        "ci_low": theta - 1.96 * se, "ci_high": theta + 1.96 * se})
    tab["rate_ratio"] = np.exp(tab.coef)
    tab["rr_ci_low"] = np.exp(tab.ci_low)
    tab["rr_ci_high"] = np.exp(tab.ci_high)
    info = dict(loglik=float(-res.fun), alpha=float(np.exp(theta[-1])), n=len(y), n_censored=int(cens.sum()),
                clusters=G, converged=bool(res.success))
    return tab, V, info


def censored_poisson(X, y, censored, lo, hi, offset=None, clusters=None, names=None):
    """Poisson regression with interval censoring (same interface and cluster-robust sandwich as
    censored_nb, without the dispersion parameter). Added for Phase 4a robustness (R3)."""
    X = np.asarray(X, float)
    y = np.nan_to_num(np.asarray(y, float))
    cens = np.asarray(censored, bool)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)

    def f_obs(beta):
        mu = np.exp(X @ beta + off)
        ll = np.empty(len(y))
        ll[~cens] = stats.poisson.logpmf(y[~cens], mu[~cens])
        if cens.any():
            prob = stats.poisson.cdf(hi[cens], mu[cens]) - stats.poisson.cdf(lo[cens] - 1, mu[cens])
            ll[cens] = np.log(np.clip(prob, 1e-300, None))
        return ll

    ys = np.where(cens, (lo + hi) / 2, y)
    start = np.linalg.lstsq(X, np.log(ys + 0.5) - off, rcond=None)[0]
    res = optimize.minimize(lambda b: -f_obs(b).sum(), start, method="BFGS", options=dict(gtol=1e-7, maxiter=5000))
    if not res.success:
        res = optimize.minimize(lambda b: -f_obs(b).sum(), res.x, method="Nelder-Mead",
                                options=dict(maxiter=20000, xatol=1e-8, fatol=1e-10))
    beta = res.x
    S = _num_grad_obs(f_obs, beta)
    A = -_num_grad_obs(lambda b: _num_grad_obs(f_obs, b).sum(axis=0), beta)
    A = (A + A.T) / 2
    g = np.arange(len(y)) if clusters is None else np.asarray(clusters)
    Sc = pd.DataFrame(S).groupby(g).sum().to_numpy()
    G = len(Sc)
    V = np.linalg.inv(A) @ (G / (G - 1) * Sc.T @ Sc) @ np.linalg.inv(A)
    se = np.sqrt(np.diag(V))
    z = beta / se
    tab = pd.DataFrame({"term": names or [f"x{i}" for i in range(X.shape[1])], "coef": beta, "se": se, "z": z,
                        "p": 2 * stats.norm.sf(np.abs(z)), "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se})
    tab["rate_ratio"] = np.exp(tab.coef)
    tab["rr_ci_low"] = np.exp(tab.ci_low)
    tab["rr_ci_high"] = np.exp(tab.ci_high)
    info = dict(loglik=float(-res.fun), alpha=0.0, n=len(y), n_censored=int(cens.sum()), clusters=G,
                converged=bool(res.success))
    return tab, V, info
