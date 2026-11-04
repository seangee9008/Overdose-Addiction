"""
Segmented (broken-stick) log-linear models for TRUE monthly death counts (CDC WONDER), with
month-of-year seasonality, for Phase 4a robustness check R5.

  log E[y_t] = b0 + b1*t + sum_k c_k*(t - tau_k)_+ + month-of-year effects

Each break tau_k is a change in the log-linear trend (a "kink"; the fitted curve is continuous).
Fitted by Poisson IRLS; overdispersion handled quasi-likelihood style with the Pearson dispersion
phi = X2 / (n - p). Break-date confidence set by profiling the scaled deviance over candidate months:
  {tau : (D(tau) - D_min) / phi_hat <= chi2_1(0.95)}.
Cross-checks: moving-block bootstrap of Pearson residuals (block >= 12, fixed seed), and a bootstrap
test of 2 breaks vs 1 break under the 1-break null.
"""
import itertools

import numpy as np
from scipy import stats

CHI2_95 = stats.chi2.ppf(0.95, 1)


def design(n, taus, months=None):
    """Columns: intercept, t, (t - tau)_+ for each tau, month-of-year dummies (January = reference)."""
    t = np.arange(n, dtype=float)
    cols = [np.ones(n), t] + [np.maximum(t - tau, 0.0) for tau in taus]
    if months is not None:
        months = np.asarray(months)
        cols += [(months == m).astype(float) for m in range(2, 13)]
    return np.column_stack(cols)


def poisson_irls(X, y, tol=1e-10, max_iter=100):
    """Poisson log-link GLM by IRLS. Returns (beta, mu, deviance, pearson_dispersion)."""
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.mean())
    for _ in range(max_iter):
        eta = X @ beta
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        WX = X * mu[:, None]
        new = np.linalg.solve(X.T @ WX, WX.T @ z)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    mu = np.exp(X @ beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.sum(np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))
    phi = np.sum((y - mu) ** 2 / mu) / (len(y) - X.shape[1])
    return beta, mu, float(dev), float(phi)


def candidates(n, k, min_seg=12):
    """All break tuples (sorted) leaving >= min_seg months at each end and between breaks."""
    grid = range(min_seg, n - min_seg + 1)
    return [c for c in itertools.combinations(grid, k) if all(b - a >= min_seg for a, b in zip(c, c[1:]))]


def profile(y, k, months=None, min_seg=12):
    """Fit every candidate break tuple. Returns dict(best, dev, phi, beta, mu, table[(taus, dev)])."""
    y = np.asarray(y, float)
    n = len(y)
    table = []
    for taus in candidates(n, k, min_seg):
        _, _, dev, _ = poisson_irls(design(n, taus, months), y)
        table.append((taus, dev))
    best, dmin = min(table, key=lambda r: r[1])
    beta, mu, dev, phi = poisson_irls(design(n, best, months), y)
    return dict(best=best, dev=dev, phi=phi, beta=beta, mu=mu, table=table)


def profile_ci(prof, which=0):
    """Profile confidence set for break `which`: all candidate values of that break appearing in any
    tuple whose scaled deviance increase is within chi2_1(0.95). Returns (low, high) indices."""
    ok = [taus[which] for taus, dev in prof["table"] if (dev - prof["dev"]) / prof["phi"] <= CHI2_95]
    return min(ok), max(ok)


def mbb_resample(resid, L, rng):
    n = len(resid)
    starts = rng.integers(0, n - L + 1, size=int(np.ceil(n / L)))
    return np.concatenate([resid[s:s + L] for s in starts])[:n]


def simulate_from_fit(mu, phi, resid, L, rng):
    """New series from a fit: mu + sqrt(phi*mu) * (block-resampled standardized Pearson residuals)."""
    r = mbb_resample(resid, L, rng)
    return np.maximum(0.0, np.round(mu + np.sqrt(phi * mu) * r))


def pearson_std(y, prof):
    r = (np.asarray(y, float) - prof["mu"]) / np.sqrt(prof["phi"] * prof["mu"])
    return (r - r.mean()) / r.std(ddof=1)


def bootstrap_break_dates(y, k, months, B, L, seed, min_seg=12):
    """Block-bootstrap distribution of the best break tuple (refit on series simulated from the fit)."""
    prof = profile(y, k, months, min_seg)
    rng = np.random.default_rng(seed)
    resid = pearson_std(y, prof)
    return prof, np.array([profile(simulate_from_fit(prof["mu"], prof["phi"], resid, L, rng), k, months, min_seg)["best"]
                           for _ in range(B)])


def bootstrap_test_extra_break(y, months, B, L, seed, k0=1, min_seg=12):
    """Test k0+1 breaks vs k0 breaks. Statistic: (D_k0 - D_k0+1) / phi_(k0+1). Null distribution from
    series simulated (block-resampled residuals) under the k0-break fit. Returns (stat, p, prof0, prof1)."""
    p0 = profile(y, k0, months, min_seg)
    p1 = profile(y, k0 + 1, months, min_seg)
    stat = (p0["dev"] - p1["dev"]) / p1["phi"]
    rng = np.random.default_rng(seed)
    resid = pearson_std(y, p0)
    null = []
    for _ in range(B):
        ys = simulate_from_fit(p0["mu"], p0["phi"], resid, L, rng)
        a, b = profile(ys, k0, months, min_seg), profile(ys, k0 + 1, months, min_seg)
        null.append((a["dev"] - b["dev"]) / b["phi"])
    p = (1 + np.sum(np.array(null) >= stat)) / (B + 1)
    return float(stat), float(p), p0, p1
