"""
Two-way fixed-effects (unit + time) OLS with cluster-robust inference for few clusters
(wild cluster bootstrap, Rademacher weights). Used for TEDS-A analysis (a): state x year panels.

  y_it = a_i + g_t + b' x_it + e_it,   errors clustered by unit i

Inference on one coefficient j:
  * CR1 cluster-robust SE (small-sample factor G/(G-1) * (N-1)/(N-K)).
  * p-value: restricted wild cluster bootstrap (WCR, null b_j = 0 imposed), bootstrap-t.
  * 95% CI: unrestricted wild cluster bootstrap-t (WCU) percentile-t interval.
With few clusters (~40 states) these are more reliable than normal-approximation CR1 intervals
(Cameron, Gelbach & Miller 2008).
"""
import numpy as np
import pandas as pd


def twfe_design(df, x_cols, unit, time):
    """[x columns, unit dummies (first dropped), time dummies (first dropped), intercept]."""
    X = [df[c].to_numpy(float) for c in x_cols]
    names = list(x_cols)
    for col in (unit, time):
        levels = sorted(df[col].unique())
        for lv in levels[1:]:
            X.append((df[col] == lv).to_numpy(float))
            names.append(f"{col}[{lv}]")
    X.append(np.ones(len(df)))
    names.append("(Intercept)")
    return np.column_stack(X), names


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def _cr1_se(X, resid, cluster_idx, n_clusters, XtX_inv):
    k = X.shape[1]
    S = np.zeros((n_clusters, k))
    np.add.at(S, cluster_idx, X * resid[:, None])
    n = len(resid)
    adj = n_clusters / (n_clusters - 1) * (n - 1) / (n - k)
    V = adj * XtX_inv @ (S.T @ S) @ XtX_inv
    return np.sqrt(np.diag(V))


WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def _weights(rng, G, kind):
    """Cluster-level wild-bootstrap weights: Rademacher (+/-1) or Webb's 6-point distribution, which
    gives many more distinct draws when there are few clusters (Webb 2023)."""
    if kind == "rademacher":
        return rng.choice([-1.0, 1.0], size=G)
    if kind == "webb":
        return rng.choice(WEBB, size=G)
    raise ValueError(kind)


def wild_cluster_bootstrap(df, y_col, x_cols, unit, time, j=0, B=9999, seed=0, weights="rademacher", ci="equal"):
    """Fit the TWFE model and bootstrap coefficient x_cols[j]. Returns dict with estimate, CR1 SE,
    WCR bootstrap p-value (H0: b_j = 0), WCU percentile-t 95% CI, n, clusters. `weights`: 'rademacher'
    or 'webb'. `ci`: 'equal' (equal-tailed percentile-t, default) or 'symmetric' (beta +/- q95(|t*|) * se)."""
    y = df[y_col].to_numpy(float)
    X, names = twfe_design(df, x_cols, unit, time)
    clusters, cidx = np.unique(df[unit].to_numpy(), return_inverse=True)
    G = len(clusters)
    if G < 2:
        raise ValueError(f"wild cluster bootstrap needs at least 2 clusters, got {G}")
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta, resid = _ols(X, y)
    se = _cr1_se(X, resid, cidx, G, XtX_inv)
    t_obs = beta[j] / se[j]
    rng = np.random.default_rng(seed)
    # restricted model (b_j = 0) for the p-value
    Xr = np.delete(X, j, axis=1)
    br, ur = _ols(Xr, y)
    fit_r = Xr @ br
    fit_u = X @ beta
    t_wcr = np.empty(B)
    t_wcu = np.empty(B)
    for b in range(B):
        v = _weights(rng, G, weights)[cidx]
        y_r = fit_r + ur * v
        b_r, e_r = _ols(X, y_r)
        t_wcr[b] = b_r[j] / _cr1_se(X, e_r, cidx, G, XtX_inv)[j]
        y_u = fit_u + resid * v
        b_u, e_u = _ols(X, y_u)
        t_wcu[b] = (b_u[j] - beta[j]) / _cr1_se(X, e_u, cidx, G, XtX_inv)[j]
    p = float(np.mean(np.abs(t_wcr) >= abs(t_obs)))
    if ci == "symmetric":
        q_hi = float(np.quantile(np.abs(t_wcu), 0.95))
        q_lo = -q_hi
    else:
        q_lo, q_hi = np.quantile(t_wcu, [0.025, 0.975])
    return dict(term=names[j], estimate=float(beta[j]), se_cr1=float(se[j]), t=float(t_obs), p_wild=p,
                ci_low=float(beta[j] - q_hi * se[j]), ci_high=float(beta[j] - q_lo * se[j]),
                n=len(y), clusters=G, B=B, seed=seed, weights=weights, ci=ci)


def cluster_bootstrap_totals(values, B, seed):
    """State-cluster bootstrap of panel totals. values: array [G clusters, ...]. Each replicate resamples
    G clusters with replacement and sums them. Returns array [B, ...] of resampled totals."""
    rng = np.random.default_rng(seed)
    G = values.shape[0]
    idx = rng.integers(0, G, size=(B, G))
    return values[idx].sum(axis=1)


def simulate_state_totals(rng, G, base_mean=5000.0, mu=-0.4, tau=0.25, noise=0.05, T=2):
    """Synthetic state totals for a start and an end year: state sizes vary (lognormal); each state's log
    change is mu + N(0, tau^2) + small year noise. Returns array [G, 2]."""
    size = base_mean * np.exp(rng.normal(0, 0.8, G))
    change = mu + rng.normal(0, tau, G)
    start = size * np.exp(rng.normal(0, noise, G))
    end = size * np.exp(change + rng.normal(0, noise, G))
    return np.column_stack([start, end])


def superpopulation_log_ratio(mu=-0.4, tau=0.25, noise=0.05, n=2_000_000, seed=0):
    """Target of the ratio-of-totals estimator: log(E[end] / E[start]) over the state superpopulation."""
    rng = np.random.default_rng(seed)
    size = np.exp(rng.normal(0, 0.8, n))
    return float(np.log((size * np.exp(mu + rng.normal(0, tau, n) + rng.normal(0, noise, n))).mean()
                        / (size * np.exp(rng.normal(0, noise, n))).mean()))


def simulate_panel(rng, G=40, T=9, beta=0.6, rho=0.5, sd=0.15):
    """Synthetic state x year panel: unit and time effects in both x and y, AR(1) errors within unit."""
    rows = []
    a = rng.normal(0, 0.5, G)
    g = rng.normal(0, 0.2, T)
    for i in range(G):
        e = np.empty(T)
        e[0] = rng.normal(0, sd)
        for t in range(1, T):
            e[t] = rho * e[t - 1] + rng.normal(0, sd * np.sqrt(1 - rho ** 2))
        x = 0.8 * a[i] + 0.5 * g + rng.normal(0, 0.3, T)
        for t in range(T):
            rows.append(dict(unit=i, time=t, x=x[t], y=a[i] + g[t] + beta * x[t] + e[t]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------
# Studentized (bootstrap-t) state-cluster bootstrap for any smooth function of panel totals.
#   values : array [G, *shape] of per-state contributions (e.g. counts by year x measure)
#   f      : vectorized function mapping totals [..., *shape] -> scalar per leading index
# SE = delete-one-cluster jackknife: se^2 = (G-1)/G * sum_s (theta_(-s) - mean)^2. (The cluster linearization
# SE was 10-14% too small for ratio statistics with very unequal state sizes in simulation; the jackknife SE
# matched the empirical spread.) Every bootstrap replicate recomputes its own jackknife SE, and the 95% CI is
# SYMMETRIC: theta +/- q95(|t*|) * se with t* = (theta* - theta) / se*. In simulations at 15, 28 and 35
# clusters this interval covered at 95.3%, 94.8% and 94.2% (equal-tailed: 92.6-93.9%).

def jackknife_se(Vb, f):
    """Delete-one-cluster jackknife SE of f(totals) for a batch of panels Vb [B, G, *shape]; returns [B]."""
    G = Vb.shape[1]
    loo = f(Vb.sum(axis=1)[:, None] - Vb)                 # [B, G] leave-one-cluster-out estimates
    return np.sqrt((G - 1) / G * ((loo - loo.mean(axis=1, keepdims=True)) ** 2).sum(axis=1))


def studentized_cluster_bootstrap(values, f, B=9999, seed=0, batch=500):
    """Symmetric bootstrap-t over clusters for theta = f(sum of values over clusters), jackknife SE.
    Returns dict(estimate, se, ci_low, ci_high, p_boot, B); p_boot is the symmetric bootstrap-t p-value
    for H0: theta = 0."""
    values = np.asarray(values, float)
    G = values.shape[0]
    if G < 3:
        raise ValueError(f"cluster bootstrap needs at least 3 clusters, got {G}")
    theta = float(f(values.sum(axis=0)[None])[0])
    se = float(jackknife_se(values[None], f)[0])
    rng = np.random.default_rng(seed)
    t = []
    for start in range(0, B, batch):
        nb = min(batch, B - start)
        Vb = values[rng.integers(0, G, size=(nb, G))]
        t.append((f(Vb.sum(axis=1)) - theta) / jackknife_se(Vb, f))
    t = np.concatenate(t)
    t = np.abs(t[np.isfinite(t)])
    q = float(np.quantile(t, 0.95))
    return dict(estimate=theta, se=se, ci_low=theta - q * se, ci_high=theta + q * se,
                p_boot=float(np.mean(t >= abs(theta / se))) if se > 0 else np.nan, B=len(t))


def simulate_state_series(rng, G, years, base_mean=5000.0, mu=-0.08, tau=0.05, noise=0.05):
    """Synthetic [G, T, 2] panel (count, population): each state's log rate has its own linear trend
    mu + N(0, tau^2) plus noise; populations constant within state."""
    T = len(years)
    t = np.arange(T, dtype=float)
    size = base_mean * np.exp(rng.normal(0, 0.8, G))
    slope = mu + rng.normal(0, tau, G)
    counts = size[:, None] * np.exp(slope[:, None] * t[None] + rng.normal(0, noise, (G, T)))
    pop = (size * 2e4)[:, None] * np.ones(T)
    return np.stack([counts, pop], axis=2)


def superpopulation_trend(years, mu=-0.08, tau=0.05):
    """Target of the log-linear trend of the ratio of totals: log E[count_t] = const + mu t + tau^2 t^2 / 2."""
    t = np.arange(len(years), dtype=float)
    y = mu * t + tau ** 2 * t ** 2 / 2
    tc = t - t.mean()
    return float((y * tc).sum() / (tc ** 2).sum())


# ---------------------------------------------------------------------------------------------------
# Wild cluster bootstrap confidence interval by TEST INVERSION (restricted WCR bootstrap; MacKinnon & Webb).
# The 95% CI is the set of b0 that the WCR bootstrap-t test of H0: b_j = b0 does not reject at 5%. The same
# cluster weights are used for every b0. With few clusters this covered better in simulation than the
# unrestricted WCU percentile-t interval (WCU equal-tailed at 15 states: 92.3%).

def _wcr_pvalue(X, y, j, b0, cidx, G, A, M, adj, V):
    """WCR bootstrap-t p-value for H0: b_j = b0. V: [G, B] cluster weights. Vectorized over draws."""
    xj = X[:, j]
    Xr = np.delete(X, j, axis=1)
    y0 = y - b0 * xj
    br, *_ = np.linalg.lstsq(Xr, y0, rcond=None)
    fr = Xr @ br
    ur = y0 - fr
    a = A[j]                                                  # b_j = a @ y

    def se_of(E):                                             # CR1 SE of b_j for residual columns E [n, B]
        S = np.zeros((G,) + E.shape[1:])
        np.add.at(S, cidx, a[:, None] * E if E.ndim == 2 else a * E)
        return np.sqrt(adj * (S ** 2).sum(axis=0))
    e = M @ y
    t_obs = (a @ y - b0) / se_of(e)
    Ys = (b0 * xj + fr)[:, None] + ur[:, None] * V[cidx]      # [n, B]
    t_b = (a @ Ys - b0) / se_of(M @ Ys)
    return float(np.mean(np.abs(t_b) >= abs(t_obs)))


def wild_cluster_ci_inverted(df, y_col, x_cols, unit, time, j=0, B=9999, seed=0, weights="webb", level=0.95,
                             tol=1e-4):
    """Estimate, CR1 SE, WCR p-value (H0: b_j = 0) and the test-inversion CI for coefficient x_cols[j]."""
    y = df[y_col].to_numpy(float)
    X, names = twfe_design(df, x_cols, unit, time)
    _, cidx = np.unique(df[unit].to_numpy(), return_inverse=True)
    G = cidx.max() + 1
    if G < 2:
        raise ValueError(f"wild cluster bootstrap needs at least 2 clusters, got {G}")
    n, k = X.shape
    A = np.linalg.pinv(X)                                     # [k, n]
    M = np.eye(n) - X @ A
    rank = np.linalg.matrix_rank(X)
    adj = G / (G - 1) * (n - 1) / (n - rank)
    rng = np.random.default_rng(seed)
    V = np.stack([_weights(rng, G, weights) for _ in range(B)], axis=1)   # [G, B]
    beta = float(A[j] @ y)
    S = np.zeros(G)
    np.add.at(S, cidx, A[j] * (M @ y))
    se = float(np.sqrt(adj * (S ** 2).sum()))
    alpha = 1 - level

    def p(b0):
        return _wcr_pvalue(X, y, j, b0, cidx, G, A, M, adj, V)

    def bound(direction):
        inside, step = beta, 2 * se
        outside = beta + direction * step
        while p(outside) > alpha:                            # expand until rejected
            inside, step = outside, step * 2
            outside = beta + direction * step
            if step > 1e3 * max(se, 1e-12):
                return np.inf * direction
        while abs(outside - inside) > tol * max(se, 1e-12):   # bisect on the acceptance boundary
            mid = (inside + outside) / 2
            if p(mid) > alpha:
                inside = mid
            else:
                outside = mid
        return (inside + outside) / 2
    return dict(term=names[j], estimate=beta, se_cr1=se, p_wild=p(0.0), ci_low=float(bound(-1)),
                ci_high=float(bound(1)), n=n, clusters=int(G), B=B, seed=seed, weights=weights,
                ci="WCR test inversion")
