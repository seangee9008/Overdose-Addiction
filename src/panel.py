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


def wild_cluster_bootstrap(df, y_col, x_cols, unit, time, j=0, B=9999, seed=0, weights="rademacher"):
    """Fit the TWFE model and bootstrap coefficient x_cols[j]. Returns dict with estimate, CR1 SE,
    WCR bootstrap p-value (H0: b_j = 0), WCU percentile-t 95% CI, n, clusters. `weights`: 'rademacher'
    or 'webb'."""
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
    q_lo, q_hi = np.quantile(t_wcu, [0.025, 0.975])
    return dict(term=names[j], estimate=float(beta[j]), se_cr1=float(se[j]), t=float(t_obs), p_wild=p,
                ci_low=float(beta[j] - q_hi * se[j]), ci_high=float(beta[j] - q_lo * se[j]),
                n=len(y), clusters=G, B=B, seed=seed, weights=weights)


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
