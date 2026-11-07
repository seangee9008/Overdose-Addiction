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


def wild_cluster_bootstrap(df, y_col, x_cols, unit, time, j=0, B=9999, seed=0):
    """Fit the TWFE model and bootstrap coefficient x_cols[j]. Returns dict with estimate, CR1 SE,
    WCR bootstrap p-value (H0: b_j = 0), WCU percentile-t 95% CI, n, clusters."""
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
        v = rng.choice([-1.0, 1.0], size=G)[cidx]
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
                n=len(y), clusters=G, B=B, seed=seed)


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
