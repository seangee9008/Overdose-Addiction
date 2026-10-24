"""
Survey-weighted logistic regression with design-based (linearization) standard errors.

Same estimator as R survey::svyglm(family = quasibinomial) with a stratified, clustered design
(with-replacement PSU approximation):

  Point estimates: maximize the weighted pseudo-log-likelihood  sum_i w_i [y_i log mu_i + (1-y_i) log(1-mu_i)]
                   (Newton-Raphson / IRLS).
  Variance:        V = A^-1 B A^-1  ("sandwich")
                   A = sum_i w_i mu_i (1-mu_i) x_i x_i'                 (weighted information)
                   u_i = w_i (y_i - mu_i) x_i                            (score contributions)
                   U_hj = sum of u_i in PSU j of stratum h
                   B = sum_h n_h/(n_h-1) sum_j (U_hj - mean_j U_hj)(U_hj - mean_j U_hj)'
  Tests:           t = beta / SE on (design df - p + 1) degrees of freedom, as svyglm does.
"""
import numpy as np
import pandas as pd
from scipy import stats


def design_matrix(df, numeric=(), categorical=None):
    """Intercept + numeric columns + treatment-coded dummies.
    categorical: dict {column: reference level}; levels sorted, reference dropped."""
    cols, names = [np.ones(len(df))], ["(Intercept)"]
    for c in numeric:
        cols.append(df[c].to_numpy(float))
        names.append(c)
    for c, ref in (categorical or {}).items():
        for lvl in sorted(df[c].dropna().unique()):
            if lvl == ref:
                continue
            cols.append((df[c] == lvl).to_numpy(float))
            names.append(f"{c}[{lvl}]")
    return np.column_stack(cols), names


def svy_logit(X, y, w, strata, psu, design_df, names=None, tol=1e-10, max_iter=50):
    X, y, w = np.asarray(X, float), np.asarray(y, float), np.asarray(w, float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        A = (X * (w * mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.solve(A, X.T @ (w * (y - mu)))
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise RuntimeError("svy_logit did not converge")
    mu = 1 / (1 + np.exp(-(X @ beta)))
    A = (X * (w * mu * (1 - mu))[:, None]).T @ X
    u = X * (w * (y - mu))[:, None]
    U = pd.DataFrame(u).groupby([np.asarray(strata), np.asarray(psu)]).sum()
    B = np.zeros((X.shape[1], X.shape[1]))
    for _, Uh in U.groupby(level=0):
        n_h = len(Uh)
        if n_h < 2:
            raise ValueError("stratum with a single PSU")
        d = Uh.to_numpy() - Uh.to_numpy().mean(axis=0)
        B += n_h / (n_h - 1) * d.T @ d
    Ainv = np.linalg.inv(A)
    V = Ainv @ B @ Ainv
    se = np.sqrt(np.diag(V))
    dof = design_df - X.shape[1] + 1
    t = beta / se
    tcrit = stats.t.ppf(0.975, dof)
    out = pd.DataFrame({"term": names or [f"x{i}" for i in range(len(beta))], "coef": beta, "se": se,
                        "t": t, "p": 2 * stats.t.sf(np.abs(t), dof),
                        "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se})
    out["odds_ratio"] = np.exp(out.coef)
    out["or_ci_low"] = np.exp(out.ci_low)
    out["or_ci_high"] = np.exp(out.ci_high)
    out["df"] = dof
    return out, V
