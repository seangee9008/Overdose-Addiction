"""
Design-based estimates of weighted proportions for stratified, clustered surveys (e.g. NSDUH).

Method: Taylor-series linearization, with-replacement PSU approximation -- the default in
SUDAAN, Stata `svy`, and R `survey`.

  p_hat = sum(w*d*y) / sum(w*d)                      (d = 1 if the respondent is in the domain)
  z_i   = w_i * d_i * (y_i - p_hat) / sum(w*d)        (linearized score)
  Z_hj  = sum of z_i in PSU j of stratum h
  Var   = sum_h  n_h/(n_h-1) * sum_j (Z_hj - mean_j Z_hj)^2

Domains (subgroups) are estimated on the FULL sample with d as an indicator, never by subsetting,
so every stratum/PSU stays in the variance calculation.

Contrasts between two domains (e.g. 2024 vs 2021) linearize the contrast itself, so any
covariance from shared PSUs (NSDUH years share one sample design) is included automatically.

Confidence intervals: computed on the logit scale and back-transformed (as SAMHSA does for
NSDUH percentages), with a t critical value on the design degrees of freedom.
"""
import numpy as np
import pandas as pd
from scipy import stats


def _psu_variance(z, strata, psu):
    """Linearization variance from per-record scores z."""
    tot = pd.DataFrame({"h": strata, "j": psu, "z": z}).groupby(["h", "j"])["z"].sum()
    var = 0.0
    for _, zh in tot.groupby(level=0):
        n_h = len(zh)
        if n_h < 2:
            raise ValueError("stratum with a single PSU; variance undefined")
        var += n_h / (n_h - 1) * ((zh - zh.mean()) ** 2).sum()
    return var


def _scores(y, w, d):
    """Return (p_hat, linearized score vector) for the domain d."""
    keep = d & ~np.isnan(y)
    wd = np.where(keep, w, 0.0)
    yy = np.where(keep, y, 0.0)
    W = wd.sum()
    if W == 0:
        raise ValueError("empty domain")
    p = (wd * yy).sum() / W
    return p, wd * (yy - p) / W, int(keep.sum())


def _logit_ci(p, se, tcrit):
    if p <= 0 or p >= 1:
        return np.nan, np.nan
    lg, se_lg = np.log(p / (1 - p)), se / (p * (1 - p))
    lo, hi = lg - tcrit * se_lg, lg + tcrit * se_lg
    return 1 / (1 + np.exp(-lo)), 1 / (1 + np.exp(-hi))


def proportion(df, y, weight, strata, psu, domain=None, df_design=None, level=0.95):
    """Weighted proportion of y == 1 in a domain, with design-based SE and logit 95% CI.

    domain: boolean Series/array (True = in subgroup) or None for everyone.
    Records with y missing are excluded from the domain (not counted as 0).
    Returns dict(p, se, ci_low, ci_high, n, df).
    """
    yv = df[y].to_numpy(float)
    w = df[weight].to_numpy(float)
    d = np.ones(len(df), bool) if domain is None else np.asarray(domain, bool)
    p, z, n = _scores(yv, w, d)
    se = np.sqrt(_psu_variance(z, df[strata].to_numpy(), df[psu].to_numpy()))
    dof = df_design if df_design is not None else _design_df(df, strata, psu)
    lo, hi = _logit_ci(p, se, stats.t.ppf(0.5 + level / 2, dof))
    return dict(p=p, se=se, ci_low=lo, ci_high=hi, n=n, df=dof)


def contrast(df, y, weight, strata, psu, domain_a, domain_b, df_design=None, level=0.95):
    """Compare a proportion between two domains (b vs a), accounting for shared design.

    Returns the difference p_b - p_a and the log ratio log(p_b / p_a), each with SE,
    95% CI, t statistic and two-sided p-value on the design degrees of freedom.
    """
    yv = df[y].to_numpy(float)
    w = df[weight].to_numpy(float)
    h, j = df[strata].to_numpy(), df[psu].to_numpy()
    pa, za, na = _scores(yv, w, np.asarray(domain_a, bool))
    pb, zb, nb = _scores(yv, w, np.asarray(domain_b, bool))
    dof = df_design if df_design is not None else _design_df(df, strata, psu)
    tcrit = stats.t.ppf(0.5 + level / 2, dof)
    out = dict(p_a=pa, p_b=pb, n_a=na, n_b=nb, df=dof)
    for name, est, z in [("diff", pb - pa, zb - za),
                         ("log_ratio", np.log(pb / pa), zb / pb - za / pa)]:
        se = np.sqrt(_psu_variance(z, h, j))
        t = est / se
        out.update({name: est, f"{name}_se": se, f"{name}_ci_low": est - tcrit * se,
                    f"{name}_ci_high": est + tcrit * se, f"{name}_t": t,
                    f"{name}_pvalue": 2 * stats.t.sf(abs(t), dof)})
    return out


def _design_df(df, strata, psu):
    """Number of PSUs minus number of strata."""
    return df.groupby([strata, psu]).ngroups - df[strata].nunique()


def by_group(df, y, weight, strata, psu, group, df_design=None):
    """proportion() for each level of `group` (domain estimation on the full sample)."""
    rows = []
    for g in sorted(df[group].dropna().unique()):
        r = proportion(df, y, weight, strata, psu, domain=(df[group] == g).to_numpy(), df_design=df_design)
        rows.append({group: g, **r})
    return pd.DataFrame(rows)
