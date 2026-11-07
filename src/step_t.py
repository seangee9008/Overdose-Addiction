#!/usr/bin/env python3
"""
Phase 4b: TEDS-A treatment admissions as an independent, AREA-LEVEL signal on opioid addiction.

    python -m src.step_t            # panels, (a), (b), (c), (d), simulation coverage at actual panel sizes

Admissions are not prevalence: they count treatment episodes (not people) at publicly funded facilities,
depend on funding, capacity and referral practice, and miss office-based buprenorphine. Never joined to
NSDUH. TEDS-A here ends in 2023 (combined 2006-2023 file), so it covers the period BEFORE the main
decline in overdose deaths (late 2022 onward) and cannot test that decline itself.

Panels (decided by the project owner)
  P35      : states meeting the completeness rule (admissions reported every year 2015-2023 and total
             admissions never changing > 50% year to year). Admissions-only trends: (a) trends, (b), (c).
  P_deaths : P35 states that also have CDC opioid-specific deaths ('Opioids (T40.0-T40.4,T40.6)') in every
             year 2019-2023. Admissions-vs-deaths: (a) association and (d), window 2019-2023.
  P15      : P35 states with opioid deaths in every year 2015-2023 (sensitivity, 2015-2023).
  All-drug deaths are never substituted for opioid deaths.

Inference
  * (a) association: log(admissions/pop) = state FE + year FE + b log(opioid deaths/pop); wild cluster
    bootstrap by state with Webb weights (src/panel.py), B = 9999; Rademacher shown for comparison.
  * (a) trends, (b), (c), (d): national-panel quantities (ratios of sums over the panel's states) with a
    state-cluster bootstrap (resample states), B = 9999, percentile 95% CI. Admissions are a census of
    reporting facilities, so the uncertainty reflects heterogeneity between states, not sampling.
  * COVID sensitivity: every trend / association is also estimated without 2020.
  * Coverage of both bootstrap methods is checked by simulation at the actual panel sizes
    (sim_coverage.csv).
"""
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import teds, warehouse
from src import survey as sv
from src.panel import (cluster_bootstrap_totals, simulate_panel, simulate_state_totals, superpopulation_log_ratio,
                       wild_cluster_bootstrap)

OUT = "outputs/teds"
SEED = 20261107
B = 9999
YEARS = list(range(2015, 2024))
DEATH_WINDOW = list(range(2019, 2024))
SRC = ("TEDS-A 2006-2023 combined PUF (SAMHSA), admissions 2015-2023; CDC VSRR state file (xkb8-kh2a) opioid deaths "
       "'Opioids (T40.0-T40.4,T40.6)', Dec 12-month-ending; Census state population (intercensal 2015-2019, "
       "Vintage 2025 2020-2023)")
MEASURES = {"primary_opioid": "primary substance heroin / non-Rx methadone / other opiates & synthetics",
            "primary_heroin": "primary substance heroin",
            "primary_other_opiates": "primary substance other opiates & synthetics (incl. fentanyl, Rx opioids, buprenorphine)",
            "any_opioid": "any opioid mention in primary, secondary or tertiary substance"}
AGE_BANDS = {"12-17": (1, 2), "18-24": (3, 4), "25-34": (5, 6), "35-44": (7, 8), "45-54": (9, 10), "55+": (11, 12)}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TEDS_2024_NOTE = ("TEDS-A 2024 is available from SAMHSA as a single-year file (released 2026; no combined 2006-2024 "
                  "file) but is NOT included: these analyses use the combined 2006-2023 file and end in 2023.")
SCOPE_NOTE = ("TEDS-A data here end in 2023, so they cover the period before the main decline in overdose deaths "
              "(late 2022 onward); they describe treatment admissions before and at the start of that decline and do "
              "not test the decline itself.")


# ---------------------------------------------------------------------------------------------- panels
def build_panels(p):
    regions = warehouse.query("select state_fips, region_name from state_regions")
    st = (p.groupby(["state_fips", "state_name"])
           .agg(is_complete=("is_complete", "first"), teds_exclusion_reason=("exclusion_reason", "first"))
           .reset_index().merge(regions, on="state_fips", how="left"))
    deaths = p.assign(has=p.opioid_deaths.notna()).pivot_table(index="state_fips", columns="year", values="has",
                                                               aggfunc="first")
    st["deaths_2019_2023"] = st.state_fips.map(deaths[DEATH_WINDOW].all(axis=1))
    st["deaths_2015_2023"] = st.state_fips.map(deaths[YEARS].all(axis=1))
    st["in_P35"] = st.is_complete
    st["in_P_deaths"] = st.is_complete & st.deaths_2019_2023
    st["in_P15"] = st.is_complete & st.deaths_2015_2023
    pop23 = p[p.year == 2023].set_index("state_fips").population
    st["population_2023"] = st.state_fips.map(pop23)

    def missing(fips, yrs):
        have = set(p[(p.state_fips == fips) & p.opioid_deaths.notna()].year)
        return ", ".join(str(y) for y in yrs if y not in have)

    st["opioid_deaths_missing_2019_2023"] = [missing(f, DEATH_WINDOW) for f in st.state_fips]
    st["opioid_deaths_missing_2015_2023"] = [missing(f, YEARS) for f in st.state_fips]
    st["source"] = SRC
    summ = []
    us_pop = st.population_2023.sum()
    for name, col, window in (("P35 (admissions trends)", "in_P35", "2015-2023"),
                              ("P_deaths (admissions vs deaths)", "in_P_deaths", "2019-2023"),
                              ("P15 (sensitivity)", "in_P15", "2015-2023"),
                              ("all 50 states + DC", None, "")):
        g = st if col is None else st[st[col]]
        r = dict(panel=name, window=window, states=len(g), population_2023=int(g.population_2023.sum()),
                 population_share_of_us=g.population_2023.sum() / us_pop)
        for reg in ("Northeast", "Midwest", "South", "West"):
            r[f"states_{reg}"] = int((g.region_name == reg).sum())
            r[f"pop_share_{reg}"] = g[g.region_name == reg].population_2023.sum() / g.population_2023.sum()
        summ.append(r)
    return st, pd.DataFrame(summ)


def panel_rows(p, states, years):
    return p[p.state_fips.isin(states) & p.year.isin(years)].copy()


def stack(d, cols, years):
    """[G states, T years, K columns] array of panel values (states sorted)."""
    states = sorted(d.state_fips.unique())
    arr = np.zeros((len(states), len(years), len(cols)))
    idx = {s: i for i, s in enumerate(states)}
    for r in d.itertuples():
        if r.year in years:
            for k, c in enumerate(cols):
                arr[idx[r.state_fips], years.index(r.year), k] = getattr(r, c)
    return arr


def pct_ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def boot_p(x):
    """Two-sided bootstrap p-value for H0: statistic = 0 (share of replicates on the other side of 0)."""
    return float(min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0))))


def log_linear_slope(rates, years):
    """OLS slope of log(rate) on year; rates shape [..., T]."""
    t = np.asarray(years, float)
    tc = t - t.mean()
    lr = np.log(rates)
    return ((lr - lr.mean(axis=-1, keepdims=True)) * tc).sum(axis=-1) / (tc ** 2).sum()


# ---------------------------------------------------------------------------------------------- (a)
def analysis_a(p, st):
    p35 = panel_rows(p, st[st.in_P35].state_fips, YEARS)
    cols = list(MEASURES) + ["population"]
    arr = stack(p35, cols, YEARS)
    tot = arr.sum(axis=0)
    rows = []
    for k, m in enumerate(MEASURES):
        rate = 1e5 * tot[:, k] / tot[:, -1]
        for y_i, y in enumerate(YEARS):
            rows.append(dict(panel="P35", measure=m, year=y, states=arr.shape[0], admissions=int(tot[y_i, k]),
                             population=int(tot[y_i, -1]), per_100k=rate[y_i], index_2015=100 * rate[y_i] / rate[0]))
    series = pd.DataFrame(rows)
    # admissions trend (log-linear slope, %/yr), with and without 2020
    boot = cluster_bootstrap_totals(arr, B, SEED + 1)
    trends = []
    for k, m in enumerate(MEASURES):
        for label, yrs in (("2015-2023", YEARS), ("2015-2023 excluding 2020", [y for y in YEARS if y != 2020])):
            ix = [YEARS.index(y) for y in yrs]
            est = float(log_linear_slope(tot[ix, k] / tot[ix, -1], yrs))
            bs = log_linear_slope(boot[:, ix, k] / boot[:, ix, -1], yrs)
            lo, hi = pct_ci(bs)
            trends.append(dict(panel="P35", measure=m, years=label, slope_log_per_year=est,
                               pct_change_per_year=100 * np.expm1(est), ci_low_pct=100 * np.expm1(lo),
                               ci_high_pct=100 * np.expm1(hi), p_boot=boot_p(bs), states=arr.shape[0], B=B,
                               method="log-linear trend of panel admission rate; state-cluster bootstrap"))
    # deaths comparison series (P_deaths 2019-2023 main; P15 2015-2023 sensitivity)
    comp = []
    for pname, col, yrs in (("P_deaths", "in_P_deaths", DEATH_WINDOW), ("P15", "in_P15", YEARS)):
        d = panel_rows(p, st[st[col]].state_fips, yrs)
        a = stack(d, ["primary_opioid", "any_opioid", "opioid_deaths", "population"], yrs).sum(axis=0)
        for y_i, y in enumerate(yrs):
            for k, m in enumerate(("primary_opioid", "any_opioid", "opioid_deaths")):
                r = 1e5 * a[:, k] / a[:, 3]
                comp.append(dict(panel=pname, measure=m, year=y, states=d.state_fips.nunique(), per_100k=r[y_i],
                                 index_first_year=100 * r[y_i] / r[0], first_year=yrs[0]))
    comp = pd.DataFrame(comp)
    # association: TWFE, wild cluster bootstrap (Webb main; Rademacher comparison)
    assoc = []
    specs = [("P_deaths", "in_P_deaths", DEATH_WINDOW, "main"), ("P15", "in_P15", YEARS, "sensitivity")]
    for s_i, (pname, col, yrs, role) in enumerate(specs):
        for excl in (False, True):
            use = [y for y in yrs if not (excl and y == 2020)]
            d = panel_rows(p, st[st[col]].state_fips, use)
            for m_i, m in enumerate(("primary_opioid", "any_opioid")):
                d["log_adm_rate"] = np.log(d[m] / d.population)
                d["log_death_rate"] = np.log(d.opioid_deaths / d.population)
                wts = ["webb", "rademacher"] if (role == "main" and not excl and m == "primary_opioid") else ["webb"]
                for w in wts:
                    r = wild_cluster_bootstrap(d, "log_adm_rate", ["log_death_rate"], "state_fips", "year", B=B,
                                               seed=SEED + 100 * s_i + 10 * excl + m_i, weights=w)
                    assoc.append(dict(panel=pname, role=role, years=f"{use[0]}-{use[-1]}" + (" excl. 2020" if excl else ""),
                                      admissions_measure=m, weights=w, elasticity=r["estimate"], ci_low=r["ci_low"],
                                      ci_high=r["ci_high"], p_wild=r["p_wild"], se_cr1=r["se_cr1"],
                                      n_state_years=r["n"], states=r["clusters"], B=B, seed=r["seed"]))
    assoc = pd.DataFrame(assoc)
    assoc["model"] = "log(admissions/pop) = state FE + year FE + b log(opioid deaths/pop); wild cluster bootstrap by state"
    return series, pd.DataFrame(trends), comp, assoc


# ---------------------------------------------------------------------------------------------- (b)
def analysis_b(p, st):
    d = panel_rows(p, st[st.in_P35].state_fips, YEARS)
    cols = ["primary_heroin", "primary_methadone_nonrx", "primary_other_opiates", "primary_opioid"]
    arr = stack(d, cols, YEARS)
    tot = arr.sum(axis=0)
    shares = []
    for y_i, y in enumerate(YEARS):
        for k, m in enumerate(cols[:3]):
            shares.append(dict(panel="P35", year=y, component=m, admissions=int(tot[y_i, k]),
                               primary_opioid=int(tot[y_i, 3]), share=tot[y_i, k] / tot[y_i, 3]))
    boot = cluster_bootstrap_totals(arr, B, SEED + 2)
    out = []
    for k, m in enumerate(cols[:3]):
        sh, bsh = tot[:, k] / tot[:, 3], boot[:, :, k] / boot[:, :, 3]
        for a_y, b_y in ((2015, 2023), (2019, 2023)):
            ia, ib = YEARS.index(a_y), YEARS.index(b_y)
            est, bs = sh[ib] - sh[ia], bsh[:, ib] - bsh[:, ia]
            lo, hi = pct_ci(bs)
            out.append(dict(component=m, statistic=f"share change {a_y}->{b_y} (percentage points)",
                            estimate=100 * est, ci_low=100 * lo, ci_high=100 * hi, p_boot=boot_p(bs)))
        for label, yrs in (("2015-2023", YEARS), ("2015-2023 excluding 2020", [y for y in YEARS if y != 2020])):
            ix = [YEARS.index(y) for y in yrs]
            odds = sh[ix] / (1 - sh[ix])
            bodds = bsh[:, ix] / (1 - bsh[:, ix])
            est, bs = float(log_linear_slope(odds, yrs)), log_linear_slope(bodds, yrs)
            lo, hi = pct_ci(bs)
            out.append(dict(component=m, statistic=f"log-odds trend per year, {label}", estimate=est, ci_low=lo,
                            ci_high=hi, p_boot=boot_p(bs)))
    out = pd.DataFrame(out)
    out["states"], out["B"] = arr.shape[0], B
    out["method"] = "share of primary-opioid admissions, P35 panel totals; state-cluster bootstrap"
    return pd.DataFrame(shares), out


# ---------------------------------------------------------------------------------------------- (c)
def analysis_c(st):
    states = list(st[st.in_P35].state_fips)
    a = warehouse.query("select * from teds_state_year_age where year between 2015 and 2023")
    pop = warehouse.query("select state_fips, year, teds_age_group as age_group, population from population_state_age "
                          "where teds_age_group between 1 and 12")
    a = a[a.state_fips.isin(states)].merge(pop, on=["state_fips", "year", "age_group"], how="left")
    if a.population.isna().any():
        raise ValueError("missing population for some state x year x age cells")
    band_of = {g: band for band, gs in AGE_BANDS.items() for g in gs}
    a["band"] = a.age_group.map(band_of)
    g = a.groupby(["state_fips", "year", "band"])[["primary_opioid_first_time", "primary_opioid_noprior_missing",
                                                   "primary_opioid", "population"]].sum().reset_index()
    rows, trends = [], []
    for b_i, band in enumerate(AGE_BANDS):
        d = g[g.band == band]
        arr = stack(d, ["primary_opioid_first_time", "primary_opioid_noprior_missing", "primary_opioid", "population"], YEARS)
        tot = arr.sum(axis=0)
        lower = tot[:, 0] / tot[:, 3]
        upper = (tot[:, 0] + tot[:, 1]) / tot[:, 3]
        for y_i, y in enumerate(YEARS):
            rows.append(dict(age_band=band, year=y, first_time=int(tot[y_i, 0]), noprior_missing=int(tot[y_i, 1]),
                             primary_opioid=int(tot[y_i, 2]), population=int(tot[y_i, 3]),
                             first_time_per_100k_lower=1e5 * lower[y_i], first_time_per_100k_upper=1e5 * upper[y_i],
                             share_noprior_missing=tot[y_i, 1] / tot[y_i, 2]))
        boot = cluster_bootstrap_totals(arr, B, SEED + 30 + b_i)
        for bound, est_rate, b_rate in (("lower (missing = not first-time)", lower, boot[:, :, 0] / boot[:, :, 3]),
                                        ("upper (missing = first-time)", upper,
                                         (boot[:, :, 0] + boot[:, :, 1]) / boot[:, :, 3])):
            for a_y, b_y in ((2015, 2023), (2019, 2023)):
                ia, ib = YEARS.index(a_y), YEARS.index(b_y)
                est, bs = np.log(est_rate[ib] / est_rate[ia]), np.log(b_rate[:, ib] / b_rate[:, ia])
                lo, hi = pct_ci(bs)
                trends.append(dict(age_band=band, bound=bound, statistic=f"log change {a_y}->{b_y}", estimate=float(est),
                                   pct_change=100 * np.expm1(est), ci_low_pct=100 * np.expm1(lo),
                                   ci_high_pct=100 * np.expm1(hi), p_boot=boot_p(bs)))
            for label, yrs in (("trend 2015-2023", YEARS), ("trend 2015-2023 excluding 2020", [y for y in YEARS if y != 2020])):
                ix = [YEARS.index(y) for y in yrs]
                est, bs = float(log_linear_slope(est_rate[ix], yrs)), log_linear_slope(b_rate[:, ix], yrs)
                lo, hi = pct_ci(bs)
                trends.append(dict(age_band=band, bound=bound, statistic=f"log-linear {label}, per year", estimate=est,
                                   pct_change=100 * np.expm1(est), ci_low_pct=100 * np.expm1(lo),
                                   ci_high_pct=100 * np.expm1(hi), p_boot=boot_p(bs)))
    tr = pd.DataFrame(trends)
    tr["states"], tr["B"] = len(states), B
    tr["method"] = ("first-time (NOPRIOR = 0) primary-opioid admissions per 100k age-band population, P35 panel totals; "
                    "bounds for missing NOPRIOR; state-cluster bootstrap")
    return pd.DataFrame(rows), tr


# ---------------------------------------------------------------------------------------------- (d)
def analysis_d(p, st):
    rows = []
    specs = [("P_deaths", "in_P_deaths", DEATH_WINDOW, (2019, 2021, 2022), "main"),
             ("P15", "in_P15", YEARS, (2015, 2019, 2021, 2022), "sensitivity")]
    for s_i, (pname, col, yrs, starts, role) in enumerate(specs):
        d = panel_rows(p, st[st[col]].state_fips, yrs)
        cols = list(MEASURES) + ["opioid_deaths", "population"]
        arr = stack(d, cols, yrs)
        tot = arr.sum(axis=0)
        boot = cluster_bootstrap_totals(arr, B, SEED + 50 + s_i)
        k_d, k_p = cols.index("opioid_deaths"), cols.index("population")
        end = yrs.index(2023)
        for start in starts:
            i0 = yrs.index(start)
            L_death = np.log((tot[end, k_d] / tot[end, k_p]) / (tot[i0, k_d] / tot[i0, k_p]))
            bL_death = np.log((boot[:, end, k_d] / boot[:, end, k_p]) / (boot[:, i0, k_d] / boot[:, i0, k_p]))
            block = []
            for k, m in enumerate(MEASURES):
                L_adm = np.log((tot[end, k] / tot[end, k_p]) / (tot[i0, k] / tot[i0, k_p]))
                bL_adm = np.log((boot[:, end, k] / boot[:, end, k_p]) / (boot[:, i0, k] / boot[:, i0, k_p]))
                est, bs = float(L_death - L_adm), bL_death - bL_adm
                lo, hi = pct_ci(bs)
                block.append(dict(panel=pname, role=role, measure=m, start=start, end=2023, states=arr.shape[0],
                                  death_pct_change=100 * np.expm1(L_death), admission_pct_change=100 * np.expm1(L_adm),
                                  d=est, ci_low=lo, ci_high=hi, p_boot=boot_p(bs), B=B))
            ps = [r["p_boot"] for r in block]
            for r, ph in zip(block, sv.holm(ps)):
                r["p_holm"] = float(ph)
            rows += block
    out = pd.DataFrame(rows)
    out["estimand"] = "d = log(opioid death rate end/start) - log(admission rate end/start); d > 0: deaths rose relative to admissions"
    return out


# ---------------------------------------------------------------------------------------------- simulation
def sim_coverage(st, n_sims=300):
    """Coverage of the two bootstrap methods at the ACTUAL panel sizes (synthetic data, planted truth)."""
    rows = []
    for pname, col, T in (("P_deaths", "in_P_deaths", len(DEATH_WINDOW)), ("P15", "in_P15", len(YEARS))):
        G = int(st[col].sum())
        for w in ("webb", "rademacher"):
            rng = np.random.default_rng([SEED, G, T, len(w)])
            cover = reject = 0
            for s in range(n_sims):
                r = wild_cluster_bootstrap(simulate_panel(rng, G=G, T=T, beta=0.6), "y", ["x"], "unit", "time",
                                           B=199, seed=s, weights=w)
                cover += r["ci_low"] <= 0.6 <= r["ci_high"]
                r0 = wild_cluster_bootstrap(simulate_panel(rng, G=G, T=T, beta=0.0), "y", ["x"], "unit", "time",
                                            B=199, seed=50_000 + s, weights=w)
                reject += r0["p_wild"] < 0.05
            rows.append(dict(method=f"wild cluster bootstrap ({w}), TWFE slope", panel=pname, clusters=G, years=T,
                             sims=n_sims, B=199, coverage_95=cover / n_sims, null_rejection_5pct=reject / n_sims))
    truth = superpopulation_log_ratio()
    for pname, col in (("P35", "in_P35"), ("P_deaths", "in_P_deaths"), ("P15", "in_P15")):
        G = int(st[col].sum())
        rng = np.random.default_rng([SEED, G, 7])
        cover = 0
        n = 1000
        for s in range(n):
            vals = simulate_state_totals(rng, G)
            t = cluster_bootstrap_totals(vals, 999, s)
            lo, hi = pct_ci(np.log(t[:, 1]) - np.log(t[:, 0]))
            cover += lo <= truth <= hi
        rows.append(dict(method="state-cluster bootstrap, percentile CI of log ratio of totals", panel=pname, clusters=G,
                         years=2, sims=n, B=999, coverage_95=cover / n, null_rejection_5pct=np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- figures
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def plot_a(series, comp):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    _style(ax)
    for c, m in zip(COLORS, MEASURES):
        s = series[series.measure == m]
        ax.plot(s.year, s.index_2015, "o-", color=c, lw=2, ms=5, label=m.replace("_", " "))
    ax.axvspan(2019.5, 2020.5, color="#e4e3df", zorder=0)
    ax.set_title(f"P35: opioid admission rates (index 2015 = 100), {int(series.states.iloc[0])} states", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5)
    ax = axes[1]
    _style(ax)
    c = comp[comp.panel == "P_deaths"]
    for col, m, lab in ((COLORS[0], "primary_opioid", "primary-opioid admissions"), (COLORS[3], "any_opioid", "any-opioid admissions"),
                        ("#52514e", "opioid_deaths", "opioid overdose deaths")):
        s = c[c.measure == m]
        ax.plot(s.year, s.index_first_year, "o-", color=col, lw=2, ms=5, label=lab)
    ax.set_title(f"P_deaths: admissions vs deaths (index 2019 = 100), {int(c.states.iloc[0])} states", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5)
    fig.text(0.01, 0.01, SCOPE_NOTE + " Shaded: 2020. Admissions are episodes at publicly funded facilities, not people.",
             fontsize=6.3, color="#52514e", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{OUT}/a_trends.png", dpi=150)
    plt.close(fig)


def plot_b(shares):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    _style(ax)
    for c, m, lab in zip(COLORS, ("primary_heroin", "primary_other_opiates", "primary_methadone_nonrx"),
                         ("heroin", "other opiates & synthetics (incl. fentanyl, Rx, buprenorphine)", "non-Rx methadone")):
        s = shares[shares.component == m]
        ax.plot(s.year, 100 * s.share, "o-", color=c, lw=2, ms=5, label=lab)
    ax.set_ylabel("% of primary-opioid admissions")
    ax.set_title("TEDS-A (b): composition of primary-opioid admissions, P35 panel", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/b_shares.png", dpi=150)
    plt.close(fig)


def plot_c(ft):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, band in zip(axes.flat, AGE_BANDS):
        _style(ax)
        s = ft[ft.age_band == band]
        ax.fill_between(s.year, s.first_time_per_100k_lower, s.first_time_per_100k_upper, color=COLORS[0], alpha=0.2, lw=0)
        ax.plot(s.year, s.first_time_per_100k_lower, "o-", color=COLORS[0], lw=2, ms=4)
        ax.set_title(f"ages {band}", loc="left", fontsize=9)
    fig.suptitle("TEDS-A (c): first-time primary-opioid admissions per 100k (line: NOPRIOR = 0; band: + missing NOPRIOR), "
                 "P35 panel", x=0.01, ha="left", fontsize=10)
    fig.text(0.01, 0.01, "First-time admissions are not new cases of addiction: entry also depends on treatment access, "
             "capacity and time from onset to treatment.", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(f"{OUT}/c_first_time.png", dpi=150)
    plt.close(fig)


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)                         # no stale files from earlier versions of (a)
    os.makedirs(OUT)
    p = teds.panel()
    p = p[p.year.between(2015, 2023)]
    st, summ = build_panels(p)
    st.to_csv(f"{OUT}/panels_states.csv", index=False)
    summ.to_csv(f"{OUT}/panels_summary.csv", index=False)
    series, trends, comp, assoc = analysis_a(p, st)
    series.to_csv(f"{OUT}/a_admissions_series_P35.csv", index=False)
    trends.to_csv(f"{OUT}/a_admissions_trend_P35.csv", index=False)
    comp.to_csv(f"{OUT}/a_admissions_vs_deaths_series.csv", index=False)
    assoc.to_csv(f"{OUT}/a_association.csv", index=False)
    plot_a(series, comp)
    shares, b_tests = analysis_b(p, st)
    shares.to_csv(f"{OUT}/b_shares_P35.csv", index=False)
    b_tests.to_csv(f"{OUT}/b_share_changes.csv", index=False)
    plot_b(shares)
    ft, c_tests = analysis_c(st)
    ft.to_csv(f"{OUT}/c_first_time_by_age_P35.csv", index=False)
    c_tests.to_csv(f"{OUT}/c_first_time_changes.csv", index=False)
    plot_c(ft)
    d = analysis_d(p, st)
    d.to_csv(f"{OUT}/d_difference_of_declines.csv", index=False)
    cov = sim_coverage(st)
    cov.to_csv(f"{OUT}/sim_coverage.csv", index=False)
    with open(f"{OUT}/meta.json", "w") as fh:
        json.dump(dict(seed=SEED, B=B, years=[YEARS[0], YEARS[-1]], death_window=[DEATH_WINDOW[0], DEATH_WINDOW[-1]],
                       completeness_rule="TEDS admissions in every year 2015-2023 and total admissions never change by "
                                         "more than 50% year to year (ratio within [0.5, 1.5])",
                       teds_2024=TEDS_2024_NOTE, scope=SCOPE_NOTE, source=SRC,
                       method=__doc__.split("Panels (decided")[0].strip()), fh, indent=2)
    print("done")


if __name__ == "__main__":
    main()
