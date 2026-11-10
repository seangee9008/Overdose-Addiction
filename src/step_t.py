#!/usr/bin/env python3
"""
Phase 4b: TEDS-A treatment admissions as an independent, AREA-LEVEL signal on opioid addiction.

    python -m src.step_t            # harmonization check, panels, (a)-(d), coverage sims, ledger rows T1-T4

Admissions are not prevalence: they count treatment episodes (not people) at publicly funded facilities,
depend on funding, capacity and referral practice, and miss office-based buprenorphine. Never joined to
NSDUH. TEDS-A now runs through 2024 (combined 2006-2023 file + 2024 single-year file, identical codes),
so 2023 -> 2024 overlaps part of the decline in overdose deaths (which began in late 2022).

Specifications
  main          : 2015-2024; completeness rule applied over 2015-2024; death window 2019-2024.
  through_2023  : 2015-2023; completeness rule over 2015-2023; death window 2019-2023 (previous results,
                  kept as sensitivity rows).
Panels (per specification; decided by the project owner)
  P35      : states meeting the completeness rule (admissions reported every year of the window and total
             admissions never changing > 50% year to year). Admissions-only: (a) trends, (b), (c).
  P_deaths : P35 states with CDC opioid-specific deaths ('Opioids (T40.0-T40.4,T40.6)') in every year of the
             death window. Admissions-vs-deaths: (a) association and (d).
  P15      : P35 states with opioid deaths in every year from 2015 (sensitivity).
  All-drug deaths are never substituted for opioid deaths.

Inference
  * (a) association (a regression): two-way FE, restricted wild cluster bootstrap by state, Webb weights,
    B = 9999; p-value from the WCR bootstrap-t test, 95% CI by inverting that test (the unrestricted percentile-t
    interval covered 92.3% at 15 states in simulation; the inverted interval 94.2%).
  * Everything that is a function of panel totals ((a) trends, (b), (c), (d)): symmetric studentized
    (bootstrap-t) state-cluster bootstrap with a delete-one-state jackknife SE, B = 9999 (src/panel.py).
  * Coverage of each method is simulated at each panel's actual number of states (target >= 94%).
  * COVID: every trend is also estimated without 2020.

(c) baseline: a reporting artifact in prior-treatment coding (NOPRIOR) inflates first-time admissions in
2018-2020, driven by California (first-time share 36-41% in 2015-2017, 87-97% in 2018-2020, ~35% from
2022) and a lasting 2018 step in Indiana. As instructed, 2019 is the baseline for (c); note that 2019 itself
falls inside California's anomaly. Sensitivity: (c) without every state flagged by a data rule (first-time
share changing > 25 percentage points in any year), with 2015 and 2019 baselines.
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
from src.panel import (simulate_panel, simulate_state_series, simulate_state_totals, studentized_cluster_bootstrap,
                       superpopulation_trend, wild_cluster_ci_inverted)
from src.teds_codebooks import compare_codebooks, compare_data

OUT = "outputs/teds"
SEED = 20261109
B = 9999
SPECS = {"main": dict(end=2024, complete_col="is_complete", reason_col="exclusion_reason", label="2015-2024 (main)"),
         "through_2023": dict(end=2023, complete_col="is_complete_2015_2023", reason_col="exclusion_reason_2015_2023",
                              label="2015-2023 (sensitivity)")}
MEASURES = {"primary_opioid": "primary substance heroin / non-Rx methadone / other opiates & synthetics",
            "primary_heroin": "primary substance heroin",
            "primary_other_opiates": "primary substance other opiates & synthetics (incl. fentanyl, Rx opioids, buprenorphine)",
            "any_opioid": "any opioid mention in primary, secondary or tertiary substance"}
AGE_BANDS = {"12-17": (1, 2), "18-24": (3, 4), "25-34": (5, 6), "35-44": (7, 8), "45-54": (9, 10), "55+": (11, 12)}
CODING_SHIFT_PP = 0.25          # data rule for NOPRIOR coding shifts (first-time share change in percentage points)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SRC = ("TEDS-A (SAMHSA): combined 2006-2023 PUF (2015-2023) + 2024 single-year PUF; CDC VSRR state file (xkb8-kh2a) "
       "opioid deaths 'Opioids (T40.0-T40.4,T40.6)', Dec 12-month-ending; Census state population (intercensal "
       "2015-2019, Vintage 2025 2020-2024)")
SCOPE_NOTE = ("TEDS-A now runs through 2024, so 2023 -> 2024 overlaps part of the decline in overdose deaths "
              "(late 2022 onward); earlier years describe treatment admissions before that decline.")

# Rules for the new findings-ledger rows, written BEFORE the 2024 re-run (verdicts: holds / partial /
# fails / not distinguishable).
TEDS_RULES = {
    "T1": "Heroin's share of primary-opioid admissions fell and 'other opiates & synthetics' rose 2019 -> end year: "
          "'holds' if both share changes have 95% CIs excluding 0 in the main specification AND in the through-2023 "
          "specification AND both log-odds trends excluding 2020 have CIs excluding 0; 'partial' if the main "
          "specification meets this but a sensitivity does not; 'fails' otherwise.",
    "T2": "First-time primary-opioid admissions per 100k aged 18-24 fell 2019 -> end year: 'holds' if the log change "
          "CI excludes 0 (negative) for both bounds (missing NOPRIOR as not first-time / first-time) in the main "
          "specification, the through-2023 specification and the coding-shift-excluded sensitivity; 'partial' if the "
          "main specification meets this but a sensitivity does not; 'fails' if the main CI includes 0.",
    "T3": "From 2019, opioid deaths rose relative to primary-opioid admissions (d > 0): 'holds' if d's CI excludes 0 "
          "(positive) for start 2019 in the main and the through-2023 specifications; 'partial' if only one; "
          "'fails' if neither.",
    "T4": "2023 -> 2024 (first overlap with the death decline), primary-opioid admissions vs opioid deaths: "
          "'not distinguishable' if d's CI includes 0; otherwise 'holds' for the observed direction (d > 0: deaths "
          "rose relative to admissions; d < 0: deaths fell relative to admissions).",
}


# ---------------------------------------------------------------------------------------------- helpers
def stack(d, cols, years):
    """[G states, T years, K columns] array (states sorted)."""
    states = sorted(d.state_fips.unique())
    arr = np.zeros((len(states), len(years), len(cols)))
    pos = {s: i for i, s in enumerate(states)}
    for r in d.itertuples():
        if r.year in years:
            for k, c in enumerate(cols):
                arr[pos[r.state_fips], years.index(r.year), k] = getattr(r, c)
    return arr


def slope(logy, years):
    t = np.asarray(years, float)
    tc = t - t.mean()
    return ((logy - logy.mean(axis=-1, keepdims=True)) * tc).sum(axis=-1) / (tc ** 2).sum()


def boot(values, f, seed):
    return studentized_cluster_bootstrap(values, f, B=B, seed=seed)


def row_from(r, **extra):
    return dict(**extra, estimate=r["estimate"], se_jackknife=r["se"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                p_boot=r["p_boot"], B=r["B"])


# ---------------------------------------------------------------------------------------------- panels
def build_panels(p, spec):
    cfg = SPECS[spec]
    years = list(range(2015, cfg["end"] + 1))
    dwin = list(range(2019, cfg["end"] + 1))
    p = p[p.year.isin(years)]
    regions = warehouse.query("select state_fips, region_name from state_regions")
    st = (p.groupby(["state_fips", "state_name"])
           .agg(is_complete=(cfg["complete_col"], "first"), teds_exclusion_reason=(cfg["reason_col"], "first"))
           .reset_index().merge(regions, on="state_fips", how="left"))
    has = p.assign(h=p.opioid_deaths.notna()).pivot_table(index="state_fips", columns="year", values="h", aggfunc="first")
    st["deaths_death_window"] = st.state_fips.map(has[dwin].all(axis=1))
    st["deaths_all_years"] = st.state_fips.map(has[years].all(axis=1))
    st["in_P35"] = st.is_complete
    st["in_P_deaths"] = st.is_complete & st.deaths_death_window
    st["in_P15"] = st.is_complete & st.deaths_all_years
    st["population_end"] = st.state_fips.map(p[p.year == cfg["end"]].set_index("state_fips").population)

    def missing(fips, yrs):
        have = set(p[(p.state_fips == fips) & p.opioid_deaths.notna()].year)
        return ", ".join(str(y) for y in yrs if y not in have)

    st["opioid_deaths_missing_death_window"] = [missing(f, dwin) for f in st.state_fips]
    st.insert(0, "spec", spec)
    summ = []
    us = st.population_end.sum()
    for name, col, window in (("P35", "in_P35", f"2015-{cfg['end']}"), ("P_deaths", "in_P_deaths", f"2019-{cfg['end']}"),
                              ("P15", "in_P15", f"2015-{cfg['end']}"), ("all 50 states + DC", None, "")):
        g = st if col is None else st[st[col]]
        r = dict(spec=spec, panel=name, window=window, states=len(g), population_end_year=int(g.population_end.sum()),
                 population_share_of_us=g.population_end.sum() / us)
        for reg in ("Northeast", "Midwest", "South", "West"):
            r[f"states_{reg}"] = int((g.region_name == reg).sum())
            r[f"pop_share_{reg}"] = g[g.region_name == reg].population_end.sum() / g.population_end.sum()
        summ.append(r)
    return st, pd.DataFrame(summ), years, dwin


def panel_changes(st_main, st_23):
    rows = []
    m = st_main.set_index("state_name")
    o = st_23.set_index("state_name")
    for name in sorted(m.index):
        for col, lab in (("in_P35", "P35"), ("in_P_deaths", "P_deaths"), ("in_P15", "P15")):
            if bool(m.loc[name, col]) != bool(o.loc[name, col]):
                why = (m.loc[name, "teds_exclusion_reason"] if not m.loc[name, "is_complete"]
                       else f"opioid deaths missing: {m.loc[name, 'opioid_deaths_missing_death_window']}")
                rows.append(dict(state_name=name, panel=lab, through_2023=bool(o.loc[name, col]),
                                 through_2024=bool(m.loc[name, col]), reason=why))
    return pd.DataFrame(rows, columns=["state_name", "panel", "through_2023", "through_2024", "reason"])


# ---------------------------------------------------------------------------------------------- (a)
def analysis_a(p, st, years, dwin, spec, s_off):
    end = years[-1]
    p35 = p[p.state_fips.isin(st[st.in_P35].state_fips) & p.year.isin(years)]
    cols = list(MEASURES) + ["population"]
    arr = stack(p35, cols, years)
    tot = arr.sum(axis=0)
    series = []
    for k, m in enumerate(MEASURES):
        rate = 1e5 * tot[:, k] / tot[:, -1]
        for i, y in enumerate(years):
            series.append(dict(spec=spec, panel="P35", measure=m, year=y, states=arr.shape[0], admissions=int(tot[i, k]),
                               population=int(tot[i, -1]), per_100k=rate[i], index_2015=100 * rate[i] / rate[0]))
    trends = []
    for k, m in enumerate(MEASURES):
        for label, yrs in ((f"2015-{end}", years), (f"2015-{end} excluding 2020", [y for y in years if y != 2020])):
            ix = [years.index(y) for y in yrs]
            v = arr[:, ix][:, :, [k, len(cols) - 1]]
            r = boot(v, lambda T, yrs=yrs: slope(np.log(T[..., 0] / T[..., 1]), yrs), SEED + s_off + 10 + k)
            trends.append(row_from(r, spec=spec, panel="P35", measure=m, years=label, states=arr.shape[0],
                                   statistic="log-linear trend of admission rate, per year",
                                   pct_per_year=100 * np.expm1(r["estimate"]),
                                   ci_low_pct=100 * np.expm1(r["ci_low"]), ci_high_pct=100 * np.expm1(r["ci_high"])))
    comp = []
    for pname, col, yrs in (("P_deaths", "in_P_deaths", dwin), ("P15", "in_P15", years)):
        d = p[p.state_fips.isin(st[st[col]].state_fips) & p.year.isin(yrs)]
        a = stack(d, ["primary_opioid", "any_opioid", "opioid_deaths", "population"], yrs).sum(axis=0)
        for i, y in enumerate(yrs):
            for k, m in enumerate(("primary_opioid", "any_opioid", "opioid_deaths")):
                r = 1e5 * a[:, k] / a[:, 3]
                comp.append(dict(spec=spec, panel=pname, measure=m, year=y, states=d.state_fips.nunique(), per_100k=r[i],
                                 index_first_year=100 * r[i] / r[0], first_year=yrs[0]))
    assoc = []
    for s_i, (pname, col, yrs, role) in enumerate((("P_deaths", "in_P_deaths", dwin, "main panel"),
                                                   ("P15", "in_P15", years, "sensitivity panel"))):
        for excl in (False, True):
            use = [y for y in yrs if not (excl and y == 2020)]
            d = p[p.state_fips.isin(st[st[col]].state_fips) & p.year.isin(use)].copy()
            for m_i, m in enumerate(("primary_opioid", "any_opioid")):
                d["log_adm_rate"] = np.log(d[m] / d.population)
                d["log_death_rate"] = np.log(d.opioid_deaths / d.population)
                r = wild_cluster_ci_inverted(d, "log_adm_rate", ["log_death_rate"], "state_fips", "year", B=B,
                                             seed=SEED + s_off + 100 * s_i + 10 * excl + m_i, weights="webb")
                assoc.append(dict(spec=spec, panel=pname, role=role,
                                  years=f"{use[0]}-{use[-1]}" + (" excl. 2020" if excl else ""), admissions_measure=m,
                                  weights="webb", ci_method=r["ci"], elasticity=r["estimate"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                                  p_wild=r["p_wild"], se_cr1=r["se_cr1"], n_state_years=r["n"], states=r["clusters"], B=B))
    return pd.DataFrame(series), pd.DataFrame(trends), pd.DataFrame(comp), pd.DataFrame(assoc)


# ---------------------------------------------------------------------------------------------- (b)
def analysis_b(p, st, years, spec, s_off):
    end = years[-1]
    d = p[p.state_fips.isin(st[st.in_P35].state_fips) & p.year.isin(years)]
    cols = ["primary_heroin", "primary_methadone_nonrx", "primary_other_opiates", "primary_opioid"]
    arr = stack(d, cols, years)
    tot = arr.sum(axis=0)
    shares = [dict(spec=spec, panel="P35", year=y, component=m, admissions=int(tot[i, k]), primary_opioid=int(tot[i, 3]),
                   share=tot[i, k] / tot[i, 3]) for i, y in enumerate(years) for k, m in enumerate(cols[:3])]
    out = []
    pairs = [(2015, end), (2019, end)] + ([(2023, 2024)] if end == 2024 else [])
    for k, m in enumerate(cols[:3]):
        for a_y, b_y in pairs:
            ia, ib = years.index(a_y), years.index(b_y)
            v = arr[:, [ia, ib]][:, :, [k, 3]]
            r = boot(v, lambda T: T[..., 1, 0] / T[..., 1, 1] - T[..., 0, 0] / T[..., 0, 1], SEED + s_off + 200 + 10 * k + a_y % 100)
            out.append(row_from(r, spec=spec, component=m, statistic=f"share change {a_y}->{b_y}", states=arr.shape[0],
                                estimate_pp=100 * r["estimate"], ci_low_pp=100 * r["ci_low"], ci_high_pp=100 * r["ci_high"]))
        for label, yrs in ((f"2015-{end}", years), (f"2015-{end} excluding 2020", [y for y in years if y != 2020])):
            ix = [years.index(y) for y in yrs]
            v = arr[:, ix][:, :, [k, 3]]

            def f(T, yrs=yrs):
                sh = T[..., 0] / T[..., 1]
                return slope(np.log(sh / (1 - sh)), yrs)
            r = boot(v, f, SEED + s_off + 300 + 10 * k + len(yrs))
            out.append(row_from(r, spec=spec, component=m, statistic=f"log-odds trend per year, {label}",
                                states=arr.shape[0], estimate_pp=np.nan, ci_low_pp=np.nan, ci_high_pp=np.nan))
    return pd.DataFrame(shares), pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------- (c)
def noprior_coding_check(states):
    """Flag state-years where the first-time share of primary-opioid admissions changes by more than
    CODING_SHIFT_PP from the previous year (data rule for NOPRIOR coding shifts)."""
    a = warehouse.query("select state_fips, year, sum(primary_opioid_first_time) ft, sum(primary_opioid_noprior_missing) mi, "
                        "sum(primary_opioid) po from teds_state_year_age where year between 2015 and 2024 group by 1, 2")
    names = warehouse.query("select distinct state_fips, state_name from teds_reporting_panel")
    a = a[a.state_fips.isin(states)].merge(names, on="state_fips").sort_values(["state_fips", "year"])
    a["first_time_share"] = a.ft / a.po
    a["missing_share"] = a.mi / a.po
    a["change_pp"] = a.groupby("state_fips").first_time_share.diff()
    a["flag"] = a.change_pp.abs() > CODING_SHIFT_PP
    return a


def jump_2018_by_state(states):
    """Per-state contribution to the 2017 -> 2018 change in first-time admissions, ages 25-34 and 55+."""
    a = warehouse.query("select * from teds_state_year_age where year in (2017, 2018)")
    names = warehouse.query("select distinct state_fips, state_name from teds_reporting_panel")
    a = a[a.state_fips.isin(states)].merge(names, on="state_fips")
    rows = []
    for band, groups in (("25-34", (5, 6)), ("55+", (11, 12))):
        g = (a[a.age_group.isin(groups)].groupby(["state_name", "year"])
             [["primary_opioid_first_time", "primary_opioid_noprior_missing", "primary_opioid"]].sum().unstack("year"))
        ft = g["primary_opioid_first_time"]
        total_change = (ft[2018] - ft[2017]).sum()
        for s in g.index:
            rows.append(dict(age_band=band, state_name=s, first_time_2017=int(ft.loc[s, 2017]),
                             first_time_2018=int(ft.loc[s, 2018]), change=int(ft.loc[s, 2018] - ft.loc[s, 2017]),
                             share_of_panel_change=(ft.loc[s, 2018] - ft.loc[s, 2017]) / total_change,
                             first_time_share_2017=ft.loc[s, 2017] / g["primary_opioid"].loc[s, 2017],
                             first_time_share_2018=ft.loc[s, 2018] / g["primary_opioid"].loc[s, 2018],
                             missing_share_2017=g["primary_opioid_noprior_missing"].loc[s, 2017] / g["primary_opioid"].loc[s, 2017],
                             missing_share_2018=g["primary_opioid_noprior_missing"].loc[s, 2018] / g["primary_opioid"].loc[s, 2018]))
    return pd.DataFrame(rows).sort_values(["age_band", "change"], ascending=[True, False])


def analysis_c(states, years, spec, s_off, variant):
    end = years[-1]
    a = warehouse.query("select * from teds_state_year_age where year between 2015 and ?", [end])
    pop = warehouse.query("select state_fips, year, teds_age_group as age_group, population from population_state_age "
                          "where teds_age_group between 1 and 12")
    a = a[a.state_fips.isin(states)].merge(pop, on=["state_fips", "year", "age_group"], how="left")
    if a.population.isna().any():
        raise ValueError("missing population for some state x year x age cells")
    band_of = {g: band for band, gs in AGE_BANDS.items() for g in gs}
    a["band"] = a.age_group.map(band_of)
    g = a.groupby(["state_fips", "year", "band"])[["primary_opioid_first_time", "primary_opioid_noprior_missing",
                                                   "primary_opioid", "population"]].sum().reset_index()
    rows, tests = [], []
    for b_i, band in enumerate(AGE_BANDS):
        arr = stack(g[g.band == band], ["primary_opioid_first_time", "primary_opioid_noprior_missing", "primary_opioid",
                                        "population"], years)
        tot = arr.sum(axis=0)
        for i, y in enumerate(years):
            rows.append(dict(spec=spec, variant=variant, age_band=band, year=y, states=arr.shape[0],
                             first_time=int(tot[i, 0]), noprior_missing=int(tot[i, 1]), primary_opioid=int(tot[i, 2]),
                             population=int(tot[i, 3]), first_time_per_100k_lower=1e5 * tot[i, 0] / tot[i, 3],
                             first_time_per_100k_upper=1e5 * (tot[i, 0] + tot[i, 1]) / tot[i, 3],
                             share_noprior_missing=tot[i, 1] / tot[i, 2]))
        for bound, num in (("lower (missing = not first-time)", lambda T: T[..., 0]),
                           ("upper (missing = first-time)", lambda T: T[..., 0] + T[..., 1])):
            for a_y in (2019, 2015):
                ia, ib = years.index(a_y), years.index(end)
                v = arr[:, [ia, ib]]
                r = boot(v, lambda T, num=num: np.log(num(T[..., 1, :]) / T[..., 1, 3]) - np.log(num(T[..., 0, :]) / T[..., 0, 3]),
                         SEED + s_off + 400 + 10 * b_i + a_y % 100 + (0 if bound.startswith("lower") else 5))
                tests.append(row_from(r, spec=spec, variant=variant, age_band=band, bound=bound,
                                      statistic=f"log change {a_y}->{end}" + (" (2015 baseline: before the 2018-2020 coding artifact)"
                                                                             if a_y == 2015 else " (2019 baseline)"),
                                      states=arr.shape[0], pct_change=100 * np.expm1(r["estimate"]),
                                      ci_low_pct=100 * np.expm1(r["ci_low"]), ci_high_pct=100 * np.expm1(r["ci_high"])))
    return pd.DataFrame(rows), pd.DataFrame(tests)


# ---------------------------------------------------------------------------------------------- (d)
def analysis_d(p, st, years, dwin, spec, s_off):
    end = years[-1]
    rows = []
    starts_main = [2019, 2021, 2022] + ([2023] if end == 2024 else [])
    for s_i, (pname, col, yrs, starts, role) in enumerate((("P_deaths", "in_P_deaths", dwin, starts_main, "main panel"),
                                                           ("P15", "in_P15", years, [2015] + starts_main, "sensitivity panel"))):
        d = p[p.state_fips.isin(st[st[col]].state_fips) & p.year.isin(yrs)]
        cols = list(MEASURES) + ["opioid_deaths"]
        arr = stack(d, cols, yrs)
        k_d = cols.index("opioid_deaths")
        for start in starts:
            block = []
            for k, m in enumerate(MEASURES):
                v = arr[:, [yrs.index(start), yrs.index(end)]][:, :, [k, k_d]]
                # d = log(D_end / D_start) - log(A_end / A_start); population cancels (same panel both terms)
                r = boot(v, lambda T: np.log(T[..., 1, 1] / T[..., 0, 1]) - np.log(T[..., 1, 0] / T[..., 0, 0]),
                         SEED + s_off + 500 + 100 * s_i + 10 * k + start % 100)
                tot = v.sum(axis=0)
                block.append(row_from(r, spec=spec, panel=pname, role=role, measure=m, start=start, end=end,
                                      states=arr.shape[0], death_pct_change=100 * (tot[1, 1] / tot[0, 1] - 1),
                                      admission_pct_change=100 * (tot[1, 0] / tot[0, 0] - 1)))
            for rr, ph in zip(block, sv.holm([x["p_boot"] for x in block])):
                rr["p_holm"] = float(ph)
            rows += block
    out = pd.DataFrame(rows)
    out["estimand"] = "d = log(opioid death change) - log(admission change); d > 0: deaths rose relative to admissions"
    return out


# ---------------------------------------------------------------------------------------------- coverage simulations
def sim_coverage(st, years, n_sims=2000, n_wcb=1000):
    """Simulated 95% CI coverage of each method at the actual (main-specification) panel sizes."""
    rows = []
    mu_a, mu_d, tau = -0.4, 0.3, 0.25
    truth_lr = mu_a + tau ** 2 / 2                 # log E[end]/E[start] for lognormal state changes
    truth_d = (mu_d + tau ** 2 / 2) - truth_lr
    truth_tr = superpopulation_trend(years)
    for pname, col in (("P35", "in_P35"), ("P_deaths", "in_P_deaths"), ("P15", "in_P15")):
        G = int(st[col].sum())
        rng = np.random.default_rng([SEED, G])
        cov = dict(log_ratio=0, d=0, trend=0)
        for s in range(n_sims):
            v = simulate_state_totals(rng, G, mu=mu_a, tau=tau)
            r = studentized_cluster_bootstrap(v, lambda T: np.log(T[..., 1] / T[..., 0]), B=499, seed=s)
            cov["log_ratio"] += r["ci_low"] <= truth_lr <= r["ci_high"]
            dv = np.stack([simulate_state_totals(rng, G, mu=mu_a, tau=tau), simulate_state_totals(rng, G, mu=mu_d, tau=tau)], 2)
            r = studentized_cluster_bootstrap(dv, lambda T: np.log(T[..., 1, 1] / T[..., 0, 1]) - np.log(T[..., 1, 0] / T[..., 0, 0]),
                                              B=499, seed=s)
            cov["d"] += r["ci_low"] <= truth_d <= r["ci_high"]
            sv_ = simulate_state_series(rng, G, years)
            r = studentized_cluster_bootstrap(sv_, lambda T: slope(np.log(T[..., 0] / T[..., 1]), years), B=499, seed=s)
            cov["trend"] += r["ci_low"] <= truth_tr <= r["ci_high"]
        for k, c in cov.items():
            rows.append(dict(method="symmetric bootstrap-t, jackknife SE (state clusters)", statistic=k, panel=pname,
                             clusters=G, sims=n_sims, B=499, coverage_95=c / n_sims,
                             mc_se=np.sqrt(0.95 * 0.05 / n_sims), null_rejection_5pct=np.nan,
                             meets_94pct_target=c / n_sims >= 0.94))
    for pname, col, T in (("P_deaths", "in_P_deaths", 6), ("P15", "in_P15", len(years))):
        G = int(st[col].sum())
        rng = np.random.default_rng([SEED, G, T])
        cover = reject = 0
        for s in range(n_wcb):
            r = wild_cluster_ci_inverted(simulate_panel(rng, G=G, T=T, beta=0.6), "y", ["x"], "unit", "time", B=399,
                                         seed=s, weights="webb")
            cover += r["ci_low"] <= 0.6 <= r["ci_high"]
            r0 = wild_cluster_ci_inverted(simulate_panel(rng, G=G, T=T, beta=0.0), "y", ["x"], "unit", "time", B=399,
                                          seed=50_000 + s, weights="webb")
            reject += r0["p_wild"] < 0.05
        rows.append(dict(method="wild cluster bootstrap (WCR, Webb weights), test-inversion CI (TWFE slope)",
                         statistic="association slope", panel=pname, clusters=G, sims=n_wcb, B=399,
                         coverage_95=cover / n_wcb, mc_se=np.sqrt(0.95 * 0.05 / n_wcb),
                         null_rejection_5pct=reject / n_wcb, meets_94pct_target=cover / n_wcb >= 0.94))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- ledger rows
def _excl0(r, sign=None):
    if sign == "neg":
        return r.ci_high < 0
    if sign == "pos":
        return r.ci_low > 0
    return r.ci_low > 0 or r.ci_high < 0


def ledger_rows(b, c, d):
    rows, verd = [], {}
    # T1
    bm = b[(b.spec == "main")]
    b23 = b[(b.spec == "through_2023")]

    def pick(t, comp, stat):
        return t[(t.component == comp) & (t.statistic == stat)].iloc[0]
    h_m, o_m = pick(bm, "primary_heroin", "share change 2019->2024"), pick(bm, "primary_other_opiates", "share change 2019->2024")
    h_s, o_s = pick(b23, "primary_heroin", "share change 2019->2023"), pick(b23, "primary_other_opiates", "share change 2019->2023")
    tr_h = pick(bm, "primary_heroin", "log-odds trend per year, 2015-2024 excluding 2020")
    tr_o = pick(bm, "primary_other_opiates", "log-odds trend per year, 2015-2024 excluding 2020")
    main_ok = _excl0(h_m, "neg") and _excl0(o_m, "pos")
    sens_ok = _excl0(h_s, "neg") and _excl0(o_s, "pos") and _excl0(tr_h, "neg") and _excl0(tr_o, "pos")
    verd["T1"] = "holds" if main_ok and sens_ok else ("partial" if main_ok else "fails")
    rows.append(dict(
        finding="T1", original_claim="TEDS-A (through 2023): heroin's share of primary-opioid admissions fell and 'other opiates & "
                                     "synthetics' rose, mainly from 2021.",
        verdict=verd["T1"],
        supported_claim=(f"From 2019 to 2024, heroin's share of primary-opioid admissions changed by {h_m.estimate_pp:.1f} "
                         f"percentage points and the 'other opiates & synthetics' share by {o_m.estimate_pp:+.1f} points in the "
                         f"{int(h_m.states)}-state panel."),
        key_evidence=(f"heroin {h_m.estimate_pp:.1f} pp (95% CI {h_m.ci_low_pp:.1f}, {h_m.ci_high_pp:.1f}); other opiates "
                      f"{o_m.estimate_pp:+.1f} pp ({o_m.ci_low_pp:.1f}, {o_m.ci_high_pp:.1f}); through 2023: heroin "
                      f"{h_s.estimate_pp:.1f} pp ({h_s.ci_low_pp:.1f}, {h_s.ci_high_pp:.1f}); log-odds trends excl. 2020: heroin "
                      f"{tr_h.estimate:.3f}/yr ({tr_h.ci_low:.3f}, {tr_h.ci_high:.3f}), other {tr_o.estimate:.3f}/yr "
                      f"({tr_o.ci_low:.3f}, {tr_o.ci_high:.3f})"),
        main_caveat=("Code 7 combines fentanyl, prescription opioids and buprenorphine (no fentanyl-specific code); "
                     "admissions are treatment episodes at publicly funded facilities, not people or prevalence.")))
    # T2
    def c_row(spec, variant, bound, stat_prefix):
        t = c[(c.spec == spec) & (c.variant == variant) & (c.age_band == "18-24") & (c.bound.str.startswith(bound))]
        return t[t.statistic.str.startswith(stat_prefix)].iloc[0]
    main_l, main_u = c_row("main", "P35", "lower", "log change 2019->2024"), c_row("main", "P35", "upper", "log change 2019->2024")
    s23_l, s23_u = (c_row("through_2023", "P35", "lower", "log change 2019->2023"),
                    c_row("through_2023", "P35", "upper", "log change 2019->2023"))
    ex_l, ex_u = (c_row("main", "P35 without coding-shift states", "lower", "log change 2019->2024"),
                  c_row("main", "P35 without coding-shift states", "upper", "log change 2019->2024"))
    ex15 = c_row("main", "P35 without coding-shift states", "lower", "log change 2015->2024")
    youth = c[(c.spec == "main") & (c.variant == "P35") & (c.age_band == "12-17") & c.bound.str.startswith("lower")
              & c.statistic.str.startswith("log change 2019->2024")].iloc[0]
    m_ok = _excl0(main_l, "neg") and _excl0(main_u, "neg")
    s_ok = all(_excl0(x, "neg") for x in (s23_l, s23_u, ex_l, ex_u))
    verd["T2"] = "holds" if m_ok and s_ok else ("partial" if m_ok else "fails")
    rows.append(dict(
        finding="T2", original_claim="TEDS-A (through 2023): first-time primary-opioid admissions per 100k fell at ages 18-24.",
        verdict=verd["T2"],
        supported_claim=(f"First-time primary-opioid admissions per 100k people aged 18-24 changed by {main_l['pct_change']:.0f}% "
                         f"from 2019 to 2024 (lower bound; {main_u['pct_change']:.0f}% upper bound) in the {int(main_l.states)}-state "
                         f"panel, and by {ex_l['pct_change']:.0f}% without states flagged for prior-treatment coding shifts."),
        key_evidence=(f"main 2019->2024 {main_l['pct_change']:.0f}% ({main_l.ci_low_pct:.0f}, {main_l.ci_high_pct:.0f}); through 2023 "
                      f"{s23_l['pct_change']:.0f}% ({s23_l.ci_low_pct:.0f}, {s23_l.ci_high_pct:.0f}); without coding-shift states "
                      f"{ex_l['pct_change']:.0f}% ({ex_l.ci_low_pct:.0f}, {ex_l.ci_high_pct:.0f}), 2015 baseline {ex15['pct_change']:.0f}% "
                      f"({ex15.ci_low_pct:.0f}, {ex15.ci_high_pct:.0f}); ages 12-17 2019->2024 {youth['pct_change']:+.0f}% "
                      f"({youth.ci_low_pct:.0f}, {youth.ci_high_pct:.0f}) from a small base"),
        main_caveat=("First-time admissions are not new cases of addiction (they also reflect access, capacity and time from "
                     "onset to treatment). 2019 lies inside a 2018-2020 prior-treatment coding artifact driven by California; "
                     "the coding-shift-excluded sensitivity addresses this.")))
    # T3 / T4
    def d_row(spec, start, panel="P_deaths", measure="primary_opioid"):
        t = d[(d.spec == spec) & (d.panel == panel) & (d.measure == measure) & (d.start == start)]
        return t.iloc[0] if len(t) else None
    m19, s19 = d_row("main", 2019), d_row("through_2023", 2019)
    hits = [x is not None and _excl0(x, "pos") for x in (m19, s19)]
    verd["T3"] = "holds" if all(hits) else ("partial" if any(hits) else "fails")
    rows.append(dict(
        finding="T3", original_claim="TEDS-A (through 2023): from 2019, opioid deaths rose relative to primary-opioid admissions.",
        verdict=verd["T3"],
        supported_claim=(f"From 2019 to 2024, opioid deaths changed by {m19.death_pct_change:+.0f}% and primary-opioid admissions "
                         f"by {m19.admission_pct_change:+.0f}% in the {int(m19.states)}-state deaths panel."),
        key_evidence=(f"main d = {m19.estimate:.3f} ({m19.ci_low:.3f}, {m19.ci_high:.3f}), Holm p = {m19.p_holm:.4f}; through 2023 "
                      f"d = {s19.estimate:.3f} ({s19.ci_low:.3f}, {s19.ci_high:.3f})"),
        main_caveat=("Admissions track treatment entry at publicly funded facilities (funding, capacity, 2020 disruption, "
                     "office-based buprenorphine not captured); deaths are mostly illicitly made fentanyl.")))
    m23 = d_row("main", 2023)
    inc = m23.ci_low <= 0 <= m23.ci_high
    verd["T4"] = "not distinguishable" if inc else "holds"
    direction = ("the data do not distinguish whether deaths or admissions declined faster" if inc else
                 ("the opioid death rate rose relative to the admission rate" if m23.estimate > 0
                  else "the opioid death rate fell relative to the admission rate"))
    rows.append(dict(
        finding="T4", original_claim="(new) 2023 -> 2024, the first TEDS-A year overlapping the decline in overdose deaths.",
        verdict=verd["T4"],
        supported_claim=(f"From 2023 to 2024, opioid deaths changed by {m23.death_pct_change:+.0f}% and primary-opioid admissions by "
                         f"{m23.admission_pct_change:+.0f}%; {direction}."),
        key_evidence=f"d = {m23.estimate:.3f} ({m23.ci_low:.3f}, {m23.ci_high:.3f}), Holm p = {m23.p_holm:.4f}; one year only",
        main_caveat="A single year; the 2024 TEDS-A file excludes South Carolina and Washington (neither is in the panel)."))
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
    s_main = series[series.spec == "main"]
    ax = axes[0]
    _style(ax)
    for c, m in zip(COLORS, MEASURES):
        s = s_main[s_main.measure == m]
        ax.plot(s.year, s.index_2015, "o-", color=c, lw=2, ms=5, label=m.replace("_", " "))
    ax.axvspan(2019.5, 2020.5, color="#e4e3df", zorder=0)
    ax.set_title(f"P35: opioid admission rates (index 2015 = 100), {int(s_main.states.iloc[0])} states", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5)
    ax = axes[1]
    _style(ax)
    c = comp[(comp.spec == "main") & (comp.panel == "P_deaths")]
    for col, m, lab in ((COLORS[0], "primary_opioid", "primary-opioid admissions"),
                        (COLORS[3], "any_opioid", "any-opioid admissions"), ("#52514e", "opioid_deaths", "opioid overdose deaths")):
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
    s0 = shares[shares.spec == "main"]
    for c, m, lab in zip(COLORS, ("primary_heroin", "primary_other_opiates", "primary_methadone_nonrx"),
                         ("heroin", "other opiates & synthetics (incl. fentanyl, Rx, buprenorphine)", "non-Rx methadone")):
        s = s0[s0.component == m]
        ax.plot(s.year, 100 * s.share, "o-", color=c, lw=2, ms=5, label=lab)
    ax.set_ylabel("% of primary-opioid admissions")
    ax.set_title("TEDS-A (b): composition of primary-opioid admissions, P35 panel, 2015-2024", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/b_shares.png", dpi=150)
    plt.close(fig)


def plot_c(ft):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, band in zip(axes.flat, AGE_BANDS):
        _style(ax)
        for variant, color in (("P35", COLORS[0]), ("P35 without coding-shift states", COLORS[1])):
            s = ft[(ft.spec == "main") & (ft.variant == variant) & (ft.age_band == band)]
            ax.fill_between(s.year, s.first_time_per_100k_lower, s.first_time_per_100k_upper, color=color, alpha=0.15, lw=0)
            ax.plot(s.year, s.first_time_per_100k_lower, "o-", color=color, lw=2, ms=3, label=variant)
        ax.axvspan(2017.5, 2020.5, color="#e4e3df", zorder=0)
        ax.set_title(f"ages {band}", loc="left", fontsize=9)
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle("TEDS-A (c): first-time primary-opioid admissions per 100k (line: NOPRIOR = 0; band: + missing NOPRIOR). "
                 "Shaded 2018-2020: prior-treatment coding artifact", x=0.01, ha="left", fontsize=9.5)
    fig.text(0.01, 0.01, "First-time admissions are not new cases of addiction: entry also depends on treatment access, capacity "
             "and time from onset to treatment.", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(f"{OUT}/c_first_time.png", dpi=150)
    plt.close(fig)


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    compare_codebooks().to_csv(f"{OUT}/harmonization_codebook_2024.csv", index=False)
    compare_data().to_csv(f"{OUT}/harmonization_data_2024.csv", index=False)
    p_all = teds.panel()
    out = {k: [] for k in ("states", "summary", "series", "trends", "comp", "assoc", "shares", "b", "ft", "c", "d")}
    sts = {}
    for s_off, spec in ((0, "main"), (5000, "through_2023")):
        st, summ, years, dwin = build_panels(p_all, spec)
        sts[spec] = (st, years, dwin)
        p = p_all[p_all.year.isin(years)]
        out["states"].append(st)
        out["summary"].append(summ)
        series, trends, comp, assoc = analysis_a(p, st, years, dwin, spec, s_off)
        out["series"].append(series)
        out["trends"].append(trends)
        out["comp"].append(comp)
        out["assoc"].append(assoc)
        shares, btests = analysis_b(p, st, years, spec, s_off)
        out["shares"].append(shares)
        out["b"].append(btests)
        states35 = list(st[st.in_P35].state_fips)
        chk = noprior_coding_check(states35)
        flagged = sorted(chk[chk.flag].state_fips.unique())
        if spec == "main":
            chk.to_csv(f"{OUT}/c_noprior_coding_check.csv", index=False)
            jump_2018_by_state(states35).to_csv(f"{OUT}/c_2018_jump_by_state.csv", index=False)
        for v_i, (variant, states) in enumerate((("P35", states35),
                                                 ("P35 without coding-shift states", [s for s in states35 if s not in flagged]))):
            ft, ct = analysis_c(states, years, spec, s_off + 1000 * v_i, variant)
            out["ft"].append(ft)
            out["c"].append(ct)
        out["d"].append(analysis_d(p, st, years, dwin, spec, s_off))
    cat = {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
    for k, f in (("states", "panels_states"), ("summary", "panels_summary"), ("series", "a_admissions_series_P35"),
                 ("trends", "a_admissions_trend_P35"), ("comp", "a_admissions_vs_deaths_series"), ("assoc", "a_association"),
                 ("shares", "b_shares_P35"), ("b", "b_share_changes"), ("ft", "c_first_time_by_age"),
                 ("c", "c_first_time_changes"), ("d", "d_difference_of_declines")):
        cat[k].to_csv(f"{OUT}/{f}.csv", index=False)
    panel_changes(sts["main"][0], sts["through_2023"][0]).to_csv(f"{OUT}/panels_changes_2023_vs_2024.csv", index=False)
    plot_a(cat["series"], cat["comp"])
    plot_b(cat["shares"])
    plot_c(cat["ft"])
    sim_coverage(sts["main"][0], sts["main"][1]).to_csv(f"{OUT}/sim_coverage.csv", index=False)
    ledger_rows(cat["b"], cat["c"], cat["d"]).to_csv(f"{OUT}/findings_rows.csv", index=False)
    chk = pd.read_csv(f"{OUT}/c_noprior_coding_check.csv")
    with open(f"{OUT}/meta.json", "w") as fh:
        json.dump(dict(seed=SEED, B=B, specs={k: v["label"] for k, v in SPECS.items()},
                       completeness_rule="TEDS admissions in every year of the window and total admissions never change by "
                                         "more than 50% year to year (ratio within [0.5, 1.5])",
                       scope=SCOPE_NOTE, coding_shift_rule=f"first-time share changes by more than {CODING_SHIFT_PP:.0%} points "
                                                           "year to year",
                       coding_shift_states=sorted(chk[chk.flag].state_name.unique().tolist()),
                       rules=TEDS_RULES, source=SRC, method=__doc__.split("Specifications")[0].strip()), fh, indent=2)
    print("done")


if __name__ == "__main__":
    main()
