#!/usr/bin/env python3
"""
Phase 4b: TEDS-A treatment admissions as an independent, AREA-LEVEL signal on opioid addiction.

    python -m src.step_t            # analysis (a)

Admissions are not prevalence: they count treatment episodes at publicly funded facilities, depend on
funding, capacity and referral practice, and miss office-based buprenorphine. Never joined to NSDUH.

(a) Opioid admissions vs opioid deaths, 2015-2023
  * Admissions: primary substance (SUB1) heroin, non-Rx methadone or other opiates/synthetics (main);
    any opioid mention in SUB1-SUB3 (variant). Code 7 includes fentanyl, prescription opioids and
    buprenorphine together (no fentanyl-specific code in the public file).
  * Deaths: CDC VSRR state 'Opioids (T40.0-T40.4,T40.6)', December 12-month-ending value = calendar year
    (NY = NY + NYC). Rates per 100k with Census July 1 population (intercensal 2015-2019, Vintage 2025
    2020-2023).
  * Main panel: states meeting the completeness rule (TEDS admissions in every year 2015-2023 and no
    year-to-year change in total admissions > 50%) AND with opioid deaths in every year. Sensitivity:
    every state-year with admissions and deaths.
  * National series: sums over the main panel (fixed set of states), rates per 100k and index 2015 = 100.
  * State-year association: log(admission rate) = state FE + year FE + b * log(death rate), clusters =
    states; wild cluster bootstrap (src/panel.py): WCR p-value, WCU percentile-t 95% CI.
  * State changes 2019 -> 2023 (pre-pandemic year to latest): change in log admission and log death rates.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import teds
from src.panel import wild_cluster_bootstrap

OUT = "outputs/teds"
SEED = 20261107
B = 9999
YEARS = list(range(teds.WINDOW[0], teds.WINDOW[1] + 1))
SRC = ("TEDS-A 2006-2023 combined PUF (SAMHSA), admissions 2015-2023; CDC VSRR state file (xkb8-kh2a) opioid deaths, "
       "Dec 12-month-ending; Census state population (intercensal 2015-2019, Vintage 2025 2020-2023)")
MEASURES = {"primary_opioid": "primary substance heroin / non-Rx methadone / other opiates & synthetics",
            "any_opioid": "any opioid mention in primary, secondary or tertiary substance"}


def build_samples(p):
    """Return (main, all_states, state_table)."""
    p = p.copy()
    p["has_deaths"] = p.opioid_deaths.notna()
    ok = p[p.reports & p.has_deaths & (p.primary_opioid > 0)]
    per_state = p.groupby(["state_fips", "state_name"]).agg(
        is_complete=("is_complete", "first"), exclusion_reason=("exclusion_reason", "first"),
        years_with_deaths=("has_deaths", "sum"), years_reported=("reports", "sum")).reset_index()
    per_state["deaths_all_years"] = per_state.years_with_deaths == len(YEARS)
    per_state["in_main_panel"] = per_state.is_complete & per_state.deaths_all_years

    def reason(r):
        out = []
        if not r.is_complete:
            out.append(r.exclusion_reason)
        if not r.deaths_all_years:
            missing = sorted(set(YEARS) - set(p[(p.state_fips == r.state_fips) & p.has_deaths].year))
            out.append("no CDC opioid death value for " + ", ".join(map(str, missing)))
        return "; ".join(out) if out else ""
    per_state["main_panel_exclusion_reason"] = per_state.apply(reason, axis=1)
    main = ok[ok.state_fips.isin(per_state[per_state.in_main_panel].state_fips)].copy()
    if main.empty:
        raise ValueError("main panel is empty: no state meets the completeness rule and has opioid deaths in every "
                         f"year {YEARS[0]}-{YEARS[-1]}")
    return main, ok.copy(), per_state


def national_series(d, label):
    g = d.groupby("year").agg(states=("state_fips", "nunique"), primary_opioid=("primary_opioid", "sum"),
                              any_opioid=("any_opioid", "sum"), opioid_deaths=("opioid_deaths", "sum"),
                              population=("population", "sum")).reset_index()
    for c in ("primary_opioid", "any_opioid", "opioid_deaths"):
        g[f"{c}_per_100k"] = 1e5 * g[c] / g.population
        g[f"{c}_index_2015"] = 100 * g[f"{c}_per_100k"] / g[f"{c}_per_100k"].iloc[0]
    g.insert(0, "sample", label)
    return g


def associations(main, all_states):
    rows = []
    for k, (sample, d) in enumerate((("main panel (complete states)", main), ("all reporting state-years", all_states))):
        for m_i, m in enumerate(MEASURES):
            d = d.copy()
            d["log_adm_rate"] = np.log(d[m] / d.population)
            d["log_death_rate"] = np.log(d.opioid_deaths / d.population)
            r = wild_cluster_bootstrap(d, "log_adm_rate", ["log_death_rate"], "state_fips", "year", B=B,
                                       seed=SEED + 10 * k + m_i)
            rows.append(dict(analysis="a", sample=sample, admissions_measure=m, measure_label=MEASURES[m],
                             elasticity=r["estimate"], ci_low=r["ci_low"], ci_high=r["ci_high"], p_wild=r["p_wild"],
                             se_cr1=r["se_cr1"], n_state_years=r["n"], states=r["clusters"], B=r["B"], seed=r["seed"],
                             model="log(admissions/pop) = state FE + year FE + b log(opioid deaths/pop); "
                                   "wild cluster bootstrap by state (WCR p, WCU percentile-t CI)",
                             source=SRC))
    return pd.DataFrame(rows)


def state_changes(main, start=2019, end=2023):
    cols = ["primary_opioid", "opioid_deaths", "population"]
    sub = main[main.year.isin([start, end])].copy()
    sub[cols] = sub[cols].astype(float)
    w = sub.pivot(index=["state_fips", "state_name"], columns="year", values=cols)
    out = pd.DataFrame(index=w.index)
    for c in ("primary_opioid", "opioid_deaths"):
        r0 = w[(c, start)] / w[("population", start)]
        r1 = w[(c, end)] / w[("population", end)]
        out[f"{c}_per_100k_{start}"] = 1e5 * r0
        out[f"{c}_per_100k_{end}"] = 1e5 * r1
        out[f"{c}_log_change"] = np.log(r1 / r0)
    out = out.reset_index()
    out["source"] = SRC
    return out


def plot_trends(series):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    main = series[series["sample"] == "main panel (complete states)"]
    allst = series[series["sample"] == "all reporting state-years"]
    for col, color, lab in (("opioid_deaths", "#2a78d6", "Opioid overdose deaths (CDC)"),
                            ("primary_opioid", "#eb6834", "Primary-opioid treatment admissions (TEDS-A)"),
                            ("any_opioid", "#1baf7a", "Any-opioid-mention admissions (TEDS-A)")):
        ax.plot(main.year, main[f"{col}_index_2015"], "o-", color=color, lw=2, ms=6, label=f"{lab}, main panel")
        ax.plot(allst.year, allst[f"{col}_index_2015"], "--", color=color, lw=1.2, alpha=0.7)
    ax.axhline(100, color="#52514e", lw=0.8)
    ax.set_ylabel("Rate per 100k, index 2015 = 100")
    n = int(main.states.iloc[0])
    ax.set_title(f"TEDS-A (a): opioid admissions vs opioid deaths, {YEARS[0]}-{YEARS[-1]} ({n} complete states; "
                 "dashed = all reporting states)", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.01, 0.01, "Admissions are treatment episodes at publicly funded facilities, not people or prevalence. "
             "Source: " + SRC, fontsize=6, color="#52514e", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{OUT}/a_trends.png", dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    p = teds.panel()
    main_d, all_d, states = build_samples(p)
    states.to_csv(f"{OUT}/a_panel_states.csv", index=False)
    series = pd.concat([national_series(main_d, "main panel (complete states)"),
                        national_series(all_d, "all reporting state-years")], ignore_index=True)
    series["source"] = SRC
    series.to_csv(f"{OUT}/a_national_series.csv", index=False)
    assoc = associations(main_d, all_d)
    assoc.to_csv(f"{OUT}/a_state_year_association.csv", index=False)
    ch = state_changes(main_d)
    ch.to_csv(f"{OUT}/a_state_changes_2019_2023.csv", index=False)
    plot_trends(series)
    with open(f"{OUT}/a_meta.json", "w") as fh:
        json.dump(dict(window=list(teds.WINDOW), seed=SEED, B=B,
                       completeness_rule="TEDS admissions in every year 2015-2023 and total admissions never change by "
                                         "more than 50% year to year (ratio within [0.5, 1.5])",
                       main_panel_states=int(states.in_main_panel.sum()), states_considered=len(states),
                       method=__doc__.split("(a)")[1].strip()), fh, indent=2)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(states[~states.in_main_panel][["state_name", "main_panel_exclusion_reason"]].to_string(index=False))
    print(series.drop(columns="source").round(1).to_string(index=False))
    print(assoc.drop(columns=["model", "source", "measure_label"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
