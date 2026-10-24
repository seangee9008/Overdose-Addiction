#!/usr/bin/env python3
"""
Step D: county-level change in overdose deaths by social vulnerability (CDC county file + SVI 2022 only).

    python -m src.step_d

Area-level analysis only: nothing from NSDUH enters here (no individual/area pooling).

Data
  * CDC VSRR provisional county overdose deaths (data.cdc.gov gb4e-yj24): ALL-DRUG deaths only (the
    file has no opioid split), 12-month-ending, 2020-01 to 2025-12. December values = calendar years.
  * CDC/ATSDR SVI 2022 county CSV: RPL_THEMES (overall national percentile, 0-1) and E_TOTPOP
    (ACS 2018-2022 population, used as the exposure for every period).

Data are read from the dbt marts county_deaths_svi and county_fips_audit (DuckDB warehouse, built by
`make data` from the raw files above). Cleaning is done in dbt staging (see data/DATA_PROFILE.md):
  * FIPS zero-padded to 5 characters.
  * Two Virginia FIPS (51005, 51019) appear twice per month (independent cities folded into their
    counties) with identical values; one row per FIPS is kept.
  * Connecticut: county file uses 8 legacy counties, SVI 2022 uses 9 planning regions -> no 1:1 match.
    Alaska 02261 (Valdez-Cordova) was split into 02063/02066 in SVI. These are dropped and listed.
  * Zeros are true zeros (literal "0", no suppression footnote). Blank cells carry the NCHS footnote
    "counts between 1-9 suppressed" and are treated as interval-censored on [1, 9].

Periods: main 2022 -> 2025 (2022 = national peak calendar year for all-drug deaths, CDC state file;
2025 = latest complete year). Sensitivity: 2023 -> 2025. Both are non-overlapping calendar years.

Model: negative binomial, interval-censored for suppressed cells (src/count_models.py):
  log E[deaths] = log(pop) + b0 + g*1[2025] + sum_q a_q*1[SVI quartile q] + sum_q d_q*1[2025]*1[quartile q]
  Quartiles of RPL_THEMES (Q1 = least vulnerable, reference). The 2025-vs-base rate ratio in
  quartile q is exp(g + d_q); d_q compares that change with Q1. Joint Wald test of all d_q = 0.
  SEs clustered by county (each county appears in both periods).
For transparency, the same model is also fit with suppressed cells (a) set to 0 and (b) dropped.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import warehouse
from src.count_models import censored_nb

OUT = "outputs"
COL = "Provisional Drug Overdose Deaths"
PERIODS = {"main": (2022, 2025), "sensitivity": (2023, 2025)}
QLAB = {1: "Q1 (least vulnerable)", 2: "Q2", 3: "Q3", 4: "Q4 (most vulnerable)"}
SRC = ("CDC VSRR county file gb4e-yj24 (all-drug, Dec 12-month-ending) x CDC/ATSDR SVI 2022 county CSV "
       "(RPL_THEMES, E_TOTPOP)")


def load():
    """December (= calendar-year) county rows joined to SVI, the join losses, and the collapsed
    duplicate-FIPS rows -- all from the dbt marts (cleaning rules: dbt/models/staging/)."""
    m = warehouse.query(f"""
        select fips as "FIPS", st_abbrev as "ST_ABBREV", county_name as "COUNTYNAME", year as "Year",
               month as "Month", deaths_reported as "{COL}", rpl_themes as "RPL_THEMES", e_totpop as "E_TOTPOP"
        from county_deaths_svi
        where month = 12 and rpl_themes is not null          -- SVI -999 (missing) is NULL in staging
        order by src_row""")
    audit = warehouse.query("select side, fips as \"FIPS\", st_abbrev as \"ST_ABBREV\", name "
                            "from county_fips_audit order by side_order, order_key")
    lost = audit[audit.side != "Duplicate FIPS rows in CDC county file"].reset_index(drop=True)
    dup_names = (audit[audit.side == "Duplicate FIPS rows in CDC county file"]
                 .rename(columns={"name": "COUNTYNAME"})[["FIPS", "ST_ABBREV", "COUNTYNAME"]])
    m["quartile"] = np.minimum(np.floor(m.RPL_THEMES * 4).astype(int) + 1, 4)
    m["censored"] = m[COL].isna()
    return m, lost, dup_names


def fit(m, base, end, how):
    d = m[m.Year.isin([base, end])].copy()
    d["post"] = (d.Year == end).astype(float)
    y = d[COL].to_numpy(float)
    cens = d.censored.to_numpy()
    if how == "zero":
        y, cens = np.where(cens, 0.0, y), np.zeros(len(d), bool)
    elif how == "drop":
        keep = ~cens
        d, y, cens = d[keep], y[keep], cens[keep]
    cols, names = [np.ones(len(d)), d.post.to_numpy()], ["(Intercept)", f"year_{end}"]
    for q in (2, 3, 4):
        cols.append((d.quartile == q).to_numpy(float))
        names.append(f"SVI_Q{q}")
    for q in (2, 3, 4):
        cols.append(((d.quartile == q) & (d.post == 1)).to_numpy(float))
        names.append(f"year_{end}:SVI_Q{q}")
    X = np.column_stack(cols)
    tab, V, info = censored_nb(X, y, cens, np.ones(len(d)), np.full(len(d), 9.0),
                               offset=np.log(d.E_TOTPOP.to_numpy(float)), clusters=d.FIPS.to_numpy(), names=names)
    # change ratio (end vs base) per quartile = exp(g + d_q), CI from V
    rows = []
    for q in (1, 2, 3, 4):
        L = np.zeros(len(tab))
        L[1] = 1
        if q > 1:
            L[names.index(f"year_{end}:SVI_Q{q}")] = 1
        est, se = L @ tab.coef.to_numpy(), np.sqrt(L @ V @ L)
        rows.append(dict(quartile=QLAB[q], counties=int(d[d.post == 1].quartile.eq(q).sum()),
                         change_rate_ratio=np.exp(est), ci_low=np.exp(est - 1.96 * se), ci_high=np.exp(est + 1.96 * se),
                         pct_change=100 * np.expm1(est)))
    inter = [names.index(f"year_{end}:SVI_Q{q}") for q in (2, 3, 4)]
    b = tab.coef.to_numpy()[inter]
    W = float(b @ np.linalg.solve(V[np.ix_(inter, inter)], b))
    meta = dict(base=base, end=end, suppressed=how, wald_interactions=W, wald_df=3,
                wald_p=float(stats.chi2.sf(W, 3)), **info)
    return tab, pd.DataFrame(rows), meta


def main():
    os.makedirs(OUT, exist_ok=True)
    m, lost, dup_names = load()
    lost.to_csv(f"{OUT}/step_d_join_losses.csv", index=False)

    # descriptive: exact deaths, suppressed cells, zero cells, bounds on totals, by quartile x year
    desc = (m.groupby(["quartile", "Year"])
              .apply(lambda g: pd.Series({
                  "counties": len(g), "population": g.E_TOTPOP.sum(), "zero_cells": int((g[COL] == 0).sum()),
                  "suppressed_cells": int(g.censored.sum()), "deaths_exact_sum": g[COL].sum(),
                  "deaths_total_lower": g[COL].sum() + 1 * g.censored.sum(),
                  "deaths_total_upper": g[COL].sum() + 9 * g.censored.sum()}), include_groups=False)
              .reset_index())
    desc["quartile"] = desc.quartile.map(QLAB)
    desc["rate_per_100k_lower"] = 1e5 * desc.deaths_total_lower / desc.population
    desc["rate_per_100k_upper"] = 1e5 * desc.deaths_total_upper / desc.population
    desc["source"] = SRC
    desc.to_csv(f"{OUT}/step_d_descriptive.csv", index=False)

    coefs, changes, metas = [], [], []
    for label, (base, end) in PERIODS.items():
        for how in ("censored", "zero", "drop"):
            if label == "sensitivity" and how != "censored":
                continue
            tab, ch, meta = fit(m, base, end, how)
            tag = dict(analysis=label, suppressed_handling=how, periods=f"{base}->{end}")
            coefs.append(tab.assign(**tag, source=SRC))
            changes.append(ch.assign(**tag, source=SRC))
            metas.append({**tag, **meta})
    coefs = pd.concat(coefs)
    changes = pd.concat(changes)
    metas = pd.DataFrame(metas)
    coefs.to_csv(f"{OUT}/step_d_coefficients.csv", index=False)
    changes.to_csv(f"{OUT}/step_d_change_by_svi.csv", index=False)
    metas.to_csv(f"{OUT}/step_d_model_info.csv", index=False)

    n_join = m.FIPS.nunique()
    n_fips = warehouse.query("select count(distinct fips) as n from stg_cdc_county_all").n.iloc[0]
    counts = dict(cdc_fips_after_padding_and_dedup=int(n_fips),
                  virginia_duplicate_rows_collapsed=json.loads(dup_names.to_json(orient="records")),
                  counties_in_analysis=int(n_join),
                  cdc_counties_lost_in_join=int((lost.side == "CDC county file, no SVI match").sum()),
                  svi_counties_unmatched=int((lost.side == "SVI, no CDC match").sum()))
    with open(f"{OUT}/step_d_summary.json", "w") as fh:
        json.dump(dict(method=__doc__.split("Model:")[1].strip(), periods=PERIODS, counts=counts, source=SRC,
                       note="All-drug deaths (county file has no opioid split). Area-level associations only; "
                            "they do not describe individuals (ecological fallacy)."), fh, indent=2)

    # chart: % change by SVI quartile, main vs sensitivity (censored model)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    sel = changes[changes.suppressed_handling == "censored"]
    for k, (lab, color) in enumerate((("main", "#2a78d6"), ("sensitivity", "#eb6834"))):
        d = sel[sel.analysis == lab]
        ypos = np.arange(4) + (k - 0.5) * 0.3
        pc = 100 * (d.change_rate_ratio - 1)
        ax.errorbar(pc, ypos, xerr=[pc - 100 * (d.ci_low - 1), 100 * (d.ci_high - 1) - pc], fmt="o", color=color,
                    ms=7, capsize=3, label=f"{d.periods.iloc[0]}")
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_yticks(np.arange(4), [QLAB[q] for q in (1, 2, 3, 4)])
    ax.invert_yaxis()
    ax.set_xlabel("% change in county overdose death rate (all drugs), 95% CI")
    ax.set_title("Change in county overdose deaths by social vulnerability (SVI 2022 quartile)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, title="periods", title_fontsize=8.5)
    fig.text(0.01, 0.01, "Censored negative binomial (suppressed 1-9 as intervals), county-clustered SEs. "
             "Source: " + SRC, fontsize=6.3, color="#52514e", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{OUT}/step_d_change_by_svi.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 220)
    print(json.dumps(counts, indent=1))
    print(lost.to_string(index=False))
    print(desc[desc.Year.isin([2022, 2023, 2025])].drop(columns="source").round(1).to_string(index=False))
    print(changes.drop(columns="source").round(3).to_string(index=False))
    print(metas.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
