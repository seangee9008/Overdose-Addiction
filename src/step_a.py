#!/usr/bin/env python3
"""
Step A: opioid deaths vs. three NSDUH opioid measures, 2021-2024 (national; no pooling of levels).

    python -m src.step_a

Question answered: is the 2021->2024 proportional decline in each NSDUH measure statistically
distinguishable from the proportional decline in opioid overdose deaths?

Method
  * Each change is a log ratio, L = ln(value_2024 / value_2021); % change = 100*(e^L - 1).
  * NSDUH log ratios and their SEs come from survey.contrast(): Taylor linearization of
    ln(p_2024) - ln(p_2021) on the shared NSDUH design (VESTR_C/VEREP), so the covariance between
    years that share PSUs is included. 50 design degrees of freedom.
  * Deaths are a complete count (not a sample), so the main test treats the deaths log ratio as
    fixed. Sensitivity: add Poisson count variance 1/D_2021 + 1/D_2024.
  * Test statistic: t = (L_nsduh - L_deaths) / SE, compared to t(50). Three measures are tested;
    p-values are not adjusted for multiple comparisons.
  * No measure is designated as "addiction"; all three are reported side by side.

Sources
  Deaths: CDC/NCHS VSRR (data.cdc.gov xkb8-kh2a), State=US, Indicator 'Opioids (T40.0-T40.4,T40.6)',
          'Data Value', December 12-month-ending value = calendar-year total.
  NSDUH:  see src/nsduh_estimates.py (comparable 2021-2024 definitions; ANALWT2_C1, VESTR_C, VEREP).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import nsduh
from src import survey as sv
from src.cdc import OPIOIDS, SYNTHETIC_OPIOIDS, load_series

OUT = "outputs"
YEARS = [2021, 2022, 2023, 2024]
BASE, END = 2021, 2024
MEASURES = list(nsduh.COMPARABLE)  # OPI_MISUSE, OPI_OUD, HERYR
SHORT = {"OPI_MISUSE": "Opioid misuse", "OPI_OUD": "Opioid use disorder", "HERYR": "Heroin use"}
DEATHS_SRC = ("CDC VSRR xkb8-kh2a, US, 'Opioids (T40.0-T40.4,T40.6)', Data Value, "
              "Dec 12-month-ending (= calendar year)")
NSDUH_SRC = (f"NSDUH 2021-2024 combined PUF (+2024 single-year PUF for OUD), {nsduh.WEIGHT}, "
             f"{nsduh.STRATA}/{nsduh.PSU}, df={nsduh.DESIGN_DF}")
REMINDER = ("NSDUH mostly captures prescription pain-reliever (pill) misuse, while opioid overdose deaths "
            "are mostly illicitly made fentanyl (synthetic opioids, T40.4). The two series measure "
            "different behaviors in different populations; a household survey also misses many people "
            "at highest risk (e.g. unhoused, incarcerated).")
COLORS = {"deaths": "#2a78d6", "OPI_MISUSE": "#eb6834", "OPI_OUD": "#1baf7a", "HERYR": "#eda100"}


def annual_deaths(indicator):
    s = load_series(indicator=indicator)
    return s[s.month == 12].set_index("year")["deaths"]


def main():
    os.makedirs(OUT, exist_ok=True)
    deaths = annual_deaths(OPIOIDS).loc[YEARS]
    synth = annual_deaths(SYNTHETIC_OPIOIDS).loc[YEARS]
    df, _ = nsduh.load_analysis(demographics=())
    W, H, J, DF = nsduh.WEIGHT, nsduh.STRATA, nsduh.PSU, nsduh.DESIGN_DF
    tcrit = stats.t.ppf(0.975, DF)
    yr = {y: (df.YEAR == y).to_numpy() for y in YEARS}

    # ---- year-by-year index (2021 = 100) with CIs from the linearized log ratio vs 2021
    rows = [dict(measure="deaths", label="Opioid overdose deaths (CDC)", year=y, value=float(deaths[y]),
                 index=100 * deaths[y] / deaths[BASE], index_ci_low=np.nan, index_ci_high=np.nan,
                 source=DEATHS_SRC) for y in YEARS]
    for v in MEASURES:
        for y in YEARS:
            p = sv.proportion(df, v, W, H, J, domain=yr[y], df_design=DF)
            if y == BASE:
                lo = hi = 100.0
            else:
                c = sv.contrast(df, v, W, H, J, yr[BASE], yr[y], df_design=DF)
                lo, hi = 100 * np.exp(c["log_ratio_ci_low"]), 100 * np.exp(c["log_ratio_ci_high"])
            rows.append(dict(measure=v, label=nsduh.COMPARABLE[v], year=y, value=100 * p["p"],
                             value_se=100 * p["se"], index=100 * p["p"] / (
                                 sv.proportion(df, v, W, H, J, domain=yr[BASE], df_design=DF)["p"]),
                             index_ci_low=lo, index_ci_high=hi, source=NSDUH_SRC))
    comp = pd.DataFrame(rows)
    comp.round(4).to_csv(f"{OUT}/step_a_comparison.csv", index=False)

    # ---- tests: NSDUH decline vs deaths decline, 2021 -> 2024
    L_d = float(np.log(deaths[END] / deaths[BASE]))
    var_pois = float(1 / deaths[BASE] + 1 / deaths[END])
    tests = []
    for v in MEASURES:
        c = sv.contrast(df, v, W, H, J, yr[BASE], yr[END], df_design=DF)
        diff = c["log_ratio"] - L_d
        t_fixed = diff / c["log_ratio_se"]
        se_p = np.sqrt(c["log_ratio_se"] ** 2 + var_pois)
        t_pois = diff / se_p
        tests.append(dict(
            measure=v, label=nsduh.COMPARABLE[v],
            nsduh_pct_2021=100 * c["p_a"], nsduh_pct_2024=100 * c["p_b"],
            nsduh_pct_change=100 * np.expm1(c["log_ratio"]),
            nsduh_pct_change_ci_low=100 * np.expm1(c["log_ratio_ci_low"]),
            nsduh_pct_change_ci_high=100 * np.expm1(c["log_ratio_ci_high"]),
            nsduh_change_pvalue_vs_zero=c["log_ratio_pvalue"],
            deaths_2021=int(deaths[BASE]), deaths_2024=int(deaths[END]),
            deaths_pct_change=100 * np.expm1(L_d),
            log_ratio_nsduh=c["log_ratio"], log_ratio_se=c["log_ratio_se"], log_ratio_deaths=L_d,
            difference_in_log_ratios=diff,
            difference_ci_low=diff - tcrit * c["log_ratio_se"], difference_ci_high=diff + tcrit * c["log_ratio_se"],
            t_deaths_fixed=t_fixed, p_deaths_fixed=2 * stats.t.sf(abs(t_fixed), DF),
            t_deaths_poisson=t_pois, p_deaths_poisson=2 * stats.t.sf(abs(t_pois), DF),
            deaths_per_case_ratio_2024_vs_2021=float(np.exp(L_d - c["log_ratio"])),
            df=DF, nsduh_source=NSDUH_SRC, deaths_source=DEATHS_SRC))
    tests = pd.DataFrame(tests)
    tests.round(6).to_csv(f"{OUT}/step_a_tests.csv", index=False)

    # ---- context for the reminder: what each source is mostly made of
    ctx = [dict(item="Share of opioid overdose deaths involving synthetic opioids (T40.4)", year=y,
                value_pct=100 * synth[y] / deaths[y], se=np.nan,
                source="CDC VSRR xkb8-kh2a, US, Dec 12-month-ending: 'Synthetic opioids, excl. methadone (T40.4)' / "
                       "'Opioids (T40.0-T40.4,T40.6)'; categories overlap (multiple drugs per death)")
           for y in (BASE, END)]
    df["PNR_ONLY"] = ((df.PNRNMYR == 1) & (df.HERYR == 0)).astype(float)
    df.loc[df.OPI_MISUSE == 0, "PNR_ONLY"] = np.nan  # domain = people who misused opioids
    for y in (BASE, END):
        r = sv.proportion(df, "PNR_ONLY", W, H, J, domain=yr[y], df_design=DF)
        ctx.append(dict(item="Share of NSDUH opioid misusers who misused Rx pain relievers but did not use heroin",
                        year=y, value_pct=100 * r["p"], se=100 * r["se"], source=NSDUH_SRC + "; PNRNMYR=1 & HERYR=0 among OPI_MISUSE=1"))
    # Illegally made fentanyl (IMF): asked 2022 onward (codebook: -9 "not collected" for 2021 records).
    imf, _ = nsduh.load(columns=["YEAR", "IMFYR", W, H, J])
    for y in (2022, 2023, 2024):
        r = sv.proportion(imf, "IMFYR", W, H, J, domain=(imf.YEAR == y).to_numpy(), df_design=DF)
        ctx.append(dict(item="Past-year use of illegally made fentanyl, people 12+ (IMFYR; not asked in 2021)",
                        year=y, value_pct=100 * r["p"], se=100 * r["se"],
                        source=f"NSDUH 2021-2024 combined PUF, IMFYR, {W}, {H}/{J}"))
    ctx = pd.DataFrame(ctx)
    ctx.round(4).to_csv(f"{OUT}/step_a_context.csv", index=False)

    # ---- chart: one axis, everything indexed to 2021 = 100
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    offsets = {"deaths": 0, "OPI_MISUSE": -0.1, "OPI_OUD": 0.05, "HERYR": 0.12}
    for m in ["deaths", *MEASURES]:
        d = comp[comp.measure == m]
        x = d.year + offsets[m]
        name = "Opioid overdose deaths (CDC)" if m == "deaths" else f"{SHORT[m]} (NSDUH)"
        ax.plot(x, d["index"], "o-" if m == "deaths" else "s-", color=COLORS[m], lw=2, ms=7, label=name)
        if m != "deaths":
            yerr = np.clip([d["index"] - d.index_ci_low, d.index_ci_high - d["index"]], 0, None)  # 2021: float noise
            ax.errorbar(x, d["index"], yerr=yerr,
                        fmt="none", ecolor=COLORS[m], elinewidth=1.2, capsize=3)
    ax.axhline(100, color="#52514e", lw=0.8)
    ax.set_xticks(YEARS)
    ax.set_ylabel("Index, 2021 = 100 (bars: 95% CI vs 2021)")
    ax.set_title("Opioid deaths vs three NSDUH opioid measures, 2021-2024", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.text(0.01, 0.01, "Reminder: " + REMINDER, fontsize=6.5, color="#52514e", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{OUT}/step_a_index.png", dpi=150)
    plt.close(fig)

    summary = dict(reminder=REMINDER, method=__doc__.split("Method")[1].split("Sources")[0].strip(),
                   deaths_source=DEATHS_SRC, nsduh_source=NSDUH_SRC,
                   tests=json.loads(tests.to_json(orient="records")),
                   context=json.loads(ctx.to_json(orient="records")),
                   files=["outputs/step_a_comparison.csv", "outputs/step_a_tests.csv",
                          "outputs/step_a_context.csv", "outputs/step_a_index.png"])
    with open(f"{OUT}/step_a_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    pd.set_option("display.width", 200)
    print(comp[["measure", "year", "value", "index", "index_ci_low", "index_ci_high"]].round(2).to_string(index=False))
    print(tests[["measure", "nsduh_pct_change", "nsduh_pct_change_ci_low", "nsduh_pct_change_ci_high",
                 "nsduh_change_pvalue_vs_zero", "deaths_pct_change", "difference_in_log_ratios",
                 "p_deaths_fixed", "p_deaths_poisson", "deaths_per_case_ratio_2024_vs_2021"]].round(4).to_string(index=False))
    print(ctx[["item", "year", "value_pct", "se"]].round(2).to_string(index=False))
    print("\nREMINDER:", REMINDER)


if __name__ == "__main__":
    main()
