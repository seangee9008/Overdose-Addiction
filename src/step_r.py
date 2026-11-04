#!/usr/bin/env python3
"""
Phase 4a robustness and sensitivity checks. Reuses the existing estimators; changes no existing output.

    python -m src.step_r            # all implemented checks
    python -m src.step_r r12        # R1 + R2 only

Outputs go to outputs/robustness/. Sign convention for the death-vs-survey comparison:

    d = log(death_end / death_start) - log(prevalence_end / prevalence_start)

so d < 0 means deaths fell (proportionally) faster than the survey measure; d = 0 means the same
proportional change.

R1 (start-year sensitivity) and R2 (uncertainty on d)
  * Start years 2021, 2022, 2023; end year 2024. NSDUH 2020 is never used (the combined 2021-2024
    file starts in 2021). Survey measures: comparable OPI_MISUSE, OPI_OUD, HERYR (src/nsduh.py).
  * Deaths: CDC VSRR 'Opioids (T40.0-T40.4,T40.6)', Data Value, December 12-month-ending value =
    calendar-year count. Death scale: counts, and rates per 100k with Census July 1 national
    population (population_annual mart).
  * Variance of d (three methods):
      shared_design (main): Taylor linearization of the survey log ratio on the shared NSDUH design
          (years share strata/PSUs per the codebook, so their covariance is included), plus Poisson
          variance of the death log ratio, 1/D_start + 1/D_end. t on 50 df.
      independent_years: var(log p_start) + var(log p_end), ignoring the shared-PSU covariance
          (the specification as originally requested; sensitivity only).
      rao_wu_bootstrap: Rao-Wu rescaling bootstrap (m_h = n_h - 1, i.e. one of the 2 PSUs per stratum
          at double weight), PSUs resampled jointly across years, B replicates, fixed seed; SE of the
          survey part from the replicates plus the same Poisson death variance.
  * Holm adjustment across the 3 survey measures within each start year x death scale x method.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from src import nsduh, warehouse
from src import survey as sv
from src.cdc import OPIOIDS, load_series

OUT = "outputs/robustness"
SEED = 20261103
B_BOOT = 2000
END = 2024
STARTS = (2021, 2022, 2023)
MEASURES = list(nsduh.COMPARABLE)                     # OPI_MISUSE, OPI_OUD, HERYR
DF = nsduh.DESIGN_DF
DEATHS_SRC = "CDC VSRR xkb8-kh2a, US, '{ind}', Data Value, Dec 12-month-ending (= calendar year)"
POP_SRC = "Census Population Estimates Vintage 2025 (co-est2025-alldata), July 1, nation = 50 states + DC"
NSDUH_SRC = (f"NSDUH 2021-2024 combined PUF (+2024 single-year PUF for OUD), {nsduh.WEIGHT}, "
             f"{nsduh.STRATA}/{nsduh.PSU}, df={DF}")


def annual_deaths(indicator):
    s = load_series(indicator=indicator)
    return s[s.month == 12].set_index("year")["deaths"]


def national_population():
    p = warehouse.query("select year, population from population_annual where level = 'nation' order by year")
    return p.set_index("year")["population"]


def _row(measure, start, scale, method, L_death, L_surv, se, deaths, pop, extra):
    d = L_death - L_surv
    t = d / se
    tcrit = stats.t.ppf(0.975, DF)
    return dict(measure=measure, label=nsduh.COMPARABLE[measure], start=start, end=END, death_scale=scale,
                variance_method=method, deaths_start=int(deaths[start]), deaths_end=int(deaths[END]),
                pop_start=int(pop[start]), pop_end=int(pop[END]),
                death_pct_change=100 * np.expm1(L_death), survey_pct_change=100 * np.expm1(L_surv),
                log_ratio_death=L_death, log_ratio_survey=L_surv, d=d, se=se,
                ci_low=d - tcrit * se, ci_high=d + tcrit * se, t=t, p=2 * stats.t.sf(abs(t), DF), df=DF, **extra)


def r12(indicator=OPIOIDS, tag="opioids"):
    """R1 + R2 for one death indicator. Returns the results DataFrame (also written to CSV)."""
    deaths = annual_deaths(indicator)
    pop = national_population()
    df, _ = nsduh.load_analysis(demographics=())
    W, H, J = nsduh.WEIGHT, nsduh.STRATA, nsduh.PSU
    yr = {y: (df.YEAR == y).to_numpy() for y in (*STARTS, END)}
    rows = []
    for start in STARTS:
        var_pois = 1 / deaths[start] + 1 / deaths[END]
        L_count = float(np.log(deaths[END] / deaths[start]))
        L_rate = float(np.log((deaths[END] / pop[END]) / (deaths[start] / pop[start])))
        for k, m in enumerate(MEASURES):
            c = sv.contrast(df, m, W, H, J, yr[start], yr[END], df_design=DF)
            ind = sv.log_ratio_independent(df, m, W, H, J, yr[start], yr[END], df_design=DF)
            seed = SEED + 100 * (start - 2000) + k
            est, reps = sv.rao_wu_log_ratio(df, m, W, H, J, yr[start], yr[END], B=B_BOOT, seed=seed)
            se_boot = float(np.std(reps, ddof=1))
            base = dict(survey_pct_start=100 * c["p_a"], survey_pct_end=100 * c["p_b"])
            for scale, L_death in (("count", L_count), ("rate_per_100k", L_rate)):
                rows.append(_row(m, start, scale, "shared_design", L_death, c["log_ratio"],
                                 np.sqrt(c["log_ratio_se"] ** 2 + var_pois), deaths, pop,
                                 dict(base, survey_log_ratio_se=c["log_ratio_se"], B=np.nan, seed=np.nan,
                                      pct_ci_low=np.nan, pct_ci_high=np.nan)))
                rows.append(_row(m, start, scale, "independent_years", L_death, ind["log_ratio"],
                                 np.sqrt(ind["log_ratio_se"] ** 2 + var_pois), deaths, pop,
                                 dict(base, survey_log_ratio_se=ind["log_ratio_se"], B=np.nan, seed=np.nan,
                                      pct_ci_low=np.nan, pct_ci_high=np.nan)))
                lo, hi = np.percentile(L_death - reps, [2.5, 97.5])   # deaths fixed in the percentile interval
                rows.append(_row(m, start, scale, "rao_wu_bootstrap", L_death, est,
                                 np.sqrt(se_boot ** 2 + var_pois), deaths, pop,
                                 dict(base, survey_log_ratio_se=se_boot, B=B_BOOT, seed=seed,
                                      pct_ci_low=lo, pct_ci_high=hi)))
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, g in out.groupby(["start", "death_scale", "variance_method"]):
        out.loc[g.index, "p_holm"] = sv.holm(g.p.to_numpy())
    out.insert(0, "death_indicator", indicator)
    out["deaths_source"] = DEATHS_SRC.format(ind=indicator)
    out["population_source"] = POP_SRC
    out["survey_source"] = NSDUH_SRC
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/r1_r2_declines_{tag}.csv", index=False)
    return out


def check_against_step_a(r):
    """The 2021 count / shared-design rows must reproduce Step A (same estimator, opposite sign)."""
    a = pd.read_csv("outputs/step_a_tests.csv").set_index("measure")
    sub = r[(r.start == 2021) & (r.death_scale == "count") & (r.variance_method == "shared_design")].set_index("measure")
    rows = []
    for m in MEASURES:
        rows.append(dict(measure=m, d_here=sub.loc[m, "d"], minus_step_a_difference=-a.loc[m, "difference_in_log_ratios"],
                         p_here=sub.loc[m, "p"], step_a_p_deaths_poisson=a.loc[m, "p_deaths_poisson"]))
    chk = pd.DataFrame(rows)
    chk["match"] = (np.isclose(chk.d_here, chk.minus_step_a_difference, atol=1e-6)
                    & np.isclose(chk.p_here, chk.step_a_p_deaths_poisson, atol=1e-6))
    chk.to_csv(f"{OUT}/r2_check_vs_step_a.csv", index=False)
    return chk


def main(checks):
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    if "r12" in checks:
        r = r12()
        chk = check_against_step_a(r)
        meta["r12"] = dict(seed=SEED, B=B_BOOT, starts=list(STARTS), end=END, measures=MEASURES,
                           step_a_reproduced=bool(chk.match.all()))
        pd.set_option("display.width", 250)
        cols = ["measure", "start", "death_scale", "variance_method", "death_pct_change", "survey_pct_change",
                "d", "ci_low", "ci_high", "p", "p_holm"]
        print(r[cols].round(4).to_string(index=False))
        print(chk.to_string(index=False))
    with open(f"{OUT}/step_r_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or ["r12"])
