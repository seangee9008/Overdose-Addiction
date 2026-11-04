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
from src.count_models import censored_nb, censored_poisson

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


# ============================================================================== R3 / R4 (county)
COUNTY_SRC = ("CDC VSRR county file gb4e-yj24 (all-drug, 12-month-ending) x CDC/ATSDR SVI 2022 (RPL_THEMES, E_TOTPOP)"
              "; Census Vintage 2025 county population where noted")


def county_frame(base, end):
    """County rows for two 12-month-ending windows: base = (year, month), end = (year, month).
    Same SVI inclusion as Step D (RPL_THEMES not missing); quartile formula as in src/step_d.py."""
    m = warehouse.query("""
        select c.fips, c.st_abbrev, c.year, c.month, c.deaths_reported, c.rpl_themes, c.e_totpop,
               p.population as census_pop
        from county_deaths_svi c
        left join population_annual p on p.level = 'county' and p.fips = c.fips and p.year = c.year
        where c.rpl_themes is not null and ((c.year = ? and c.month = ?) or (c.year = ? and c.month = ?))
        order by c.src_row""", [base[0], base[1], end[0], end[1]])
    m["post"] = ((m.year == end[0]) & (m.month == end[1])).astype(float)
    m["quartile"] = np.minimum(np.floor(m.rpl_themes * 4).astype(int) + 1, 4)
    m["censored"] = m.deaths_reported.isna()
    m["state"] = m.fips.str[:2]
    return m


def _county_fit(d, family="nb", cluster="fips", svi="quartile", y=None, censored=None, offset=None):
    """deaths ~ post x SVI with offset log(pop); suppressed cells interval-censored on [1, 9] unless an
    imputed y is given. Returns (coef table, V, info, names)."""
    yv = d.deaths_reported.to_numpy(float) if y is None else y
    cens = d.censored.to_numpy() if censored is None else censored
    cols, names = [np.ones(len(d)), d.post.to_numpy()], ["(Intercept)", "post"]
    if svi == "quartile":
        for q in (2, 3, 4):
            cols.append((d.quartile == q).to_numpy(float))
            names.append(f"SVI_Q{q}")
        for q in (2, 3, 4):
            cols.append(((d.quartile == q) & (d.post == 1)).to_numpy(float))
            names.append(f"post:SVI_Q{q}")
    else:
        cols += [d.rpl_themes.to_numpy(), d.rpl_themes.to_numpy() * d.post.to_numpy()]
        names += ["SVI", "post:SVI"]
    X = np.column_stack(cols)
    off = np.log(d.e_totpop.to_numpy(float)) if offset is None else offset
    fn = censored_nb if family == "nb" else censored_poisson
    tab, V, info = fn(X, yv, cens, np.ones(len(d)), np.full(len(d), 9.0), offset=off,
                      clusters=d[cluster].to_numpy(), names=names)
    return tab, V, info, names


def _gap_row(variant, group, tab, V, info, names, d, note="", term="post:SVI_Q4", scale=1.0, estimand=None):
    t = tab.set_index("term")
    est, se = t.loc[term, "coef"] * scale, t.loc[term, "se"] * scale
    row = dict(check="R3" if group != "R4" else "R4", variant=variant, group=group,
               estimand=estimand or "ratio of change ratios, SVI Q4 vs Q1 (>1 = Q4 declined less)",
               gap_ratio=np.exp(est), ci_low=np.exp(est - 1.96 * se), ci_high=np.exp(est + 1.96 * se),
               log_gap=est, se=se, p=2 * stats.norm.sf(abs(est / se)), n=info["n"], n_censored=info["n_censored"],
               clusters=info["clusters"], converged=info["converged"], note=note)
    if "post:SVI_Q2" in names and term == "post:SVI_Q4":
        idx = [names.index(f"post:SVI_Q{q}") for q in (2, 3, 4)]
        b = tab.coef.to_numpy()[idx]
        W = float(b @ np.linalg.solve(V[np.ix_(idx, idx)], b))
        row["wald_all_quartiles_p"] = float(stats.chi2.sf(W, 3))
        g = tab.coef.to_numpy()[names.index("post")]
        row["q1_change_ratio"] = float(np.exp(g))
        row["q4_change_ratio"] = float(np.exp(g + t.loc["post:SVI_Q4", "coef"]))
    return row


def r3():
    rows = []
    d = county_frame((2022, 12), (2025, 12))
    for fam in ("nb", "poisson"):
        for cl in ("fips", "state"):
            tab, V, info, names = _county_fit(d, fam, cl)
            variant = f"{'negative binomial' if fam == 'nb' else 'Poisson'}, censored, clusters={'county' if cl == 'fips' else 'state'}"
            rows.append(_gap_row(variant, "main" if (fam, cl) == ("nb", "fips") else "model", tab, V, info, names, d))
    tab, V, info, names = _county_fit(d, svi="continuous")
    for scale, lab in ((1.0, "per full 0-1 range of SVI percentile"), (0.75, "per 0.75 (Q4 vs Q1 midpoints)")):
        rows.append(_gap_row(f"continuous SVI, {lab}", "svi", tab, V, info, names, d, term="post:SVI", scale=scale,
                             estimand="ratio of change ratios per unit SVI percentile x scale (>1 = more vulnerable declined less)"))
    for k in (1, 5, 9):
        y = d.deaths_reported.fillna(k).to_numpy(float)
        tab, V, info, names = _county_fit(d, y=y, censored=np.zeros(len(d), bool))
        rows.append(_gap_row(f"suppressed set to {k} (no censoring)", "suppression", tab, V, info, names, d))
    for lab, q4, other in (("bound: gap maximised (Q4 base=1,end=9; others base=9,end=1)", (1, 9), (9, 1)),
                           ("bound: gap minimised (Q4 base=9,end=1; others base=1,end=9)", (9, 1), (1, 9))):
        fill = np.where(d.quartile == 4, np.where(d.post == 0, q4[0], q4[1]), np.where(d.post == 0, other[0], other[1]))
        y = np.where(d.censored, fill, d.deaths_reported).astype(float)
        tab, V, info, names = _county_fit(d, y=y, censored=np.zeros(len(d), bool))
        rows.append(_gap_row(lab, "suppression", tab, V, info, names, d))
    exact = warehouse.query("""select fips from county_deaths_svi where month = 12 and rpl_themes is not null
                               group by fips having sum(is_suppressed::int) = 0""").fips
    ex = d[d.fips.isin(set(exact))].reset_index(drop=True)
    tab, V, info, names = _county_fit(ex)
    rows.append(_gap_row("only counties never suppressed in any Dec 2020-2025 window", "sample", tab, V, info, names, ex,
                         note=f"{ex.fips.nunique()} counties; no censoring needed"))
    nla = d[d.st_abbrev != "LA"].reset_index(drop=True)
    tab, V, info, names = _county_fit(nla)
    rows.append(_gap_row("excluding Louisiana (64 parishes)", "sample", tab, V, info, names, nla))
    cp = d[d.census_pop.notna()].reset_index(drop=True)
    tab, V, info, names = _county_fit(cp, offset=np.log(cp.census_pop.to_numpy(float)))
    rows.append(_gap_row("Census population for each year as exposure", "sample", tab, V, info, names, cp,
                         note=f"{len(d) - len(cp)} county-years without a Census match dropped"))
    for lab, base, end, grp in (("end = 12 months ending Jun 2025 (last 6 months dropped)", (2022, 12), (2025, 6), "R4"),
                                ("end = 12 months ending Dec 2024 (last 12 months dropped)", (2022, 12), (2024, 12), "R4"),
                                ("2023 -> 2025 (Step D sensitivity)", (2023, 12), (2025, 12), "period")):
        dd = county_frame(base, end)
        tab, V, info, names = _county_fit(dd)
        rows.append(_gap_row(lab, grp, tab, V, info, names, dd))
    rows += r3_baseline(d)
    out = pd.DataFrame(rows)
    out["source"] = COUNTY_SRC
    out["connecticut"] = "excluded in every variant: 8 legacy CT counties have no SVI 2022 match (planning regions)"
    out.to_csv(f"{OUT}/r3_r4_county_gap.csv", index=False)
    return out


def r3_baseline(d):
    """Regression-to-the-mean check: model the END window only, adjusting for each county's 2020-2022
    baseline rate (mean of the three Dec 12-month-ending counts). Suppressed baseline cells are set to
    1, 5 or 9 (three variants). Estimand: Q4 vs Q1 rate ratio at the end, at equal baseline rate."""
    base = warehouse.query("""
        select fips, year, deaths_reported from county_deaths_svi
        where month = 12 and year in (2020, 2021, 2022) and rpl_themes is not null""")
    rows = []
    end = d[d.post == 1].reset_index(drop=True)
    for k in (1, 5, 9):
        b = base.assign(v=base.deaths_reported.fillna(k)).groupby("fips").v.sum()
        e = end.merge(b.rename("base3").reset_index(), on="fips", how="inner")
        logbase = np.log((e.base3.to_numpy() / 3 + 0.5) / e.e_totpop.to_numpy(float))
        X = np.column_stack([np.ones(len(e))] + [(e.quartile == q).to_numpy(float) for q in (2, 3, 4)] + [logbase])
        names = ["(Intercept)", "SVI_Q2", "SVI_Q3", "SVI_Q4", "log_baseline_rate"]
        tab, V, info = censored_nb(X, e.deaths_reported.to_numpy(float), e.censored.to_numpy(), np.ones(len(e)),
                                   np.full(len(e), 9.0), offset=np.log(e.e_totpop.to_numpy(float)),
                                   clusters=e.fips.to_numpy(), names=names)
        slope = float(tab.set_index("term").loc["log_baseline_rate", "coef"])
        rows.append(_gap_row(f"baseline-adjusted (2020-22 mean rate; suppressed baseline = {k})", "baseline",
                             tab, V, info, names, e, term="SVI_Q4",
                             estimand="Q4 vs Q1 end-period rate ratio at equal 2020-22 baseline rate (>1 = Q4 higher at end)",
                             note=(f"NOT COMPARABLE, excluded from decision rule: log-baseline slope {slope:.3f} (<1); "
                                   f"{base.deaths_reported.isna().mean():.1%} of baseline cells suppressed, so the noisy baseline "
                                   "lets the Q4 term absorb Q4's higher overall level (baseline measurement error)")))
    return rows


def check_r3_against_step_d(r):
    s = pd.read_csv("outputs/step_d_coefficients.csv")
    s = s[(s.analysis == "main") & (s.suppressed_handling == "censored") & (s.term == "year_2025:SVI_Q4")].iloc[0]
    m = r[r.group == "main"].iloc[0]
    chk = pd.DataFrame([dict(step_d_rate_ratio=s.rate_ratio, here=m.gap_ratio, step_d_se=s.se, here_se=m.se,
                             match=bool(np.isclose(s.rate_ratio, m.gap_ratio, rtol=1e-6) and np.isclose(s.se, m.se, rtol=1e-4)))])
    chk.to_csv(f"{OUT}/r3_check_vs_step_d.csv", index=False)
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
    if "r3" in checks:
        r = r3()
        chk = check_r3_against_step_d(r)
        meta["r3"] = dict(step_d_reproduced=bool(chk.match.all()))
        pd.set_option("display.width", 250)
        print(r[["group", "variant", "gap_ratio", "ci_low", "ci_high", "p", "q1_change_ratio", "q4_change_ratio",
                 "n_censored", "converged"]].round(4).to_string(index=False))
        print(chk.to_string(index=False))
    with open(f"{OUT}/step_r_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or ["r12"])
