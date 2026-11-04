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
from src import segmented as seg
from src.breaks import fast_fit, fit_no_break, fit_one_break, fit_two_breaks
from src.cdc import ALL_DRUG, OPIOIDS, SYNTHETIC_OPIOIDS, load_series
from src.count_models import censored_nb, censored_poisson
from src.step_b import mbb, stat10, stat21

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


def annual_deaths(indicator, value_col="Data Value"):
    s = load_series(indicator=indicator, value_col=value_col)
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


def r12(indicator=OPIOIDS, tag="opioids", value_col="Data Value"):
    """R1 + R2 for one death indicator. Returns the results DataFrame (also written to CSV)."""
    deaths = annual_deaths(indicator, value_col)
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
    out.insert(1, "death_value_col", value_col)
    out["deaths_source"] = DEATHS_SRC.format(ind=indicator).replace("Data Value", value_col)
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


# ============================================================================== F3: R4 / R5 / R6
B_ROLL = 1000          # block-bootstrap replicates for the rolling-series tests (Step B used 2000)
B_MONTHLY = 500        # replicates for the monthly-count bootstrap tests
BLOCK = 12
WONDER_SERIES = "wonder_drug_induced_monthly"


def rolling_break_tests(y, dates, seed):
    """Step B's procedure (block bootstrap, L = 12) on a rolling 12-month-ending series."""
    obs = fast_fit(y, 12)
    t1, t2, fit2, _ = fit_two_breaks(y, 12)
    _, fit1, _ = fit_one_break(y, 12)
    fit0, _ = fit_no_break(y)
    rng = np.random.default_rng(seed)
    f21 = np.array([stat21(fast_fit(fit1 + mbb(y - fit1, BLOCK, rng), 12)) for _ in range(B_ROLL)])
    f10 = np.array([stat10(fast_fit(fit0 + mbb(y - fit0, BLOCK, rng), 12)) for _ in range(B_ROLL)])
    bt = np.array([fast_fit(fit2 + mbb(y - fit2, BLOCK, rng), 12)["tau2"] for _ in range(B_ROLL)])
    ci = [np.percentile(bt[:, k], [2.5, 97.5]) for k in (0, 1)]
    n = len(y)
    return dict(break_1=dates[t1], break_1_ci_low=dates[int(np.floor(ci[0][0]))], break_1_ci_high=dates[min(int(np.ceil(ci[0][1])), n - 1)],
                break_2=dates[t2], break_2_ci_low=dates[int(np.floor(ci[1][0]))], break_2_ci_high=dates[min(int(np.ceil(ci[1][1])), n - 1)],
                p_1_vs_0=(1 + (f10 >= stat10(obs)).sum()) / (B_ROLL + 1), p_2_vs_1=(1 + (f21 >= stat21(obs)).sum()) / (B_ROLL + 1),
                B=B_ROLL, block=BLOCK, seed=seed, n_months=n, first=dates[0], last=dates[-1], t1=t1, t2=t2)


def hac_break_tests(y, t1, t2, lags):
    """Segmented OLS on the rolling series at the ESTIMATED breaks (each segment its own intercept and
    slope, as in Step B); Wald tests that adjacent segments are equal, Newey-West HAC covariance.
    Conditional on the estimated break dates, so these p-values are optimistic (anti-conservative)."""
    import statsmodels.api as sm
    n = len(y)
    t = np.arange(n, dtype=float)
    seg_id = np.where(t < t1, 0, np.where(t < t2, 1, 2))
    X = np.column_stack([(seg_id == k).astype(float) for k in range(3)] + [t * (seg_id == k) for k in range(3)])
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {}
    for name, (a, b) in (("break_1", (0, 1)), ("break_2", (1, 2))):
        R = np.zeros((2, 6))
        R[0, a], R[0, b], R[1, 3 + a], R[1, 3 + b] = 1, -1, 1, -1
        out[name] = float(fit.wald_test(R, scalar=True).pvalue)
    return out


def r5_rolling():
    """F3 on the CDC rolling series: R4 truncation, Predicted Value, R6 synthetic opioids, HAC (R5a),
    and annual non-overlapping windows (R5b)."""
    rows, hac_rows, annual_rows = [], [], []
    variants = [("opioids", OPIOIDS, "Data Value", None), ("all_drug", ALL_DRUG, "Data Value", None),
                ("opioids", OPIOIDS, "Data Value", 6), ("opioids", OPIOIDS, "Data Value", 12),
                ("all_drug", ALL_DRUG, "Data Value", 6), ("all_drug", ALL_DRUG, "Data Value", 12),
                ("opioids", OPIOIDS, "Predicted Value", None), ("synthetic_opioids", SYNTHETIC_OPIOIDS, "Data Value", None)]
    for i, (tag, ind, vcol, drop) in enumerate(variants):
        s = load_series(indicator=ind, value_col=vcol)
        if drop:
            s = s.iloc[:-drop]
        y = s.deaths.to_numpy(float)
        dates = s.date.dt.strftime("%Y-%m").to_numpy()
        r = rolling_break_tests(y, dates, SEED + 7 * i)
        label = f"{tag}, {vcol}" + (f", last {drop} months dropped" if drop else "")
        grp = "R4" if (drop or vcol != "Data Value") else ("R6" if tag == "synthetic_opioids" else "reference")
        rows.append(dict(check=grp, series=tag, variant=label, indicator=ind, value_col=vcol, months_dropped=drop or 0,
                         **{k: v for k, v in r.items() if k not in ("t1", "t2")},
                         source=f"CDC VSRR xkb8-kh2a, US, '{ind}', {vcol}, 12-month-ending"))
        if drop is None and vcol == "Data Value":
            for lags in (12, 24):
                h = hac_break_tests(y, r["t1"], r["t2"], lags)
                hac_rows.append(dict(check="R5a", series=tag, hac_lags=lags, break_1=r["break_1"], break_2=r["break_2"],
                                     p_break_1_hac=h["break_1"], p_break_2_hac=h["break_2"],
                                     note="Wald test that adjacent segments are equal, at the estimated breaks "
                                          "(conditional on estimated dates: optimistic p-values)",
                                     source=f"CDC VSRR xkb8-kh2a, US, '{ind}', Data Value"))
            ann = s[s.month == 12]
            ya = ann.deaths.to_numpy(float)
            years = ann.year.to_numpy()
            p1 = seg.profile(ya, 1, None, min_seg=3)
            lo, hi = seg.profile_ci(p1)
            p0 = seg.profile(ya, 0, None, min_seg=3)
            annual_rows.append(dict(check="R5b", series=tag, years=f"{years[0]}-{years[-1]}", n_years=len(ya),
                                    break_year=int(years[p1['best'][0]]), break_year_ci_low=int(years[lo]),
                                    break_year_ci_high=int(years[hi]),
                                    scaled_deviance_drop_1_vs_0=(p0["dev"] - p1["dev"]) / p1["phi"], dispersion=p1["phi"],
                                    note="non-overlapping calendar years (Dec 12-month-ending values); quasi-Poisson kink "
                                         "model, profile CI; too few points for a 2-break model",
                                    source=f"CDC VSRR xkb8-kh2a, US, '{ind}', Data Value, December values"))
    pd.DataFrame(rows).to_csv(f"{OUT}/r4_r5_r6_rolling_breaks.csv", index=False)
    pd.DataFrame(hac_rows).to_csv(f"{OUT}/r5a_hac_break_tests.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(f"{OUT}/r5b_annual_breaks.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(hac_rows), pd.DataFrame(annual_rows)


def r5_monthly():
    """F3 on TRUE monthly counts (CDC WONDER). Partial months are excluded. Kink model with
    month-of-year seasonality; profile and block-bootstrap break-date CIs; bootstrap tests."""
    w = warehouse.query("select * from wonder_deaths_monthly where series = ? order by month_date", [WONDER_SERIES])
    qp = w.query_parameters.iloc[0]
    w = w[~w.is_partial].reset_index(drop=True)
    rows = []
    for i, drop in enumerate((0, 6, 12)):
        d = w.iloc[:len(w) - drop] if drop else w
        y = d.deaths.to_numpy(float)
        months = d.month.to_numpy()
        dates = pd.to_datetime(d.month_date).dt.strftime("%Y-%m").to_numpy()
        seed = SEED + 1000 + i
        stat10_, p10, _, p1 = seg.bootstrap_test_extra_break(y, months, B_MONTHLY, BLOCK, seed, k0=0)
        stat21_, p21, _, p2 = seg.bootstrap_test_extra_break(y, months, B_MONTHLY, BLOCK, seed + 1, k0=1)
        lo1, hi1 = seg.profile_ci(p1)
        _, bt1 = seg.bootstrap_break_dates(y, 1, months, B_MONTHLY, BLOCK, seed + 2)
        b1lo, b1hi = np.percentile(bt1[:, 0], [2.5, 97.5])
        lo2a, hi2a = seg.profile_ci(p2, 0)
        lo2b, hi2b = seg.profile_ci(p2, 1)
        rows.append(dict(check="R5" if not drop else "R4", series=WONDER_SERIES, months_dropped=drop,
                         first=dates[0], last=dates[-1], n_months=len(y),
                         one_break=dates[p1["best"][0]], one_break_profile_ci=f"{dates[lo1]} to {dates[hi1]}",
                         one_break_bootstrap_ci=f"{dates[int(np.floor(b1lo))]} to {dates[int(np.ceil(b1hi))]}",
                         two_breaks=f"{dates[p2['best'][0]]}, {dates[p2['best'][1]]}",
                         two_break_1_profile_ci=f"{dates[lo2a]} to {dates[hi2a]}",
                         two_break_2_profile_ci=f"{dates[lo2b]} to {dates[hi2b]}",
                         stat_1_vs_0=stat10_, p_1_vs_0=p10, stat_2_vs_1=stat21_, p_2_vs_1=p21,
                         dispersion_1_break=p1["phi"], B=B_MONTHLY, block=BLOCK, seed=seed,
                         model="log E[y] = trend + kink(s) + month-of-year; quasi-Poisson; partial months excluded",
                         wonder_query=qp, source="CDC WONDER Provisional Mortality, monthly (final 2019-2024, provisional 2025+)"))
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/r5_monthly_breaks.csv", index=False)
    return out


# ============================================================================== decision rule + summary
DECISION_RULE = {
    "F1": "d < 0 in every variant; 95% CI excludes 0 in the main specification and in >= 75% of variants; "
          "Holm-adjusted p < 0.05 in the main specification. 'partially' if signs hold but CI criterion fails; "
          "'no' if any sign flips.",
    "F2": "95% CI for d includes 0 in the main specification and in >= 75% of variants ('partially' if 50-75%; "
          "'no' if the main CI excludes 0).",
    "F3a": "1 break beats 0 at p < 0.05 by every method AND every break-date CI overlaps calendar 2023 "
           "('partially' if significant everywhere but some CI does not overlap 2023; 'no' if not significant "
           "by the main method).",
    "F3b": "'2020 break not robust' holds if 2 breaks vs 1 gives p >= 0.05 by the corrected method (true monthly "
           "counts); 'no' if p < 0.05.",
    "F5": "Q4/Q1 gap ratio > 1 in every variant; 95% CI excludes 1 in the main specification and in >= 75% of "
          "variants ('partially' if signs hold but CI criterion fails; 'no' if any sign flips).",
}


def _fmt(est, lo, hi, nd=3):
    return f"{est:.{nd}f} ({lo:.{nd}f}, {hi:.{nd}f})"


def _overlaps_2023(lo, hi):
    return str(lo)[:4] <= "2023" <= str(hi)[:4]


def summarize():
    files = {"opioids": "r1_r2_declines_opioids.csv", "synthetic_opioids": "r1_r2_declines_synthetic_opioids.csv",
             "all_drug": "r1_r2_declines_all_drug.csv", "opioids_predicted_value": "r1_r2_declines_opioids_predicted_value.csv"}
    decl = pd.concat([pd.read_csv(f"{OUT}/{f}").assign(outcome=k) for k, f in files.items()], ignore_index=True)
    decl["is_main"] = ((decl.outcome == "opioids") & (decl.start == 2021) & (decl.death_scale == "count")
                       & (decl.variance_method == "shared_design"))
    var_rows, verdicts = [], []

    def variant_label(r):
        return f"{r.outcome}; start {r.start}; {r.death_scale}; {r.variance_method}"

    for finding, measure in (("F1 (opioid misuse)", "OPI_MISUSE"), ("F1 (OUD)", "OPI_OUD"), ("F2 (heroin)", "HERYR")):
        g = decl[decl.measure == measure]
        main = g[g.is_main].iloc[0]
        rest = g[~g.is_main]
        for r in g.itertuples():
            var_rows.append(dict(finding=finding, variant=("MAIN: " if r.is_main else "") + variant_label(r),
                                 estimate=r.d, ci_low=r.ci_low, ci_high=r.ci_high, p=r.p, p_holm=r.p_holm,
                                 estimate_95ci=_fmt(r.d, r.ci_low, r.ci_high),
                                 sign_ok=bool(r.d < 0) if finding.startswith("F1") else np.nan,
                                 ci_criterion_met=bool(r.ci_high < 0) if finding.startswith("F1") else bool(r.ci_low <= 0 <= r.ci_high)))
        if finding.startswith("F1"):
            signs = int((g.d < 0).sum())
            share = float((rest.ci_high < 0).mean())
            if signs < len(g):
                v = "no"
            elif main.ci_high < 0 and main.p_holm < 0.05 and share >= 0.75:
                v = "yes"
            else:
                v = "partially"
            note = (f"d<0 in {signs}/{len(g)}; CI excludes 0 in {int((rest.ci_high < 0).sum())}/{len(rest)} variants "
                    f"({share:.0%}); main Holm p = {main.p_holm:.4f}")
        else:
            inc = (rest.ci_low <= 0) & (rest.ci_high >= 0)
            share = float(inc.mean())
            main_inc = main.ci_low <= 0 <= main.ci_high
            v = "no" if not main_inc else ("yes" if share >= 0.75 else ("partially" if share >= 0.5 else "no"))
            note = (f"CI includes 0 in {int(inc.sum())}/{len(rest)} variants ({share:.0%}); sign of d: "
                    f"{int((g.d > 0).sum())} positive, {int((g.d < 0).sum())} negative")
        verdicts.append(dict(finding=finding, original=f"d = {_fmt(main.d, main.ci_low, main.ci_high)}, p = {main.p:.4f}",
                             n_variants=len(rest), holds=v, notes=note, rule=DECISION_RULE[finding[:2]]))

    roll = pd.read_csv(f"{OUT}/r4_r5_r6_rolling_breaks.csv")
    hac = pd.read_csv(f"{OUT}/r5a_hac_break_tests.csv")
    ann = pd.read_csv(f"{OUT}/r5b_annual_breaks.csv")
    mon = pd.read_csv(f"{OUT}/r5_monthly_breaks.csv")
    f3 = []
    for r in roll.itertuples():
        f3.append(dict(variant=f"rolling: {r.variant}", sig=r.p_1_vs_0 < 0.05, p=r.p_1_vs_0,
                       timing=f"{r.break_2} ({r.break_2_ci_low} to {r.break_2_ci_high})",
                       overlaps=_overlaps_2023(r.break_2_ci_low, r.break_2_ci_high),
                       is_main=(r.variant == "opioids, Data Value")))
    for r in hac.itertuples():
        f3.append(dict(variant=f"HAC lag {r.hac_lags}: {r.series} (conditional on estimated date)", sig=r.p_break_2_hac < 0.05,
                       p=r.p_break_2_hac, timing=f"{r.break_2} (no CI)", overlaps=np.nan, is_main=False))
    for r in ann.itertuples():
        lo, hi = r.break_year_ci_low + 1, r.break_year_ci_high + 1
        f3.append(dict(variant=f"annual non-overlapping: {r.series}", sig=r.scaled_deviance_drop_1_vs_0 > stats.chi2.ppf(0.95, 1),
                       p=float(stats.chi2.sf(r.scaled_deviance_drop_1_vs_0, 1)),
                       timing=f"bend at {r.break_year} -> change during {r.break_year + 1} (CI {lo}-{hi})",
                       overlaps=_overlaps_2023(lo, hi), is_main=False))
    for r in mon.itertuples():
        lab = f"monthly counts (WONDER), {r.months_dropped} months dropped" if r.months_dropped else "monthly counts (WONDER)"
        a, b = r.one_break_profile_ci.split(" to ")
        f3.append(dict(variant=f"{lab}: one-break profile CI", sig=r.p_1_vs_0 < 0.05, p=r.p_1_vs_0,
                       timing=f"{r.one_break} ({r.one_break_profile_ci})", overlaps=_overlaps_2023(a, b), is_main=False))
        a, b = r.one_break_bootstrap_ci.split(" to ")
        f3.append(dict(variant=f"{lab}: one-break bootstrap CI", sig=r.p_1_vs_0 < 0.05, p=r.p_1_vs_0,
                       timing=f"{r.one_break} ({r.one_break_bootstrap_ci})", overlaps=_overlaps_2023(a, b), is_main=False))
        a, b = r.two_break_2_profile_ci.split(" to ")
        f3.append(dict(variant=f"{lab}: later break of 2-break model", sig=r.p_1_vs_0 < 0.05, p=r.p_1_vs_0,
                       timing=f"{r.two_breaks.split(', ')[1]} ({r.two_break_2_profile_ci})", overlaps=_overlaps_2023(a, b),
                       is_main=False))
    f3 = pd.DataFrame(f3)
    for r in f3.itertuples():
        var_rows.append(dict(finding="F3a (2023 break)", variant=("MAIN: " if r.is_main else "") + r.variant,
                             estimate_95ci=r.timing, p=r.p, sign_ok=r.sig, ci_criterion_met=r.overlaps))
    main = f3[f3.is_main].iloc[0]
    with_ci = f3[f3.overlaps.notna()]
    v = ("no" if not main.sig else ("yes" if f3.sig.all() and with_ci.overlaps.astype(bool).all() else "partially"))
    sb = pd.read_csv("outputs/step_b_tests.csv")
    sb = sb[sb.block_len == 12].set_index("series")
    verdicts.append(dict(finding="F3a (2023 break)",
                         original=(f"Step B (B=2000): opioids 1 vs 0 p = {sb.loc['opioids', 'p_1vs0']:.4f}, all-drug "
                                   f"{sb.loc['all_drug', 'p_1vs0']:.4f}; re-run reference here (B={B_ROLL}): {main.p:.4f}"),
                         n_variants=len(f3) - 1, holds=v,
                         notes=(f"1 vs 0 significant in {int(f3.sig.sum())}/{len(f3)} rows; break-date CI overlaps 2023 in "
                                f"{int(with_ci.overlaps.astype(bool).sum())}/{len(with_ci)} rows with a CI. Annual rows map a "
                                "bend at year Y to a change during Y+1."),
                         rule=DECISION_RULE["F3a"]))

    f3b = [dict(variant=f"rolling: {r.variant}", p=r.p_2_vs_1, is_corrected=False) for r in roll.itertuples()]
    f3b += [dict(variant=(f"monthly counts (WONDER), {r.months_dropped} months dropped" if r.months_dropped
                          else "MAIN (corrected method): monthly counts (WONDER)"), p=r.p_2_vs_1,
                 is_corrected=(r.months_dropped == 0), timing=f"{r.two_breaks}; earlier break CI {r.two_break_1_profile_ci}")
             for r in mon.itertuples()]
    for r in f3b:
        var_rows.append(dict(finding="F3b (2020 break not robust)", variant=r["variant"], p=r["p"],
                             estimate_95ci=r.get("timing", ""), ci_criterion_met=bool(r["p"] >= 0.05)))
    corr = [r for r in f3b if r["is_corrected"]][0]
    roll_ns = sum(r["p"] >= 0.05 for r in f3b if r["variant"].startswith("rolling"))
    verdicts.append(dict(finding="F3b (2020 break not robust)",
                         original=(f"Step B (B=2000): 2 vs 1 p = {sb.loc['all_drug', 'p_2vs1']:.4f} (all-drug), "
                                   f"{sb.loc['opioids', 'p_2vs1']:.4f} (opioids)"),
                         n_variants=len(f3b) - 1, holds="yes" if corr["p"] >= 0.05 else "no",
                         notes=(f"corrected method (true monthly counts, all-drug drug-induced): 2 vs 1 p = {corr['p']:.4f}, "
                                f"earlier break {corr['timing']}; rolling series: p >= 0.05 in {roll_ns}/8 variants. "
                                "Opioid-specific monthly counts not yet available."),
                         rule=DECISION_RULE["F3b"]))

    cg = pd.read_csv(f"{OUT}/r3_r4_county_gap.csv")
    cg = cg[cg.group != "baseline"]
    for r in cg.itertuples():
        var_rows.append(dict(finding="F5 (SVI gap)", variant=("MAIN: " if r.group == "main" else "") + r.variant,
                             estimate=r.gap_ratio, ci_low=r.ci_low, ci_high=r.ci_high, p=r.p,
                             estimate_95ci=_fmt(r.gap_ratio, r.ci_low, r.ci_high), sign_ok=bool(r.gap_ratio > 1),
                             ci_criterion_met=bool(r.ci_low > 1)))
    main = cg[cg.group == "main"].iloc[0]
    rest = cg[cg.group != "main"]
    signs = int((cg.gap_ratio > 1).sum())
    share = float((rest.ci_low > 1).mean())
    v = "no" if signs < len(cg) else ("yes" if main.ci_low > 1 and share >= 0.75 else "partially")
    flips = ", ".join(cg[cg.gap_ratio <= 1].variant)
    verdicts.append(dict(finding="F5 (SVI gap)", original=f"gap ratio = {_fmt(main.gap_ratio, main.ci_low, main.ci_high)}",
                         n_variants=len(rest), holds=v,
                         notes=(f"gap > 1 in {signs}/{len(cg)}; CI excludes 1 in {int((rest.ci_low > 1).sum())}/{len(rest)} "
                                f"variants ({share:.0%}). Sign flips: {flips or 'none'}. Baseline-adjusted variants excluded "
                                "(not comparable)."),
                         rule=DECISION_RULE["F5"]))
    verdicts.append(dict(finding="F4 (within-group decline)", original="see Step C", n_variants=0,
                         holds="not re-tested", notes="Not part of R1-R6.", rule=""))
    pd.DataFrame(var_rows).to_csv(f"{OUT}/summary_variants.csv", index=False)
    vd = pd.DataFrame(verdicts)
    vd.to_csv(f"{OUT}/summary_verdicts.csv", index=False)
    return vd


# ============================================================================== figures
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)


def plot_forest():
    """Forest plot of d (shared-design, counts) by survey measure, death outcome and start year."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outcomes = [("opioids", "Opioid deaths (T40.0-T40.4,T40.6)"), ("synthetic_opioids", "Synthetic-opioid deaths (T40.4)"),
                ("all_drug", "All-drug overdose deaths")]
    meas = [("OPI_MISUSE", "Opioid misuse"), ("OPI_OUD", "Opioid use disorder"), ("HERYR", "Heroin use")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), sharey=True)
    for ax, (m, mlab) in zip(axes, meas):
        _style(ax)
        ax.axvline(0, color="#52514e", lw=0.8)
        ypos, labels = 0, []
        for start in STARTS:
            for k, (o, olab) in enumerate(outcomes):
                r = pd.read_csv(f"{OUT}/r1_r2_declines_{o}.csv")
                r = r[(r.measure == m) & (r.start == start) & (r.death_scale == "count") & (r.variance_method == "shared_design")].iloc[0]
                ax.errorbar(r.d, ypos, xerr=[[r.d - r.ci_low], [r.ci_high - r.d]], fmt="o", color=COLORS[k], ms=6,
                            capsize=2, lw=1.4, label=olab if (start == STARTS[0] and m == "OPI_MISUSE") else None)
                labels.append(f"{start}->{END}")
                ypos += 1
            ypos += 0.6
        ax.set_title(mlab, fontsize=10, loc="left")
        ax.set_xlabel("d = log(death change) - log(survey change)", fontsize=8)
    ticks = [i + 0.6 * (i // 3) for i in range(9)]
    axes[0].set_yticks(ticks, labels)
    axes[0].invert_yaxis()
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("R1/R2/R6: death decline vs survey decline (d < 0: deaths fell faster); 95% CI, shared NSDUH design",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(f"{OUT}/forest_d.png", dpi=150)
    plt.close(fig)


def plot_break_dates():
    """Break-date point estimates and 95% intervals across methods (later and earlier break)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    roll = pd.read_csv(f"{OUT}/r4_r5_r6_rolling_breaks.csv")
    mon = pd.read_csv(f"{OUT}/r5_monthly_breaks.csv")
    ann = pd.read_csv(f"{OUT}/r5b_annual_breaks.csv")
    rows = []
    for r in roll.itertuples():
        lab = r.variant.replace(", Data Value", "").replace("last ", "-").replace(" months dropped", " mo")
        rows.append((f"rolling: {lab}", r.break_1, r.break_1_ci_low, r.break_1_ci_high,
                     r.break_2, r.break_2_ci_low, r.break_2_ci_high))
    for r in mon.itertuples():
        b1, b2 = r.two_breaks.split(", ")
        lab = "WONDER monthly" + (f", -{r.months_dropped} mo" if r.months_dropped else "")
        rows.append((lab + ", 2-break", b1, *r.two_break_1_profile_ci.split(" to "), b2, *r.two_break_2_profile_ci.split(" to ")))
        rows.append((lab + ", 1-break", None, None, None, r.one_break, *r.one_break_profile_ci.split(" to ")))
    for r in ann.itertuples():
        rows.append((f"annual: {r.series} (year after bend)", None, None, None, f"{r.break_year + 1}-07",
                     f"{r.break_year_ci_low + 1}-01", f"{r.break_year_ci_high + 1}-12"))
    def dt(x):
        return pd.Timestamp(x + "-01") if isinstance(x, str) and len(x) == 7 else (pd.Timestamp(x) if isinstance(x, str) else None)
    fig, ax = plt.subplots(figsize=(12, 0.32 * len(rows) + 1.4))
    _style(ax)
    for i, (lab, b1, l1, h1, b2, l2, h2) in enumerate(rows):
        for (b, lo, hi), color in (((b1, l1, h1), COLORS[1]), ((b2, l2, h2), COLORS[0])):
            if b:
                ax.plot([dt(lo), dt(hi)], [i, i], color=color, lw=2.5, solid_capstyle="round")
                ax.plot(dt(b), i, "o", color=color, ms=6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"), color="#e4e3df", zorder=0)
    ax.plot([], [], color=COLORS[1], lw=2.5, label="earlier break (around 2020)")
    ax.plot([], [], color=COLORS[0], lw=2.5, label="later break (downturn)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("R4/R5: break dates with 95% intervals by method (shaded: calendar 2023). "
                 "WONDER monthly = all-drug 'drug-induced' counts", loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/break_dates.png", dpi=150)
    plt.close(fig)


def main(checks):
    os.makedirs(OUT, exist_ok=True)
    mp = f"{OUT}/step_r_meta.json"
    meta = json.load(open(mp)) if os.path.exists(mp) else {}      # merge: partial runs keep other checks' entries
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
    if "r6" in checks:
        for ind, tag in ((SYNTHETIC_OPIOIDS, "synthetic_opioids"), (ALL_DRUG, "all_drug")):
            r12(ind, tag)
        r12(OPIOIDS, "opioids_predicted_value", value_col="Predicted Value")
        meta["r6"] = dict(age_adjusted="not done: needs a CDC WONDER annual export grouped by Year with age-adjusted rates")
    if "r5" in checks:
        roll, hac, ann = r5_rolling()
        mon = r5_monthly()
        pd.set_option("display.width", 250)
        print(roll[["variant", "break_1", "break_1_ci_low", "break_1_ci_high", "break_2", "break_2_ci_low",
                    "break_2_ci_high", "p_1_vs_0", "p_2_vs_1"]].to_string(index=False))
        print(hac.drop(columns=["note", "source"]).to_string(index=False))
        print(ann.drop(columns=["note", "source"]).to_string(index=False))
        print(mon.drop(columns=["model", "wonder_query", "source"]).T.to_string())
        meta["r5"] = dict(monthly_series=WONDER_SERIES, B_rolling=B_ROLL, B_monthly=B_MONTHLY, block=BLOCK,
                          opioid_monthly="not done: needs a CDC WONDER export with the T40 multiple-cause filter")
    if "summary" in checks:
        plot_forest()
        plot_break_dates()
        vd = summarize()
        pd.set_option("display.width", 250, "display.max_colwidth", 200)
        print(vd.drop(columns="rule").to_string(index=False))
    with open(f"{OUT}/step_r_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or ["r12", "r3", "r5", "r6", "summary"])
