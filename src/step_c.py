#!/usr/bin/env python3
"""
Step C: survey-weighted logistic regression, NSDUH 2021-2024 only (individual level).

    python -m src.step_c          # fits models, writes R extract, runs R cross-check if Rscript exists

Question: did prevalence fall WITHIN demographic groups, or did the population mix shift?
  * crude model:    outcome ~ year
  * adjusted model: outcome ~ year + age group (CATAG3) + sex (IRSEX) + race/ethnicity (NEWRACE2)
  If the 2024-vs-2021 year effect is about the same in both, the change happened within groups;
  if adjusting shrinks it, part of the change came from a shift in demographic composition.
  Age, sex and race/ethnicity are not consequences of survey year, so adjusting for them does not
  control for a mediator. Nothing area-level (CDC/SVI) enters these models.

Outcomes: OPI_MISUSE, OPI_OUD, HERYR (comparable 2021-2024 definitions, src/nsduh.py).
Design: weight ANALWT2_C1, strata VESTR_C, PSU VEREP; linearization SEs (src/survey_glm.py);
t tests on design df (50) - p + 1, as R svyglm does. Reference levels: 2021, 12-17, male, NH White.

Checks: (1) hand calculation -- in the crude model the 2024 coefficient must equal
logit(p_2024) - logit(p_2021) from the weighted proportions, with SE equal to the linearized SE of
that difference; (2) every coefficient and SE is compared with R survey::svyglm.
"""
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

from src import nsduh
from src import survey as sv
from src.survey_glm import design_matrix, svy_logit

OUT = "outputs"
EXTRACT = "data/raw/derived/nsduh_step_c_extract.csv"   # microdata extract, git-ignored
DEMOG = {"CATAG3": 1, "IRSEX": 1, "NEWRACE2": 1}
YEARS = [2021, 2022, 2023, 2024]
SOURCE = (f"NSDUH 2021-2024 combined PUF (+2024 single-year PUF for OUD), {nsduh.WEIGHT}, "
          f"{nsduh.STRATA}/{nsduh.PSU}; survey-weighted logistic regression, linearization SEs")


def hand_check(df, var):
    """Crude-model 2024 coefficient and SE computed directly from weighted proportions."""
    W, H, J = nsduh.WEIGHT, nsduh.STRATA, nsduh.PSU
    y, w = df[var].to_numpy(float), df[W].to_numpy(float)
    p21, z21, _ = sv._scores(y, w, (df.YEAR == 2021).to_numpy())
    p24, z24, _ = sv._scores(y, w, (df.YEAR == 2024).to_numpy())
    coef = np.log(p24 / (1 - p24)) - np.log(p21 / (1 - p21))
    z = z24 / (p24 * (1 - p24)) - z21 / (p21 * (1 - p21))       # linearized logit difference
    se = np.sqrt(sv._psu_variance(z, df[H].to_numpy(), df[J].to_numpy()))
    return dict(p_2021=p21, p_2024=p24, coef_hand=coef, se_hand=se)


def main():
    os.makedirs(OUT, exist_ok=True)
    df, labels = nsduh.load_analysis(demographics=tuple(DEMOG))
    W, H, J = nsduh.WEIGHT, nsduh.STRATA, nsduh.PSU

    coefs, effects, checks = [], [], []
    for var, label in nsduh.COMPARABLE.items():
        fits = {}
        for model, cats in (("crude", {"YEAR": 2021}), ("adjusted", {"YEAR": 2021, **DEMOG})):
            X, names = design_matrix(df, categorical=cats)
            res, _ = svy_logit(X, df[var], df[W], df[H], df[J], nsduh.DESIGN_DF, names)
            res.insert(0, "model", model)
            res.insert(0, "outcome", var)
            res["source"] = SOURCE
            coefs.append(res)
            fits[model] = res.set_index("term")
        for y in YEARS[1:]:
            t = f"YEAR[{y}]"
            c, a = fits["crude"].loc[t], fits["adjusted"].loc[t]
            effects.append(dict(outcome=var, label=label, year_vs_2021=y,
                                crude_or=c.odds_ratio, crude_ci_low=c.or_ci_low, crude_ci_high=c.or_ci_high, crude_p=c.p,
                                adjusted_or=a.odds_ratio, adjusted_ci_low=a.or_ci_low, adjusted_ci_high=a.or_ci_high,
                                adjusted_p=a.p, log_or_change_from_adjustment=a.coef - c.coef, source=SOURCE))
        h = hand_check(df, var)
        c = fits["crude"].loc["YEAR[2024]"]
        checks.append(dict(outcome=var, **h, coef_model=c.coef, se_model=c.se,
                           abs_diff_coef=abs(c.coef - h["coef_hand"]), abs_diff_se=abs(c.se - h["se_hand"])))

    coefs = pd.concat(coefs)
    effects = pd.DataFrame(effects)
    checks = pd.DataFrame(checks)
    coefs.to_csv(f"{OUT}/step_c_coefficients.csv", index=False)
    effects.to_csv(f"{OUT}/step_c_year_effects.csv", index=False)
    checks.to_csv(f"{OUT}/step_c_hand_check.csv", index=False)

    # composition: weighted share of each demographic level, 2021 vs 2024
    comp = []
    for g in DEMOG:
        for lvl in sorted(df[g].unique()):
            df["_ind"] = (df[g] == lvl).astype(float)
            for y in (2021, 2024):
                r = sv.proportion(df, "_ind", W, H, J, domain=(df.YEAR == y).to_numpy(), df_design=nsduh.DESIGN_DF)
                comp.append(dict(variable=g, level=labels[g][lvl], year=y, pct=100 * r["p"], se=100 * r["se"],
                                 source=SOURCE.split(";")[0]))
    df.drop(columns="_ind", inplace=True)
    comp = pd.DataFrame(comp).pivot_table(index=["variable", "level"], columns="year", values=["pct", "se"])
    comp.columns = [f"{a}_{b}" for a, b in comp.columns]
    comp = comp.reset_index()
    comp["pct_point_change"] = comp.pct_2024 - comp.pct_2021
    comp["source"] = SOURCE.split(";")[0]
    comp.to_csv(f"{OUT}/step_c_composition.csv", index=False)

    # R cross-check
    os.makedirs(os.path.dirname(EXTRACT), exist_ok=True)
    df[["YEAR", W, H, J, *nsduh.COMPARABLE, *DEMOG]].to_csv(EXTRACT, index=False)
    if shutil.which("Rscript"):
        subprocess.run(["Rscript", "R/step_c_svyglm.R"], check=True)
        r = pd.read_csv(f"{OUT}/step_c_r_check.csv")
        m = coefs.merge(r, left_on=["outcome", "model", "term"], right_on=["outcome", "model", "term"], how="outer",
                        indicator=True)
        if (m["_merge"] != "both").any():
            raise ValueError(f"term mismatch with R: {m.loc[m._merge != 'both', ['outcome', 'model', 'term']]}")
        cmp = m.groupby(["outcome", "model"]).apply(lambda g: pd.Series({
            "n_terms": len(g), "max_abs_diff_coef": (g.coef - g.coef_r).abs().max(),
            "max_rel_diff_se": ((g.se - g.se_r).abs() / g.se_r).max(),
            "df_python": g.df.iloc[0], "df_r": g.df_r.iloc[0]}), include_groups=False).reset_index()
        cmp.to_csv(f"{OUT}/step_c_r_comparison.csv", index=False)
        print(cmp.to_string(index=False))
    else:
        print("Rscript not found: R cross-check skipped")

    pd.set_option("display.width", 220)
    print(checks.round(8).to_string(index=False))
    print(effects.drop(columns=["label", "source"]).round(4).to_string(index=False))
    print(comp.drop(columns="source").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
