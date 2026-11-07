#!/usr/bin/env python3
"""
Collect every number in outputs/ into results/RESULTS.md. No interpretation.

    python -m src.build_results

Each known output gets a section with its source. Any file in outputs/ that no section claims is
rendered automatically under "Other outputs", so nothing is silently left out. Output is
deterministic (no timestamps): re-running on the same outputs/ gives the same file.
"""
import json
import os

import pandas as pd

OUT_DIR = "outputs"
RES_DIR = "results"
RES = os.path.join(RES_DIR, "RESULTS.md")
FINDINGS = os.path.join(RES_DIR, "FINDINGS.md")
LEDGER = os.path.join(OUT_DIR, "robustness", "findings_ledger.csv")
MANIFEST = "data/raw/MANIFEST.json"


def fmt(v, nd=3):
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        return f"{v:,.{nd}f}" if abs(v) < 1e6 else f"{v:,.0f}"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}" if abs(v) >= 10000 else str(v)  # keeps years (2021) unformatted
    return str(v)


def table(df, nd=3, drop=()):
    df = df.drop(columns=[c for c in drop if c in df.columns])
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "".join("| " + " | ".join(fmt(v, nd).replace("|", "/") for v in row) + " |\n"
                   for row in df.itertuples(index=False))
    return head + body


def sources(df, cols=("source",)):
    """List distinct source strings once, instead of repeating them on every row."""
    out = []
    for c in cols:
        if c in df.columns:
            for s in sorted(df[c].dropna().unique()):
                out.append(f"- `{c}`: {s}")
    return "\n".join(out) + "\n" if out else ""


def img(name):
    return f"![{name}](../{OUT_DIR}/{name})\n"


def flatten(d, prefix=""):
    rows = []
    if isinstance(d, dict):
        for k, v in d.items():
            rows += flatten(v, f"{prefix}{k}.")
    elif isinstance(d, list) and d and all(isinstance(x, dict) for x in d):
        for i, v in enumerate(d):
            rows += flatten(v, f"{prefix}{i}.")
    else:
        rows.append((prefix.rstrip("."), json.dumps(d) if isinstance(d, list) else d))
    return rows


# ------------------------------------------------------------------ sections
def ledger_note(*findings):
    """One generated line pointing to FINDINGS.md with the current robustness verdict(s)."""
    if not os.path.exists(LEDGER):
        return ""
    led = pd.read_csv(LEDGER).set_index("finding")
    parts = [f"{f}: **{led.loc[f, 'verdict']}**" for f in findings if f in led.index]
    return ("> Robustness status (from `results/FINDINGS.md`): " + "; ".join(parts)
            + ". Read these estimates together with the ledger.\n") if parts else ""


def sec_cdc_breaks(used):
    parts = ["## CDC national death series: break-point fits\n",
             "Produced by `analyze_real.py`. Values are 12-month-ending totals.\n"]
    rows = []
    for tag in ("all_drug", "opioids"):
        f = f"{OUT_DIR}/{tag}_summary.json"
        if not os.path.exists(f):
            continue
        used.add(os.path.basename(f))
        s = json.load(open(f))
        bf = s["break_fit"]
        rows.append(dict(series=tag, indicator=s["series"]["indicator"], value_col=s["series"]["value_col"],
                         months=s["n_months"], range=" to ".join(s["date_range"]), break_1=bf["tau1_date"],
                         break_2=bf["tau2_date"], one_break=bf["one_break_date"], sse_0=bf["sse_flat"],
                         sse_1=bf["sse_one_break"], sse_2=bf["sse_two_break"], peak=f'{s["peak"]["value"]:,} ({s["peak"]["date"]})',
                         latest=f'{s["latest"]["value"]:,} ({s["latest"]["date"]})',
                         pct_below_peak=s["pct_decline_from_peak"], source=s["series"]["source"]))
        for chart in s.get("charts", []):
            used.add(os.path.basename(chart))
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    parts += [table(df, 0, drop=("source",)), "\nSources:\n", sources(df)]
    for tag in ("all_drug", "opioids"):
        for kind in ("breakpoints", "netflow_proxy", "vs_nsduh"):
            if os.path.exists(f"{OUT_DIR}/{tag}_{kind}.png"):
                parts.append(f"\n**{tag} / {kind}**\n\n" + img(f"{tag}_{kind}.png"))
    parts.append("\nNote: the `*_vs_nsduh` charts use hand-entered published NSDUH figures "
                 "(see `nsduh_source` in the JSON); Step A below replaces them with microdata estimates.\n")
    return "\n".join(parts)


def sec_nsduh(used):
    f = f"{OUT_DIR}/nsduh_estimates.csv"
    if not os.path.exists(f):
        return ""
    used.update({"nsduh_estimates.csv", "nsduh_variables_by_year.csv", "nsduh_validation.csv"})
    est = pd.read_csv(f)
    parts = ["## Step 1: NSDUH 2021-2024 weighted estimates\n",
             "Produced by `src/nsduh_estimates.py`. Percent of people aged 12+, design-based SE and 95% CI "
             "(logit), `rse` = SE / estimate.\n"]
    if os.path.exists(f"{OUT_DIR}/nsduh_variables_by_year.csv"):
        parts += ["### Respondents with data, by variable and year\n",
                  table(pd.read_csv(f"{OUT_DIR}/nsduh_variables_by_year.csv"))]
    cols = ["variable", "year", "level", "pct", "se", "ci_low", "ci_high", "n", "rse", "low_precision"]
    parts += ["\n### National, by year\n", table(est[est.group == "all"][cols].drop(columns="level"))]
    for g, title in (("CATAG3", "age group"), ("IRSEX", "sex")):
        parts += [f"\n### By year and {title} (`{g}`)\n", table(est[est.group == g][cols])]
    if os.path.exists(f"{OUT_DIR}/nsduh_validation.csv"):
        val = pd.read_csv(f"{OUT_DIR}/nsduh_validation.csv")
        parts += ["\n### Validation against previously hand-entered published figures\n",
                  table(val, 2, drop=("published_source",)), "\nSources:\n", sources(val, ("published_source",))]
    parts += ["\nSources:\n", sources(est)]
    return "\n".join(parts)


def sec_step_a(used):
    f = f"{OUT_DIR}/step_a_tests.csv"
    if not os.path.exists(f):
        return ""
    used.update({"step_a_tests.csv", "step_a_comparison.csv", "step_a_context.csv", "step_a_summary.json",
                 "step_a_index.png"})
    s = json.load(open(f"{OUT_DIR}/step_a_summary.json"))
    tests = pd.read_csv(f)
    comp = pd.read_csv(f"{OUT_DIR}/step_a_comparison.csv")
    ctx = pd.read_csv(f"{OUT_DIR}/step_a_context.csv")
    parts = ["## Step A: opioid deaths vs NSDUH opioid measures, 2021-2024\n", ledger_note("F1-misuse", "F1-OUD", "F2"),
             "Produced by `src/step_a.py`.\n",
             f"> **Reminder:** {s['reminder']}\n",
             "### Method (from the script)\n", "```\n" + s["method"] + "\n```\n",
             img("step_a_index.png"),
             "### Levels and index (2021 = 100)\n",
             table(comp.drop(columns=["label"]), 3, drop=("source",)),
             "\n### Tests: 2021 -> 2024 change in each NSDUH measure vs change in deaths\n",
             table(tests[["measure", "nsduh_pct_2021", "nsduh_pct_2024", "nsduh_pct_change", "nsduh_pct_change_ci_low",
                          "nsduh_pct_change_ci_high", "nsduh_change_pvalue_vs_zero", "deaths_2021", "deaths_2024",
                          "deaths_pct_change", "difference_in_log_ratios", "difference_ci_low", "difference_ci_high",
                          "p_deaths_fixed", "p_deaths_poisson", "deaths_per_case_ratio_2024_vs_2021", "df"]], 4),
             "\n### Context: what each source mostly measures\n", table(ctx, 2, drop=("source",)),
             "\nSources:\n", sources(comp), sources(tests, ("nsduh_source", "deaths_source")), sources(ctx)]
    return "\n".join(parts)


def sec_step_b(used):
    f = f"{OUT_DIR}/step_b_tests.csv"
    if not os.path.exists(f):
        return ""
    s = json.load(open(f"{OUT_DIR}/step_b_summary.json"))
    used.update({"step_b_tests.csv", "step_b_break_ci.csv", "step_b_bic_naive.csv", "step_b_summary.json"})
    tests, cis, bics = (pd.read_csv(f"{OUT_DIR}/{x}") for x in
                        ("step_b_tests.csv", "step_b_break_ci.csv", "step_b_bic_naive.csv"))
    parts = ["## Step B: break-point inference (moving block bootstrap)\n", ledger_note("F3a", "F3b"),
             f"Produced by `src/step_b.py`. B = {s['B']} resamples per test, seed {s['seed']}, "
             f"minimum segment {s['min_seg']} months, block lengths {s['block_lengths']} "
             f"(first = main analysis).\n",
             "### Method (from the script)\n", "```\n" + s["method"] + "\n```\n",
             "### Model comparison (bootstrap p-values)\n", table(tests, 4, drop=("source",)),
             "\n### Break dates with 95% bootstrap intervals\n", table(cis, 3, drop=("source",)),
             "\n### Naive BIC (assumes independent errors; reference only)\n", table(bics, 1, drop=("source",)),
             "\nSources:\n", sources(tests)]
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("step_b_bootstrap_") and f.endswith(".png"):
            used.add(f)
            parts.append(img(f))
    return "\n".join(parts)


def sec_step_c(used):
    f = f"{OUT_DIR}/step_c_year_effects.csv"
    if not os.path.exists(f):
        return ""
    names = ["step_c_year_effects.csv", "step_c_coefficients.csv", "step_c_composition.csv",
             "step_c_hand_check.csv", "step_c_r_check.csv", "step_c_r_comparison.csv"]
    used.update(names)
    eff, coef, comp, hand = (pd.read_csv(f"{OUT_DIR}/{x}") for x in names[:4])
    parts = ["## Step C: survey-weighted logistic regression (NSDUH only)\n", ledger_note("F4"),
             "Produced by `src/step_c.py` (models: `src/survey_glm.py`; R cross-check: `R/step_c_svyglm.R`). "
             "Crude = outcome ~ year; adjusted = + age group, sex, race/ethnicity. Reference: 2021, 12-17, male, "
             "NH White. Odds ratios with 95% CI.\n",
             "### Year effects (odds ratio vs 2021), crude vs adjusted\n", table(eff, 4, drop=("label", "source")),
             "\n### Demographic composition, 2021 vs 2024 (weighted %)\n", table(comp, 2, drop=("source",)),
             "\n### Hand-calculation check (crude 2024 coefficient)\n", table(hand, 6)]
    if os.path.exists(f"{OUT_DIR}/step_c_r_comparison.csv"):
        parts += ["\n### Cross-check against R survey::svyglm\n",
                  table(pd.read_csv(f"{OUT_DIR}/step_c_r_comparison.csv"), 10)]
    parts += ["\n### All coefficients\n", table(coef, 4, drop=("source",)), "\nSources:\n", sources(eff)]
    rcheck = pd.read_csv(f"{OUT_DIR}/step_c_r_check.csv") if os.path.exists(f"{OUT_DIR}/step_c_r_check.csv") else None
    if rcheck is not None:
        parts += ["\n### R svyglm coefficients (raw)\n", table(rcheck, 6)]
    return "\n".join(parts)


def sec_step_d(used):
    f = f"{OUT_DIR}/step_d_change_by_svi.csv"
    if not os.path.exists(f):
        return ""
    names = ["step_d_change_by_svi.csv", "step_d_model_info.csv", "step_d_descriptive.csv",
             "step_d_join_losses.csv", "step_d_coefficients.csv", "step_d_summary.json", "step_d_change_by_svi.png"]
    used.update(names)
    ch, info, desc, lost, coef = (pd.read_csv(f"{OUT_DIR}/{x}") for x in names[:5])
    s = json.load(open(f"{OUT_DIR}/step_d_summary.json"))
    counts = s["counts"]
    parts = ["## Step D: county change in overdose deaths by social vulnerability (CDC county + SVI only)\n", ledger_note("F5"),
             "Produced by `src/step_d.py` (model: `src/count_models.py`).\n",
             f"> {s['note']}\n",
             "### Model (from the script)\n", "```\n" + s["method"] + "\n```\n",
             img("step_d_change_by_svi.png"),
             "### Change in death rate by SVI quartile (rate ratio end vs base)\n", table(ch, 3, drop=("source",)),
             "\n### Model information (Wald test: does the change differ across quartiles?)\n", table(info, 4),
             "\n### Counties: cleaning and join\n",
             table(pd.DataFrame([dict(item=k, value=json.dumps(v) if isinstance(v, list) else v)
                                 for k, v in counts.items()])),
             "\n#### Counties lost in the CDC-SVI join\n", table(lost),
             "\n### Descriptive totals by quartile and year (bounds: suppressed cells counted as 1 and as 9)\n",
             table(desc, 1, drop=("source",)),
             "\n### All coefficients\n", table(coef, 4, drop=("source",)),
             "\nSources:\n", sources(ch)]
    return "\n".join(parts)


ROB = os.path.join(OUT_DIR, "robustness")


def sec_robustness(used):
    vf = os.path.join(ROB, "summary_verdicts.csv")
    if not os.path.exists(vf):
        return ""
    used.add("robustness")
    verd = pd.read_csv(vf)
    var = pd.read_csv(os.path.join(ROB, "summary_variants.csv"))
    meta = json.load(open(os.path.join(ROB, "step_r_meta.json")))
    parts = ["## Robustness (Phase 4a)\n",
             "Produced by `src/step_r.py` (`make robustness`). Each finding is re-estimated under alternative choices; "
             "the decision rule below was fixed before any check was run. Wording is descriptive: 'holds' means the "
             "pre-set rule is met, not that the finding is true.\n",
             "### Decision rule (set in advance)\n",
             table(verd[verd.rule.notna() & (verd.rule != "")][["finding", "rule"]].drop_duplicates("finding")),
             "\n### Summary\n", table(verd.drop(columns="rule")),
             "\n### Per-variant results\n",
             "`sign_ok`: F1 d < 0, F3a 1-vs-0 significant, F5 gap > 1. `ci_criterion_met`: F1 CI excludes 0, F2 CI "
             "includes 0, F3a break-date CI overlaps 2023, F3b p >= 0.05, F5 CI excludes 1.\n",
             table(var[["finding", "variant", "estimate_95ci", "p", "p_holm", "sign_ok", "ci_criterion_met"]], 4),
             "\n" + img("robustness/forest_d.png"), img("robustness/break_dates.png"),
             "### Not done / pending\n",
             table(pd.DataFrame([dict(item=k, status=v.get(x)) for k, v in meta.items() for x in v
                                 if isinstance(v.get(x), str) and v.get(x).startswith("not done")])),
             "\n### Detailed tables\n"]
    for f in sorted(os.listdir(ROB)):
        if f.endswith(".csv") and not f.startswith("summary_"):
            parts += [f"#### `robustness/{f}`\n", table(pd.read_csv(os.path.join(ROB, f)), 4)]
    parts += ["#### `robustness/step_r_meta.json`\n",
              table(pd.DataFrame(flatten(meta), columns=["key", "value"]))]
    return "\n".join(parts)


SECTIONS = [sec_cdc_breaks, sec_nsduh, sec_step_a, sec_step_b, sec_step_c, sec_step_d, sec_robustness]
EXTRA_SECTIONS = []  # later steps register here


def sec_other(used):
    rest = sorted(f for f in os.listdir(OUT_DIR) if f not in used and not f.startswith(".")
                  and os.path.isfile(os.path.join(OUT_DIR, f)))
    if not rest:
        return ""
    parts = ["## Other outputs (auto-rendered)\n"]
    for f in rest:
        p = f"{OUT_DIR}/{f}"
        parts.append(f"### `{f}`\n")
        if f.endswith(".csv"):
            parts.append(table(pd.read_csv(p)))
        elif f.endswith(".json"):
            parts.append(table(pd.DataFrame(flatten(json.load(open(p))), columns=["key", "value"])))
        elif f.endswith(".png"):
            parts.append(img(f))
        else:
            parts.append(f"(not rendered: {f})\n")
    return "\n".join(parts)


def write_findings():
    """results/FINDINGS.md: the findings ledger, generated from outputs/robustness/findings_ledger.csv."""
    if not os.path.exists(LEDGER):
        return
    led = pd.read_csv(LEDGER)
    vd = pd.read_csv(os.path.join(OUT_DIR, "robustness", "summary_verdicts.csv"))
    text = ["# Findings ledger\n",
            "Generated by `python -m src.build_results` from `outputs/robustness/findings_ledger.csv` (written by "
            "`src/step_r.py`). Every number is filled in from the output files; the wording is templated and "
            "descriptive. Verdicts apply the decision rules fixed before the robustness checks were run (listed below); "
            "'holds' means the rule is met, not that the claim is established.\n",
            "## Ledger\n",
            table(led.rename(columns={"original_claim": "original claim", "supported_claim": "what the data support now",
                                      "key_evidence": "key evidence", "main_caveat": "main caveat"})),
            "\n## Decision rules\n",
            table(vd[vd.rule.notna() & (vd.rule != "")][["finding", "rule"]]),
            "\nFull per-variant results, figures and detailed tables: `results/RESULTS.md`, section "
            "\"Robustness (Phase 4a)\".\n"]
    with open(FINDINGS, "w") as fh:
        fh.write("\n".join(text).rstrip() + "\n")


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    write_findings()
    used = set()
    body = [s(used) for s in SECTIONS + EXTRA_SECTIONS]
    head = ["# Results\n",
            "Generated by `python -m src.build_results` from the files in `outputs/`. Every number below comes "
            "from a file in `outputs/`; each table lists its source. This file contains no interpretation.\n"]
    if os.path.exists(MANIFEST):
        man = json.load(open(MANIFEST))
        head += ["## Raw data files (`data/raw/MANIFEST.json`)\n",
                 table(pd.DataFrame([dict(source=k, file=v["file"], sha256=v["sha256"][:16] + "...",
                                          fetched=v["fetched"]) for k, v in sorted(man.items())]))]
    text = "\n".join(head + [b for b in body if b] + [sec_other(used)])
    with open(RES, "w") as fh:
        fh.write(text.rstrip() + "\n")
    print(f"wrote {RES} ({len(text.splitlines())} lines); outputs rendered: {sorted(used)}")


if __name__ == "__main__":
    main()
