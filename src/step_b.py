#!/usr/bin/env python3
"""
Step B: inference for the structural breaks in the CDC national 12-month-ending death series.

    python -m src.step_b

Problem: each value is a 12-month rolling total, so neighbouring values share 11 of 12 months and
the residuals around any trend fit are strongly serially correlated. Textbook F-tests / BIC assume
independent errors and would overstate the evidence, and break dates that are themselves estimated
do not give a standard F distribution anyway.

Approach: moving block bootstrap (MBB) of residuals.
  * Resample residuals in contiguous blocks of L months (main L = 12, the length of the overlap
    window; sensitivity L = 6 and 24), so the autocorrelation inside a block is preserved.
  * Model comparison (2 vs 1 break, 1 vs 0 breaks): statistic F = (SSE_simple - SSE_complex) / SSE_complex.
    Null distribution: fit the simpler model, add block-resampled residuals of that fit to its
    fitted values, refit both models, recompute F. p = (1 + #{F* >= F_obs}) / (B + 1).
    (Under "0 breaks" the residuals still contain the trend changes, which inflates the null noise and
    makes that test conservative.)
  * Break-date confidence intervals: add block-resampled residuals of the 2-break fit to its fitted
    values, re-estimate both break dates, take the 2.5% / 97.5% percentiles.
  * Naive BIC (independent-errors assumption) is reported for reference only.

Caveat for reading break dates: a sudden change in monthly deaths at month m shows up in a
12-month rolling total as a bend spread over months m..m+11, so a break date in this series
can lag the underlying change by up to about a year.

Source: CDC/NCHS VSRR (data.cdc.gov xkb8-kh2a), State=US, 'Data Value', 12-month-ending.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.breaks import fast_fit, fit_no_break, fit_one_break, fit_two_breaks
from src.cdc import ALL_DRUG, OPIOIDS, load_series

OUT = "outputs"
SERIES = {"all_drug": ALL_DRUG, "opioids": OPIOIDS}
BLOCKS = (12, 6, 24)          # first = main analysis
B = 2000
MIN_SEG = 12
SEED = 20261023


def mbb(resid, L, rng):
    """One moving-block-bootstrap resample of resid (length preserved)."""
    n = len(resid)
    starts = rng.integers(0, n - L + 1, size=int(np.ceil(n / L)))
    return np.concatenate([resid[s:s + L] for s in starts])[:n]


def stat21(f):
    return (f["sse1"] - f["sse2"]) / f["sse2"]


def stat10(f):
    return (f["sse0"] - f["sse1"]) / f["sse1"]


def bic(sse, n, k):
    return n * np.log(sse / n) + k * np.log(n)


def run(tag, indicator):
    s = load_series(indicator=indicator)
    y = s.deaths.to_numpy(float)
    n = len(y)
    dates = s.date.dt.strftime("%Y-%m").to_numpy()
    obs = fast_fit(y, MIN_SEG)
    t1, t2, fit2, _ = fit_two_breaks(y, MIN_SEG)
    _, fit1, _ = fit_one_break(y, MIN_SEG)
    fit0, _ = fit_no_break(y)
    r2, r1, r0 = y - fit2, y - fit1, y - fit0
    src = f"CDC VSRR xkb8-kh2a, US, '{indicator}', Data Value, 12-month-ending, {dates[0]} to {dates[-1]}"

    tests, cis, draws = [], [], {}
    for L in BLOCKS:
        rng = np.random.default_rng([SEED, L, len(tag)])
        f21 = np.array([stat21(fast_fit(fit1 + mbb(r1, L, rng), MIN_SEG)) for _ in range(B)])
        f10 = np.array([stat10(fast_fit(fit0 + mbb(r0, L, rng), MIN_SEG)) for _ in range(B)])
        bt = np.array([fast_fit(fit2 + mbb(r2, L, rng), MIN_SEG)["tau2"] for _ in range(B)])
        draws[L] = bt
        tests.append(dict(series=tag, block_len=L, B=B,
                          F_2vs1=stat21(obs), p_2vs1=(1 + (f21 >= stat21(obs)).sum()) / (B + 1),
                          F_1vs0=stat10(obs), p_1vs0=(1 + (f10 >= stat10(obs)).sum()) / (B + 1),
                          source=src))
        for k, (name, est) in enumerate((("break_1", t1), ("break_2", t2))):
            lo, hi = np.percentile(bt[:, k], [2.5, 97.5])
            lo, hi = int(np.floor(lo)), int(np.ceil(hi))
            cis.append(dict(series=tag, block_len=L, B=B, break_=name, estimate=dates[est],
                            ci_low=dates[lo], ci_high=dates[min(hi, n - 1)],
                            ci_width_months=hi - lo,
                            share_within_3_months=float(np.mean(np.abs(bt[:, k] - est) <= 3)),
                            source=src))
    bics = pd.DataFrame([dict(series=tag, model=m, sse=sse, params=k, bic_naive=bic(sse, n, k), source=src)
                         for m, sse, k in (("0 breaks", obs["sse0"], 2), ("1 break", obs["sse1"], 5),
                                           ("2 breaks", obs["sse2"], 8))])
    # k = 2 coefficients per segment + 1 per estimated break location

    # chart: bootstrap distribution of break dates (main block length)
    bt = draws[BLOCKS[0]]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    bins = np.arange(n + 1) - 0.5
    for k, (est, color, name) in enumerate(((t1, "#2a78d6", "break 1"), (t2, "#eb6834", "break 2"))):
        ax.hist(bt[:, k], bins=bins, color=color, alpha=0.85, label=f"{name} (estimate {dates[est]})")
        ax.axvline(est, color=color, lw=1, ls=":")
    ticks = [i for i in range(n) if dates[i].endswith("-01")]
    ax.set_xticks(ticks, [dates[i][:4] for i in ticks])
    ax.set_ylabel(f"bootstrap draws (of {B})")
    ax.set_title(f"{tag}: block-bootstrap distribution of break dates (block = {BLOCKS[0]} months)",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/step_b_bootstrap_{tag}.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame(tests), pd.DataFrame(cis).rename(columns={"break_": "break"}), bics


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = [run(tag, ind) for tag, ind in SERIES.items()]
    tests = pd.concat([p[0] for p in parts])
    cis = pd.concat([p[1] for p in parts])
    bics = pd.concat([p[2] for p in parts])
    tests.to_csv(f"{OUT}/step_b_tests.csv", index=False)
    cis.to_csv(f"{OUT}/step_b_break_ci.csv", index=False)
    bics.to_csv(f"{OUT}/step_b_bic_naive.csv", index=False)
    with open(f"{OUT}/step_b_summary.json", "w") as fh:
        json.dump(dict(method=__doc__.split("Approach:")[1].split("Source:")[0].strip(),
                       seed=SEED, B=B, min_seg=MIN_SEG, block_lengths=list(BLOCKS),
                       files=["outputs/step_b_tests.csv", "outputs/step_b_break_ci.csv",
                              "outputs/step_b_bic_naive.csv"] +
                             [f"outputs/step_b_bootstrap_{t}.png" for t in SERIES]), fh, indent=2)
    pd.set_option("display.width", 200)
    print(tests.drop(columns="source").round(4).to_string(index=False))
    print(cis.drop(columns="source").round(3).to_string(index=False))
    print(bics.drop(columns="source").round(1).to_string(index=False))


if __name__ == "__main__":
    main()
