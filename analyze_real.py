#!/usr/bin/env python3
"""
analyze_real.py -- break-point detection and NSDUH-vs-deaths comparison on a real CDC series.

Input: dbt mart state_deaths_long (DuckDB warehouse), built from data/raw/cdc_vsrr_overdose_national_state.csv
  CDC/NCHS "Provisional Drug Overdose Death Counts" (data.cdc.gov dataset xkb8-kh2a),
  downloaded by fetch_data.py. Each value is a 12-MONTH-ENDING total (deaths in the
  12 months ending in that calendar month) -- NOT the count of deaths in that single month.

The series is a parameter (state x indicator x value column). Examples:

  python analyze_real.py                                   # US, all-drug deaths (tag: all_drug)
  python analyze_real.py --indicator "Opioids (T40.0-T40.4,T40.6)" --tag opioids
  python analyze_real.py --indicator "Opioids (T40.0-T40.4,T40.6)" --value-col "Predicted Value" --tag opioids_pred

What this script does, and is honest about not doing:
  1. Plots the real rolling series.
  2. Fits a two-break piecewise-linear trend to the rolling series itself (valid: the
     rolling series is real data, and "when did its trend change slope" is a well-posed
     question even though the series is smoothed).
  3. Computes the first difference of the rolling series (S_t - S_{t-1}) as a NET-FLOW
     PROXY (S_t - S_{t-1} = D_t - D_{t-12}, i.e. this month's deaths minus the same month
     a year earlier), which is not the same as true monthly deaths.
  4. Builds the NSDUH-opioid-misuse-% vs. CDC-deaths comparison, flagging definitional
     differences (self-reported survey misuse vs. death certificates) rather than papering
     over them.

Outputs: data/processed/<state>_<tag>_series.csv and outputs/<tag>_*.png, outputs/<tag>_summary.json.
This is a worked demonstration of the pipeline, not a finished result for submission.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.breaks import fit_no_break, fit_one_break, fit_two_breaks
from src.cdc import ALL_DRUG, STATE_CSV, VALUE_COLS, load_series

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
PROCESSED = os.path.join(ROOT, "data", "processed")

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"

# NSDUH: % of people 12+ reporting past-year opioid misuse (self-report survey).
# PROVENANCE: hard-coded published aggregates, entered by hand from the SAMHSA 2024 NSDUH
# annual national report and the 2021-2024 companion trend report. NOT computed from
# microdata by this pipeline and not re-verified against the reports here.
# 2023 is left missing: the report gives it as a 3.0-3.1 range, and filling in a midpoint
# would be an invented number. (For reference only, the 2023 PUF microdata gives weighted
# OPINMYR = 3.19%; see data/DATA_PROFILE.md. That is a different source and is not mixed in.)
NSDUH_OPIOID_MISUSE = pd.DataFrame({
    "year": [2021, 2022, 2023, 2024],
    "opioid_misuse_pct": [3.2, 3.0, np.nan, 2.7],
})
NSDUH_SOURCE = ("SAMHSA NSDUH published estimates (2024 annual national report + 2021-2024 "
                "trend report), hard-coded; 2023 omitted (published only as 3.0-3.1 range)")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="US")
    ap.add_argument("--indicator", default=ALL_DRUG)
    ap.add_argument("--value-col", default="Data Value", choices=VALUE_COLS)
    ap.add_argument("--tag", default="all_drug", help="prefix for output files")
    ap.add_argument("--min-seg", type=int, default=12, help="minimum months per segment")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PROCESSED, exist_ok=True)
    provenance = (f"CDC/NCHS VSRR Provisional Drug Overdose Death Counts (data.cdc.gov xkb8-kh2a), "
                  f"file {STATE_CSV}, State={a.state!r}, Indicator={a.indicator!r}, "
                  f"column={a.value_col!r}; 12-month-ending totals")

    # ------------------------------------------------------------ load
    df = load_series(state=a.state, indicator=a.indicator, value_col=a.value_col)
    n = len(df)
    series_path = os.path.join(PROCESSED, f"{a.state.lower()}_{a.tag}_series.csv")
    df.assign(source=provenance).to_csv(series_path, index=False)
    print(f"Loaded {n} months: {df.date.iloc[0]:%Y-%m} to {df.date.iloc[-1]:%Y-%m}\n  {provenance}")
    print(f"wrote {series_path}")

    # ------------------------------------------------------ break fits
    y = df["deaths"].to_numpy(dtype=float)
    tau1, tau2, fitted, sse = fit_two_breaks(y, min_seg=a.min_seg)
    tau_single, _, sse_single = fit_one_break(y, min_seg=a.min_seg)
    _, sse_flat = fit_no_break(y)
    d1, d2 = df.date.iloc[tau1], df.date.iloc[tau2]
    print(f"Two-break fit: {d1:%Y-%m}, {d2:%Y-%m}")
    print(f"SSE: no-break={sse_flat:,.0f}  one-break(@{df.date.iloc[tau_single]:%Y-%m})={sse_single:,.0f}  "
          f"two-break={sse:,.0f}")

    # ------------------------------------------------- net-flow proxy
    df["net_flow_proxy"] = df["deaths"].diff()  # = D_t - D_{t-12}; NOT true monthly deaths

    span = f"{df.date.iloc[0]:%b %Y} – {df.date.iloc[-1]:%b %Y}"
    label = f"{a.indicator} [{a.value_col}]"

    # --------------------------------------------------------- plot 1
    fig, ax = plt.subplots(figsize=(11, 5.5))
    style(ax)
    ax.plot(df.date, df.deaths, color=BLUE, lw=2, label="12-month-ending total (CDC VSRR)")
    ax.plot(df.date, fitted, color=ORANGE, lw=2, ls="--", label="2-break piecewise-linear fit")
    for tau, lbl in [(tau1, "break 1"), (tau2, "break 2")]:
        ax.axvline(df.date.iloc[tau], color=MUTED, lw=1, ls=":")
        ax.annotate(f" {lbl}: {df.date.iloc[tau]:%Y-%m}", (df.date.iloc[tau], 0.02),
                    xycoords=("data", "axes fraction"), ha="left", va="bottom", fontsize=8, color=MUTED)
    ax.set_title(f"{a.state} overdose deaths, {label}, {span}", color=INK, fontsize=11, loc="left")
    ax.set_ylabel("Deaths in trailing 12 months", color=MUTED)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.text(0.01, 0.01, f"Source: {provenance}", fontsize=6.5, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p1 = os.path.join(OUT, f"{a.tag}_breakpoints.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # --------------------------------------------------------- plot 2
    fig, ax = plt.subplots(figsize=(11, 4.5))
    style(ax)
    ax.axhline(0, color=INK, lw=0.8)
    ax.plot(df.date, df.net_flow_proxy, color=BLUE, lw=2)
    ax.set_title(f"Month-over-month change in the 12-month-ending total ({label})\n"
                 "net-flow proxy = this month's deaths minus same month a year earlier; NOT monthly deaths",
                 color=INK, fontsize=10, loc="left")
    ax.set_ylabel("Δ 12-month-ending total", color=MUTED)
    fig.tight_layout()
    p2 = os.path.join(OUT, f"{a.tag}_netflow_proxy.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # ---------------------------------------- NSDUH vs deaths comparison
    # December value of a 12-month-ending series == that calendar year's total.
    annual = df[df.date.dt.month == 12][["year", "deaths"]]
    comp = NSDUH_OPIOID_MISUSE.merge(annual, on="year", how="left")
    base = comp.iloc[0]
    comp["deaths_idx"] = (100 * comp.deaths / base.deaths).round(1)
    comp["opioid_misuse_idx"] = (100 * comp.opioid_misuse_pct / base.opioid_misuse_pct).round(1)
    print(f"\nNSDUH opioid misuse % vs CDC {label}, 2021 = 100:")
    print(comp.to_string(index=False))
    last = comp.dropna().iloc[-1]
    pct = dict(
        from_year=int(base.year), to_year=int(last.year),
        deaths_pct_change=round(100 * (last.deaths / base.deaths - 1), 1),
        opioid_misuse_pct_change=round(100 * (last.opioid_misuse_pct / base.opioid_misuse_pct - 1), 1),
    )
    print(f"{pct['from_year']}->{pct['to_year']}: deaths {pct['deaths_pct_change']:+}%, "
          f"NSDUH opioid misuse {pct['opioid_misuse_pct_change']:+}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    style(ax)
    ax.plot(comp.year, comp.deaths_idx, "o-", color=BLUE, lw=2, ms=8, label=f"Deaths: {a.indicator} (CDC)")
    # NaN breaks the line, so no segment is drawn through the missing 2023 value.
    ax.plot(comp.year, comp.opioid_misuse_idx, "s-", color=ORANGE, lw=2, ms=8,
            label="Past-year opioid misuse % (NSDUH, published)")
    ax.set_title(f"CDC deaths vs NSDUH opioid misuse, indexed 2021 = 100\n"
                 f"(NSDUH 2023 omitted: published only as a range)", color=INK, fontsize=10, loc="left")
    ax.set_ylabel("Index (2021 = 100)", color=MUTED)
    ax.set_xticks(comp.year)
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    p3 = os.path.join(OUT, f"{a.tag}_vs_nsduh.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    # ------------------------------------------------------ save numbers
    peak_i = int(df.deaths.idxmax())
    summary = dict(
        series=dict(state=a.state, indicator=a.indicator, value_col=a.value_col, source=provenance,
                    processed_file=os.path.relpath(series_path, ROOT)),
        n_months=n,
        date_range=[f"{df.date.iloc[0]:%Y-%m}", f"{df.date.iloc[-1]:%Y-%m}"],
        break_fit=dict(min_seg=a.min_seg, tau1_date=f"{d1:%Y-%m}", tau2_date=f"{d2:%Y-%m}",
                       sse_flat=sse_flat, sse_one_break=sse_single, sse_two_break=sse,
                       one_break_date=f"{df.date.iloc[tau_single]:%Y-%m}"),
        peak=dict(date=f"{df.date.iloc[peak_i]:%Y-%m}", value=int(df.deaths.iloc[peak_i])),
        latest=dict(date=f"{df.date.iloc[-1]:%Y-%m}", value=int(df.deaths.iloc[-1])),
        pct_decline_from_peak=round(100 * (1 - df.deaths.iloc[-1] / df.deaths.max()), 1),
        nsduh_source=NSDUH_SOURCE,
        nsduh_vs_deaths=json.loads(comp.to_json(orient="records")),
        nsduh_vs_deaths_change=pct,
        charts=[os.path.relpath(p, ROOT) for p in (p1, p2, p3)],
    )
    sp = os.path.join(OUT, f"{a.tag}_summary.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {sp}, {p1}, {p2}, {p3}")


if __name__ == "__main__":
    main()
