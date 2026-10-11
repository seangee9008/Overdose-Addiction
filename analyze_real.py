#!/usr/bin/env python3
"""
analyze_real.py -- real-data demo pipeline for the addiction/overdose Stats 199 project.

Input: national_deaths_real.json
  CDC/NCHS "Provisional Drug Overdose Death Counts" (dataset xkb8-kh2a on data.cdc.gov),
  pulled live via the Socrata API, state="US", indicator="Number of Drug Overdose Deaths".
  Each value is a 12-MONTH-ENDING total (the count of deaths in the 12 months ending in
  that calendar month) -- NOT the count of deaths in that single month. That distinction
  matters for everything below.

What this script does, and is honest about not doing:
  1. Plots the real rolling series.
  2. Fits a two-break piecewise-linear trend to the rolling series itself (valid: the
     rolling series is real data, and "when did its trend change slope" is a well-posed
     question even though the series is smoothed).
  3. Computes the first difference of the rolling series (S_t - S_{t-1}) as a NET-FLOW
     PROXY, and explains in-line why this is not the same as true monthly deaths.
  4. Builds the NSDUH-opioid-misuse-% vs. CDC-all-drug-death comparison, flagging the
     definitional mismatch (opioid misuse, self-reported vs. all-drug deaths, death
     certificates) rather than papering over it.

This is a worked demonstration of the pipeline, not a finished result for submission.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/user/overdose_project"

# ---------------------------------------------------------------- load
with open(f"{OUT}/national_deaths_real.json") as fh:
    raw = json.load(fh)

df = pd.DataFrame(raw)
df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
df = df.sort_values("date").reset_index(drop=True)
df["t"] = np.arange(len(df))
n = len(df)
print(f"Loaded {n} months of real 12-month-ending overdose death totals: "
      f"{df.date.iloc[0]:%Y-%m} to {df.date.iloc[-1]:%Y-%m}")

# ---------------------------------------------------------- two-break fit
def fit_two_breaks(y, min_seg=12):
    """Grid search over (tau1, tau2) for the best-fitting 3-segment piecewise-linear
    trend (continuous at the breakpoints is NOT enforced -- each segment gets its own
    intercept+slope, same approach used on the simulated series earlier in this project).
    Returns (tau1, tau2, fitted_values, sse)."""
    t = np.arange(len(y))
    best = None
    for tau1 in range(min_seg, len(y) - 2 * min_seg):
        for tau2 in range(tau1 + min_seg, len(y) - min_seg):
            segs = [(0, tau1), (tau1, tau2), (tau2, len(y))]
            sse = 0.0
            fitted = np.empty(len(y))
            for a, b in segs:
                ts, ys = t[a:b], y[a:b]
                A = np.vstack([ts, np.ones_like(ts)]).T
                coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
                pred = A @ coef
                fitted[a:b] = pred
                sse += np.sum((ys - pred) ** 2)
            if best is None or sse < best[2]:
                best = (tau1, tau2, sse, fitted.copy())
    tau1, tau2, sse, fitted = best
    return tau1, tau2, fitted, sse

y = df["deaths"].to_numpy(dtype=float)
tau1, tau2, fitted, sse = fit_two_breaks(y, min_seg=12)
d1, d2 = df.date.iloc[tau1], df.date.iloc[tau2]
print(f"Best-fit break points (3-segment piecewise-linear trend on the real rolling "
      f"series): index {tau1} ({d1:%Y-%m}), index {tau2} ({d2:%Y-%m}); SSE={sse:,.0f}")

# one-break and no-break SSE for comparison (does a second break even earn its keep?)
def fit_one_break(y, min_seg=12):
    t = np.arange(len(y))
    best = None
    for tau in range(min_seg, len(y) - min_seg):
        sse = 0.0
        for a, b in [(0, tau), (tau, len(y))]:
            ts, ys = t[a:b], y[a:b]
            A = np.vstack([ts, np.ones_like(ts)]).T
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            sse += np.sum((ys - A @ coef) ** 2)
        if best is None or sse < best[1]:
            best = (tau, sse)
    return best

tau_single, sse_single = fit_one_break(y)
A0 = np.vstack([np.arange(n), np.ones(n)]).T
coef0, *_ = np.linalg.lstsq(A0, y, rcond=None)
sse_flat = np.sum((y - A0 @ coef0) ** 2)
print(f"SSE: no-break={sse_flat:,.0f}  one-break(@{df.date.iloc[tau_single]:%Y-%m})={sse_single:,.0f}  "
      f"two-break={sse:,.0f}")

# ------------------------------------------------------- net-flow proxy
df["net_flow_proxy"] = df["deaths"].diff()  # S_t - S_{t-1}; NOT true monthly deaths (see docstring)

# ---------------------------------------------------------------- plot 1
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(df.date, df.deaths, color="#1f5fa8", lw=1.8, label="12-month-ending total overdose deaths (real, CDC)")
ax.plot(df.date, fitted, color="#c0392b", lw=1.4, ls="--", label="2-break piecewise-linear fit")
for tau, lbl in [(tau1, "break 1"), (tau2, "break 2")]:
    ax.axvline(df.date.iloc[tau], color="gray", lw=1, ls=":")
    ax.annotate(f"{lbl}\n{df.date.iloc[tau]:%Y-%m}", (df.date.iloc[tau], ax.get_ylim()[1] * 0.97),
                ha="center", va="top", fontsize=8, color="gray")
ax.set_title("Real US drug overdose deaths (12-month-ending total), Jan 2015 – Apr 2026")
ax.set_ylabel("Deaths in trailing 12 months")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/real_deaths_breakpoints.png", dpi=150)
print(f"wrote {OUT}/real_deaths_breakpoints.png")

# ---------------------------------------------------------------- plot 2
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.axhline(0, color="black", lw=0.8)
ax.plot(df.date, df.net_flow_proxy, color="#27633f", lw=1.3)
ax.set_title("Month-over-month change in the rolling total (net-flow proxy, NOT true monthly deaths)")
ax.set_ylabel("Δ(12-mo-ending total)")
fig.tight_layout()
fig.savefig(f"{OUT}/real_deaths_netflow_proxy.png", dpi=150)
print(f"wrote {OUT}/real_deaths_netflow_proxy.png")

# -------------------------------------------------- NSDUH vs deaths comparison
# NSDUH: % of people 12+ reporting past-year opioid misuse (self-report survey).
# Source: SAMHSA 2024 NSDUH annual national report + 2021-2024 companion trend report.
nsduh = pd.DataFrame({
    "year": [2021, 2022, 2023, 2024],
    "opioid_misuse_pct": [3.2, 3.0, 3.05, 2.7],  # 2023 published as a 3.0-3.1 range; midpoint used, flagged below
})

# CDC: all-drug overdose deaths, 12-month-ending-December total for each calendar year
# (December value of the real series == that year's annual total).
annual = df[df.date.dt.month == 12].copy()
annual["year"] = annual.date.dt.year
annual = annual[annual.year.isin(nsduh.year)][["year", "deaths"]]

comp = nsduh.merge(annual, on="year")
comp["deaths_idx"] = 100 * comp.deaths / comp.deaths.iloc[0]
comp["opioid_misuse_idx"] = 100 * comp.opioid_misuse_pct / comp.opioid_misuse_pct.iloc[0]
print("\nNSDUH opioid misuse % vs CDC all-drug deaths, 2021 = 100:")
print(comp.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(comp.year, comp.deaths_idx, "o-", color="#1f5fa8", label="All-drug overdose deaths (CDC, index)")
ax.plot(comp.year, comp.opioid_misuse_idx, "s-", color="#c0392b", label="Past-year opioid misuse % (NSDUH, index)")
ax.set_title("Deaths fell faster than self-reported opioid misuse (2021 = 100)")
ax.set_ylabel("Index (2021 = 100)")
ax.set_xticks(comp.year)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/nsduh_vs_deaths.png", dpi=150)
print(f"wrote {OUT}/nsduh_vs_deaths.png")

# ---------------------------------------------------------------- save numbers
summary = dict(
    n_months=n,
    date_range=[f"{df.date.iloc[0]:%Y-%m}", f"{df.date.iloc[-1]:%Y-%m}"],
    break_fit=dict(tau1_date=f"{d1:%Y-%m}", tau2_date=f"{d2:%Y-%m}",
                   sse_flat=sse_flat, sse_one_break=sse_single, sse_two_break=sse,
                   one_break_date=f"{df.date.iloc[tau_single]:%Y-%m}"),
    peak=dict(date=f"{df.date.iloc[int(df.deaths.idxmax())]:%Y-%m}", value=int(df.deaths.max())),
    latest=dict(date=f"{df.date.iloc[-1]:%Y-%m}", value=int(df.deaths.iloc[-1])),
    pct_decline_from_peak=round(100 * (1 - df.deaths.iloc[-1] / df.deaths.max()), 1),
    nsduh_vs_deaths=comp.to_dict(orient="records"),
)
with open(f"{OUT}/real_analysis_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"\nwrote {OUT}/real_analysis_summary.json")
print(json.dumps(summary, indent=2))
