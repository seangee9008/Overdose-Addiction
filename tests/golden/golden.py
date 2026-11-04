"""
Golden snapshot of every number the pipeline writes (outputs/*.csv, outputs/*.json,
data/processed/*.csv), stored as the exact text written, so comparisons are exact.

    python -m tests.golden.golden write     # (re)create tests/golden/golden_results.json
    python -m tests.golden.golden compare   # print differences vs the snapshot
"""
import json
import os
import sys

import pandas as pd

GOLDEN = "tests/golden/golden_results.json"
SOURCES = ("outputs", "data/processed")
# Phase 4a robustness numbers: a SEPARATE snapshot, so the original one above never changes.
ROBUST_GOLDEN = "tests/golden/robustness_golden.json"
ROBUST_SOURCES = ("outputs/robustness",)
ROBUST_PIPELINE = ["make data", "make results", "python -m src.step_r"]
PIPELINE = [
    "python analyze_real.py",
    'python analyze_real.py --indicator "Opioids (T40.0-T40.4,T40.6)" --tag opioids',
    "python -m src.nsduh_estimates", "python -m src.step_a", "python -m src.step_b",
    "python -m src.step_c", "python -m src.step_d", "python -m src.build_results",
]


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}{i}."))
        return out
    return {prefix.rstrip("."): obj}


def collect(sources=SOURCES):
    """{file: {"row.column" or "json.path": exact value}} for every data file."""
    snap = {}
    for d in sources:
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if f.endswith(".csv"):
                df = pd.read_csv(p, dtype=str, keep_default_na=False)
                snap[p] = {f"{i}.{c}": v for i, row in df.iterrows() for c, v in row.items()}
            elif f.endswith(".json"):
                snap[p] = _flatten(json.load(open(p)))
    return snap


def compare(golden, current):
    """List of (file, key, golden_value, current_value) differences, incl. missing/extra."""
    diffs = []
    for f in sorted(set(golden) | set(current)):
        g, c = golden.get(f), current.get(f)
        if g is None or c is None:
            diffs.append((f, "<file>", "present" if g else "missing", "present" if c else "missing"))
            continue
        for k in sorted(set(g) | set(c)):
            if g.get(k, "<missing>") != c.get(k, "<missing>"):
                diffs.append((f, k, g.get(k, "<missing>"), c.get(k, "<missing>")))
    return diffs


if __name__ == "__main__":
    robust = "--robustness" in sys.argv
    path, sources, pipeline = (ROBUST_GOLDEN, ROBUST_SOURCES, ROBUST_PIPELINE) if robust else (GOLDEN, SOURCES, PIPELINE)
    if "write" in sys.argv:
        snap = collect(sources)
        n = sum(len(v) for v in snap.values())
        json.dump(dict(produced_by=pipeline, files=len(snap), values=n, snapshot=snap),
                  open(path, "w"), indent=1, sort_keys=True)
        print(f"wrote {path}: {len(snap)} files, {n} values")
    else:
        d = compare(json.load(open(path))["snapshot"], collect(sources))
        print(f"{len(d)} differences")
        for row in d[:50]:
            print(row)
