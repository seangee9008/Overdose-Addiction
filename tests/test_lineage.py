"""Area-level TEDS-A / county models must never depend on individual-level NSDUH models (and vice versa).
Reads dbt's dependency graph (dbt/target/manifest.json, written by any dbt build/parse)."""
import json
import os

import pytest

MANIFEST = "dbt/target/manifest.json"
pytestmark = pytest.mark.skipif(not os.path.exists(MANIFEST), reason="run dbt (make data / make fixtures-data) first")


def _graph():
    m = json.load(open(MANIFEST))
    parents = {k: v.get("depends_on", {}).get("nodes", []) for k, v in {**m["nodes"], **m["sources"]}.items()}
    return m, parents


def _ancestors(node, parents):
    seen, stack = set(), [node]
    while stack:
        for p in parents.get(stack.pop(), []):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def test_teds_models_never_depend_on_nsduh():
    m, parents = _graph()
    teds = [k for k, v in m["nodes"].items() if v["resource_type"] == "model" and "teds" in v["name"]]
    assert teds, "no TEDS models found"
    for k in teds:
        bad = [a for a in _ancestors(k, parents) if "nsduh" in a]
        assert not bad, f"{k} depends on NSDUH: {bad}"


def test_nsduh_models_never_depend_on_area_level_sources():
    m, parents = _graph()
    nsduh = [k for k, v in m["nodes"].items() if v["resource_type"] == "model" and "nsduh" in v["name"]]
    for k in nsduh:
        bad = [a for a in _ancestors(k, parents) if any(s in a for s in ("teds", "county", "svi", "census"))]
        assert not bad, f"{k} depends on area-level data: {bad}"
