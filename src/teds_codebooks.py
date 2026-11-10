"""
Harmonization check for TEDS-A 2024 (single-year file) against the combined 2006-2023 file.

  * Codebook level: parse the value-label tables of both codebook PDFs for every variable the pipeline
    uses and compare code by code (same code + same label / label differs / code in only one codebook).
  * Data level: compare the set of codes observed in the 2023 rows (combined file) and the 2024 rows.

Writes outputs/teds/harmonization_codebook_2024.csv and outputs/teds/harmonization_data_2024.csv.
"""
import re

import pandas as pd
import pypdf

from src import warehouse

CB_COMBINED = "data/raw/teds_a_2006_2023_codebook.pdf"
CB_2024 = "data/raw/teds_a_2024_codebook.pdf"
VARS = ["AGE", "SEX", "RACE", "ETHNIC", "STFIPS", "SUB1", "SUB2", "SUB3", "HERFLG", "METHFLG", "OPSYNFLG",
        "NOPRIOR", "METHUSE", "SERVICES", "PSOURCE", "ROUTE1", "IDU"]
VALUE_LINE = re.compile(r"^(-?\d+)\s+(.+?)\s+[\d,]+\s+[\d.]+%\s*$")
HEADER = re.compile(r"^([A-Z][A-Z0-9_]+):\s")


def codebook_labels(path):
    """{variable: {code: label}} from the value-label tables of a TEDS-A codebook PDF."""
    lines = []
    for page in pypdf.PdfReader(path).pages:
        lines += (page.extract_text() or "").splitlines()
    out, current = {}, None
    for raw in lines:
        line = raw.strip().replace("–", "-")
        h = HEADER.match(line)
        if h and h.group(1) in VARS:
            current = h.group(1)
            out.setdefault(current, {})
            continue
        if h:
            current = None
            continue
        m = VALUE_LINE.match(line)
        if current and m:
            out[current].setdefault(int(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip())
    return out


def compare_codebooks():
    a, b = codebook_labels(CB_COMBINED), codebook_labels(CB_2024)
    rows = []
    for v in VARS:
        ca, cb = a.get(v, {}), b.get(v, {})
        for code in sorted(set(ca) | set(cb)):
            la, lb = ca.get(code), cb.get(code)
            if la is None:
                status = "only in 2024 codebook"
            elif lb is None:
                status = "only in 2006-2023 codebook"
            elif la.lower() == lb.lower():
                status = "same"
            else:
                status = "label differs"
            rows.append(dict(variable=v, code=code, label_2006_2023=la, label_2024=lb, status=status))
    out = pd.DataFrame(rows)
    out["source"] = f"{CB_COMBINED} vs {CB_2024} (value-label tables)"
    return out


def compare_data():
    rows = []
    for v in VARS:
        col = {"AGE": "age_group", "STFIPS": "state_fips"}.get(v, v.lower())
        q = warehouse.query(f"select year, {col} as code, count(*) as n from stg_teds_a where year in (2023, 2024) "
                            "group by 1, 2")
        q["code"] = q.code.astype(str).replace({"<NA>": "missing (-9)", "None": "missing (-9)", "nan": "missing (-9)"})
        piv = q.pivot_table(index="code", columns="year", values="n", aggfunc="sum").fillna(0)
        for code, r in piv.iterrows():
            n23, n24 = int(r.get(2023, 0)), int(r.get(2024, 0))
            rows.append(dict(variable=v, code=code, admissions_2023=n23, admissions_2024=n24,
                             status="both years" if n23 and n24 else ("2023 only" if n23 else "2024 only")))
    out = pd.DataFrame(rows)
    out["source"] = "stg_teds_a (combined 2006-2023 file for 2023; single-year 2024 file for 2024)"
    return out
