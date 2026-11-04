#!/usr/bin/env python3
"""
fetch_data.py -- download every raw dataset for the addiction/overdose project into data/raw/.

Run this on YOUR machine (the cloud sandbox this was drafted in cannot reach these hosts).
Standard library only. Re-running skips files that already exist; use --force to refetch.

  python fetch_data.py              # everything automatic
  python fetch_data.py --only cdc_state cdc_county
  python fetch_data.py --list

STATUS OF EACH SOURCE
  cdc_state    VERIFIED  (national series from this dataset was pulled via the same API)
  cdc_county   VERIFIED 2026-10-10
  nsduh_2023   VERIFIED 2026-10-10. Stata bundle (.dta). The SAS bundle on the same page is a
               SAS CPORT (.stc) file that Python cannot read without SAS, so it is not used.
  nsduh_2021_2024  Combined (concatenated) 2021-2024 PUF, Stata bundle, + its codebook. The SAS
               version of this file is CPORT (.stc) and unreadable in Python.
  nsduh_2024   Single-year 2024 PUF (Stata) + codebook, to check 2024 opioid misuse / OUD definitions.
  census_pop   Census Population Estimates, Vintage 2025: July 1 resident population 2020-2025 for
               states (SUMLEV 040) and counties (SUMLEV 050). National = sum of 50 states + DC.
  svi_2022     VERIFIED 2026-10-10. Official ATSDR CSV from svi.cdc.gov (no geometry; join on FIPS).
  svi_2022_arcgis_geo  OPT-IN ONLY (--only svi_2022_arcgis_geo). Polygons from an ArcGIS Online
               layer whose metadata credits CDC/ATSDR GRASP, but it is hosted in a shared public
               org, so treat it as a mirror. Worked 2026-10-10 (3,144 features).
Nothing here is parsed or cleaned; this script only fetches and records checksums.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
UA = "UCLA-Stats199-research/0.1 (student project)"

FILES = {
    "cdc_state": ("https://data.cdc.gov/api/views/xkb8-kh2a/rows.csv?accessType=DOWNLOAD",
                  "cdc_vsrr_overdose_national_state.csv"),
    "cdc_county": ("https://data.cdc.gov/api/views/gb4e-yj24/rows.csv?accessType=DOWNLOAD",
                   "cdc_vsrr_overdose_county.csv"),
    "nsduh_2023": ("https://www.samhsa.gov/data/system/files/media-puf-file/nsduh-2023-ds0001-bndl-data-stata_v2.zip",
                   "nsduh_2023_puf_stata.zip"),
    "nsduh_2023_codebook": ("https://www.samhsa.gov/data/system/files/media-puf-file/nsduh-2023-ds0001-info-codebook_v2.pdf",
                            "nsduh_2023_codebook.pdf"),
    "nsduh_2021_2024": ("https://www.samhsa.gov/data/system/files/media-puf-file/NSDUH-2021_2024-DS0001-bndl-data-stata_v1.zip",
                        "nsduh_2021_2024_concat_puf_stata.zip"),
    "nsduh_2021_2024_codebook": ("https://www.samhsa.gov/data/system/files/media-puf-file/nsduh-2021_2024-ds0001-info-codebook_v1.pdf",
                                 "nsduh_2021_2024_concat_codebook.pdf"),
    "nsduh_2024": ("https://www.samhsa.gov/data/system/files/media-puf-file/NSDUH-2024-DS0001-bndl-data-stata_v1.zip",
                   "nsduh_2024_puf_stata.zip"),
    "nsduh_2024_codebook": ("https://www.samhsa.gov/data/system/files/media-puf-file/nsduh-2024-ds0001-info-codebook_v1.pdf",
                            "nsduh_2024_codebook.pdf"),
    "census_pop": ("https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/counties/totals/co-est2025-alldata.csv",
                   "census_co_est2025_alldata.csv"),
    "svi_2022": ("https://svi.cdc.gov/Documents/Data/2022/csv/states_counties/SVI_2022_US_county.csv",
                 "svi_2022_us_county.csv"),
}
SVI_LAYER = "https://services7.arcgis.com/arZnhQhtvIXpgVPD/ArcGIS/rest/services/SVI2022_US_county/FeatureServer/0/query"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, force=False):
    if os.path.exists(dest) and not force:
        print(f"  skip (exists): {dest}")
        return True
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    print(f"  ok: {dest} ({os.path.getsize(dest):,} bytes)")
    return True


def fetch_svi(dest, force=False):
    """Page through the ArcGIS FeatureServer layer as GeoJSON (non-official mirror; opt-in)."""
    if os.path.exists(dest) and not force:
        print(f"  skip (exists): {dest}")
        return True
    feats, offset, page = [], 0, 1000
    while True:
        q = urllib.parse.urlencode(dict(where="1=1", outFields="*", f="geojson",
                                        resultOffset=offset, resultRecordCount=page))
        req = urllib.request.Request(f"{SVI_LAYER}?{q}", headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
        except Exception as e:
            print(f"  FAILED at offset {offset}: {type(e).__name__}: {e}")
            return False
        got = data.get("features", [])
        feats += got
        print(f"  svi: {len(feats)} counties so far")
        if not got or len(got) < page and not data.get("exceededTransferLimit"):
            break
        offset += len(got)
        time.sleep(1)
    with open(dest, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    print(f"  ok: {dest} ({len(feats)} features; expect ~3,144 counties)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    names = list(FILES)
    optional = ["svi_2022_arcgis_geo"]
    if a.list:
        print("\n".join(names + [f"{o} (opt-in)" for o in optional]))
        return
    todo = a.only or names
    os.makedirs(RAW, exist_ok=True)
    manifest = {}
    for name in todo:
        print(f"[{name}]")
        if name == "svi_2022_arcgis_geo":
            dest = os.path.join(RAW, "svi_2022_county.geojson")
            ok = fetch_svi(dest, a.force)
        elif name in FILES:
            url, fn = FILES[name]
            dest = os.path.join(RAW, fn)
            ok = download(url, dest, a.force)
        else:
            print("  unknown source")
            continue
        if ok and os.path.exists(dest):
            manifest[name] = dict(file=os.path.basename(dest), bytes=os.path.getsize(dest), sha256=sha256(dest),
                                  fetched=time.strftime("%Y-%m-%d"))
    mp = os.path.join(RAW, "MANIFEST.json")
    old = json.load(open(mp)) if os.path.exists(mp) else {}
    old.update(manifest)
    json.dump(old, open(mp, "w"), indent=2)
    print(f"\nwrote {mp}")


if __name__ == "__main__":
    main()
