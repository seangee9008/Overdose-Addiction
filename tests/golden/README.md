# Golden snapshot

`golden_results.json` holds every value written to `outputs/*.csv|json` and `data/processed/*.csv`
as exact text, captured on 2026-10-23 before the data-engineering refactor (git commit noted in the
commit that added it). The full pipeline that produced it (`produced_by` in the JSON) took **70 s**
on the author's machine. `tests/test_golden.py` fails on any difference; never loosen it to make a
change pass -- investigate instead.
