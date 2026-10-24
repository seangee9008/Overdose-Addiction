"""Every number in outputs/ and data/processed/ must equal the golden snapshot EXACTLY
(text as written; no tolerance). Regenerate outputs with `make results` before running."""
import json

from tests.golden.golden import GOLDEN, collect, compare


def test_outputs_match_golden_exactly():
    golden = json.load(open(GOLDEN))
    diffs = compare(golden["snapshot"], collect())
    assert not diffs, f"{len(diffs)} values differ from golden, first 20:\n" + "\n".join(map(str, diffs[:20]))
