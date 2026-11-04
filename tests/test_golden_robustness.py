"""Every Phase 4a robustness number (outputs/robustness/) must equal its own golden snapshot EXACTLY.
Kept separate from tests/golden/golden_results.json, which must never change."""
import json
import os

import pytest

from tests.golden.golden import ROBUST_GOLDEN, ROBUST_SOURCES, collect, compare

pytestmark = pytest.mark.skipif(not os.path.exists(ROBUST_GOLDEN), reason="robustness golden not captured yet")


def test_robustness_outputs_match_golden_exactly():
    golden = json.load(open(ROBUST_GOLDEN))
    diffs = compare(golden["snapshot"], collect(ROBUST_SOURCES))
    assert not diffs, f"{len(diffs)} robustness values differ, first 20:\n" + "\n".join(map(str, diffs[:20]))
