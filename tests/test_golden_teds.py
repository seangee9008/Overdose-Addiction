"""Every TEDS-A number (outputs/teds/) must equal its own golden snapshot EXACTLY."""
import json
import os

import pytest

from tests.golden.golden import TEDS_GOLDEN, TEDS_SOURCES, collect, compare

pytestmark = pytest.mark.skipif(not os.path.exists(TEDS_GOLDEN), reason="TEDS golden not captured yet")


def test_teds_outputs_match_golden_exactly():
    golden = json.load(open(TEDS_GOLDEN))
    diffs = compare(golden["snapshot"], collect(TEDS_SOURCES))
    assert not diffs, f"{len(diffs)} TEDS values differ, first 20:\n" + "\n".join(map(str, diffs[:20]))
