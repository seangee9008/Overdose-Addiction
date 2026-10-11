"""
Piecewise-linear structural-break detection by exhaustive grid search.

Each segment gets its own intercept and slope (continuity at the break points is NOT
enforced). The break locations are the indices that minimise total within-segment SSE.
A break at index tau means the new segment starts at observation tau.
"""
import numpy as np


def _segment_sse(t, y, a, b):
    """SSE and fitted values of an OLS line fitted to y[a:b] on t[a:b]."""
    ts, ys = t[a:b], y[a:b]
    A = np.vstack([ts, np.ones_like(ts)]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    return float(np.sum((ys - pred) ** 2)), pred


def fit_no_break(y):
    """Single straight line. Returns (fitted, sse)."""
    y = np.asarray(y, dtype=float)
    sse, fitted = _segment_sse(np.arange(len(y), dtype=float), y, 0, len(y))
    return fitted, sse


def fit_one_break(y, min_seg=12):
    """Best single break. Returns (tau, fitted, sse)."""
    y = np.asarray(y, dtype=float)
    t = np.arange(len(y), dtype=float)
    best = None
    for tau in range(min_seg, len(y) - min_seg + 1):
        sse_a, pred_a = _segment_sse(t, y, 0, tau)
        sse_b, pred_b = _segment_sse(t, y, tau, len(y))
        sse = sse_a + sse_b
        if best is None or sse < best[2]:
            best = (tau, np.concatenate([pred_a, pred_b]), sse)
    return best


def fit_two_breaks(y, min_seg=12):
    """Best pair of breaks (tau1 < tau2), every segment at least min_seg long.
    Returns (tau1, tau2, fitted, sse)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    t = np.arange(n, dtype=float)
    best = None
    for tau1 in range(min_seg, n - 2 * min_seg + 1):
        for tau2 in range(tau1 + min_seg, n - min_seg + 1):
            sse, preds = 0.0, []
            for a, b in [(0, tau1), (tau1, tau2), (tau2, n)]:
                s, p = _segment_sse(t, y, a, b)
                sse += s
                preds.append(p)
            if best is None or sse < best[3]:
                best = (tau1, tau2, np.concatenate(preds), sse)
    return best
