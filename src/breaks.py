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


# ---------------------------------------------------------------------------------------------
# Fast exact versions (same answers as above) for bootstrapping. Every segment's OLS SSE is
# computed in closed form from cumulative sums:
#   SSE(a,b) = Syy - Sy^2/m - (Sty - St*Sy/m)^2 / (Stt - St^2/m),   m = b - a


def segment_sse_matrix(y):
    """M[a, b] = SSE of an OLS line fitted to y[a:b] (b exclusive); NaN where b - a < 3."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    scale = y.std() or 1.0                      # center/scale for numerical stability;
    y = (y - y.mean()) / scale                  # SSE is rescaled by scale**2 at the end
    t = np.arange(n, dtype=float) - (n - 1) / 2
    c = [np.concatenate([[0.0], np.cumsum(v)]) for v in (np.ones(n), t, t * t, y, t * y, y * y)]
    a = np.arange(n + 1)[:, None]
    b = np.arange(n + 1)[None, :]
    S = [ci[b] - ci[a] for ci in c]
    m, St, Stt, Sy, Sty, Syy = S
    with np.errstate(divide="ignore", invalid="ignore"):
        sse = Syy - Sy ** 2 / m - (Sty - St * Sy / m) ** 2 / (Stt - St ** 2 / m)
    sse[(b - a) < 3] = np.nan
    return np.maximum(sse, 0.0) * scale ** 2


def fast_fit(y, min_seg=12):
    """Return dict(sse0, tau1, sse1, tau2=(t1, t2), sse2) using segment_sse_matrix."""
    n = len(y)
    M = segment_sse_matrix(y)
    sse0 = M[0, n]
    taus = np.arange(min_seg, n - min_seg + 1)
    s1 = M[0, taus] + M[taus, n]
    i = int(np.nanargmin(s1))
    t1 = np.arange(min_seg, n - 2 * min_seg + 1)[:, None]
    t2 = np.arange(n + 1)[None, :]
    s2 = M[0, t1] + M[t1, t2] + M[t2, n]
    s2 = np.where((t2 >= t1 + min_seg) & (t2 <= n - min_seg), s2, np.inf)
    k1, k2 = np.unravel_index(int(np.argmin(s2)), s2.shape)
    return dict(sse0=float(sse0), tau1=int(taus[i]), sse1=float(s1[i]),
                tau2=(int(t1[k1, 0]), int(k2)), sse2=float(s2[k1, k2]))
