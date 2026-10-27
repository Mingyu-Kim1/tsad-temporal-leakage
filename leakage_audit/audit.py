"""leakage_audit.audit — adaptation–evaluation temporal leakage audit (standalone, NumPy only).

This module implements the evaluation control of the paper (Section 3.5, Algorithm 1)
independently of the experiment pipeline, so it can be applied to any sliding-window
time-series experiment.

Definitions (time indices are integer sample positions in one series)
    window x with start s and length L covers the span [s, s + L - 1]
    E0      : all evaluation windows (no control)
    E1      : remove evaluation windows identical to a pool window (exact match)
    E2(g)   : remove evaluation windows whose span lies within distance g of any pool
              window span. g = 0 removes every window that shares at least one time step.
    distance between spans [a, b] and [c, d] = max(0, c - b, a - d)
              (0 when they overlap, 1 when they are adjacent)

Two implementations of E2(g) are provided:
    span_keep      direct span comparison; works for any strides and window lengths
    dilation_keep  Algorithm 1 fast path; valid when evaluation windows have stride 1 and
                   pool windows lie on the same start grid (radius = g + L - 1)
They agree whenever both apply (see tests/test_leakage_audit.py).
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "window_starts", "exact_keep", "exact_keep_from_arrays", "span_keep",
    "dilation_keep", "evaluability", "AuditResult", "audit",
]


# ---------------------------------------------------------------------------
def window_starts(n: int, length: int, stride: int = 1, offset: int = 0) -> np.ndarray:
    """Start indices of sliding windows of `length` over a series of `n` samples."""
    if n < length:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, n - length + 1, stride, dtype=np.int64) + int(offset)


def exact_keep(pool_starts, eval_starts, pool_len: int, eval_len: int | None = None) -> np.ndarray:
    """E1 on index level: drop evaluation windows with the same start and length as a pool window."""
    eval_len = pool_len if eval_len is None else eval_len
    eval_starts = np.asarray(eval_starts, dtype=np.int64)
    if eval_len != pool_len:
        return np.ones(eval_starts.size, dtype=bool)
    return ~np.isin(eval_starts, np.asarray(pool_starts, dtype=np.int64))


def exact_keep_from_arrays(pool_windows: np.ndarray, eval_windows: np.ndarray, decimals: int = 6) -> np.ndarray:
    """E1 on value level (hash comparison of window arrays), as used when only arrays are available."""
    def h(w):
        w = np.ascontiguousarray(np.round(np.asarray(w, dtype=np.float64), decimals))
        return hashlib.blake2b(w.tobytes(), digest_size=16).digest()
    pool = {h(w) for w in pool_windows}
    return np.array([h(w) not in pool for w in eval_windows], dtype=bool)


def span_keep(pool_starts, eval_starts, pool_len: int, eval_len: int | None = None, gap: int = 0) -> np.ndarray:
    """E2(g) by direct span comparison. Returns True for evaluation windows that are kept."""
    eval_len = pool_len if eval_len is None else eval_len
    p = np.sort(np.asarray(pool_starts, dtype=np.int64))
    e = np.asarray(eval_starts, dtype=np.int64)
    if p.size == 0:
        return np.ones(e.size, dtype=bool)
    # pool spans expanded by g, merged into disjoint segments [lo_k, hi_k]
    lo = p - gap
    hi = p + pool_len - 1 + gap
    seg_lo, seg_hi = [lo[0]], [hi[0]]
    for a, b in zip(lo[1:], hi[1:]):
        if a <= seg_hi[-1] + 1:
            seg_hi[-1] = max(seg_hi[-1], b)
        else:
            seg_lo.append(a)
            seg_hi.append(b)
    seg_lo, seg_hi = np.array(seg_lo), np.array(seg_hi)
    a, b = e, e + eval_len - 1
    # first segment whose end >= a; window is removed if that segment starts <= b
    k = np.searchsorted(seg_hi, a, side="left")
    hit = (k < seg_lo.size)
    hit[hit] = seg_lo[k[hit]] <= b[hit]
    return ~hit


def dilation_keep(exact_inpool: np.ndarray, length: int, gap: int = 0) -> np.ndarray:
    """E2(g) by dilating the exact-match indicator (Algorithm 1, line 3).

    exact_inpool[i] is True when evaluation window i (stride 1) equals a pool window.
    Every window within radius g + L - 1 of such a position shares time with it.
    """
    m = np.asarray(exact_inpool, dtype=bool)
    radius = gap + length - 1
    if radius <= 0 or not m.any():
        return ~m
    n = m.size
    diff = np.zeros(n + 1, dtype=np.int64)
    idx = np.flatnonzero(m)
    np.add.at(diff, np.maximum(idx - radius, 0), 1)
    np.add.at(diff, np.minimum(idx + radius + 1, n), -1)
    return ~(np.cumsum(diff[:-1]) > 0)


# ---------------------------------------------------------------------------
def evaluability(y_window, keep) -> dict:
    """Positive and negative windows left after control; a pair is scorable only if both remain."""
    y = np.asarray(y_window).astype(bool)
    k = np.asarray(keep, dtype=bool)
    pos, neg = int((y & k).sum()), int((~y & k).sum())
    return {"pos_kept": pos, "neg_kept": neg, "evaluable": bool(pos > 0 and neg > 0)}


@dataclass
class AuditResult:
    n_eval: int
    n_pool: int
    removed_exact: int
    removed_temporal: int
    frac_removed_exact: float
    frac_removed_temporal: float
    leaked_anomaly_share: float | None
    pos_kept: int | None
    neg_kept: int | None
    evaluable: bool | None
    gap: int

    def to_dict(self):
        return asdict(self)


def audit(pool_starts, eval_starts, length: int, y_window=None, gap: int = 0,
          eval_len: int | None = None) -> AuditResult:
    """Full audit of one transfer pair (Algorithm 1): E1/E2(g) removal, leaked anomaly share, evaluability."""
    pool_starts = np.asarray(pool_starts, dtype=np.int64)
    eval_starts = np.asarray(eval_starts, dtype=np.int64)
    k1 = exact_keep(pool_starts, eval_starts, length, eval_len)
    k2 = span_keep(pool_starts, eval_starts, length, eval_len, gap)
    n = eval_starts.size
    share = pos = neg = ev = None
    if y_window is not None:
        y = np.asarray(y_window).astype(bool)
        share = float((y & ~k2).sum() / y.sum()) if y.any() else float("nan")
        e = evaluability(y, k2)
        pos, neg, ev = e["pos_kept"], e["neg_kept"], e["evaluable"]
    return AuditResult(
        n_eval=int(n), n_pool=int(pool_starts.size),
        removed_exact=int((~k1).sum()), removed_temporal=int((~k2).sum()),
        frac_removed_exact=float((~k1).mean()) if n else 0.0,
        frac_removed_temporal=float((~k2).mean()) if n else 0.0,
        leaked_anomaly_share=share, pos_kept=pos, neg_kept=neg, evaluable=ev, gap=int(gap),
    )
