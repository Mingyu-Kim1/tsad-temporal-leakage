"""Tests for leakage_audit. Run from the repository root:  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leakage_audit import (audit, dilation_keep, evaluability, exact_keep,  # noqa: E402
                           exact_keep_from_arrays, span_keep, window_starts)


def test_stride_asymmetry_one_pool_window():
    """One pool window (L=100) vs stride-1 evaluation: 1 exact duplicate, 4 windows shifted by 1-4
    samples share 96-99% of the span, and 2L-1 windows overlap in total (Fig. 1a)."""
    L = 100
    ev = window_starts(1000, L, 1)
    pool = np.array([400])
    assert (~exact_keep(pool, ev, L)).sum() == 1
    removed = ~span_keep(pool, ev, L)
    assert removed.sum() == 2 * L - 1
    for d in range(1, 5):
        assert removed[400 + d] and removed[400 - d]


@pytest.mark.parametrize("gap", [0, 100, 400])
@pytest.mark.parametrize("seed", range(5))
def test_dilation_equals_span_comparison(gap, seed):
    rng = np.random.default_rng(seed)
    L, n = 100, 5000
    ev = window_starts(n, L, 1)
    pool = np.sort(rng.choice(window_starts(n, L, 5), size=60, replace=False))
    inpool = ~exact_keep(pool, ev, L)
    assert np.array_equal(dilation_keep(inpool, L, gap), span_keep(pool, ev, L, gap=gap))


def test_hash_exact_matches_index_exact():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(800, 3))
    L = 20
    ev_s, pool_s = window_starts(800, L, 1), window_starts(400, L, 5)
    ev = np.stack([x[s:s + L] for s in ev_s])
    pool = np.stack([x[s:s + L] for s in pool_s])
    assert np.array_equal(exact_keep_from_arrays(pool, ev), exact_keep(pool_s, ev_s, L))


# Boiler release: each test file contains the train file as a prefix (Section 1).
# Pool = stride-5 windows over the target train file, evaluation = stride-1 windows over the
# test file, L = 100. Counts below are the stored audit values (work/tables_final/A1, rescore_summary).
BOILER = {  # pair: (target train length, n_eval, n_in(E2), frac E1, n_keep g=L, n_keep g=4L)
    "Boiler_1to2": (70712, 88291, 70710, 0.1599596787894576, 17481, 17181),
    "Boiler_1to3": (17926, 89540, 17925, 0.03982577618941252, 71515, 71215),
    "Boiler_2to1": (72096, 90021, 72095, 0.1599626753757457, None, None),
}


@pytest.mark.parametrize("pair", list(BOILER))
def test_reproduces_boiler_audit(pair):
    n_train, n_eval, n_in, frac_e1, keep_L, keep_4L = BOILER[pair]
    L = 100
    pool = window_starts(n_train, L, 5)
    ev = np.arange(n_eval)
    r = audit(pool, ev, L)
    assert r.removed_temporal == n_in
    assert r.frac_removed_exact == pytest.approx(frac_e1, abs=1e-12)
    if keep_L is not None:
        assert n_eval - audit(pool, ev, L, gap=L).removed_temporal == keep_L
        assert n_eval - audit(pool, ev, L, gap=4 * L).removed_temporal == keep_4L


def test_unscorable_when_pool_covers_evaluation():
    """Run-to-failure case: the pool spans the whole evaluation segment -> no windows remain."""
    L = 20
    ev = window_starts(300, L, 1)
    pool = window_starts(300, L, 5)
    y = np.zeros(ev.size, bool)
    y[-30:] = True
    r = audit(pool, ev, L, y_window=y)
    assert r.frac_removed_temporal == 1.0
    assert (r.pos_kept, r.neg_kept, r.evaluable) == (0, 0, False)
    assert evaluability(y, np.ones_like(y))["evaluable"]
