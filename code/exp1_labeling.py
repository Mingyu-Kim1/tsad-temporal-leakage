# -*- coding: utf-8 -*-
"""
exp1_labeling.py — 실험 1: 이상 라벨링
========================================
확정 설계:
  - 배관(Sim→Real): CSV에 라벨 컬럼 없음. 원본 기록의 fault_point_idx=2549 를
    코드에서 직접 지정 → labels[2549:] = 1 (2600 중 마지막 51개 이상)
  - 베어링(XJTU/PHM): '원본(비정규화)' CSV의 rms_h·rms_v에 대해
    초기 정상구간(앞 10%) μ + kσ 최초 초과 시점 이후 = 이상 (k=3 기본)
  - persistence: 스파이크 방어 옵션 (1=계획서 원정의, 5=연속 초과 조건)
    ※ XB1은 index 2에 이미 rms 3.2 스파이크 존재 → 두 정의 비교 보고 권장
  - window 라벨: window(20) 내 이상 point 1개라도 있으면 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BEARING_RMS_COLS = ["rms_h", "rms_v"]
PIPE_FAULT_IDX_DEFAULT = 2549  # 사람 기준 2550번째 시점 (0-based 2549), 원본 기록 확인값


# ---------------------------------------------------------------------------
# 배관: known onset
# ---------------------------------------------------------------------------

def pipe_labels(n_rows: int, fault_idx: int = PIPE_FAULT_IDX_DEFAULT) -> dict:
    labels = np.zeros(n_rows, dtype=int)
    labels[fault_idx:] = 1
    return {"labels": labels, "onset_index": int(fault_idx),
            "type": "known_onset", "n_anomaly": int(labels.sum())}


# ---------------------------------------------------------------------------
# 베어링: RMS μ + kσ pseudo label (원본 CSV 기준)
# ---------------------------------------------------------------------------

def _first_persistent(binary: np.ndarray, persistence: int, start: int = 0):
    """start index 이후에서 연속 persistence개 초과가 시작되는 첫 위치."""
    b = binary[start:]
    if persistence <= 1:
        hits = np.flatnonzero(b)
        return int(start + hits[0]) if hits.size else None
    run = np.convolve(b.astype(int), np.ones(persistence, int), mode="valid")
    hits = np.flatnonzero(run == persistence)
    return int(start + hits[0]) if hits.size else None


def bearing_pseudo_labels(
    csv_path: str,
    k: float = 3.0,
    healthy_frac: float = 0.1,
    persistence: int = 1,
    rms_cols=BEARING_RMS_COLS,
    robust: bool = False,
    segments: list | None = None,
) -> dict:
    """
    run-to-failure 베어링 pseudo label.
    두 RMS 채널 중 '하나라도' 자기 채널의 threshold를 초과하면 초과로 판정.
    onset 이후 전 구간 = 1 (열화는 비가역 가정).

    핵심 원칙: 초기 정상구간(앞 healthy_frac)은 '정의상 정상'이므로
    onset 탐색은 정상구간 종료 이후(index >= h)부터 시작.
    (run-in 구간에서 RMS가 높게 시작해 내려가는 실데이터 특성 때문에
     정상구간 내부 포인트가 threshold를 넘어 onset=0으로 오검출되는 것 방지)

    robust=True: μ+kσ 대신 median + k·(1.4826·MAD) — 정상구간 내 스파이크·드리프트에 강건.

    segments: combined CSV가 여러 베어링의 이어붙임인 경우 각 베어링 길이 리스트.
      미지정 시 행 수가 SEGMENT_PRESETS와 정확히 일치하면 자동 적용.
      세그먼트마다 (정상구간 → threshold → onset)을 독립 수행하고,
      각 세그먼트의 onset~세그먼트 끝 구간만 이상(1)으로 라벨.
    """
    df = pd.read_csv(csv_path)
    rms = df[rms_cols].to_numpy(float)
    T = len(rms)
    segs, seg_src = resolve_segments(T, segments)

    labels = np.zeros(T, dtype=int)
    seg_meta, pos = [], 0
    for si, L in enumerate(segs):
        r = rms[pos:pos + L]
        h = max(int(L * healthy_frac), 10)
        if robust:
            mu = np.median(r[:h], axis=0)
            sd = np.median(np.abs(r[:h] - mu), axis=0) * 1.4826 + 1e-12
        else:
            mu = r[:h].mean(axis=0)
            sd = r[:h].std(axis=0, ddof=1)
        thr = mu + k * sd
        exceed = (r > thr).any(axis=1)
        onset = _first_persistent(exceed, persistence, start=h)  # 세그먼트-로컬
        if onset is not None:
            labels[pos + onset:pos + L] = 1
        seg_meta.append({"segment": si, "start": pos, "length": L,
                         "onset_local": onset,
                         "onset_global": None if onset is None else pos + onset,
                         "thresholds": {c: float(t) for c, t in zip(rms_cols, thr)}})
        pos += L

    onsets_g = [m["onset_global"] for m in seg_meta]
    return {"labels": labels,
            "onset_index": onsets_g[0] if len(segs) == 1 else onsets_g,  # 하위호환
            "segments": segs, "segment_source": seg_src, "segment_meta": seg_meta,
            "k": k, "persistence": persistence, "robust": robust,
            "healthy_frac": healthy_frac,
            "type": "pseudo_rms" + ("" if len(segs) == 1 else "_segmented"),
            "n_anomaly": int(labels.sum())}


def onset_table(csv_path: str, k_values=(2.0, 3.0, 4.0),
                persistences=(1, 5), robusts=(False, True), **kw) -> pd.DataFrame:
    """k × persistence × robust 조합별 onset 비교표 (민감도 분석 부속 자료)."""
    rows = []
    for k in k_values:
        for p in persistences:
            for rb in robusts:
                r = bearing_pseudo_labels(csv_path, k=k, persistence=p, robust=rb, **kw)
                oi = r["onset_index"]
                rows.append({"k": k, "persistence": p, "robust": rb,
                             "onset_index": str(oi),
                             "anomaly_ratio": r["labels"].mean()})
    return pd.DataFrame(rows)




# ---------------------------------------------------------------------------
# 다중 베어링 세그먼트 (combined CSV = 여러 run-to-failure 이어붙임)
# ---------------------------------------------------------------------------
# ⚠ 아래 preset은 XJTU-SY/PHM2012 공식 베어링별 샘플 수와 행 수가 '정확히 일치'해서
#   추정한 값. combined CSV 생성 스크립트와 반드시 대조 확인할 것 (순서 포함).
SEGMENT_PRESETS = {
    # rows=1566: XJTU condition2 B2_1..B2_5
    1566: ("XB2", [491, 161, 533, 42, 339]),
    # rows=494: XJTU condition1 B1_1,B1_2,B1_3,B1_5 (B1_4=122 제외 시 합=494) — 확인 필요
    494:  ("XB1", [123, 161, 158, 52]),
    # rows=13219: PHM condition1 B1_1..B1_7 중 B1_4(1428) 제외 6개 합 — 확인 필요
    13219: ("PB1", [2803, 871, 2375, 2463, 2448, 2259]),
    # rows=5345: PHM condition2 중 B2_5(2311) 제외 6개 합 — 확인 필요
    5345: ("PB2", [911, 797, 1955, 751, 701, 230]),
}


def resolve_segments(n_rows: int, segments=None):
    """명시 segments 우선, 없으면 행 수가 preset과 정확히 일치할 때만 적용."""
    if segments is not None:
        assert sum(segments) == n_rows, f"segments 합({sum(segments)}) != 행수({n_rows})"
        return list(segments), "explicit"
    if n_rows in SEGMENT_PRESETS:
        name, segs = SEGMENT_PRESETS[n_rows]
        return list(segs), f"preset:{name}"
    return [n_rows], "single"


def suggest_segment_boundaries(csv_path: str, drop_ratio: float = 0.4,
                               min_seg: int = 30, rms_cols=BEARING_RMS_COLS):
    """
    경계 후보 자동 탐지 (사용자 검증용): 고장 직후 새 베어링 시작 시
    RMS가 급락하는 지점. rms_mean[i] < drop_ratio * max(직전 min_seg 구간) 인 i.
    """
    df = pd.read_csv(csv_path)
    r = df[rms_cols].to_numpy(float).mean(axis=1)
    cands = []
    for i in range(min_seg, len(r)):
        prev_max = r[max(0, i - min_seg):i].max()
        if r[i] < drop_ratio * prev_max and (not cands or i - cands[-1] >= min_seg):
            cands.append(i)
    return cands


# ---------------------------------------------------------------------------
# point → window 라벨 (make_windows: stride 1, 개수 = T - W + 1)
# ---------------------------------------------------------------------------

def point_to_window_labels(point_labels: np.ndarray, window_size: int,
                           rule: str = "any") -> np.ndarray:
    p = np.asarray(point_labels, dtype=int)
    n = len(p) - window_size + 1
    if rule == "last":
        return p[window_size - 1:]
    idx = np.arange(window_size)[None, :] + np.arange(n)[:, None]
    return p[idx].max(axis=1).astype(int)
