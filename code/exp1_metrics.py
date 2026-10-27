# -*- coding: utf-8 -*-
"""
eval_metrics.py — 실험 1: 평가 지표 (확정안)
=============================================
  - 주지표  : AUROC, AUPR (threshold-free)
  - 보조    : Validation threshold(정상 adapt_val score의 99.9% quantile) 기반 F1
  - 참고값  : Best F1 (test에서 최적 threshold — DACAD 비교용으로만 보고)
  - Point Adjustment 미사용
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, precision_recall_curve,
)

VAL_QUANTILE = 0.999  # 확정: 99.9%


@dataclass
class EvalResult:
    auroc: float
    aupr: float
    f1_val_thr: float
    precision_val_thr: float
    recall_val_thr: float
    best_f1: float            # 참고값 (낙관적)
    threshold: float          # adapt_val에서 고정된 threshold
    anomaly_ratio: float      # 평가 구간 이상 비율 (해석용)

    def as_dict(self):
        return asdict(self)


def threshold_from_normal_val(val_scores: np.ndarray, quantile: float = VAL_QUANTILE) -> float:
    """정상 adapt_val score의 상위 quantile → threshold 고정 (소량 정상만으로 결정 가능)."""
    return float(np.quantile(np.asarray(val_scores, float), quantile))


def best_f1_score(scores: np.ndarray, labels: np.ndarray) -> float:
    p, r, _ = precision_recall_curve(labels, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(np.nanmax(f1))


def evaluate(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    val_scores: np.ndarray,
    quantile: float = VAL_QUANTILE,
) -> EvalResult:
    s = np.asarray(test_scores, float)
    y = np.asarray(test_labels, int)
    assert len(s) == len(y), f"score({len(s)})와 label({len(y)}) 길이 불일치 — window 매핑 확인"

    single_class = len(np.unique(y)) < 2
    auroc = np.nan if single_class else roc_auc_score(y, s)
    aupr = np.nan if single_class else average_precision_score(y, s)
    bf1 = np.nan if single_class else best_f1_score(s, y)

    thr = threshold_from_normal_val(val_scores, quantile)
    pred = (s >= thr).astype(int)
    return EvalResult(
        auroc=float(auroc), aupr=float(aupr),
        f1_val_thr=float(f1_score(y, pred, zero_division=0)),
        precision_val_thr=float(precision_score(y, pred, zero_division=0)),
        recall_val_thr=float(recall_score(y, pred, zero_division=0)),
        best_f1=float(bf1), threshold=thr,
        anomaly_ratio=float(y.mean()),
    )
