# -*- coding: utf-8 -*-
"""
efficiency.py — 실험 4 병행: 성능-효율 로깅
=============================================
(pair, method, seed)마다 기록:
  total_params, trainable_params, trainable_pct
  adapt_time_s   : 적응(또는 fine-tune) 학습 벽시계 시간
  infer_ms_per_window : test 추론 평균 latency
  peak_mem_mb    : CUDA peak memory (CPU면 NaN)
→ Pareto plot (x=trainable params log, y=AUROC)은 analyze_results.py에서 생성
"""

from __future__ import annotations

import time
import numpy as np
import torch


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }


class AdaptTimer:
    """with AdaptTimer() as t: ... ; t.elapsed (초). CUDA sync 포함."""
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0
        self.peak_mem_mb = (torch.cuda.max_memory_allocated() / 1024**2
                            if torch.cuda.is_available() else float("nan"))


@torch.no_grad()
def measure_inference_latency(score_fn, loader, device, warmup: int = 2) -> float:
    """
    score_fn(batch)->scores 를 test loader 전체에 대해 실행,
    window당 평균 latency(ms) 반환. warmup 배치는 측정 제외.
    """
    n_windows, elapsed = 0, 0.0
    for i, batch in enumerate(loader):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = score_fn(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if i >= warmup:
            elapsed += time.perf_counter() - t0
            n_windows += len(x)
    return 1000.0 * elapsed / max(n_windows, 1)
