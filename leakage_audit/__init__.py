"""Standalone adaptation–evaluation temporal leakage audit (Algorithm 1 of the paper)."""
from .audit import (AuditResult, audit, dilation_keep, evaluability, exact_keep,
                    exact_keep_from_arrays, span_keep, window_starts)

__all__ = ["AuditResult", "audit", "dilation_keep", "evaluability", "exact_keep",
           "exact_keep_from_arrays", "span_keep", "window_starts"]
