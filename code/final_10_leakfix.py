"""
final_10_leakfix.py
===================
"적응 pool 에 들어간 window 를 평가 집합에서 제거하고 다시 채점한다"를
기존 파이프라인을 **재구현하지 않고** 런타임 패치로 구현하는 모듈.

왜 이렇게 하는가
----------------
window 생성 / scaler / best-head MAE 점수 계산을 다시 짜면 기존 결과와
미세하게 달라질 위험이 있다. 그래서 파이프라인이 실제로 만든 배열을
가로채서 해시만 비교한다. 학습 경로는 한 줄도 건드리지 않으므로
auroc(기존 지표)는 이전 실행과 비트 단위로 같아야 한다.
-> 그 자체가 이 패치가 안전한지 검증하는 지표가 된다(회귀 테스트).

무엇을 하는가
-------------
1) exp3_final.make_pool 을 감싼다.
   make_pool 에 들어오는 D 에는 clean / anom / eval_x / eval_y 가 있다.
   여기서 pair 별 마스크를 만든다:
       keep = (eval window 해시가 target-train pool 후보 해시에 없음)
   * 오염률/seed 와 무관하게 pair 마다 하나의 고정 마스크를 쓴다.
     방법 간 AUROC 를 비교하려면 평가 집합이 동일해야 하기 때문이다.
     clean + anom 후보 전체를 제거하는 가장 보수적인 정의를 쓴다.

2) exp1_metrics.evaluate 를 감싼다.
   원래 반환 dict 에 다음을 추가한다:
       auroc_nopool, aupr_nopool, n_eval_nopool, pos_nopool, neg_nopool,
       frac_removed, mask_source, f1_far0.01_nopool, prec/rec_far0.01_nopool
   러너가 dict 를 그대로 CSV 로 흘리면 results.csv 에 바로 실린다.
   흘리지 않더라도 사이드카 CSV(leakfix_rows.csv)에 같은 값을 적어두므로
   final_30_analysis.py 가 (n_eval_windows, auroc) 로 사후 병합한다.

3) 마스크는 디스크에 캐시된다(leakfix_masks.npz).
   한 번 만들어지면 이후 실행/스테이지에서 재사용된다.

사용법 (직접 실행하지 말고 final_20_run.py 가 import 한다)
    import final_10_leakfix as LF
    LF.install(outdir=r"jobs_out\\final\\S2")
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402  누수 마스크 위치 (work/leakfix_masks.npz)

import numpy as np

# ----------------------------------------------------------------------
# 전역 상태
# ----------------------------------------------------------------------
_LOCK = threading.Lock()
_MASKS: dict[str, np.ndarray] = {}          # y_hash -> bool mask (eval window 유지 여부)
_MASK_META: dict[str, dict] = {}            # y_hash -> meta
_LEN_INDEX: dict[int, list[str]] = {}       # len(y) -> [y_hash...]  (보조 조회)
_ORIG_EVALUATE = None
_ORIG_MAKE_POOL = None
_ORIG_PREPARE = None
_ORIG_EVAL_ALL = [None]
_CURRENT_PAIR = {"name": "?"}
_CALL_IDX = {"n": 0}
_STATE = {
    "outdir": None,
    "mask_path": None,
    "sidecar": None,
    "installed": False,
    "abort_after_masks": False,
    "hash_round": None,   # None=바이트 정확 비교, 정수면 그 자릿수로 반올림 후 비교
    "n_eval_calls": 0,
    "n_mask_hit": 0,
    "n_mask_miss": 0,
}

SIDECAR_COLS = [
    "call_idx", "pair", "y_hash", "n_eval", "n_eval_nopool", "frac_removed",
    "pos_all", "pos_nopool", "auroc", "auroc_nopool", "aupr", "aupr_nopool",
    "f1_far0.01", "f1_far0.01_nopool", "mask_source",
]


class MaskProbeDone(Exception):
    """--masks-only 모드에서 마스크만 만들고 해당 pair 를 빠져나가기 위한 신호."""


# ----------------------------------------------------------------------
# 해시
# ----------------------------------------------------------------------
def _win_hashes(arr) -> np.ndarray:
    """(N, T, F) 또는 (N, D) 배열 -> window 별 16바이트 해시의 (N,) object 배열."""
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    a = np.ascontiguousarray(a, dtype=np.float32)
    rnd = _STATE["hash_round"]
    if rnd is not None:
        a = np.round(a, int(rnd)).astype(np.float32)
    n = a.shape[0]
    flat = a.reshape(n, -1)
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = hashlib.blake2b(flat[i].tobytes(), digest_size=16).digest()
    return out


def _labels_hash(y) -> str:
    y = np.ascontiguousarray(np.asarray(y).astype(np.int8))
    h = hashlib.blake2b(y.tobytes(), digest_size=12).hexdigest()
    return f"{len(y)}_{h}"


# ----------------------------------------------------------------------
# 마스크 캐시 입출력
# ----------------------------------------------------------------------
def _mask_path() -> Path:
    return Path(_STATE["mask_path"])


def _load_masks() -> None:
    p = _mask_path()
    if not p.exists():
        return
    try:
        z = np.load(p, allow_pickle=True)
        meta = json.loads(str(z["__meta__"]))
        for k in z.files:
            if k == "__meta__":
                continue
            _MASKS[k] = z[k].astype(bool)
            _MASK_META[k] = meta.get(k, {})
            _LEN_INDEX.setdefault(int(_MASKS[k].size), []).append(k)
        _log(f"마스크 캐시 로드: {len(_MASKS)}개 ({p.name})")
    except Exception as e:
        _log(f"마스크 캐시 로드 실패(무시하고 새로 만듦): {e!r}")


def _save_masks() -> None:
    p = _mask_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v.astype(bool) for k, v in _MASKS.items()}
        payload["__meta__"] = np.array(json.dumps(_MASK_META, ensure_ascii=False))
        np.savez_compressed(p, **payload)
    except Exception as e:
        _log(f"마스크 저장 실패: {e!r}")


def _log(msg: str) -> None:
    # cp949 콘솔에서 죽지 않도록 ASCII + 한글만 사용한다
    print(f"[leakfix] {msg}", flush=True)


# ----------------------------------------------------------------------
# D dict 에서 필요한 배열 찾기
# ----------------------------------------------------------------------
def _pick(D, names):
    for n in names:
        if isinstance(D, dict) and n in D and D[n] is not None:
            return D[n]
        if hasattr(D, n):
            v = getattr(D, n)
            if v is not None:
                return v
    return None


def _register_mask_from_D(D, pair_hint: str = "?") -> str | None:
    """make_pool 에 넘어온 D 로부터 pair 마스크를 만들어 등록. y_hash 반환."""
    ev_x = _pick(D, ["eval_x", "ev_x", "test_x", "eval_windows"])
    ev_y = _pick(D, ["eval_y", "ev_y", "test_y", "eval_labels"])
    if ev_x is None or ev_y is None:
        return None

    yh = _labels_hash(ev_y)
    if yh in _MASKS:
        return yh

    clean = _pick(D, ["clean", "pool_clean", "train_clean"])
    anom = _pick(D, ["anom", "pool_anom", "train_anom"])
    parts = [p for p in (clean, anom) if p is not None and len(p) > 0]
    if not parts:
        return None

    try:
        pool_h = set()
        for p in parts:
            pool_h.update(_win_hashes(p).tolist())
        ev_h = _win_hashes(ev_x)
        inpool = np.array([h in pool_h for h in ev_h], dtype=bool)
    except Exception as e:
        _log(f"마스크 계산 실패 pair={pair_hint}: {e!r}")
        return None

    keep = ~inpool
    y = np.asarray(ev_y).astype(int)
    meta = {
        "pair": pair_hint,
        "n_eval": int(len(y)),
        "n_removed": int(inpool.sum()),
        "frac_removed": float(inpool.mean()),
        "n_clean": int(len(clean) if clean is not None else 0),
        "n_anom": int(len(anom) if anom is not None else 0),
        "pos_all": int(y.sum()),
        "pos_keep": int(y[keep].sum()),
        "neg_keep": int((1 - y[keep]).sum()),
        "hash_round": _STATE["hash_round"],
    }

    with _LOCK:
        _MASKS[yh] = keep
        _MASK_META[yh] = meta
        _LEN_INDEX.setdefault(int(len(y)), []).append(yh)
        _save_masks()

    _log(
        f"마스크 등록 pair={pair_hint} n_eval={meta['n_eval']} "
        f"제거={meta['n_removed']}({meta['frac_removed']*100:.1f}%) "
        f"잔여 pos/neg={meta['pos_keep']}/{meta['neg_keep']}"
    )
    if meta["pos_keep"] == 0 or meta["neg_keep"] == 0:
        _log(f"  경고: pair={pair_hint} 는 pool 제거 후 단일 클래스. AUROC 계산 불가(NaN 기록)")
    return yh


def _lookup_mask(y) -> tuple[np.ndarray | None, str]:
    yh = _labels_hash(y)
    m = _MASKS.get(yh)
    if m is not None:
        return m, "exact"
    # 보조: 길이가 같고 마스크가 하나뿐이면 그것을 쓴다(동일 target 을 공유하는 pair 대비)
    cands = _LEN_INDEX.get(int(len(y)), [])
    uniq = {c for c in cands}
    if len(uniq) == 1:
        return _MASKS[next(iter(uniq))], "by_length"
    return None, "none"


# ----------------------------------------------------------------------
# 지표 계산
# ----------------------------------------------------------------------
def _metrics_on_subset(scores, y, keep, base_result: dict) -> dict:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    s = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    out: dict = {}

    if keep is None or keep.size != y.size:
        out.update(
            auroc_nopool=np.nan, aupr_nopool=np.nan,
            n_eval_nopool=int(y.size), pos_nopool=int(y.sum()),
            neg_nopool=int((1 - y).sum()), frac_removed=np.nan,
            mask_source="none",
        )
        return out

    ss, yy = s[keep], y[keep]
    pos, neg = int(yy.sum()), int((1 - yy).sum())
    out["n_eval_nopool"] = int(yy.size)
    out["pos_nopool"] = pos
    out["neg_nopool"] = neg
    out["frac_removed"] = float(1.0 - keep.mean())

    if pos == 0 or neg == 0:
        out["auroc_nopool"] = np.nan
        out["aupr_nopool"] = np.nan
    else:
        try:
            out["auroc_nopool"] = float(roc_auc_score(yy, ss))
            out["aupr_nopool"] = float(average_precision_score(yy, ss))
        except Exception:
            out["auroc_nopool"] = np.nan
            out["aupr_nopool"] = np.nan

    # 기존 결과에 있는 FAR 임계값을 그대로 적용해서 F1 재계산
    for tag in ("0.05", "0.01", "0.001"):
        thr = base_result.get(f"thr_far{tag}")
        if thr is None or pos == 0:
            continue
        try:
            pred = (ss > float(thr)).astype(int)
            tp = int(((pred == 1) & (yy == 1)).sum())
            fp = int(((pred == 1) & (yy == 0)).sum())
            fn = int(((pred == 0) & (yy == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            out[f"f1_far{tag}_nopool"] = float(f1_score(yy, pred, zero_division=0))
            out[f"prec_far{tag}_nopool"] = float(prec)
            out[f"rec_far{tag}_nopool"] = float(rec)
        except Exception:
            pass
    return out


# ----------------------------------------------------------------------
# 사이드카 CSV
# ----------------------------------------------------------------------
def _sidecar_append(row: dict) -> None:
    p = Path(_STATE["sidecar"])
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.exists()
        with open(p, "a", encoding="utf-8", newline="") as f:
            if new:
                f.write(",".join(SIDECAR_COLS) + "\n")
            vals = []
            for c in SIDECAR_COLS:
                v = row.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v!r}" if np.isnan(v) else repr(float(v)))
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")
    except Exception as e:
        _log(f"사이드카 기록 실패: {e!r}")


# ----------------------------------------------------------------------
# 인자 추출 (evaluate 시그니처가 달라도 견디도록)
# ----------------------------------------------------------------------
def _extract_scores_labels(args, kwargs):
    import inspect

    scores = labels = None
    try:
        sig = inspect.signature(_ORIG_EVALUATE)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        d = dict(bound.arguments)
        for k, v in d.items():
            lk = k.lower()
            if labels is None and ("label" in lk or lk in ("y", "y_true", "gt")):
                labels = v
            elif scores is None and "score" in lk and "val" not in lk:
                scores = v
        if scores is None or labels is None:
            pos = [v for v in args]
            if len(pos) >= 2:
                scores = scores if scores is not None else pos[0]
                labels = labels if labels is not None else pos[1]
    except Exception:
        if len(args) >= 2:
            scores, labels = args[0], args[1]

    try:
        s = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(labels).ravel()
        if s.size != y.size:
            return None, None
        u = set(np.unique(y).tolist())
        if not u.issubset({0, 1, 0.0, 1.0, True, False}):
            return None, None
        return s, y.astype(int)
    except Exception:
        return None, None


# ----------------------------------------------------------------------
# 점수 배열 덤프 (한 번 돌리면 다시 학습할 필요가 없도록)
# ----------------------------------------------------------------------
def _dump_scores(pair, y, s_arr) -> None:
    if not _STATE.get("dump_scores"):
        return
    try:
        d = Path(_STATE["outdir"]) / "scores_dump"
        d.mkdir(parents=True, exist_ok=True)
        n = _CALL_IDX["n"]
        np.savez_compressed(
            d / f"call{n:05d}_{pair}.npz",
            y=np.asarray(y, dtype=np.int8),
            s=np.asarray(s_arr, dtype=np.float32),
            pair=np.array(pair),
        )
    except Exception as e:
        _log(f"score dump 실패: {e!r}")


# ----------------------------------------------------------------------
# 패치
# ----------------------------------------------------------------------
def _rebind_everywhere(name: str, old, new) -> int:
    """이미 'from x import name' 으로 old 를 붙들고 있는 모듈들을 갱신."""
    n = 0
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            if getattr(mod, name, None) is old:
                setattr(mod, name, new)
                n += 1
        except Exception:
            pass
    return n


def install(outdir: str, hash_round=None, abort_after_masks: bool = False,
            dump_scores: bool = True) -> None:
    """
    반드시 exp1_benchmark / exp3_final / 러너를 import 하기 전에 호출할 것.
    """
    global _ORIG_EVALUATE, _ORIG_MAKE_POOL, _ORIG_PREPARE

    if _STATE["installed"]:
        _log("이미 설치됨")
        return

    outdir = str(outdir)
    _STATE["outdir"] = outdir
    _STATE["hash_round"] = hash_round
    _STATE["abort_after_masks"] = bool(abort_after_masks)
    _STATE["dump_scores"] = bool(dump_scores)
    # 마스크는 실행 폴더가 달라도 공유되어야 하므로 work/ 한 곳(paths.MASKS)에 둔다
    _STATE["mask_path"] = str(paths.MASKS)
    _STATE["sidecar"] = str(Path(outdir) / "leakfix_rows.csv")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    _load_masks()

    # --- exp1_metrics.evaluate -----------------------------------------
    import exp1_metrics as M

    _ORIG_EVALUATE = M.evaluate

    def evaluate_patched(*args, **kwargs):
        res = _ORIG_EVALUATE(*args, **kwargs)
        _STATE["n_eval_calls"] += 1
        if not isinstance(res, dict):
            return res
        try:
            s, y = _extract_scores_labels(args, kwargs)
            if s is None:
                return res
            keep, src = _lookup_mask(y)
            if keep is None:
                _STATE["n_mask_miss"] += 1
            else:
                _STATE["n_mask_hit"] += 1
            extra = _metrics_on_subset(s, y, keep, res)
            extra["mask_source"] = src
            res.update(extra)

            _CALL_IDX["n"] += 1
            _sidecar_append({
                "call_idx": _CALL_IDX["n"],
                "pair": _CURRENT_PAIR["name"],
                "y_hash": _labels_hash(y),
                "n_eval": int(y.size),
                "n_eval_nopool": extra.get("n_eval_nopool", ""),
                "frac_removed": extra.get("frac_removed", float("nan")),
                "pos_all": int(y.sum()),
                "pos_nopool": extra.get("pos_nopool", ""),
                "auroc": res.get("auroc", float("nan")),
                "auroc_nopool": extra.get("auroc_nopool", float("nan")),
                "aupr": res.get("aupr", float("nan")),
                "aupr_nopool": extra.get("aupr_nopool", float("nan")),
                "f1_far0.01": res.get("f1_far0.01", float("nan")),
                "f1_far0.01_nopool": extra.get("f1_far0.01_nopool", float("nan")),
                "mask_source": src,
            })
        except Exception:
            _log("evaluate 패치 내부 오류(원본 결과는 그대로 반환):")
            traceback.print_exc()
        return res

    M.evaluate = evaluate_patched
    n = _rebind_everywhere("evaluate", _ORIG_EVALUATE, evaluate_patched)
    _log(f"exp1_metrics.evaluate 패치 완료 (기존 참조 {n}건 재바인딩)")

    # --- exp2_robust.evaluate_all (실제 채점 경로) -----------------------
    # master 의 컬럼 구성(f1_far*, f1_primary, aupr_lift)이 이 함수의 반환값과
    # 일치한다. exp1_metrics.evaluate 만 패치하면 한 번도 호출되지 않는다.
    try:
        import exp2_robust as E2

        _ORIG_EVAL_ALL[0] = E2.evaluate_all

        def evaluate_all_patched(te_s, te_y, va_s, *a, **kw):
            res = _ORIG_EVAL_ALL[0](te_s, te_y, va_s, *a, **kw)
            _STATE["n_eval_calls"] += 1
            if not isinstance(res, dict):
                return res
            try:
                s_arr = np.asarray(te_s, dtype=float).ravel()
                y = np.asarray(te_y).astype(int).ravel()
                keep, src = _lookup_mask(y)
                if keep is None:
                    _STATE["n_mask_miss"] += 1
                else:
                    _STATE["n_mask_hit"] += 1
                extra = _metrics_on_subset(s_arr, y, keep, res)
                extra["mask_source"] = src
                res.update(extra)

                _CALL_IDX["n"] += 1
                _dump_scores(_CURRENT_PAIR["name"], y, s_arr)
                _sidecar_append({
                    "call_idx": _CALL_IDX["n"],
                    "pair": _CURRENT_PAIR["name"],
                    "y_hash": _labels_hash(y),
                    "n_eval": int(y.size),
                    "n_eval_nopool": extra.get("n_eval_nopool", ""),
                    "frac_removed": extra.get("frac_removed", float("nan")),
                    "pos_all": int(y.sum()),
                    "pos_nopool": extra.get("pos_nopool", ""),
                    "auroc": res.get("auroc", float("nan")),
                    "auroc_nopool": extra.get("auroc_nopool", float("nan")),
                    "aupr": res.get("aupr", float("nan")),
                    "aupr_nopool": extra.get("aupr_nopool", float("nan")),
                    "f1_far0.01": res.get("f1_far0.01", float("nan")),
                    "f1_far0.01_nopool": extra.get("f1_far0.01_nopool", float("nan")),
                    "mask_source": src,
                })
            except Exception:
                _log("evaluate_all 패치 내부 오류(원본 결과는 그대로 반환):")
                traceback.print_exc()
            return res

        E2.evaluate_all = evaluate_all_patched
        n2 = _rebind_everywhere("evaluate_all", _ORIG_EVAL_ALL[0], evaluate_all_patched)
        _log(f"exp2_robust.evaluate_all 패치 완료 (재바인딩 {n2}건)")
    except Exception as e:
        _log(f"evaluate_all 패치 실패: {e!r}  (exp1_metrics.evaluate 경로만 동작)")

    # --- exp9_unified.prepare (pair 이름 추적용) -------------------------
    try:
        import exp9_unified as E9

        _ORIG_PREPARE = E9.prepare

        def prepare_patched(pair, *a, **kw):
            _CURRENT_PAIR["name"] = str(pair)
            return _ORIG_PREPARE(pair, *a, **kw)

        E9.prepare = prepare_patched
        m = _rebind_everywhere("prepare", _ORIG_PREPARE, prepare_patched)
        _log(f"exp9_unified.prepare 패치 완료 (재바인딩 {m}건)")
    except Exception as e:
        _log(f"prepare 패치 생략: {e!r} (pair 이름만 '?' 로 기록됨)")

    # --- exp3_final.make_pool (마스크 생성 지점) --------------------------
    import exp3_final as E3

    _ORIG_MAKE_POOL = E3.make_pool

    def make_pool_patched(D, *a, **kw):
        try:
            _register_mask_from_D(D, _CURRENT_PAIR["name"])
        except Exception:
            _log("make_pool 패치 내부 오류:")
            traceback.print_exc()
        if _STATE["abort_after_masks"]:
            raise MaskProbeDone(
                f"[leakfix] 마스크 전용 모드: pair={_CURRENT_PAIR['name']} 등록 후 중단"
            )
        return _ORIG_MAKE_POOL(D, *a, **kw)

    E3.make_pool = make_pool_patched
    k = _rebind_everywhere("make_pool", _ORIG_MAKE_POOL, make_pool_patched)
    _log(f"exp3_final.make_pool 패치 완료 (재바인딩 {k}건)")

    _STATE["installed"] = True
    _log(f"설치 완료. 마스크={_mask_path()}  사이드카={_STATE['sidecar']}")


def summary() -> dict:
    return {
        "n_evaluate_calls": _STATE["n_eval_calls"],
        "n_mask_hit": _STATE["n_mask_hit"],
        "n_mask_miss": _STATE["n_mask_miss"],
        "n_masks": len(_MASKS),
        "masks": _MASK_META,
    }


def dump_summary(path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary(), ensure_ascii=False, indent=2), encoding="utf-8")
    s = summary()
    _log(
        f"요약: evaluate 호출 {s['n_evaluate_calls']}회, "
        f"마스크 적용 {s['n_mask_hit']}회 / 미적용 {s['n_mask_miss']}회, "
        f"등록 마스크 {s['n_masks']}개"
    )
    if s["n_mask_miss"]:
        _log(
            "미적용이 있습니다. S2 를 먼저 돌려 마스크를 만든 뒤 S1/S4/S5 를 "
            "돌리면 0 이 됩니다(마스크는 디스크에 캐시됨)."
        )


def masks_table():
    """pair 별 마스크 요약을 DataFrame 으로."""
    import pandas as pd

    if not _MASK_META:
        _load_masks()
    rows = []
    for yh, m in _MASK_META.items():
        r = dict(m)
        r["y_hash"] = yh
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 단독 실행 시에는 캐시된 마스크 요약만 보여준다
    _STATE["mask_path"] = str(paths.MASKS)
    _load_masks()
    df = masks_table()
    if len(df):
        cols = [c for c in ["pair", "n_eval", "n_removed", "frac_removed",
                            "pos_all", "pos_keep", "neg_keep"] if c in df.columns]
        print(df[cols].to_string(index=False))
    else:
        print("등록된 마스크가 없습니다. final_20_run.py 를 먼저 실행하세요.")
