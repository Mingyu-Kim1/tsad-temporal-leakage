#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_to_windows.py
실행 중 저장된 window 단위 점수 npz 를 a3_e0_matched_subsample.py 의 입력
(windows.csv) 으로 변환한다.

a3 가 요구하는 열
    pair, rho, start, label, score, in_e2
        pair   transfer pair 이름
        rho    주입 수준(contam). 0 만 있어도 무방하나 기울기 비교에는 여러 값이 필요하다
        start  window 시작 index (시간 순서로 정렬 가능한 정수)
        label  window 이상 여부 1/0
        score  anomaly score (클수록 이상)
        in_e2  E2(g) 에 남는 window 면 1

[1단계] 먼저 npz 안에 어떤 배열이 있는지 본다
    python dump_to_windows.py --inspect --root jobs_out\\final

[2단계] 확인한 key 이름을 지정해 변환한다
    python dump_to_windows.py --root jobs_out\\final --out windows.csv \\
        --score-key score --label-key y --start-key start --mask-key keep_temporal

key 이름을 주지 않으면 흔한 후보를 스스로 찾는다. 찾지 못한 열은 다음 규칙으로 채운다.
    start  없으면 배열 순서를 그대로 시작 index 로 본다
    in_e2  없으면 leakfix 마스크 파일(--masks)에서 pair 별 mask 를 찾아 사용한다
"""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd

CAND = {
    "score": ["score", "scores", "s", "anomaly_score", "err", "error", "recon_err"],
    "label": ["label", "labels", "y", "y_true", "y_eval", "target"],
    "start": ["start", "starts", "start_idx", "idx", "index", "t", "t0"],
    "mask":  ["in_e2", "keep_temporal", "mask_temporal", "keep", "mask", "nopool_temporal"],
}


def parse_meta(path):
    """파일 이름에서 pair, method, seed, contam 을 뽑는다.

    pair 이름에 하이픈이 들어가므로(SMD_1-2to1-1) method 를 먼저 찾아
    그 앞부분을 pair 로 본다.
    """
    name = os.path.basename(path).replace(".npz", "")
    out = {"pair": None, "method": None, "seed": None, "rho": None}
    stem = re.sub(r"^(scores?|dump|window|win)[_\-]", "", name)

    m = re.search(r"(adapter_latent|adapter_input|adapter_isca|adapter_output|"
                  r"full_ft|source_only)", stem)
    if m:
        out["method"] = m.group(1)
        head = stem[:m.start()].rstrip("_-")
    else:                                   # method 표기가 없으면 seed/contam 앞까지
        head = re.split(r"_(?:seed|contam|rho)\b", stem)[0].rstrip("_-")
    if head:
        out["pair"] = head

    m = re.search(r"seed[_\-]?(\d+)", name)
    if m:
        out["seed"] = int(m.group(1))
    m = re.search(r"(?:contam|rho)[_\-]?([0-9p\.]+)", name)
    if m:
        try:
            out["rho"] = float(m.group(1).rstrip(".").replace("p", "."))
        except ValueError:
            pass
    return out, name


def pick(keys, kind, given):
    if given:
        return given if given in keys else None
    low = {k.lower(): k for k in keys}
    for c in CAND[kind]:
        if c in low:
            return low[c]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="jobs_out")
    ap.add_argument("--out", default="windows.csv")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--score-key", default=None)
    ap.add_argument("--label-key", default=None)
    ap.add_argument("--start-key", default=None)
    ap.add_argument("--mask-key", default=None)
    ap.add_argument("--masks", default="leakfix_masks.npz")
    ap.add_argument("--method", default="source_only",
                    help="변환할 method. all 이면 전부")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="변환할 seed 목록. 지정하지 않으면 전부(파일이 커질 수 있음)")
    ap.add_argument("--pairs", nargs="*", default=None, help="변환할 pair 목록")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="pair·method·seed 조합당 최대 행 수(0이면 제한 없음)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.root, "**", "*.npz"), recursive=True))
    files = [f for f in files if "leakfix_masks" not in os.path.basename(f)]
    if not files:
        sys.exit(f"[없음] {a.root} 아래에서 npz 를 찾지 못했습니다.")
    print(f"[찾음] npz {len(files)}개")

    if a.inspect:
        for f in files[:5]:
            z = np.load(f, allow_pickle=True)
            meta, name = parse_meta(f)
            print(f"\n--- {f}")
            print(f"    파일명 해석: {meta}")
            for k in list(z.keys())[:15]:
                arr = z[k]
                print(f"    {k:24s} shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")
        print("\n위 key 이름을 --score-key 등으로 지정해 다시 실행하세요.")
        return

    masks = {}
    if os.path.exists(a.masks):
        try:
            mz = np.load(a.masks, allow_pickle=True)
            masks = {k: mz[k] for k in mz.keys()}
            print(f"[마스크] {a.masks} 에서 {len(masks)}개 로드")
        except Exception as e:
            print(f"[마스크] 읽기 실패: {e}")

    rows, skipped = [], 0
    for f in files:
        meta, name = parse_meta(f)
        if a.method != "all" and meta["method"] and meta["method"] != a.method:
            continue
        if a.seeds is not None and meta["seed"] is not None and meta["seed"] not in a.seeds:
            continue
        if a.pairs and meta["pair"] not in a.pairs:
            continue
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            skipped += 1; continue
        keys = list(z.keys())
        ks = pick(keys, "score", a.score_key)
        kl = pick(keys, "label", a.label_key)
        if ks is None or kl is None:
            skipped += 1; continue
        score = np.asarray(z[ks]).ravel()
        label = np.asarray(z[kl]).ravel().astype(int)
        n = min(len(score), len(label))
        kt = pick(keys, "start", a.start_key)
        start = (np.asarray(z[kt]).ravel()[:n] if kt else np.arange(n))
        km = pick(keys, "mask", a.mask_key)
        if km:
            in_e2 = np.asarray(z[km]).ravel()[:n].astype(int)
        else:
            mk = masks.get(meta["pair"])
            in_e2 = (np.asarray(mk).ravel()[:n].astype(int) if mk is not None
                     else np.ones(n, int))
        if a.max_rows and n > a.max_rows:
            sel = np.linspace(0, n - 1, a.max_rows).astype(int)
            score, label, start, in_e2 = score[sel], label[sel], start[sel], in_e2[sel]
            n = a.max_rows
        rows.append(pd.DataFrame({
            "pair": meta["pair"] or name, "method": meta["method"] or "unknown",
            "seed": meta["seed"], "rho": meta["rho"] if meta["rho"] is not None else 0.0,
            "start": start[:n], "label": label[:n], "score": score[:n], "in_e2": in_e2}))
    if not rows:
        sys.exit("[실패] 변환할 배열을 찾지 못했습니다. --inspect 로 key 이름을 확인하세요.")
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(a.out, index=False)
    print(f"[저장] {a.out}  {len(df):,}행, pair {df.pair.nunique()}개, rho {sorted(df.rho.unique())}")
    print(f"       in_e2 비율 {df.in_e2.mean():.3f}, 이상 비율 {df.label.mean():.4f}, 건너뛴 파일 {skipped}개")
    if df.rho.nunique() < 3:
        print("\n[주의] rho 값이 3개 미만이라 기울기 비교가 어렵습니다. "
              "오염 수준별 dump 가 있는지 확인하세요(S2 단계).")
    print("\n다음 단계:")
    print(f"  python a3_e0_matched_subsample.py --windows {a.out} --n-boot 200 --out results_a3")


if __name__ == "__main__":
    main()
