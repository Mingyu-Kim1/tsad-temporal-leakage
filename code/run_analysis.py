#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_analysis.py
남은 분석 두 건을 한 번에 돌린다.

    1) window 단위 점수 npz 를 windows.csv 로 변환
    2) a4  순위 구간별 분석  (AUROC 는 유의한데 AUPR 은 유의하지 않은 이유 확인)
    3) a3  E0 부분표본 기울기 비교 (오염 수준이 3개 이상일 때만 자동 실행)

같은 폴더에 dump_to_windows.py, a4_ranking_region.py, a3_e0_matched_subsample.py 가
있어야 한다.

[실행]
    python run_analysis.py --root jobs_out
    python run_analysis.py --root jobs_out --score-key score --label-key y \
        --start-key start --mask-key keep_temporal
    python run_analysis.py --windows windows.csv          # 변환을 이미 해 둔 경우

[결과]
    out\\windows.csv         변환된 window 단위 점수
    out\\a4\\a4_summary.md    순위 구간별 결과
    out\\a3\\a3_summary.md    E0 부분표본 결과 (실행된 경우)
    out\\RESULT.md           위 결과를 한 파일로 모은 요약
"""
import argparse, os, subprocess, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, args, title):
    cmd = [sys.executable, os.path.join(HERE, script)] + args
    print("\n" + "=" * 68)
    print(f"[{title}]")
    print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.rstrip())
    return r.returncode == 0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="jobs_out", help="npz 가 들어 있는 상위 폴더")
    ap.add_argument("--windows", default=None, help="이미 만든 windows.csv 가 있으면 지정")
    ap.add_argument("--out", default="out")
    ap.add_argument("--score-key", default=None)
    ap.add_argument("--label-key", default=None)
    ap.add_argument("--start-key", default=None)
    ap.add_argument("--mask-key", default=None)
    ap.add_argument("--masks", default="leakfix_masks.npz")
    ap.add_argument("--baseline", default="source_only")
    ap.add_argument("--method", default="adapter_latent")
    ap.add_argument("--n-boot", default="200")
    ap.add_argument("--seeds", nargs="*", default=["0", "1", "2"],
                    help="변환에 사용할 seed. 기본 0 1 2 (파일 크기 조절용)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logs = []

    # ---------------- 1. 변환 ----------------
    win = a.windows
    if not win:
        win = os.path.join(a.out, "windows.csv")
        args = ["--root", a.root, "--out", win, "--method", "all", "--masks", a.masks]
        for k, v in (("--score-key", a.score_key), ("--label-key", a.label_key),
                     ("--start-key", a.start_key), ("--mask-key", a.mask_key)):
            if v:
                args += [k, v]
        if a.seeds and a.seeds != ["all"]:
            args += ["--seeds", *a.seeds]
        ok, out = run("dump_to_windows.py", args, "1. npz -> windows.csv 변환")
        logs.append(("변환", out))
        if not ok:
            print("\n[중단] 변환에 실패했습니다. 아래를 먼저 실행해 key 이름을 확인하세요.")
            print(f"  python dump_to_windows.py --root {a.root} --inspect")
            sys.exit(1)
    if not os.path.exists(win):
        sys.exit(f"[오류] {win} 이 없습니다.")

    df = pd.read_csv(win)
    rhos = sorted(pd.to_numeric(df.get("rho", pd.Series([0])), errors="coerce").dropna().unique())
    methods = sorted(df["method"].unique()) if "method" in df.columns else []
    print(f"\n[확인] windows.csv  {len(df):,}행 | pair {df['pair'].nunique()}개 | "
          f"rho {rhos} | method {methods}")

    # ---------------- 2. a4 ----------------
    if a.baseline in methods and a.method in methods:
        ok, out = run("a4_ranking_region.py",
                      ["--windows", win, "--baseline", a.baseline, "--method", a.method,
                       "--rho", "0", "--out", os.path.join(a.out, "a4")],
                      "2. 순위 구간별 분석 (a4)")
        logs.append(("a4", out))
    else:
        msg = (f"[건너뜀] a4 실행에 필요한 method 가 없습니다. "
               f"필요: {a.baseline}, {a.method} / 실제: {methods}")
        print("\n" + msg); logs.append(("a4", msg))

    # ---------------- 3. a3 ----------------
    if len(rhos) >= 3:
        ok, out = run("a3_e0_matched_subsample.py",
                      ["--windows", win, "--n-boot", a.n_boot,
                       "--method", a.baseline, "--out", os.path.join(a.out, "a3")],
                      "3. E0 부분표본 기울기 비교 (a3)")
        logs.append(("a3", out))
    else:
        msg = (f"[건너뜀] a3 는 오염 수준이 3개 이상이어야 합니다. 현재 rho = {rhos}.\n"
               "        S2 단계를 점수 저장을 켠 채로 다시 돌린 뒤 같은 명령을 실행하세요.")
        print("\n" + msg); logs.append(("a3", msg))

    # ---------------- 4. 요약 ----------------
    parts = ["# 분석 결과\n"]
    for key, title in (("a4", "순위 구간별 분석"), ("a3", "E0 부분표본 기울기 비교")):
        md = os.path.join(a.out, key, f"{key}_summary.md")
        parts.append(f"\n## {title}\n")
        parts.append(open(md, encoding="utf-8").read().split("\n", 1)[-1]
                     if os.path.exists(md) else "실행되지 않았습니다.\n")
    parts.append("\n## 실행 로그\n")
    for name, out in logs:
        parts.append(f"\n### {name}\n```\n{out.strip()[-1500:]}\n```\n")
    with open(os.path.join(a.out, "RESULT.md"), "w", encoding="utf-8") as f:
        f.write("".join(parts))

    print("\n" + "=" * 68)
    print(f"요약 파일: {os.path.abspath(os.path.join(a.out, 'RESULT.md'))}")
    print("이 파일과 out 폴더의 csv 를 보내 주시면 원고에 반영합니다.")


if __name__ == "__main__":
    main()
