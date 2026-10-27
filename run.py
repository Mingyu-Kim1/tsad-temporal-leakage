#!/usr/bin/env python3
"""run.py — code/ 의 스크립트를 work/ 를 실행 폴더로 삼아 실행한다.

    python run.py final_95_final_analysis.py --results results_rescored.csv
    python run.py final_20_run.py --plan
    (논문 수치 검증은 저장소 루트에서 python tools/verify_paper_numbers.py)

Windows 에서 torch 를 불러올 때 'OMP: Error #15 (libiomp5md.dll)' 가 나면
    python run.py --kmp <script> ...
처럼 --kmp 를 붙인다 (KMP_DUPLICATE_LIB_OK=TRUE 설정).
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
import paths  # noqa: E402

args = sys.argv[1:]
if args and args[0] == "--kmp":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    args = args[1:]
if not args:
    print(__doc__)
    sys.exit(0)

script = (CODE / args[0]).resolve()
if not script.exists():
    sys.exit(f"스크립트를 찾을 수 없습니다: {script}")

paths.WORK.mkdir(parents=True, exist_ok=True)
os.chdir(paths.WORK)
sys.argv = [str(script)] + args[1:]
runpy.run_path(str(script), run_name="__main__")
