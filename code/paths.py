"""paths.py — 저장소 안의 모든 경로와 seed 규약을 한 곳에 모은다.

폴더 규약
    <ROOT>/code/   이 파일과 모든 스크립트
    <ROOT>/work/   실행 폴더. 스크립트의 상대경로(data/…, jobs_out/…, results_rescored.csv …)는
                   모두 여기를 기준으로 풀린다. run.py 가 실행 전에 이 폴더로 이동한다.
    <ROOT>/work/data/   원본 데이터 (저장소에 포함하지 않음. 원본 폴더의 data/ 를 복사)

다른 위치를 쓰려면 환경변수 AD_WORK 로 실행 폴더를 바꾼다.
"""
from __future__ import annotations

import os
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
WORK = Path(os.environ.get("AD_WORK", ROOT / "work")).resolve()

DATA = WORK / "data"
JOBS = WORK / "jobs_out" / "final"          # final_20_run.py 의 job 출력
MASKS = WORK / "leakfix_masks.npz"          # final_10_leakfix.py 가 만드는 누수 마스크 캐시

# ---------------------------------------------------------------------------
# seed 규약 (논문 v3/v4 기준)
#   S1 (주 결과: Table 7 이득·8·9·12–14)          : seed 0–14
#   S2 / S4 / S5 / SW (오염 곡선·α·외부 비교·sweep) : seed 0–2
# ---------------------------------------------------------------------------
SEEDS_MAIN = list(range(15))
SEEDS_HEAVY = [0, 1, 2]
