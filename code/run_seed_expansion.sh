#!/usr/bin/env bash
# ==========================================================================
# run_seed_expansion.sh
# --------------------------------------------------------------------------
# 목적 : 논문 4.5절의 "최대 편차 5.7x10^-2" 가 seed 를 늘리면 해소되는지 확인
#
# 절차 : 0단계 진단 -> 1단계 seed 5개 추가 -> 판정 -> 필요하면 2단계 5개 더
#
# 전제 : run_all_xjtu_fixed.py 가 (stage, pair, seed, method, contam, alpha)
#        를 key 로 이어달리기(resume)를 지원하므로, seed 목록만 늘려서 다시
#        실행하면 이미 끝난 seed 0-4 는 건너뛰고 새 seed 만 계산한다.
# ==========================================================================
set -euo pipefail

RESULTS="results_rescored.csv"          # 기존 결과 (재채점본)
OUTDIR="run_all_out_seed"               # 새 실행 출력 폴더 (기존과 분리)
PY="${PY:-python}"

# 채점 가능한 14개 pair 만 돌린다. 제외된 9개는 어차피 채점되지 않는다.
PAIRS="Boiler_1to2 Boiler_1to3 Boiler_2to1 Boiler_2to3 Boiler_3to1 Boiler_3to2 \
MSL_G-6toE-1 MSL_G-6toE-13 MSL_P-10toP-11 MSL_P-11toP-10 \
SMD_1-1to1-2 SMD_1-2to1-1 SMD_2-1to2-2 SMD_3-1to3-2"

echo "=============================================================="
echo "0단계  재학습 없이 진단"
echo "=============================================================="
$PY seed_stability.py diagnose --results "$RESULTS" --out seed_diag

echo
echo "=============================================================="
echo "1단계  seed 5개 추가 (0-4 -> 0-9).  S1 stage 만 돌린다."
echo "        S1 = 무적응 대비 적응 이득이 계산되는 주 결과 stage"
echo "=============================================================="
$PY run_all_xjtu_fixed.py \
    --stages S1 \
    --pairs $PAIRS \
    --seeds_main 0 1 2 3 4 5 6 7 8 9 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --outdir "$OUTDIR"

echo
echo "=============================================================="
echo "1단계 판정"
echo "=============================================================="
$PY seed_stability.py decide \
    --results "$RESULTS" \
    --new "$OUTDIR/results.csv" \
    --split 5 \
    --out seed_decide_s10

echo
echo "판정 결과를 확인한다."
echo "  · '해결됨' 이면 여기서 멈춘다."
echo "  · '아직 부족' 이면 아래 2단계를 이어서 실행한다."
echo
read -r -p "2단계(seed 10-14 추가)를 실행할까요? [y/N] " ans
if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
  echo
  echo "=============================================================="
  echo "2단계  seed 5개 추가 (0-9 -> 0-14)"
  echo "=============================================================="
  $PY run_all_xjtu_fixed.py \
      --stages S1 \
      --pairs $PAIRS \
      --seeds_main 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
      --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
      --outdir "$OUTDIR"

  $PY seed_stability.py decide \
      --results "$RESULTS" \
      --new "$OUTDIR/results.csv" \
      --split 5 \
      --out seed_decide_s15
fi

echo
echo "=============================================================="
echo "완료. 확인할 파일"
echo "  seed_diag/D1_cell_variability.csv   편차가 큰 cell 목록"
echo "  seed_diag/D4_pair_sem_plan.csv      pair 별 표준오차와 필요한 seed 수"
echo "  seed_diag/D5_aggregate_plan.csv     pair 평균 기준 필요한 seed 수"
echo "  seed_decide_s10/E1_headline_before_after.csv   주 수치 변화"
echo "  seed_decide_s10/E2_cell_sem_change.csv         cell 별 표준오차 변화"
echo "=============================================================="
