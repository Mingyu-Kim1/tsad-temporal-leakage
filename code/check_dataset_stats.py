"""
check_dataset_stats.py
======================
논문 Section 4.1(Table)에 적을 데이터셋 통계를, 공개 논문의 보고값이 아니라
**본 연구가 실제로 로드한 데이터**에서 직접 산출한다.

왜 필요한가
-----------
Boiler 데이터는 선행 연구들이 274 dim / anomaly ratio 15 %(point 단위)로
보고한다. 그런데 본 연구의 T1 table 은 Boiler_1to2 0.0341, 2to3 0.1067,
3to1 0.1112 로 서로 다르다. 이는 (i) window 단위 라벨 vs point 단위 라벨,
(ii) train/test split 중 어느 쪽인지, (iii) 전처리 차이 때문일 수 있다.
reviewer 는 반드시 이 불일치를 묻는다. 따라서 **본 연구 기준 값**을 표에 적고,
공개 보고값과 다르면 각주로 이유를 설명해야 한다.

실행
    set PYTHONUTF8=1
    python check_dataset_stats.py
출력
    dataset_stats.csv   (Section 4.1 Table 에 그대로 옮길 수 있는 형태)
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd


def main() -> int:
    try:
        import exp1_benchmark as B
        import exp9_unified as E9
    except Exception as e:
        print(f"[오류] 모듈 import 실패: {e!r}")
        return 1

    # MSL 은 all_pairs() 안에서 런타임 등록되므로 먼저 등록해 준다
    try:
        import run_all_xjtu_fixed as RA

        class _Stub:
            msl_max = 4

        RA.configure_output("stats_tmp")
        RA.all_pairs(_Stub())
    except Exception as e:
        print(f"[경고] MSL 등록 생략: {e!r}")

    rows = []
    for name, cfg in sorted(B.BENCH_PAIRS.items()):
        try:
            S, T = B.load_pair(cfg)
        except Exception as e:
            print(f"[skip] {name}: {e!r}")
            continue

        def stat(d, key_x, key_y):
            x = d.get(key_x)
            y = d.get(key_y)
            if x is None:
                return None
            n = len(x)
            dim = x.shape[1] if x.ndim > 1 else 1
            ratio = float(np.mean(np.asarray(y) > 0.5)) if y is not None else np.nan
            return n, dim, ratio

        s_tr = stat(S, "train", "train_label")
        t_tr = stat(T, "train", "train_label")
        t_te = stat(T, "test", "test_label")

        # window 단위 라벨 비율 (본 연구가 실제로 채점하는 단위)
        W = E9.W_BENCH
        try:
            wlab_te = B.win_labels(T["test_label"], W, 1)
            w_ratio = float(np.mean(wlab_te > 0.5))
            n_win = int(len(wlab_te))
        except Exception:
            w_ratio, n_win = np.nan, -1

        rows.append(dict(
            pair=name,
            src_dim=s_tr[1] if s_tr else None,
            src_train_len=s_tr[0] if s_tr else None,
            tgt_train_len=t_tr[0] if t_tr else None,
            tgt_test_len=t_te[0] if t_te else None,
            point_anom_ratio_tgt_train=round(t_tr[2], 4) if t_tr else None,
            point_anom_ratio_tgt_test=round(t_te[2], 4) if t_te else None,
            window_anom_ratio_tgt_test=round(w_ratio, 4),
            n_eval_windows=n_win,
            window_len=W,
        ))
        print(f"[ok] {name}: dim={rows[-1]['src_dim']} "
              f"point_ratio(test)={rows[-1]['point_anom_ratio_tgt_test']} "
              f"window_ratio(test)={rows[-1]['window_anom_ratio_tgt_test']}")

    df = pd.DataFrame(rows)
    df.to_csv("dataset_stats.csv", index=False, encoding="utf-8-sig")
    print("\n저장: dataset_stats.csv")
    if len(df):
        print(df.to_string(index=False))

    print("""
해석 가이드
-----------
- 논문 Table 에는 **window_anom_ratio_tgt_test** 를 적으세요. 본 연구가
  채점하는 단위가 window 이기 때문입니다.
- 선행 연구가 보고한 point 단위 비율과 다르면 각주로 다음과 같이 적으세요:
  "Anomaly ratios are reported at the window level (window length L, stride 1),
   which differs from the point-level ratios reported in the original dataset
   papers."
- Boiler 의 dim 이 274 가 아니면 전처리에서 변수 선택이 있었다는 뜻이므로
  Section 4.1 에 그 사실을 명시해야 합니다.
""")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
