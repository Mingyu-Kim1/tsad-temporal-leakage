# -*- coding: utf-8 -*-
"""
setup_msl_smap.py — MSL/SMAP 배치 확인 + pair 자동 선정
=========================================================
실행 위치: anomaly_detection/data/benchmark/MSL_SMAP/
    python setup_msl_smap.py

하는 일:
  1) train/ test/ labeled_anomalies.csv 배치가 올바른지 검사 (없으면 받는 법 안내)
  2) 채널별 train 길이·차원·이상비율 집계
  3) 본 실험 조건에 맞는 source→target pair를 자동 추천
     조건: 같은 spacecraft(=차원 동일) · train 충분 · target 이상비율 2~25%
  4) exp4/exp3에 그대로 넣을 --msl_pairs 인자 문자열 출력
"""
from pathlib import Path
import ast
import sys
import numpy as np
import pandas as pd

W, STRIDE = 100, 5          # 실험 설정과 동일
MIN_WIN = 300               # 적응에 필요한 최소 window 수
RATIO_LO, RATIO_HI = 0.02, 0.25

root = Path(".")
csv = root / "labeled_anomalies.csv"
if not csv.exists():
    sys.exit("❌ labeled_anomalies.csv 없음. 이 스크립트를 MSL_SMAP 폴더에서 실행하세요.")

tr_dir, te_dir = root / "train", root / "test"
if not tr_dir.exists() or not te_dir.exists():
    print("❌ train/ test/ 폴더가 없습니다. 텔레메트리 데이터(.npy)를 아직 안 받았습니다.\n")
    print("PowerShell에서 아래를 실행하세요 (이 폴더에서):\n")
    print('  curl.exe -L -o data.zip https://s3-us-west-2.amazonaws.com/telemanom/data.zip')
    print('  Expand-Archive data.zip -DestinationPath . -Force')
    print('  Remove-Item data.zip')
    print('  # 압축 안에 data\\train, data\\test 로 들어있으면 한 단계 올리기:')
    print('  Move-Item data\\train .\\train ; Move-Item data\\test .\\test')
    print('  Remove-Item data -Recurse -Force -ErrorAction SilentlyContinue')
    print('  Remove-Item 2018-05-19_15.00.10 -Recurse -Force -ErrorAction SilentlyContinue')
    print("\n그다음 이 스크립트를 다시 실행하세요.")
    sys.exit(1)

la = pd.read_csv(csv)
rows = []
for _, r in la.iterrows():
    f_tr, f_te = tr_dir / f"{r.chan_id}.npy", te_dir / f"{r.chan_id}.npy"
    if not (f_tr.exists() and f_te.exists()):
        continue
    a, b = np.load(f_tr), np.load(f_te)
    seq = ast.literal_eval(r.anomaly_sequences)
    n_anom = sum(e - s + 1 for s, e in seq)
    rows.append(dict(chan=r.chan_id, sc=r.spacecraft, dim=a.shape[1],
                     train_len=len(a), test_len=len(b),
                     train_win=(len(a) - W) // STRIDE + 1,
                     test_win=len(b) - W + 1,
                     anom_ratio=round(n_anom / max(r.num_values, 1), 4),
                     n_seq=len(seq)))
df = pd.DataFrame(rows)
if df.empty:
    sys.exit("❌ .npy 파일을 찾지 못했습니다. train/ test/ 안에 채널별 npy가 있어야 합니다.")

print(f"✅ 채널 {len(df)}개 확인\n")
print(df.groupby(["sc", "dim"]).agg(n=("chan", "count"),
                                    train_win=("train_win", "mean"),
                                    anom=("anom_ratio", "mean")).round(2).to_string())

ok = df[(df.train_win >= MIN_WIN)]
print(f"\n=== 적응 가능(train window ≥ {MIN_WIN}) 채널 ===")
print(ok.sort_values(["sc", "train_win"], ascending=[True, False])
      [["chan", "sc", "dim", "train_win", "test_win", "anom_ratio", "n_seq"]]
      .to_string(index=False))

print("\n" + "=" * 70)
print("추천 pair (source=train 긴 채널, target=이상비율 적정 채널, 같은 spacecraft·차원)")
rec = []
for (sc, dim), g in ok.groupby(["sc", "dim"]):
    src_c = g.sort_values("train_win", ascending=False)
    tgt_c = g[(g.anom_ratio.between(RATIO_LO, RATIO_HI)) & (g.n_seq >= 1)] \
        .sort_values("test_win", ascending=False)
    for _, t in tgt_c.head(3).iterrows():
        s = src_c[src_c.chan != t.chan].head(1)
        if len(s):
            s = s.iloc[0]
            rec.append(dict(sc=sc, dim=dim, src=s.chan, tgt=t.chan,
                            src_train_win=s.train_win, tgt_test_win=t.test_win,
                            tgt_anom=t.anom_ratio))
rdf = pd.DataFrame(rec)
if rdf.empty:
    print("조건을 만족하는 조합이 없습니다. RATIO_LO/HI 또는 MIN_WIN을 완화하세요.")
else:
    print(rdf.to_string(index=False))
    arg = " ".join(f"{r.src}:{r.tgt}" for _, r in rdf.iterrows())
    print("\n▶ 실행 인자로 그대로 사용:\n")
    print(f"  --msl_pairs {arg}")
df.to_csv("msl_smap_channels.csv", index=False)
print("\n[저장] msl_smap_channels.csv")
