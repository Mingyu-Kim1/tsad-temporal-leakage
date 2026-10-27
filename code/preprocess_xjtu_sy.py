"""
preprocess_xjtu_sy.py - XJTU-SY 원시 진동 CSV → bearing별 feature 시계열 CSV

실행: python preprocess_xjtu_sy.py
출력: data/xjtu_sy/bearing{조건}_{번호}_features.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

XJTU_ROOT = 'data/xjtu_sy'

GROUPS = {
    'XB1': {
        'folder': '35Hz12kN',
        'bearings': ['Bearing1_1', 'Bearing1_2', 'Bearing1_3', 'Bearing1_5']  # 1_4 제외
    },
    'XB2': {
        'folder': '37.5Hz11kN',
        'bearings': ['Bearing2_1', 'Bearing2_2', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5']
    },
}


def extract_features(csv_path):
    """한 CSV 파일에서 진동 feature 추출"""
    df = pd.read_csv(csv_path)
    
    # 컬럼명 확인 후 처리
    cols = df.columns.tolist()
    if len(cols) >= 2:
        h = df.iloc[:, 0].values.astype(float)  # Horizontal
        v = df.iloc[:, 1].values.astype(float)  # Vertical
    else:
        raise ValueError(f"Expected 2 columns, got {len(cols)} in {csv_path}")
    
    return {
        'rms_h': np.sqrt(np.mean(h**2)),
        'rms_v': np.sqrt(np.mean(v**2)),
        'peak_h': np.max(np.abs(h)),
        'peak_v': np.max(np.abs(v)),
        'kurtosis_h': float(kurtosis(h)),
        'kurtosis_v': float(kurtosis(v)),
        'std_h': np.std(h),
        'std_v': np.std(v),
    }


def process_bearing(bearing_dir, save_path):
    """한 bearing 폴더의 모든 CSV → feature 시계열 CSV"""
    csv_files = sorted([f for f in os.listdir(bearing_dir) if f.endswith('.csv')])
    
    if len(csv_files) == 0:
        print(f"  [경고] CSV 파일 없음: {bearing_dir}")
        return None
    
    rows = []
    for f in csv_files:
        try:
            features = extract_features(os.path.join(bearing_dir, f))
            rows.append(features)
        except Exception as e:
            print(f"  [에러] {f}: {e}")
            continue
    
    if len(rows) == 0:
        return None
    
    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result.to_csv(save_path, index=False)
    print(f"  저장: {save_path} | shape={result.shape}")
    return result


def main():
    print("=" * 60)
    print("  XJTU-SY 전처리")
    print("=" * 60)
    
    for group_name, group_info in GROUPS.items():
        print(f"\n[{group_name}] {group_info['folder']}")
        
        for bearing in group_info['bearings']:
            bearing_dir = os.path.join(XJTU_ROOT, group_info['folder'], bearing)
            save_path = os.path.join(XJTU_ROOT, f'{bearing}_features.csv')
            
            if not os.path.exists(bearing_dir):
                print(f"  [경고] 폴더 없음: {bearing_dir}")
                continue
            
            process_bearing(bearing_dir, save_path)
    
    # 그룹별 합친 CSV도 생성 (Cross-Domain용)
    print("\n[그룹별 합치기]")
    for group_name, group_info in GROUPS.items():
        all_rows = []
        for bearing in group_info['bearings']:
            feat_path = os.path.join(XJTU_ROOT, f'{bearing}_features.csv')
            if os.path.exists(feat_path):
                df = pd.read_csv(feat_path)
                all_rows.append(df)
        
        if len(all_rows) > 0:
            combined = pd.concat(all_rows, ignore_index=True)
            save_path = os.path.join(XJTU_ROOT, f'{group_name}_combined.csv')
            combined.to_csv(save_path, index=False)
            print(f"  {group_name}: {save_path} | shape={combined.shape}")
    
    print("\n[완료]")


if __name__ == '__main__':
    main()
