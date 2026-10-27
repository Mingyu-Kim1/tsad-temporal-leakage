"""
preprocess_phm2012_cross.py - PHM2012 PB1/PB2 그룹별 feature 시계열 CSV 생성

실행: python preprocess_phm2012_cross.py
출력: data/phm2012/PB1_combined.csv, PB2_combined.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

PHM_ROOT = 'data/phm2012'

# 교수님 기준 그룹 분할
GROUPS = {
    'PB1': {
        'bearings': ['Bearing1_1', 'Bearing1_2', 'Bearing1_3', 'Bearing1_4',
                     'Bearing1_5', 'Bearing1_6', 'Bearing1_7'],
        'search_dirs': [
            os.path.join(PHM_ROOT, 'Training_set', 'Learning_set'),
            os.path.join(PHM_ROOT, 'Validation_Set', 'Full_Test_Set'),
            os.path.join(PHM_ROOT, 'Test_set'),
        ]
    },
    'PB2': {
        'bearings': ['Bearing2_1', 'Bearing2_2', 'Bearing2_3', 'Bearing2_4',
                     'Bearing2_6', 'Bearing2_7'],  # 2_5 제외 (교수님 기준)
        'search_dirs': [
            os.path.join(PHM_ROOT, 'Training_set', 'Learning_set'),
            os.path.join(PHM_ROOT, 'Validation_Set', 'Full_Test_Set'),
            os.path.join(PHM_ROOT, 'Test_set'),
        ]
    },
}


def extract_features(csv_path):
    df = pd.read_csv(csv_path, header=None)
    h = df.iloc[:, 4].values.astype(float)  # 수평 진동
    v = df.iloc[:, 5].values.astype(float)  # 수직 진동
    
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


def find_bearing_dir(bearing_name, search_dirs):
    for d in search_dirs:
        path = os.path.join(d, bearing_name)
        if os.path.exists(path):
            return path
    return None


def process_bearing(bearing_dir, save_path):
    acc_files = sorted([f for f in os.listdir(bearing_dir) if f.startswith('acc_')])
    
    if len(acc_files) == 0:
        print(f"  [경고] acc 파일 없음: {bearing_dir}")
        return None
    
    rows = []
    for f in acc_files:
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
    print("  PHM2012 Cross-Domain 전처리")
    print("=" * 60)
    
    for group_name, group_info in GROUPS.items():
        print(f"\n[{group_name}]")
        all_rows = []
        
        for bearing in group_info['bearings']:
            bearing_dir = find_bearing_dir(bearing, group_info['search_dirs'])
            
            if bearing_dir is None:
                print(f"  [경고] 폴더 못 찾음: {bearing}")
                continue
            
            save_path = os.path.join(PHM_ROOT, f'{bearing}_features.csv')
            df = process_bearing(bearing_dir, save_path)
            
            if df is not None:
                all_rows.append(df)
        
        if len(all_rows) > 0:
            combined = pd.concat(all_rows, ignore_index=True)
            save_path = os.path.join(PHM_ROOT, f'{group_name}_combined.csv')
            combined.to_csv(save_path, index=False)
            print(f"  {group_name} 합침: {save_path} | shape={combined.shape}")
    
    print("\n[완료]")


if __name__ == '__main__':
    main()
