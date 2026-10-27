"""
config.py  ─  실험 설정 파일
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipe_columns import SIM_COLS as _PIPE_SIM, REAL_COLS as _PIPE_REAL  # NDA
import torch


@dataclass
class DataConfig:
    data_root: str = 'data'
    real_dataset_name: str = 'realCase1'
    sim_dataset_name: str = 'simCase1'
    real_data_path: str = os.path.join('data', 'pipe', 'realCase1_pfei.csv')
    sim_data_path:  str = os.path.join('data', 'pipe', 'simCase1_pfei.csv')
    data_strategy: str = 'real_only'

    real_dataset_options: Dict[str, str] = field(default_factory=lambda: {
        'realCase1': os.path.join('data', 'pipe', 'realCase1_pfei.csv'),
        'realCase2': os.path.join('data', 'pipe','realCase2_pfei.csv'),
        'cmapss_fd001_train': os.path.join('data', 'cmapss', 'cmapss_fd001_train.csv'),
        'cmapss_fd001_test': os.path.join('data', 'cmapss', 'cmapss_fd001_test.csv'),

        'cmapss_fd002_train': os.path.join('data', 'cmapss', 'cmapss_fd002_train.csv'),
        'cmapss_fd002_test': os.path.join('data', 'cmapss', 'cmapss_fd002_test.csv'),

        'cmapss_fd003_train': os.path.join('data', 'cmapss', 'cmapss_fd003_train.csv'),
        'cmapss_fd003_test': os.path.join('data', 'cmapss', 'cmapss_fd003_test.csv'),

        'cmapss_fd004_train': os.path.join('data', 'cmapss', 'cmapss_fd004_train.csv'),
        'cmapss_fd004_test': os.path.join('data', 'cmapss', 'cmapss_fd004_test.csv'),

        'phm2012': os.path.join('data', 'phm2012'),
        'phm2012_b11': os.path.join('data', 'phm2012', 'bearing1_1_features.csv'),
    })
    sim_dataset_options: Dict[str, str] = field(default_factory=lambda: {
        'simCase1': os.path.join('data', 'pipe', 'simCase1_pfei.csv'),
    })

    real_sensor_cols: List[str] = field(default_factory=lambda: list(_PIPE_REAL))  # NDA

    sim_sensor_cols: List[str] = field(default_factory=lambda: list(_PIPE_SIM))   # NDA

    # CMAPSS 센서 컬럼 (24개)
    cmapss_sensor_cols: List[str] = field(default_factory=lambda: [
        f's{i}' for i in range(1, 25)
    ])
    cmapss_mode: bool = False
    cmapss_unit_col: str = 'engine_id'
    cmapss_time_col: str = 'cycle'

    n_features: int = 5
    window_size: int = 15
    train_ratio: float = 0.50
    val_ratio:   float = 0.20
    test_ratio:  float = 0.30

    #PHM 2012은 시점 기준으로 train/val/test 나뉘어져 있어서 따로 경로 설정, 모드 플래그 추가
    phm2012_mode: bool = False
    phm2012_train_dir: str = os.path.join('data', 'phm2012', 'Training_set', 'Learning_set')
    phm2012_val_dir: str   = os.path.join('data', 'phm2012', 'Validation_Set', 'Full_Test_Set')
    phm2012_test_dir: str  = os.path.join('data', 'phm2012', 'Test_set')

    # PHM2012는 보통 수평/수직 진동 2채널
    phm2012_sensor_cols: List[str] = field(default_factory=lambda: [
        'acc_h', 'acc_v'
    ])

    # 사람 기준 2550번째 시점, Python 0-based index 2549
    fault_point_idx: int = 2549

    sim_drypump_end: int = 310
    outlier_sigma: float = 3.0
    scaler_type: str = 'minmax'   # 'minmax' / 'standard' / 'robust'

    def __post_init__(self):
        self.resolve_data_paths()

    def resolve_data_paths(self):
        if self.real_dataset_name in self.real_dataset_options:
            self.real_data_path = self.real_dataset_options[self.real_dataset_name]
        if self.sim_dataset_name in self.sim_dataset_options:
            self.sim_data_path = self.sim_dataset_options[self.sim_dataset_name]

    def set_real_dataset(self, dataset_name: str):
        if dataset_name not in self.real_dataset_options:
            raise ValueError(f"Unknown real dataset: {dataset_name}. Available: {list(self.real_dataset_options.keys())}")
        self.real_dataset_name = dataset_name
        self.real_data_path = self.real_dataset_options[dataset_name]
        
        # 데이터셋별 센서 설정
        if dataset_name == 'phm2012':
            self.phm2012_mode = True
            self.cmapss_mode = False
            self.real_sensor_cols = self.phm2012_sensor_cols
            self.n_features = 2
        elif dataset_name == 'phm2012_b11':         
            self.real_sensor_cols = ['rms_h', 'rms_v', 'peak_h', 'peak_v',
                                     'kurtosis_h', 'kurtosis_v', 'std_h', 'std_v']
            self.n_features = 8
            self.cmapss_mode = False
            self.phm2012_mode = False   

        elif dataset_name.startswith('cmapss'):
            self.real_sensor_cols = self.cmapss_sensor_cols
            self.n_features = 24
            self.cmapss_mode = True
            self.phm2012_mode = False

        else:
            self.real_sensor_cols = list(_PIPE_REAL)
            self.n_features = 5
            self.cmapss_mode = False
            self.phm2012_mode = False

    def set_sim_dataset(self, dataset_name: str):
        if dataset_name not in self.sim_dataset_options:
            raise ValueError(f"Unknown sim dataset: {dataset_name}. Available: {list(self.sim_dataset_options.keys())}")
        self.sim_dataset_name = dataset_name
        self.sim_data_path = self.sim_dataset_options[dataset_name]


@dataclass
class ModelConfig:
    n_heads: int = 5
    input_size: int = 5
    window_size: int = 15

    lstm_hidden: int = 64
    lstm_latent: int = 32
    lstm_layers: int = 1
    lstm_dropout: float = 0.0

    d_model: int = 64
    nhead: int = 4
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    dim_ff: int = 256
    tf_dropout: float = 0.1

    tcn_channels: List[int] = field(default_factory=lambda: [64, 64, 64])
    tcn_kernel: int = 3
    tcn_dilations: List[int] = field(default_factory=lambda: [1, 2, 4])
    tcn_dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 250
    lr: float = 5e-4
    weight_decay: float = 0.0
    patience: int = 40

    threshold_mode: str = '95'   # 'max' / '99' / '95'
    median_filter_size: int = 1

    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir: str = 'checkpoints'
    result_dir: str = 'results'


MODEL_LIST = {
    'lstm_ae': '논문 제안 방법 (LSTM Encoder-Decoder + N heads)',
    'transformer_ae': 'Transformer Encoder-Decoder + N heads',
    'tcn_ae': 'TCN Encoder-Decoder + N heads',
    'lstm_pred': 'LSTM Predictor + N heads',
    'transformer_pred': 'Transformer Predictor + N heads',
    'tcn_pred': 'TCN Predictor + N heads',
}

STRATEGY_DESC: Dict[str, str] = {
    'real_only': 'realCase1만 사용',
    'sim_train_real_test': 'simCase1 학습 → realCase1 검증/테스트',
    'combined_train_real_test': 'sim+real 합쳐서 학습 → realCase1 테스트',
}

data_cfg = DataConfig()
model_cfg = ModelConfig()
train_cfg = TrainConfig()
