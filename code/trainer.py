"""
trainer.py  ─  모델 학습 및 검증
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from config import TrainConfig, ModelConfig
from models import BaseAnomalyModel, create_model


class Trainer:
    def __init__(self, model: BaseAnomalyModel, train_cfg: TrainConfig, model_name: str = 'model'):
        self.model = model.to(train_cfg.device)
        self.cfg = train_cfg
        self.name = model_name

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=15, factor=0.5
        )

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.threshold: Optional[float] = None

        os.makedirs(train_cfg.save_dir, exist_ok=True)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            x = batch.to(self.cfg.device)
            self.optimizer.zero_grad()
            loss, _, _ = self.model.compute_loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in loader:
            x = batch.to(self.cfg.device)
            loss, _, _ = self.model.compute_loss(x)
            total_loss += loss.item()

        return total_loss / len(loader)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = True) -> Dict:
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        ckpt_path = os.path.join(self.cfg.save_dir, f'{self.name}_best.pt')
        start_time = time.time()

        print(f"\n{'='*55}")
        print(f"[학습 시작] {self.name} | device: {self.cfg.device}")
        print(f"{'='*55}")

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                patience_counter += 1

            if verbose and (epoch % 10 == 0 or epoch == 1):
                elapsed = time.time() - start_time
                lr_now = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch:3d}/{self.cfg.epochs} | "
                      f"train={train_loss:.5f} | val={val_loss:.5f} | "
                      f"best_val={best_val_loss:.5f} | "
                      f"lr={lr_now:.2e} | {elapsed:.1f}s")

            if patience_counter >= self.cfg.patience:
                print(f"\n  [Early Stop] Epoch {epoch} (best epoch: {best_epoch}, val: {best_val_loss:.5f})")
                break

        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.cfg.device))
        print(f"\n  [완료] 최적 epoch={best_epoch} | best val loss={best_val_loss:.5f}")
        print(f"  학습 시간: {time.time() - start_time:.1f}초")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
        }


def train_model(model_type: str, model_cfg: ModelConfig, train_cfg: TrainConfig,
                train_loader: DataLoader, val_loader: DataLoader, verbose: bool = True):
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    model = create_model(model_type, model_cfg)
    trainer = Trainer(model, train_cfg, model_name=model_type)
    history = trainer.fit(train_loader, val_loader, verbose=verbose)
    return model, trainer, history
