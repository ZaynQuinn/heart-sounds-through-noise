"""
dataset.py - PhysioNet 2016 心音数据集加载器
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple


class HeartSoundDataset(Dataset):
    """心音数据集"""
    
    def __init__(
        self,
        csv_path: str,
        split: Optional[str] = None,
        augment: bool = False,
        normalize: bool = True,
        norm_dir: Optional[str] = None,
    ):
        self.augment = augment
        self.normalize = normalize
        
        # 加载元数据
        df = pd.read_csv(csv_path)
        
        # 筛选 split
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        
        # 标签: -1 (正常) -> 0, 1 (异常) -> 1
        df["label_id"] = (df["label"] == 1).astype(int)
        
        self.df = df
        self.class_names = ["normal", "abnormal"]
        self.num_classes = 2
        
        # 加载标准化参数
        if normalize and norm_dir:
            norm_path = Path(norm_dir)
            self.mean = np.load(norm_path / "mean.npy")[:, np.newaxis]
            self.std = np.load(norm_path / "std.npy")[:, np.newaxis]
        else:
            self.mean = None
            self.std = None
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _augment(self, spec: np.ndarray) -> np.ndarray:
        """频谱增强"""
        if not self.augment:
            return spec
        
        spec = spec.copy()
        n_mels, n_frames = spec.shape
        
        # 时间遮蔽
        if np.random.random() < 0.5:
            t = np.random.randint(0, min(20, n_frames // 4))
            t0 = np.random.randint(0, max(1, n_frames - t))
            spec[:, t0:t0+t] = 0
        
        # 频率遮蔽
        if np.random.random() < 0.5:
            f = np.random.randint(0, min(8, n_mels // 4))
            f0 = np.random.randint(0, max(1, n_mels - f))
            spec[f0:f0+f, :] = 0
        
        # 随机增益
        if np.random.random() < 0.3:
            gain = np.random.uniform(0.8, 1.2)
            spec = spec * gain
        
        return spec
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        
        # 加载特征
        spec = np.load(row["path"]).astype(np.float32)
        
        # 标准化
        if self.normalize and self.mean is not None:
            spec = (spec - self.mean) / self.std
        
        # 数据增强
        spec = self._augment(spec)
        
        # 转为 tensor [1, n_mels, frames]
        x = torch.from_numpy(spec[np.newaxis, :, :])
        label = int(row["label_id"])
        
        return x, label
    
    def get_labels(self) -> np.ndarray:
        return self.df["label_id"].values
    
    def get_sample_weights(self) -> np.ndarray:
        labels = self.get_labels()
        class_counts = np.bincount(labels, minlength=self.num_classes)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        return class_weights[labels]
