"""
tiny_cnn.py - 超小型 CNN 模型 (<10KB)

用于心音分类，模型权重 < 10KB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyHeartCNN(nn.Module):
    """
    超小型心音分类 CNN
    目标: 模型权重 < 10KB (FP32)
    
    结构:
    - 3 个深度可分离卷积块
    - 全局平均池化
    - 全连接分类器
    """
    
    def __init__(self, num_classes: int = 2, in_ch: int = 1):
        super().__init__()
        
        # 深度可分离卷积块
        self.block1 = self._make_dw_block(in_ch, 8, stride=2)
        self.block2 = self._make_dw_block(8, 16, stride=2)
        self.block3 = self._make_dw_block(16, 24, stride=2)
        
        # 全局池化 + 分类器
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(24, num_classes)
        
        self._init_weights()
    
    def _make_dw_block(self, in_ch: int, out_ch: int, stride: int = 1):
        """深度可分离卷积块"""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x
    
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def estimate_size_kb(self) -> float:
        """估算 FP32 模型大小 (KB)"""
        return self.num_params * 4 / 1024


class MicroHeartCNN(nn.Module):
    """
    微型心音分类 CNN - 更小版本
    目标: 模型权重 < 10KB
    """
    
    def __init__(self, num_classes: int = 2, in_ch: int = 1):
        super().__init__()
        
        # 使用最少的通道
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 6, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(6, 6, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(6, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x
    
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def estimate_size_kb(self) -> float:
        return self.num_params * 4 / 1024


if __name__ == "__main__":
    # 测试模型大小
    print("TinyHeartCNN:")
    m1 = TinyHeartCNN(num_classes=2)
    print(f"  Parameters: {m1.num_params}")
    print(f"  Estimated size: {m1.estimate_size_kb():.2f} KB")
    
    print("\nMicroHeartCNN:")
    m2 = MicroHeartCNN(num_classes=2)
    print(f"  Parameters: {m2.num_params}")
    print(f"  Estimated size: {m2.estimate_size_kb():.2f} KB")
    
    # 测试前向传播
    x = torch.randn(2, 1, 64, 94)  # [B, C, n_mels, frames]
    print(f"\nInput shape: {x.shape}")
    print(f"TinyHeartCNN output: {m1(x).shape}")
    print(f"MicroHeartCNN output: {m2(x).shape}")
