"""
train.py - 训练心音分类模型

Adapted reproduction entry; no guaranteed training accuracy.
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import HeartSoundDataset
from src.models.tiny_cnn import TinyHeartCNN, MicroHeartCNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        
        pbar.set_postfix(loss=loss.item(), acc=correct/total)
    
    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x)
        loss = criterion(logits, y)
        
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    return {
        "loss": total_loss / len(all_labels),
        "acc": acc,
        "f1": f1,
        "preds": all_preds,
        "labels": all_labels
    }


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    size = Path(path).stat().st_size
    return size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(ROOT / "manifests/segments.csv"))
    parser.add_argument("--norm_dir", type=str, default=str(ROOT / "normalization"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arch", type=str, choices=["tiny", "micro"], default="micro")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch_size must be positive")
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    if output.is_relative_to(data_root) or data_root.is_relative_to(output):
        parser.error("Input and output trees must be separate")
    if output.exists():
        parser.error("Output directory already exists")
    args.data_root = str(data_root)
    args.output_dir = str(output)
    torch.set_num_threads(2)
    
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")
    
    # 实验目录
    if args.exp_name is None:
        args.exp_name = f"heart_{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = output
    exp_dir.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] Experiment: {exp_dir}")
    
    # 保存配置
    with open(exp_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # 加载数据
    print("\n[INFO] Loading datasets...")
    train_ds = HeartSoundDataset(args.csv, split="train", augment=True, norm_dir=args.norm_dir)
    val_ds = HeartSoundDataset(args.csv, split="val", augment=False, norm_dir=args.norm_dir)
    test_ds = HeartSoundDataset(args.csv, split="test", augment=False, norm_dir=args.norm_dir)
    
    print(f"[INFO] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Original dataset resolves relative paths against cwd; bind them explicitly.
    for dataset in (train_ds, val_ds, test_ds):
        paths = [(data_root / path).resolve() for path in dataset.df["path"]]
        if any(not path.is_relative_to(data_root) or not path.is_file() for path in paths):
            raise ValueError("Missing feature or feature path outside data root")
        dataset.df["path"] = [str(path) for path in paths]

    # 加权采样
    train_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 创建模型
    if args.arch == "micro":
        model = MicroHeartCNN(num_classes=2).to(device)
    else:
        model = TinyHeartCNN(num_classes=2).to(device)
    
    print(f"[INFO] Model: {args.arch}")
    print(f"[INFO] Parameters: {model.num_params}")
    print(f"[INFO] Estimated size: {model.estimate_size_kb():.2f} KB")
    
    # Loss & Optimizer
    class_weights = get_class_weights(train_ds.get_labels(), 2)
    print(f"[INFO] Class weights: {class_weights.tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练
    print("\n[INFO] Starting training...")
    best_val_acc = float("-inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1"].append(val_metrics["f1"])
        
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            size = save_model(model, str(exp_dir / "best_model.pth"))
            print(f"  -> Saved best model (Acc={best_val_acc:.4f}, Size={size} bytes = {size/1024:.2f} KB)")
    
    # 保存最终模型
    save_model(model, str(exp_dir / "final_model.pth"))
    
    # 保存历史
    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # 测试集评估
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(exp_dir / "best_model.pth", weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Accuracy: {test_metrics['acc']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_metrics["labels"], test_metrics["preds"], 
                                target_names=["normal", "abnormal"], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(test_metrics["labels"], test_metrics["preds"]))
    
    # 保存测试结果
    test_results = {
        "acc": float(test_metrics["acc"]),
        "f1": float(test_metrics["f1"]),
        "confusion_matrix": confusion_matrix(test_metrics["labels"], test_metrics["preds"]).tolist()
    }
    with open(exp_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    # 检查模型大小
    model_size = (exp_dir / "best_model.pth").stat().st_size
    print(f"\n[RESULT] Model size: {model_size} bytes = {model_size/1024:.2f} KB")
    print(f"[RESULT] Test Accuracy: {test_metrics['acc']*100:.2f}%")
    


if __name__ == "__main__":
    main()
