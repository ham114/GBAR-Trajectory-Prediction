import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import random
from models.Classifier_v2 import PhaseClassifier
from utils import load_config_from_json
import os
from datetime import datetime
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import torch._dynamo
torch._dynamo.config.suppress_errors = True

scaler = GradScaler()

config = load_config_from_json("config_v20.json")
M = PhaseClassifier

class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y  # keep as one-hot [N, 4]
        self.labels = y.argmax(dim=-1)  # [N]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # keep y as one-hot for CE loss


class CustomBalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, rare_classes=None, rare_ratios=None):
        self.labels = labels
        self.batch_size = batch_size

        if rare_classes is None:
            rare_classes = [3, 4, 5, 6, 7, 8, 9]
        if rare_ratios is None:
            rare_ratios = [0.05] * len(rare_classes)

        self.rare_classes = rare_classes
        self.rare_ratios = rare_ratios

        self.class_indices = {}
        for class_id in range(config.num_classes):
            self.class_indices[class_id] = torch.where(labels == class_id)[0].tolist()

        self.rare_per_batch = {}
        for class_id, ratio in zip(rare_classes, rare_ratios):
            self.rare_per_batch[class_id] = max(1, int(batch_size * ratio))

        total_rare_per_batch = sum(self.rare_per_batch.values())
        self.common_per_batch = batch_size - total_rare_per_batch
        self.common_classes = [class_id for class_id in range(config.num_classes)
                               if class_id not in rare_classes]

        # 固定为较大的批次数量
        self.num_batches = 4096  # 或者 len(labels) // batch_size

        print(f"固定批次数量: {self.num_batches}")

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []

            # 稀缺类别：每次都从原始数据中随机选择（允许重复）
            for class_id in self.rare_classes:
                required = self.rare_per_batch[class_id]
                indices = random.choices(self.class_indices[class_id], k=required)
                batch_indices.extend(indices)

            # 普通类别：每次都从原始数据中随机选择
            if self.common_per_batch > 0:
                all_common_indices = []
                for class_id in self.common_classes:
                    all_common_indices.extend(self.class_indices[class_id])
                common_indices = random.choices(all_common_indices, k=self.common_per_batch)
                batch_indices.extend(common_indices)

            yield batch_indices

    def __len__(self):
        return self.num_batches

# 在你的 get_dataloaders 函数中
def get_dataloaders(train_path, val_path, batch_size=128, rare_classes=[4, 5, 6, 7, 8, 9]):
    # 加载已划分的训练集和验证集
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)

    X_train, y_train = train_data["geo_codes_bits"], train_data["labels_onehot"]
    X_val, y_val = val_data["geo_codes_bits"], val_data["labels_onehot"]

    train_dataset = FlightDataset(X_train, y_train)
    val_dataset = FlightDataset(X_val, y_val)

    # 新的平衡采样器，支持多个稀缺类别
    sampler = CustomBalancedBatchSampler(
        labels=train_dataset.labels,
        batch_size=batch_size,
        rare_classes=[4, 5, 6, 7, 8, 9],  # 所有稀缺类别
        rare_ratios=[0.05] * len(rare_classes)  # 每个占5%
    )

    train_loader = DataLoader(train_dataset, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in tqdm(loader, desc="Training", leave=False):
        if X.dim() == 4:
            X = X.squeeze(0)
            y = y.squeeze(0)
        perm = torch.randperm(X.size(0))
        X, y = X[perm], y[perm]
        X, y = X.float().to(device), y.to(device)
        with torch.amp.autocast(device_type='cuda'):
            logits = model(X)  # [B, num_classes]

            # 修复：将 one-hot 转换为类别索引
            targets = y.argmax(dim=-1)  # [B]

            loss = criterion(logits, targets)  # 使用类别索引

        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in loader:
            if X.dim() == 4:
                X = X.squeeze(0)
                y = y.squeeze(0)
            X, y = X.float().to(device), y.to(device)
            logits = model(X)  # [B, num_classes]
            logits = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            # 修复：将 one-hot 转换为类别索引
            labels = y.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    report = classification_report(all_labels, all_preds, digits=4)
    return report


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = M(config).to(device)
    model.get_num_params()
    train_loader, val_loader = get_dataloaders(
        train_path=config.train_tensor_path,
        val_path=config.test_tensor_path,
        batch_size=config.batch_size
    )

    # DEBUG: check first batch shapes and class counts
    for X, y in train_loader:
        print("DEBUG batch shape:", X.shape)
        labels = y.argmax(dim=-1)
        labels_flat = labels.view(-1) if labels.ndim > 1 else labels
        for i in range(config.num_classes):
            count = (labels_flat == i).sum().item()
            print(f"  Class {i}: {count}")
        break

    # 计算类别权重
    train_data = torch.load(config.train_tensor_path)
    y_train = train_data["labels_onehot"]
    labels = y_train.argmax(dim=-1)

    # 方法1: 基于类别频率计算权重
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts.float())

    # 方法2: 手动设置权重（根据你的类别分布调整）
    # class_weights = torch.tensor([1.0, 1.0, 1.0, 5.0, 10.0, 10.0, 15.0, 20.0, 20.0, 30.0])

    class_weights = class_weights.to(device)

    print("类别权重:")
    for i, weight in enumerate(class_weights):
        print(f"  类别 {i}: {weight:.2f} (样本数: {class_counts[i]})")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    best_f1 = 0.0
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_root = os.path.join(config.save_dir, timestamp)
    os.makedirs(save_root, exist_ok=True)
    log_file = os.path.join(save_root, "training_log.txt")

    with open(log_file, "w") as f:
        for epoch in range(1, config.epochs + 1):
            print(f"Epoch {epoch}/{config.epochs}")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}")
            f.write(f"\nEpoch {epoch}: train_loss={train_loss:.4f}\n")

            report = evaluate(model, val_loader, device)
            print(report)
            f.write(report + "\n")


            # 保存最佳模型
            report_lines = report.strip().split('\n')
            avg_line = [l for l in report_lines if 'accuracy' not in l][-1]
            f1 = float(avg_line.strip().split()[-2])

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), os.path.join(save_root, "best_model.pt"))
                print("Saved new best model!\n")
                f.write("Saved new best model!\n\n")

            f.flush()  # ⬅⬅⬅ 立即将缓冲写入磁盘


if __name__ == '__main__':
    train()

"""
Label  0: 001000 :   230953  (51.73%)  -> ['high_cruise']
Label  1: 000010 :   109577  (24.54%)  -> ['descent']
Label  2: 010000 :    44970  (10.07%)  -> ['climb']
Label  3: 000100 :    33583  ( 7.52%)  -> ['midlow_level']
Label  4: 001010 :     7082  ( 1.59%)  -> ['high_cruise', 'descent']
Label  5: 011000 :     6716  ( 1.50%)  -> ['climb', 'high_cruise']
Label  6: 000001 :     4991  ( 1.12%)  -> ['approach']
Label  7: 000011 :     3656  ( 0.82%)  -> ['descent', 'approach']
Label  8: 000110 :     3340  ( 0.75%)  -> ['midlow_level', 'descent']
Label  9: 010100 :     1578  ( 0.35%)  -> ['climb', 'midlow_level']
"""

"""
              precision    recall  f1-score   support

           0     0.9973    0.9950    0.9962    226659
           1     0.9931    0.9854    0.9892    109629
           2     0.9913    0.9832    0.9872     43488
           3     0.9892    0.9797    0.9844     37316
           4     0.8673    0.9443    0.9041      7141
           5     0.8390    0.9035    0.8701      6745
           6     0.9491    0.9491    0.9491      5126
           7     0.8715    0.9624    0.9147      3861
           8     0.7935    0.8757    0.8325      3571
           9     0.8420    0.9128    0.8760      1675

    accuracy                         0.9860    445211
   macro avg     0.9133    0.9491    0.9304    445211
weighted avg     0.9866    0.9860    0.9862    445211

"""
