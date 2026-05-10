import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import random
from models.Classifier import MambaClassifier
from models.Classifier_v2 import PhaseClassifier
from utils import load_config_from_json
import os
from datetime import datetime
from tqdm import tqdm

config = load_config_from_json("/home/userdata/2024_hyn/project/python/FlightGPT/config_v15.json")
# M = MambaClassifier
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
    def __init__(self, labels, batch_size, rare_class=3, rare_ratio=0.05):
        self.labels = labels
        self.batch_size = batch_size
        self.rare_class = rare_class
        self.rare_ratio = rare_ratio

        self.rare_indices = torch.where(labels == rare_class)[0].tolist()
        self.common_indices = torch.where(labels != rare_class)[0].tolist()

        # self.rare_per_batch = max(1, int(batch_size * rare_ratio))

        self.rare_per_batch = 42

        self.common_per_batch = batch_size - self.rare_per_batch

        self.num_batches = len(labels) // batch_size

    def __iter__(self):
        rare_pool = self.rare_indices.copy()
        common_pool = self.common_indices.copy()
        random.shuffle(rare_pool)
        random.shuffle(common_pool)

        for _ in range(self.num_batches):
            if len(rare_pool) < self.rare_per_batch:
                rare_pool = self.rare_indices.copy()
                random.shuffle(rare_pool)
            if len(common_pool) < self.common_per_batch:
                common_pool = self.common_indices.copy()
                random.shuffle(common_pool)

            batch_rare = [rare_pool.pop() for _ in range(self.rare_per_batch)]
            batch_common = [common_pool.pop() for _ in range(self.common_per_batch)]
            yield batch_rare + batch_common

    def __len__(self):
        return self.num_batches

def get_dataloaders(train_path, val_path, batch_size=128):
    # 加载已划分的训练集和验证集
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)

    X_train, y_train = train_data["X_bin"], train_data["y_cls"]
    X_val, y_val = val_data["X_bin"], val_data["y_cls"]

    train_dataset = FlightDataset(X_train, y_train)
    val_dataset = FlightDataset(X_val, y_val)

    # 平衡采样器，打乱训练样本
    sampler = CustomBalancedBatchSampler(
        labels=train_dataset.labels,
        batch_size=batch_size,
        rare_class=3,
        rare_ratio=0.1
    )

    train_loader = DataLoader(train_dataset, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # for X, y in train_loader:
    #     if X.dim() == 4:
    #         X = X.squeeze(0)
    #         y = y.squeeze(0)
    #     print("DEBUG batch shape:", X.shape)
    #     print(y.shape)
    #     print(y)
    #
    #     class_counts = y.argmax(dim=-1).bincount(minlength=config.num_classes)
    #     for i, count in enumerate(class_counts.tolist()):
    #         print(f"  Class {i}: {count}")
    #     break

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
        logits = model(X)  # [B, num_classes], softmaxed
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
            preds = logits.argmax(dim=-1)
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

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
PH=4: 838074 ( 28.9%)
PH=5: 1415980 ( 48.8%)
PH=6: 646140 ( 22.3%)
PH=7:     46 (  0.0%)
"""
