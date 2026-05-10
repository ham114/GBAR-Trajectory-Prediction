import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import random
from models.Classifier_v3 import PhaseClassifier
from utils import load_config_from_json
import os
from datetime import datetime
from tqdm import tqdm

config = load_config_from_json("config_v20.json")
M = PhaseClassifier

class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.float()                # multi-hot [N, C]
        self.labels = y                   # 直接保存多热矩阵，供采样器使用

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CustomBalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, rare_class=3, rare_ratio=0.05):
        """
        labels: [N, C] multi-hot。按 rare_class 列是否为1划分样本池
        """
        self.labels = labels
        self.batch_size = batch_size
        self.rare_class = rare_class
        self.rare_ratio = rare_ratio

        # 多热 → 稀有/普通索引
        rare_mask = (labels[:, rare_class] > 0.5)
        self.rare_indices = torch.where(rare_mask)[0].tolist()
        self.common_indices = torch.where(~rare_mask)[0].tolist()

        # 按比例决定每批稀有样本数
        self.rare_per_batch = max(1, int(batch_size * rare_ratio))
        self.common_per_batch = batch_size - self.rare_per_batch

        # 一个 epoch 的批次数
        self.num_batches = len(labels) // batch_size

        if len(self.rare_indices) == 0:
            raise ValueError("数据中不存在指定稀有类样本。")

    def __iter__(self):
        import random
        rare_pool = self.rare_indices.copy()
        common_pool = self.common_indices.copy()
        random.shuffle(rare_pool)
        random.shuffle(common_pool)

        for _ in range(self.num_batches):
            if len(rare_pool) < self.rare_per_batch:
                # 允许在一个 epoch 内重复使用稀有样本
                rare_pool = self.rare_indices.copy()
                random.shuffle(rare_pool)
            if len(common_pool) < self.common_per_batch:
                common_pool = self.common_indices.copy()
                random.shuffle(common_pool)

            batch_rare = [rare_pool.pop() for _ in range(self.rare_per_batch)]
            batch_common = [common_pool.pop() for _ in range(self.common_per_batch)]
            batch = batch_rare + batch_common
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def get_dataloaders(train_path, val_path, batch_size=128):
    # 加载已划分的训练集和验证集
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)

    X_train, y_train = train_data["geo_codes_bits"], train_data["labels_phase"]
    X_val, y_val = val_data["geo_codes_bits"], val_data["labels_phase"]

    train_dataset = FlightDataset(X_train, y_train)
    val_dataset = FlightDataset(X_val, y_val)

    # 平衡采样器，打乱训练样本
    sampler = CustomBalancedBatchSampler(
        labels=train_dataset.labels,  # 现在是多热 [N, C]
        batch_size=batch_size,
        rare_class=0,  # 例如你要针对“类别0”为稀有类
        rare_ratio=0.1
    )

    train_loader = DataLoader(train_dataset, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in tqdm(loader, desc="Training", leave=False):
        if X.dim() == 4:
            X = X.squeeze(0)
            y = y.squeeze(0)

        # 可选：打乱 batch 内顺序
        perm = torch.randperm(X.size(0))
        X, y = X[perm], y[perm]

        X = X.float().to(device)
        y = y.float().to(device)  # 多热→float

        out = model(X)
        logits = out[0] if isinstance(out, tuple) else out  # 兼容 return_hidden=True 的情况

        # 如果 forward 返回 logits：用 BCEWithLogitsLoss
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, threshold=0.5, use_logits=True):
    """
    use_logits=True：模型输出为 logits，需要 sigmoid
    use_logits=False：模型已 sigmoid，直接用概率
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            if X.dim() == 4:
                X = X.squeeze(0)
                y = y.squeeze(0)

            X = X.float().to(device)
            y = y.float().to(device)

            out = model(X)
            logits = out[0] if isinstance(out, tuple) else out

            probs = torch.sigmoid(logits) if use_logits else logits
            preds = (probs >= threshold).int().cpu()
            labels = (y >= 0.5).int().cpu()

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    report = classification_report(
        all_labels, all_preds, digits=4, zero_division=0
    )
    return report


def _extract_weighted_f1_from_report(report_text: str) -> float:
    # 从 classification_report 文本里抓取 "weighted avg" 行的 F1
    for line in report_text.strip().split('\n'):
        if line.strip().startswith("weighted avg"):
            parts = line.split()
            # 行格式: weighted avg  precision  recall  f1-score  support
            return float(parts[-2])
    # 兜底：取最后一行的倒数第二个字段（不含 accuracy 行）
    lines = [l for l in report_text.strip().split('\n') if 'accuracy' not in l and l.strip()]
    return float(lines[-1].split()[-2])


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = M(config).to(device)
    model.get_num_params()

    train_loader, val_loader = get_dataloaders(
        train_path=config.train_tensor_path,
        val_path=config.test_tensor_path,
        batch_size=config.batch_size
    )

    # 多标签调试统计：首个 batch 中每类“1”的数量
    for X, y in train_loader:
        print("DEBUG batch shape:", X.shape)
        print("Label shape:", y.shape)

        # y: 可能是 [1, B, C] 或 [B, C]，先挤掉外层 batch
        if y.dim() > 2 and y.size(0) == 1:
            y = y.squeeze(0)           # [B, C]
        y = y.float()

        # 将除最后一维外全部展平 → [N_eff, C]，适配 [B, C] / [B, T, C]
        y_flat = y.view(-1, y.size(-1))
        counts = y_flat.sum(dim=0)     # [C]

        num_classes = counts.numel()
        for i in range(num_classes):
            print(f"  Class {i}: {int(counts[i].item())}")
        break

    # === 计算 pos_weight，用于 BCEWithLogitsLoss ===
    # 直接从完整训练 tensor 计算一次
    train_data = torch.load(config.train_tensor_path, map_location="cpu")
    y_all = train_data["labels_phase"].float()   # [N, C] 或 [*, C]
    y_all_flat = y_all.view(-1, y_all.size(-1))  # [N_eff, C]

    pos_counts = y_all_flat.sum(dim=0)                  # 每类正样本数 [C]
    total = y_all_flat.size(0)
    neg_counts = total - pos_counts                     # 每类负样本数 [C]
    eps = 1e-6
    pos_weight = (neg_counts + eps) / (pos_counts + eps)  # [C]
    print("pos_weight:", pos_weight)

    # 模型输出 logits，使用 BCEWithLogitsLoss + pos_weight
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

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
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            scheduler.step()
            print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}")
            f.write(f"\nEpoch {epoch}: train_loss={train_loss:.4f}\n")

            # 评估：多标签，模型输出 logits
            report = evaluate(
                model, val_loader, device,
                threshold=0.5, use_logits=True
            )
            print(report)
            f.write(report + "\n")

            # 保存最佳模型：取 weighted avg 的 F1
            f1 = _extract_weighted_f1_from_report(report)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(
                    model.state_dict(),
                    os.path.join(save_root, "best_model.pt")
                )
                print("Saved new best model!\n")
                f.write("Saved new best model!\n\n")

            f.flush()


if __name__ == '__main__':
    train()
