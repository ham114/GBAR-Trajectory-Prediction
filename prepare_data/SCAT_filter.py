import torch
import os
import numpy as np
from collections import Counter


def filter_and_reencode_patterns(pt_path, patterns_to_keep):
    """
    只保留指定的模式，删除其他所有模式，并重新编码为独热编码

    Args:
        pt_path: pt文件路径
        patterns_to_keep: 要保留的模式列表，如 ['001000', '000010', '010000', ...]
    """
    # 读取数据
    obj = torch.load(pt_path)
    windows = obj["windows"]
    labels_phase = obj["labels_phase"]
    meta = obj["meta"]
    vocab = meta["phase_vocab"]

    print(f"原始窗口数量: {len(windows)}")

    # 将标签转换为模式字符串
    labels_np = labels_phase.numpy()
    patterns = []
    for i in range(len(labels_np)):
        bits = ''.join(str(int(b)) for b in labels_np[i])
        patterns.append(bits)

    # 创建过滤掩码：只保留指定模式
    keep_mask = [pattern in patterns_to_keep for pattern in patterns]

    # 应用过滤
    filtered_windows = windows[keep_mask]
    filtered_labels = labels_phase[keep_mask]
    filtered_patterns = [p for p, keep in zip(patterns, keep_mask) if keep]

    removed_count = len(windows) - len(filtered_windows)
    print(f"过滤后窗口数量: {len(filtered_windows)}")
    print(f"移除了 {removed_count} 个不匹配指定模式的窗口")

    # ===== 新增：创建新的独热编码 =====
    # 为保留的10种模式创建新的标签映射
    pattern_to_new_label = {pattern: i for i, pattern in enumerate(patterns_to_keep)}

    # 创建新的独热编码标签 [N, 10]
    new_onehot_labels = torch.zeros(len(filtered_windows), len(patterns_to_keep), dtype=torch.long)

    for i, pattern in enumerate(filtered_patterns):
        label_idx = pattern_to_new_label[pattern]
        new_onehot_labels[i, label_idx] = 1

    # 统计新的标签分布
    new_label_counts = new_onehot_labels.sum(dim=0).tolist()

    print("\n【新的独热编码分布】")
    total_kept = len(filtered_windows)
    for i, pattern in enumerate(patterns_to_keep):
        cnt = int(new_label_counts[i])
        phases_in_pattern = [vocab[j] for j, b in enumerate(pattern) if b == '1']
        print(f"Label {i:2d}: {pattern} : {cnt:8d}  ({cnt / total_kept:6.2%})  -> {phases_in_pattern}")

    # 更新meta信息
    meta["num_windows"] = int(filtered_windows.shape[0])
    meta["filtered_patterns"] = patterns_to_keep
    meta["original_num_windows"] = len(windows)
    meta["new_label_mapping"] = pattern_to_new_label
    meta["new_label_vocab"] = [f"pattern_{i}_{pattern}" for i, pattern in enumerate(patterns_to_keep)]
    meta["new_label_descriptions"] = [
        f"{pattern} -> {[vocab[j] for j, b in enumerate(pattern) if b == '1']}"
        for pattern in patterns_to_keep
    ]

    # 覆盖保存，同时保存原始多热编码和新的独热编码
    torch.save({
        "windows": filtered_windows,
        "labels_phase": filtered_labels,  # 原始多热编码 [N, 6]
        "labels_onehot": new_onehot_labels,  # 新的独热编码 [N, 10]
        "meta": meta
    }, pt_path)

    print(f"\n✅ 已覆盖保存过滤后的数据到: {pt_path}")
    print(f"    - 保留了原始多热编码 labels_phase: {filtered_labels.shape}")
    print(f"    - 新增了独热编码 labels_onehot: {new_onehot_labels.shape}")

    return removed_count


def filter_and_reencode_patterns_detailed(pt_path, patterns_to_keep):
    """
    详细版本的特定模式过滤和重新编码函数
    """
    # 读取原始数据
    obj = torch.load(pt_path)
    windows = obj["windows"]
    labels_phase = obj["labels_phase"]
    meta = obj["meta"]
    vocab = meta["phase_vocab"]

    print("=" * 80)
    print(f"处理文件: {pt_path}")
    print("=" * 80)

    # 过滤前的统计
    labels_np = labels_phase.numpy()
    pattern_counter_before = Counter()
    for i in range(len(labels_np)):
        bits = ''.join(str(int(b)) for b in labels_np[i])
        pattern_counter_before[bits] += 1

    print("\n【过滤前模式分布】")
    total_before = len(windows)
    for pattern, cnt in pattern_counter_before.most_common():
        phases_in_pattern = [vocab[j] for j, b in enumerate(pattern) if b == '1']
        pct = cnt / total_before
        mark = "✓" if pattern in patterns_to_keep else "✗"
        print(f"{mark} {pattern} : {cnt:8d}  ({pct:6.2%})  -> {phases_in_pattern}")

    # 创建过滤掩码
    patterns = [''.join(str(int(b)) for b in labels_np[i]) for i in range(len(labels_np))]
    keep_mask = [pattern in patterns_to_keep for pattern in patterns]

    # 应用过滤
    filtered_windows = windows[keep_mask]
    filtered_labels = labels_phase[keep_mask]
    filtered_patterns = [p for p, keep in zip(patterns, keep_mask) if keep]

    # ===== 新增：创建新的独热编码 =====
    pattern_to_new_label = {pattern: i for i, pattern in enumerate(patterns_to_keep)}
    new_onehot_labels = torch.zeros(len(filtered_windows), len(patterns_to_keep), dtype=torch.long)

    for i, pattern in enumerate(filtered_patterns):
        label_idx = pattern_to_new_label[pattern]
        new_onehot_labels[i, label_idx] = 1

    # 过滤后的统计
    print(f"\n【过滤结果】")
    print(f"原始窗口数量: {len(windows)}")
    print(f"保留窗口数量: {len(filtered_windows)}")
    print(f"删除窗口数量: {len(windows) - len(filtered_windows)}")
    print(f"保留比例: {len(filtered_windows) / len(windows):.2%}")

    # 新的独热编码分布
    new_label_counts = new_onehot_labels.sum(dim=0).tolist()

    print("\n【新的独热编码分布】")
    total_after = len(filtered_windows)
    for i, pattern in enumerate(patterns_to_keep):
        cnt = int(new_label_counts[i])
        phases_in_pattern = [vocab[j] for j, b in enumerate(pattern) if b == '1']
        pct = cnt / total_after if total_after > 0 else 0
        print(f"Label {i:2d}: {pattern} : {cnt:8d}  ({pct:6.2%})  -> {phases_in_pattern}")

    # 更新meta信息
    meta["num_windows"] = int(filtered_windows.shape[0])
    meta["filtered_patterns"] = patterns_to_keep
    meta["original_num_windows"] = len(windows)
    meta["new_label_mapping"] = pattern_to_new_label
    meta["new_label_vocab"] = [f"pattern_{i}" for i in range(len(patterns_to_keep))]
    meta["new_label_descriptions"] = [
        f"{pattern} -> {[vocab[j] for j, b in enumerate(pattern) if b == '1']}"
        for pattern in patterns_to_keep
    ]
    meta["filter_date"] = "2024-11-15"

    # 覆盖保存
    torch.save({
        "windows": filtered_windows,
        "labels_phase": filtered_labels,  # 原始多热编码
        "labels_onehot": new_onehot_labels,  # 新的独热编码
        "meta": meta
    }, pt_path)

    print(f"\n✅ 已成功覆盖保存到: {pt_path}")
    print(f"📊 标签信息:")
    print(f"   - labels_phase (原始多热): {filtered_labels.shape} -> 6个航段的多热编码")
    print(f"   - labels_onehot (新独热): {new_onehot_labels.shape} -> 10个类别的独热编码")

    return len(windows) - len(filtered_windows)


# 验证函数
def verify_pt_file(pt_path):
    """
    验证处理后的pt文件
    """
    print("\n" + "=" * 50)
    print("验证处理结果")
    print("=" * 50)

    obj = torch.load(pt_path)
    windows = obj["windows"]
    labels_phase = obj["labels_phase"]  # 原始多热
    labels_onehot = obj["labels_onehot"]  # 新独热
    meta = obj["meta"]

    print(f"窗口数量: {len(windows)}")
    print(f"原始多热标签形状: {labels_phase.shape}")
    print(f"新独热标签形状: {labels_onehot.shape}")

    # 验证新标签是否正确
    labels_np = labels_phase.numpy()
    patterns = []
    for i in range(len(labels_np)):
        bits = ''.join(str(int(b)) for b in labels_np[i])
        patterns.append(bits)

    # 检查是否所有模式都在允许的列表中
    unique_patterns = set(patterns)
    allowed_patterns = set(meta["filtered_patterns"])

    print(f"\n验证结果:")
    print(f"唯一模式数量: {len(unique_patterns)}")
    print(f"允许的模式数量: {len(allowed_patterns)}")
    print(f"是否有不允许的模式: {bool(unique_patterns - allowed_patterns)}")

    # 检查独热编码是否正确
    correct_count = 0
    for i, pattern in enumerate(patterns):
        expected_label_idx = meta["new_label_mapping"][pattern]
        if labels_onehot[i, expected_label_idx] == 1:
            correct_count += 1

    accuracy = correct_count / len(patterns) if len(patterns) > 0 else 0
    print(f"独热编码正确率: {accuracy:.4f} ({correct_count}/{len(patterns)})")


# 使用示例
if __name__ == '__main__':
    pt_path = '/home/h3c/dataset/SCAT/valid.pt'

    # 定义要保留的10种模式
    patterns_to_keep = [
        '001000',  # high_cruise
        '000010',  # descent
        '010000',  # climb
        '000100',  # midlow_level
        '001010',  # high_cruise, descent
        '011000',  # climb, high_cruise
        '000001',  # approach
        '000011',  # descent, approach
        '000110',  # midlow_level, descent
        '010100',  # climb, midlow_level
    ]

    # 执行过滤和重新编码
    filter_and_reencode_patterns_detailed(pt_path, patterns_to_keep)

    # 验证结果
    verify_pt_file(pt_path)

