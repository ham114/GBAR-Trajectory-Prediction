import torch
import numpy as np
import pandas as pd

def inspect_pt_feature_ranges(pt_path: str):
    """
    读取保存的 .pt 文件，打印每个特征的最小值和最大值。
    兼容 meta['feature_stats'] 格式。
    """
    obj = torch.load(pt_path, map_location="cpu")
    meta = obj.get("meta", {})
    stats = meta.get("feature_stats", None)

    if stats is None:
        print("⚠️ 未找到 meta['feature_stats']，改为直接计算窗口最值。")
        windows = obj["windows"]
        arr = windows.cpu().numpy()
        feat_min = arr.min(axis=(0, 1))
        feat_max = arr.max(axis=(0, 1))
        cols = meta.get("columns", [f"f{i}" for i in range(len(feat_min))])
        df = pd.DataFrame({"feature": cols, "min": feat_min, "max": feat_max})
        print(df.to_string(index=False))
        return df

    # 从 meta 中直接读取
    rows = []
    for k, v in stats.items():
        rows.append({"feature": k, "min": v["min"], "max": v["max"]})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df

def downsample_pt_by_top_patterns(
    in_pt_path: str,
    out_pt_path: str,
    max_samples: int = 4_000_000,
    top_k: int = 4,
    seed: int = 42,
):
    """
    从 in_pt_path 加载 {windows, labels_phase, meta}，
    对出现次数最多的前 top_k 个多热 pattern 进行随机下采样，
    其它 pattern 全量保留，使总样本数约为 max_samples。
    结果保存到 out_pt_path。
    """
    rng = np.random.default_rng(seed)

    obj = torch.load(in_pt_path)
    windows = obj["windows"]           # [N, L, C]
    labels = obj["labels_phase"]       # [N, 6] 多热
    meta = obj["meta"]

    N, K = labels.shape
    assert K == len(meta["phase_vocab"]), "labels_phase 维度与 phase_vocab 不一致"

    # ---------- 1. 将多热编码压成 0..(2^K-1) 的整数 pattern_id ----------
    labels_np = labels.cpu().numpy().astype(int)   # [N, K]
    # 权重: 例如 K=6 时 [32, 16, 8, 4, 2, 1]，高位是第一类
    weights = (1 << np.arange(K-1, -1, -1)).astype(int)
    pattern_id = (labels_np * weights).sum(axis=1)  # [N]

    # 统计每个 pattern 出现次数
    max_id = int(pattern_id.max())
    counts = np.bincount(pattern_id, minlength=max_id+1)

    # 找出出现次数最多的前 top_k 个 pattern
    nonzero_ids = np.where(counts > 0)[0]
    # (pattern_id, count) 按 count 降序
    top_pairs = sorted(
        [(pid, int(counts[pid])) for pid in nonzero_ids],
        key=lambda x: x[1],
        reverse=True
    )
    top_pairs_k = top_pairs[:top_k]

    print("=== Top patterns before downsampling ===")
    def pid2str(pid):
        return format(pid, f"0{K}b")
    for pid, cnt in top_pairs_k:
        print(f"{pid2str(pid)} : {cnt}  ({cnt / N:6.2%})")

    # ---------- 2. 计算需要保留多少样本 ----------
    top_ids = [pid for pid, _ in top_pairs_k]
    top_total = sum(cnt for _, cnt in top_pairs_k)

    # 其它所有 pattern 的索引（全部保留）
    mask_top = np.isin(pattern_id, top_ids)
    other_idx = np.where(~mask_top)[0]
    N_other = other_idx.size

    print(f"\nTotal windows        : {N}")
    print(f"Top-{top_k} total    : {top_total}")
    print(f"Other patterns total : {N_other}")

    if max_samples <= 0:
        raise ValueError("max_samples 必须为正数")

    if N_other >= max_samples:
        # 其它模式已经 >= 目标数了，只能从所有样本里随机抽 max_samples
        print("\n[警告] 其它模式数量已 >= max_samples，仅做全局随机抽样")
        keep_idx = rng.choice(N, size=max_samples, replace=False)
    else:
        # 为前 top_k 留出的预算
        budget_for_top = max_samples - N_other
        if budget_for_top <= 0:
            raise ValueError("max_samples 太小，连其它模式都容不下")

        print(f"\nBudget for top-{top_k} patterns: {budget_for_top}")

        keep_idx_list = [other_idx]  # 先放入其它模式的全部索引

        # 按各 top pattern 的占比分配预算，然后在各自内部随机采样
        for pid, cnt in top_pairs_k:
            idxs = np.where(pattern_id == pid)[0]
            # 理想抽样数量按比例分配
            k_target = int(round(budget_for_top * (cnt / top_total)))
            # 避免超过当前 pattern 数量
            k_target = min(k_target, idxs.size)
            if k_target <= 0:
                continue
            choose = rng.choice(idxs, size=k_target, replace=False)
            keep_idx_list.append(choose)

        keep_idx = np.concatenate(keep_idx_list)
        # 如果因为 round 误差导致多/少于 max_samples，微调一下
        if keep_idx.size > max_samples:
            keep_idx = rng.choice(keep_idx, size=max_samples, replace=False)
        elif keep_idx.size < max_samples:
            # 随机再补一些（从未选过的索引中抽）
            remaining = max_samples - keep_idx.size
            all_idx = np.arange(N)
            mask_keep = np.zeros(N, dtype=bool)
            mask_keep[keep_idx] = True
            leftover = all_idx[~mask_keep]
            if leftover.size > 0:
                extra = rng.choice(leftover, size=min(remaining, leftover.size), replace=False)
                keep_idx = np.concatenate([keep_idx, extra])

    # ---------- 3. 生成下采样后的数据 ----------
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    keep_idx.sort()

    windows_ds = windows[keep_idx]            # [N_ds, L, C]
    labels_ds = labels[keep_idx]              # [N_ds, K]

    # 同步相关 meta 字段
    win_file_ids = np.array(meta["win_file_ids"])
    win_start_rows = np.array(meta["win_start_rows"])

    meta_ds = dict(meta)  # 浅拷贝
    meta_ds["num_windows"] = int(windows_ds.shape[0])
    meta_ds["win_file_ids"] = win_file_ids[keep_idx].tolist()
    meta_ds["win_start_rows"] = win_start_rows[keep_idx].tolist()

    # 保存
    torch.save(
        {
            "windows": windows_ds,
            "labels_phase": labels_ds,
            "meta": meta_ds,
        },
        out_pt_path,
    )

    print(f"\n[完成] 下采样保存到: {out_pt_path}")
    print(f"原始样本数: {N}  ->  下采样后: {windows_ds.shape[0]}")

    # 简单再打一下新分布
    labels_ds_np = labels_ds.cpu().numpy().astype(int)
    weights = (1 << np.arange(K-1, -1, -1)).astype(int)
    pattern_id_ds = (labels_ds_np * weights).sum(axis=1)
    counts_ds = np.bincount(pattern_id_ds, minlength=max_id+1)

    print("\n=== Pattern distribution after downsampling ===")
    for pid, _ in top_pairs_k:
        cnt_old = counts[pid]
        cnt_new = counts_ds[pid]
        pat_str = pid2str(pid)
        print(f"{pat_str} : {cnt_new:8d} (before {cnt_old})")

if __name__ == "__main__":
    in_pt = "/home/userdata/2024_hyn/dataset/SCAT/Test.pt"
    out_pt = in_pt

    # downsample_pt_by_top_patterns(
    #     in_pt_path=in_pt,
    #     out_pt_path=out_pt,
    #     max_samples=5_000_000,
    #     top_k=4,
    #     seed=42,
    # )

    downsample_pt_by_top_patterns(
        in_pt_path=in_pt,
        out_pt_path=out_pt,
        max_samples=700_000,
        top_k=4,
        seed=42,
    )

    # inspect_pt_feature_ranges("/home/userdata/2024_hyn/dataset/SCAT/Train.pt")


"""
=== Pattern distribution (multi-hot) ===
001000 : 16827541  (51.57%)  -> ['high_cruise']
000010 : 8116554  (24.88%)  -> ['descent']
010000 : 3232848  ( 9.91%)  -> ['climb']
000100 : 2388141  ( 7.32%)  -> ['midlow_level']
001010 : 525820  ( 1.61%)  -> ['high_cruise', 'descent']
011000 : 495505  ( 1.52%)  -> ['climb', 'high_cruise']
000001 : 376851  ( 1.15%)  -> ['approach']
000011 : 264735  ( 0.81%)  -> ['descent', 'approach']
000110 : 242684  ( 0.74%)  -> ['midlow_level', 'descent']
010100 : 114706  ( 0.35%)  -> ['climb', 'midlow_level']
000101 :  31375  ( 0.10%)  -> ['midlow_level', 'approach']
001100 :   7894  ( 0.02%)  -> ['high_cruise', 'midlow_level']
010010 :   2011  ( 0.01%)  -> ['climb', 'descent']
100000 :   1325  ( 0.00%)  -> ['takeoff_initial_climb']
100100 :    390  ( 0.00%)  -> ['takeoff_initial_climb', 'midlow_level']
110000 :    169  ( 0.00%)  -> ['takeoff_initial_climb', 'climb']
100001 :    164  ( 0.00%)  -> ['takeoff_initial_climb', 'approach']

=== Pattern distribution after downsampling ===
001000 :  1066065 (before 16827541)
000010 :   514203 (before 8116554)
010000 :   204809 (before 3232848)
000100 :   151294 (before 2388141)

32295940
"""

"""
              feature         min          max
                  lat   54.262680    69.038857
                  lon   10.498563    24.127489
measured_flight_level  -60.959999 13136.879883
                   vx -284.250000   309.250000
                   vy -307.500000   302.500000
                 rocd -121.919998    54.451248
                 
              feature         min          max
                  lat   54.262981    68.967422
                  lon   10.519195    24.126223
measured_flight_level  -91.440002 13114.019531
                   vx -268.750000   299.000000
                   vy -290.500000   278.750000
                 rocd  -96.012001    33.750252
"""
