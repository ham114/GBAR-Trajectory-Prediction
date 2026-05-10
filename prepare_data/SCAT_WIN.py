import os
import glob
import time
import pandas as pd
import numpy as np
import torch
import json
from collections import Counter

def dict_to_obj(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, dict_to_obj(j))
        elif isinstance(j, seqs):
            setattr(top, i,
                    type(j)(dict_to_obj(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top


def load_config_from_json(json_path='config.json'):
    with open(json_path, 'r') as fr:
        data = json.load(fr)
    return dict_to_obj(data)


# 六类与编码
PHASES = [
    "takeoff_initial_climb",
    "climb",
    "high_cruise",
    "midlow_level",
    "descent",
    "approach",
]
PHASE2ID = {p: i for i, p in enumerate(PHASES)}

FT_TO_M = 0.3048
FPM_TO_MPS = 0.00508

vx_range = (-500, 500)
vy_range = (-500, 500)
rocd_range = (-150, 150)


def dict_to_obj(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, dict_to_obj(j))
        elif isinstance(j, seqs):
            setattr(top, i,
                    type(j)(dict_to_obj(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top


def load_config_from_json(json_path='config.json'):
    with open(json_path, 'r') as fr:
        data = json.load(fr)
    return dict_to_obj(data)


def build_windows_from_dir(
    root_dir: str,
    out_pt: str,
    columns=["lat", "lon", "measured_flight_level", "vx", "vy", "rocd"],
    window_len: int = 128,
    stride: int = 16,
    sample_freq: int = 1,
    csv_pattern: str = "*.csv",
    dropna: bool = True,
    dtype: str = "float32",
    # ==== 新增 ====
    phase_col: str = "phase",     # 航段列名
    pred_len: int = 16,           # 用窗口末尾多少个点来统计标签
    phase_ratio: float = 0.3,     # 某一类在预测段中占比超过该比例就置 1
):
    """
    从 root_dir 的一级子目录中收集所有 CSV 轨迹，按滑动窗口采样，并保存为 .pt。

    多热标签:
    - 每个窗口生成 6 维多热标签 labels_phase[i] ∈ {0,1}^6
      对于第 k 维，对应航段 PHASES[k]:
      如果在窗口“预测段”（末尾 pred_len 个点）中出现比例 >= phase_ratio，则置 1。
      若所有比例都 < phase_ratio，则将出现次数最多的航段对应位强制置 1（避免全 0）。
    """
    root_dir = os.path.abspath(root_dir)
    assert os.path.isdir(root_dir), f"目录不存在: {root_dir}"
    assert window_len > 0 and stride > 0 and sample_freq > 0, "window_len/stride/sample_freq 必须为正整数"
    assert 1 <= pred_len <= window_len, "pred_len 必须在 [1, window_len]"
    assert 0.0 < phase_ratio <= 1.0, "phase_ratio 必须在 (0,1]"


    def _read_df(csv_path):
        """容错读取：rcod 与 rocd 兼容；同时读取 phase 列。"""
        try_cols = list(required_cols)
        usecols_1 = try_cols + [phase_col]
        try:
            return pd.read_csv(csv_path, usecols=usecols_1)
        except ValueError as e1:
            # 如果要求里有 rcod，则尝试用 rocd 替代
            if "rcod" in try_cols:
                alt_cols = [("rocd" if c == "rcod" else c) for c in try_cols]
                usecols_2 = alt_cols + [phase_col]
                try:
                    df = pd.read_csv(csv_path, usecols=usecols_2)
                    if "rocd" in df.columns and "rcod" not in df.columns:
                        df = df.rename(columns={"rocd": "rcod"})
                    return df
                except Exception:
                    raise e1
            else:
                raise e1

    required_cols = list(columns)
    raw_span = window_len * sample_freq
    num_drop_dist = 0

    try:
        lat_idx = required_cols.index("lat")
        lon_idx = required_cols.index("lon")
    except ValueError:
        raise ValueError("columns 中必须包含 'lat' 和 'lon' 用于距离筛选")

    R_EARTH = 6371000.0  # 地球半径，单位：m

    base_dt = 10.0       # 相邻原始轨迹点时间间隔 10 s
    v_limit = 320.0      # m/s，民航客机速度上限 + 裕度
    dist_thresh_m = 4000
    # 例如 sample_freq=1 时 ~3200 m，可按需改成显式 4000.0
    # dist_thresh_m = 4000.0 * sample_freq

    all_windows = []
    win_file_ids = []
    win_start_rows = []
    phase_labels_bits = []   # [N, 6] 多热编码
    file_list = []

    # 只遍历一级子目录
    subdirs = [d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]
    if not subdirs:
        raise ValueError("未发现任何一级子目录。")

    file_id_counter = 0
    for sd in sorted(subdirs):
        csvs = sorted(glob.glob(os.path.join(sd, csv_pattern)))
        for csv_path in csvs:
            try:
                df_full = _read_df(csv_path)
            except Exception as e:
                print(f"[跳过] {csv_path}: 读取失败或缺列 {e}")
                continue

            T_raw = len(df_full)

            # 数值化 required_cols，phase 保持为字符串
            for c in required_cols:
                if c in df_full.columns:
                    df_full[c] = pd.to_numeric(df_full[c], errors="coerce")

            # 单位转换
            # 1) 单位转换
            if "measured_flight_level" in df_full.columns:
                df_full["measured_flight_level"] = df_full["measured_flight_level"] * 100.0 * FT_TO_M
            if "rocd" in df_full.columns:
                df_full["rocd"] = df_full["rocd"] * FPM_TO_MPS

            # 丢 NaN：仅依据数值列
            if dropna:
                mask = df_full[required_cols].notna().all(axis=1)
                df_full = df_full.loc[mask]

            if df_full.empty:
                print(f"[跳过] {csv_path}: 数据为空 (原始行数={T_raw})")
                continue

            # 拆出数值矩阵与 phase 序列（保持同长度）
            df_vals = df_full[required_cols]
            phase_seq = df_full[phase_col].astype(str).fillna("")

            arr = df_vals.to_numpy()
            T, C = arr.shape

            if T < raw_span:
                print(f"[跳过] {csv_path}: 有效长度 {T} < raw_span {raw_span} (原始行数={T_raw})")
                continue

            starts = range(0, T - raw_span + 1, stride)

            num_candidates = 0         # 候选窗口数
            num_drop_range = 0         # 因 vx/vy/rocd 超界被丢弃
            num_bad_len = 0            # win.shape[0] != window_len
            num_kept = 0               # 最终保留

            for s in starts:
                num_candidates += 1
                # 下采样窗口
                win = arr[s: s + raw_span: sample_freq]
                # 对应 phase 窗口
                ph_win = phase_seq.iloc[s: s + raw_span: sample_freq].tolist()

                # 预测段 = 窗口末尾 pred_len 个点
                tail = ph_win[-pred_len:]
                if len(tail) == 0:
                    # 理论上不会发生，因为 pred_len <= window_len 且 raw_span 对应 window_len 个点
                    continue

                cnt = Counter(tail)

                bits = [0] * len(PHASES)
                total = len(tail)

                # 按比例置 1
                for i, ph in enumerate(PHASES):
                    if ph in cnt and cnt[ph] / total >= phase_ratio:
                        bits[i] = 1

                # 如果所有位都是 0，则强制将出现次数最多的那一类置 1
                if sum(bits) == 0:
                    top_phase = None
                    top_num = -1
                    for i, ph in enumerate(PHASES):
                        n = cnt.get(ph, 0)
                        if n > top_num:
                            top_num = n
                            top_phase = ph
                    if top_phase is not None:
                        bits[PHASE2ID[top_phase]] = 1
                    else:
                        # tail 里根本没有合法 phase，跳过此窗口
                        continue

                # --- 新增：按经纬度距离筛选窗口 ---
                # win 形状 [window_len, C]，lat/lon 为度
                lat = win[:, lat_idx]
                lon = win[:, lon_idx]

                # 转弧度
                lat_rad = np.deg2rad(lat)
                lon_rad = np.deg2rad(lon)

                # 相邻点差分
                dlat = lat_rad[1:] - lat_rad[:-1]
                dlon = lon_rad[1:] - lon_rad[:-1]

                # Haversine 公式计算每一步的大圆距离 [window_len-1]
                a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon / 2.0) ** 2
                c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
                step_dists = R_EARTH * c  # 单位 m

                # 若任一 10*sample_freq 秒步长的距离超过阈值，则认为该窗口不可信
                if np.any(step_dists > dist_thresh_m):
                    num_drop_dist += 1
                    continue

                # --- 原有范围过滤 ---
                try:
                    vx = win[:, required_cols.index("vx")]
                    vy = win[:, required_cols.index("vy")]
                    rc_idx = required_cols.index("rocd") if "rocd" in required_cols else required_cols.index("rcod")
                    rc = win[:, rc_idx]
                except Exception:
                    # 列名配置不一致时这里会出错；先不计入 range 过滤
                    pass
                else:
                    if (
                            (vx.min() < vx_range[0] or vx.max() > vx_range[1]) or
                            (vy.min() < vy_range[0] or vy.max() > vy_range[1]) or
                            (rc.min() < rocd_range[0] or rc.max() > rocd_range[1])
                    ):
                        num_drop_range += 1
                        continue

                all_windows.append(win.astype(dtype, copy=False))
                win_file_ids.append(file_id_counter)
                win_start_rows.append(s)
                phase_labels_bits.append(bits)
                num_kept += 1

            if num_kept > 0:
                file_list.append(csv_path)
                file_id_counter += 1
            else:
                print(
                    f"[文件无窗口] {csv_path} | "
                    f"原始行数={T_raw}, 有效行数={T}, "
                    f"候选窗口={num_candidates}, "
                    f"range过滤={num_drop_range}, "
                    f"长度不符={num_bad_len}, "
                    f"保留=0"
                )

    if not all_windows:
        raise ValueError("没有生成任何窗口，检查参数或数据。")

    windows_np = np.stack(all_windows, axis=0)   # [N, L, C]
    windows = torch.from_numpy(windows_np)       # [N, L, C]
    labels_mh = torch.tensor(phase_labels_bits, dtype=torch.long)  # [N, 6] 多热

    # 统计各特征全局最小/最大值
    feat_min = windows_np.min(axis=(0, 1)).tolist()
    feat_max = windows_np.max(axis=(0, 1)).tolist()
    feature_stats = {col: {"min": float(mn), "max": float(mx)}
                     for col, mn, mx in zip(required_cols, feat_min, feat_max)}

    meta = {
        "source_root": root_dir,
        "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "columns": required_cols,
        "units": {
            "lat": "degree",
            "lon": "degree",
            "measured_flight_level": "meter",
            "vx": "m/s",
            "vy": "m/s",
            "rcod": "m/s",
        },
        "window_len": window_len,
        "sample_freq": sample_freq,
        "stride": stride,
        "num_windows": int(windows.shape[0]),
        "num_features": int(windows.shape[2]),
        "files": file_list,
        "win_file_ids": win_file_ids,
        "win_start_rows": win_start_rows,
        "dtype": dtype,
        "raw_span": raw_span,
        "feature_stats": feature_stats,
        # 标签相关
        "phase_col": phase_col,
        "pred_len": pred_len,
        "phase_vocab": PHASES,
        "phase_ratio": phase_ratio,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_pt)), exist_ok=True)
    torch.save({"windows": windows,
                "labels_phase": labels_mh,   # [N,6] 多热
                "meta": meta}, out_pt)

    print(f"[完成] 保存 {windows.shape[0]} 个窗口到: {out_pt}")
    print("[特征最值]")
    for k in required_cols:
        s = feature_stats[k]
        print(f"  {k:>24s} | min={s['min']:.6g}  max={s['max']:.6g}")
    print(f"[标签] pred_len={pred_len}  phase_ratio={phase_ratio}  vocab={PHASES}  计数={labels_mh.shape[0]}")

    return out_pt, meta, num_drop_dist

if __name__ == '__main__':

    config = load_config_from_json('/home/h3c/project/python/FlightGPT/config_v24.json')

    root = '/home/h3c/dataset/SCAT/train/'
    out = '/home/h3c/dataset/SCAT/train_f2v.pt'

    _, _, drop = build_windows_from_dir(root,
                                        out,
                                        window_len=config.inp_seq_len + config.horizon,
                                        stride=2,
                                        sample_freq=config.data_period,
                                        phase_col="phase",
                                        columns=config.features,
                                        pred_len=config.horizon,
                                        phase_ratio=0.4)

    print("Windows dropout: ", drop)

    obj = torch.load(out)

    labels = obj["labels_phase"]  # [N, 6] 多热
    meta = obj["meta"]
    vocab = meta["phase_vocab"]  # ['takeoff_initial_climb', ...]
    N, K = labels.shape

    # ========== 1) 单航段计数（你已有的） ==========
    counts_per_phase = labels.sum(dim=0).tolist()

    print("=== Phase window counts (multi-hot) ===")
    for idx, name in enumerate(vocab):
        n = int(counts_per_phase[idx])
        print(f"{name:25s} : {n:6d}")
    print(f"Total windows: {N}")

    print("\n--- Window-level percentage (multi-hot) ---")
    for idx, name in enumerate(vocab):
        n = int(counts_per_phase[idx])
        print(f"{name:25s} : {n:6d}  ({n / N:6.2%})")

    # ========== 2) 单独出现分布（只含这一类，其他全 0） ==========
    labels_np = labels.cpu().numpy()
    single_counts = [0] * K
    for i in range(N):
        row = labels_np[i]
        s = row.sum()
        if s == 1:
            k = int(row.argmax())
            single_counts[k] += 1

    print("\n=== Single-phase only windows (exactly one 1) ===")
    total_single = sum(single_counts)
    for idx, name in enumerate(vocab):
        n = single_counts[idx]
        pct = n / N if N > 0 else 0.0
        pct_single = n / total_single if total_single > 0 else 0.0
        print(f"{name:25s} : {n:6d}  "
              f"(of all: {pct:6.2%}, of singles: {pct_single:6.2%})")
    print(f"Total single-phase windows: {total_single}")

    # ========== 3) 组合模式分布（模式字符串 + 对应航段集合） ==========
    pattern_counter = Counter()

    for i in range(N):
        bits = ''.join(str(int(b)) for b in labels_np[i])
        pattern_counter[bits] += 1

    print("\n=== Pattern distribution (multi-hot) ===")
    # 按出现次数从多到少排序
    for pattern, cnt in pattern_counter.most_common():
        phases_in_pattern = [
            vocab[j] for j, b in enumerate(pattern) if b == '1'
        ]
        pct = cnt / N if N > 0 else 0.0
        print(f"{pattern} : {cnt:6d}  ({pct:6.2%})  -> {phases_in_pattern}")

"""
Label  0: 001000 :  2038269  (50.83%)  -> ['high_cruise']
Label  1: 000010 :  1001071  (24.97%)  -> ['descent']
Label  2: 010000 :   399919  ( 9.97%)  -> ['climb']
Label  3: 000100 :   320507  ( 7.99%)  -> ['midlow_level']
Label  4: 001010 :    65406  ( 1.63%)  -> ['high_cruise', 'descent']
Label  5: 011000 :    61117  ( 1.52%)  -> ['climb', 'high_cruise']
Label  6: 000001 :    44182  ( 1.10%)  -> ['approach']
Label  7: 000011 :    33708  ( 0.84%)  -> ['descent', 'approach']
Label  8: 000110 :    30065  ( 0.75%)  -> ['midlow_level', 'descent']
Label  9: 010100 :    15488  ( 0.39%)  -> ['climb', 'midlow_level']

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
