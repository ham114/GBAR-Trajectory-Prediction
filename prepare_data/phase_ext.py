import glob
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict
from scipy.stats import skew, kurtosis
import json
import os


def merge_all_statistics_to_json(
    mean_std_stats: Dict,
    osc_stats: Dict,
    angle_stats: Dict,
    output_path: str = "ph_feature_statistics.json"
):
    """
    将 mean/std、oscillation、circular 三类统计合并保存为 JSON 文件。

    参数
    ----
    mean_std_stats : 均值与标准差
    osc_stats : 偏度、峰度、zcr、sign
    angle_stats : 平均方向、方向一致性 R、方向方差 V
    output_path : 输出 json 路径
    """
    merged_result = {}

    all_phs = set(mean_std_stats.keys()) | set(osc_stats.keys()) | set(angle_stats.keys())

    for ph in sorted(all_phs):
        merged_result[ph] = {}
        all_feats = set()

        if ph in mean_std_stats:
            all_feats |= set(mean_std_stats[ph].keys())
        if ph in osc_stats:
            all_feats |= set(osc_stats[ph].keys())
        if ph in angle_stats:
            all_feats |= set(angle_stats[ph].keys())

        for feat in sorted(all_feats):
            merged_result[ph][feat] = {}

            # Mean & Std
            if feat in mean_std_stats.get(ph, {}):
                mean, std = mean_std_stats[ph][feat]
                merged_result[ph][feat]["mean"] = mean
                merged_result[ph][feat]["std"] = std

            # Skewness & Kurtosis & Oscillation
            if feat in osc_stats.get(ph, {}):
                merged_result[ph][feat].update(osc_stats[ph][feat])

            # Circular stats
            if feat in angle_stats.get(ph, {}):
                merged_result[ph][feat].update(angle_stats[ph][feat])

    # 保存为 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_result, f, indent=2, ensure_ascii=False)

    print(f"✅ 合并统计已保存到 {output_path}")



# ---------- 特征配置 ----------
FEATURE_CFG = {
    "E": {
        "bits": 18,
        "scale": 1000,
        "min": -1650.0,
        "max": 1650.0,
        "delta_lo": -150,
        "delta_hi": 150
    },

    "N": {
        "bits": 18,
        "scale": 1000,
        "min": -1100.0,
        "max": 1100.0,
        "delta_lo": -150,
        "delta_hi": 150
    },

    "RALT": {
        "bits": 16,
        "scale": 1,
        "min": -5.0,
        "max": 5510.0,
        "delta_lo": -500,
        "delta_hi": 500
    },

    "PTCH": {
        "bits": 8,
        "scale": 10,
        "min": -16,
        "max": 21,
        "delta_lo": -1000,
        "delta_hi": 1000
    },

    "ROLL": {
        "bits": 10,
        "scale": 1,
        "min": -45,
        "max": 45,
        "delta_lo": -1000,
        "delta_hi": 1000
    },

    "MH": {
        "bits": 10,
        "scale": 10,
        "min": -180.01,
        "max": 180,
        "delta_lo": -1000,
        "delta_hi": 1000
    },

    "TRK": {
        "bits": 10,
        "scale": 10,
        "min": -180,
        "max": 180,
        "delta_lo": -1000,
        "delta_hi": 1000
    },

    "GS": {
        "bits": 10,
        "scale": 1,
        "min": -1,
        "max": 540,
        "delta_lo": -1000,
        "delta_hi": 1000
    },

    "ALTR": {
        "bits": 9,
        "scale": 1,
        "min": -40,
        "max": 40,
        "delta_lo": -500,
        "delta_hi": 500
    },

    "LONG": {
        "bits": 9,
        "scale": 1,
        "min": -1.1,
        "max": 0.35,
        "delta_lo": -500,
        "delta_hi": 500
    },

    "VRTG": {
        "bits": 8,
        "scale": 1,
        "min": -3.4,
        "max": 2.2,
        "delta_lo": -500,
        "delta_hi": 500
    },

    "LATG": {
        "bits": 9,
        "scale": 1,
        "min": -1.1,
        "max": 0.21,
        "delta_lo": -500,
        "delta_hi": 500
    },
}

# ---------- 主接口 ----------
def collect_diff_intervals(
    directory: str,
    ph_col: str = "PH",
    step: int = 5,
    pattern: str = "*.csv",
    feature_cfg: Dict = FEATURE_CFG,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    遍历目录下所有 CSV，按 PH(4-7) + 特征计算差分区间序列。

    Returns
    -------
    result : dict
        {ph: {feature: numpy.ndarray([...]), ...}, ...}
    """
    # 计算每个特征对应的编码“区间长度”
    bin_size = {
        f: (cfg["max"] - cfg["min"]) / (2 ** cfg["bits"])
        for f, cfg in feature_cfg.items()
    }

    # 先用 list 收集，最后一次性转 numpy
    tmp_store = {ph: defaultdict(list) for ph in [4, 5, 6, 7]}

    # 遍历文件
    for csv_path in glob.glob(os.path.join(directory, pattern)):
        df = pd.read_csv(csv_path)
        # 仅保留 PH ∈ {4,5,6,7}
        df = df[df[ph_col].isin({4, 5, 6, 7})]
        if df.empty:
            continue

        # 按文件内部 PH 分段
        for ph in [4, 5, 6, 7]:
            seg = df[df[ph_col] == ph]
            if len(seg) <= step:
                continue

            for f, cfg in feature_cfg.items():
                # 相隔 step 行差分
                diff_vals = seg[f].iloc[step:].values - seg[f].iloc[:-step].values
                diff_bins = np.rint(diff_vals / bin_size[f]).astype(int)

                # 过滤过大/过小的区间差分
                mask = (diff_bins >= cfg["delta_lo"]) & (diff_bins <= cfg["delta_hi"])
                if mask.any():
                    tmp_store[ph][f].append(diff_bins[mask])

    # 合并同 ph+feature 的多文件结果并转 numpy
    result = {ph: {} for ph in [4, 5, 6, 7]}
    for ph in [4, 5, 6, 7]:
        for f in feature_cfg:
            if tmp_store[ph][f]:                          # 至少有数据
                result[ph][f] = np.concatenate(tmp_store[ph][f])
            else:
                result[ph][f] = np.empty(0, dtype=int)     # 无数据则空数组

    return result

def normalize_diff_bins(
    ph_feature_diffs: Dict[int, Dict[str, np.ndarray]],
    feature_cfg: Dict[str, Dict] = FEATURE_CFG
) -> (Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, np.ndarray]]):
    """
    对每个航段、每个特征的差分整数做：
    - 0-1归一化（norm）
    - -1到1标准化（std）

    返回
    ----
    norm_results : 差分映射到 [0, 1]
    std_results  : 差分映射到 [-1, 1]
    """
    norm_results = {ph: {} for ph in [4, 5, 6, 7]}
    std_results = {ph: {} for ph in [4, 5, 6, 7]}

    for ph in ph_feature_diffs:
        for feat, arr in ph_feature_diffs[ph].items():
            if arr.size == 0:
                norm_arr = np.empty(0)
                std_arr = np.empty(0)
            else:
                delta_lo = feature_cfg[feat]["delta_lo"]
                delta_hi = feature_cfg[feat]["delta_hi"]
                # 0-1 normalization
                norm_arr = (arr - delta_lo) / (delta_hi - delta_lo)
                # -1 to 1 standardization
                std_arr = norm_arr * 2 - 1
            norm_results[ph][feat] = norm_arr
            std_results[ph][feat] = std_arr

    return norm_results, std_results

def compute_mean_std(
    normalized_data: Dict[int, Dict[str, np.ndarray]]
) -> Dict[int, Dict[str, tuple]]:
    """
    对每个航段下每个特征的归一化差分序列，计算均值和标准差

    返回
    ----
    result : dict
        {PH: {feature: (mean, std), ...}, ...}
    """
    result = {ph: {} for ph in normalized_data.keys()}

    for ph in normalized_data:
        for feat, arr in normalized_data[ph].items():
            if arr.size == 0:
                mean, std = np.nan, np.nan
            else:
                mean = float(np.mean(arr))
                std = float(np.std(arr))
            result[ph][feat] = (mean, std)

    return result

def compute_skew_kurtosis_oscillation(
    normalized_data: Dict[int, Dict[str, np.ndarray]],
    selected_features = None
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    针对归一化差分序列（建议使用 std_results），仅对动态变量统计：
    - 偏度（skewness）
    - 峰度（kurtosis）
    - 零交叉率（zero_crossing_rate）
    - 符号一致性比率（sign_consistency）

    参数
    ----
    normalized_data : Dict[PH, Dict[feature, np.ndarray]]
        归一化后的差分数据（通常为 std_results）

    selected_features : List[str]
        要统计的动态类特征，默认是推荐的 8 个

    返回
    ----
    result : dict
        {PH: {feature: {"skew": x, "kurt": y, "zcr": z, "sign": w}, ...}, ...}
    """


    # 默认特征筛选（动态变化率类）
    if selected_features is None:
        selected_features = ["RALT", "PTCH", "ROLL", "GS", "ALTR", "LONG", "VRTG", "LATG"]

    result = {ph: {} for ph in normalized_data.keys()}

    for ph in normalized_data:
        for feat in selected_features:
            arr = normalized_data[ph].get(feat, np.array([]))
            stats = {"skew": np.nan, "kurt": np.nan, "zcr": np.nan, "sign": np.nan}

            if arr.size < 3:
                result[ph][feat] = stats
                continue

            # 偏度 & 峰度
            stats["skew"] = float(skew(arr))
            stats["kurt"] = float(kurtosis(arr, fisher=True))

            # 零交叉与符号一致性
            signs = np.sign(arr)
            zero_crossings = np.sum(signs[1:] * signs[:-1] < 0)
            stats["zcr"] = zero_crossings / (len(signs) - 1)
            stats["sign"] = np.sum(signs[1:] == signs[:-1]) / (len(signs) - 1)

            result[ph][feat] = stats

    return result


def compute_circular_diff_stats(
        original_data: Dict[int, Dict[str, np.ndarray]],
        selected_features = ["MH", "TRK"]
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    针对角度类变量（如 MH, TRK）在 [-1, 1] 归一化后还原角度差分，执行单位圆统计。

    参数
    ----
    original_data : std_results 中原始的差分（建议标准化到 [-1, 1]）
    selected_features : 限定为角度类特征（默认是 ["MH", "TRK"]）

    返回
    ----
    {PH: {feature: {"mean_direction": ..., "R": ..., "V": ...}, ...}, ...}
    """


    if selected_features is None:
        selected_features = ["MH", "TRK"]

    result = {ph: {} for ph in original_data.keys()}

    for ph in original_data:
        for feat in selected_features:
            arr = original_data[ph].get(feat, np.array([]))
            stats = {"mean_direction": np.nan, "R": np.nan, "V": np.nan}

            if arr.size < 2:
                result[ph][feat] = stats
                continue

            # 转回角度单位（从 [-1,1] → [-180°, 180°]）
            delta_theta = arr * 180

            # 单位圆向量化
            cos_vals = np.cos(np.deg2rad(delta_theta))
            sin_vals = np.sin(np.deg2rad(delta_theta))

            r_x = np.mean(cos_vals)
            r_y = np.mean(sin_vals)

            # 平均方向
            mean_direction = np.arctan2(r_y, r_x)  # 单位：弧度
            mean_direction_deg = np.rad2deg(mean_direction)  # 转角度方便解释

            # 方向集中度
            R = np.sqrt(r_x ** 2 + r_y ** 2)
            V = 1 - R

            stats["mean_direction"] = float(mean_direction_deg)
            stats["R"] = float(R)
            stats["V"] = float(V)

            result[ph][feat] = stats

    return result


if __name__ == '__main__':

    dir = '/home/userdata/2024_hyn/dataset/nasa_dashlink/traj_constraint/'

    delta = collect_diff_intervals(dir)

    norm_results, std_results = normalize_diff_bins(delta)

    # print(norm_results, std_results)

    mean_std_stats  = compute_mean_std(norm_results)

    for ph in mean_std_stats:
        print(f"PH={ph}")
        for feat in sorted(mean_std_stats[ph]):
            mean, std = mean_std_stats[ph][feat]
            print(f"  {feat:5s}  Mean: {mean:.4f},  Std: {std:.4f}")

    # 前提：你已经有 std_results
    # std_results = normalize_diff_bins(...)[1]

    osc_stats = compute_skew_kurtosis_oscillation(std_results)

    print(osc_stats)

    angle_stats = compute_circular_diff_stats(std_results)
    print(angle_stats)

merge_all_statistics_to_json(mean_std_stats, osc_stats, angle_stats, "all_stats.json")
