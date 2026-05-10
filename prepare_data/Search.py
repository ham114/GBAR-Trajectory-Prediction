import os, glob
import pandas as pd
import numpy as np

def _normalize_cell(x):
    # 统一到字符串键：去空格 + 小写
    if pd.isna(x):
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    # 数值 → 紧凑字符串
    if isinstance(x, (int, float)):
        # 避免 1.0 与 1 不一致
        return str(int(x)) if float(x).is_integer() else ("%.12g" % float(x))
    # 其他 → 去空格 + 小写
    s = str(x).strip().lower()
    # 常见真值/假值映射
    if s in {"true","t","yes","y","1"}:
        return "true"
    if s in {"false","f","no","n","0"}:
        return "false"
    return s

def _normalize_target(value):
    # 目标值也规范化
    return _normalize_cell(value)

def find_value_rows_in_csvs(
    root_dir: str,
    column: str,
    value,
    csv_pattern: str = "*.csv",
    chunksize: int = 100_000,
):
    """
    在 root_dir 及其下一层子目录中查找所有 CSV。
    返回 [(file_path, [row_indices...]), ...]；索引为 0 基（不含表头）。
    """
    target_key = _normalize_target(value)

    files = []
    files += glob.glob(os.path.join(root_dir, csv_pattern))
    files += glob.glob(os.path.join(root_dir, "*", csv_pattern))

    results = []
    for fp in files:
        try:
            header_cols = pd.read_csv(fp, nrows=0).columns
            if column not in header_cols:
                continue

            hit_indices = []
            for chunk in pd.read_csv(fp, usecols=[column], chunksize=chunksize):
                keys = chunk[column].map(_normalize_cell)
                mask = keys == target_key
                if mask.any():
                    # chunk.index 是全局行号（0 基）
                    hit_indices.extend(chunk.index[mask].tolist())

            if hit_indices:
                results.append((fp, hit_indices))
        except Exception:
            # 出错文件跳过；需要调试可改为打印异常
            continue
    return results

def stats_value_range_multi(
    root_dir: str,
    columns: list[str],
    csv_pattern: str = "*.csv",
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    遍历 root_dir 及其下一层子目录所有 CSV。
    对 columns 中每个列做数值统计，返回 DataFrame:
      columns: [col, min, max, abs_min, abs_max, count, files_covered]
    说明：
      - 非数值将被忽略（to_numeric(coerce)）
      - 某列若在所有文件都不存在，统计结果为 NaN/0
    """
    # 目标列去重保持顺序
    cols_target = list(dict.fromkeys(columns))

    # 结果累积器
    stats = {
        c: {
            "min": np.inf,
            "max": -np.inf,
            "abs_min": np.inf,
            "abs_max": 0.0,
            "count": 0,
            "files_covered": 0,
        } for c in cols_target
    }

    files  = glob.glob(os.path.join(root_dir, csv_pattern))
    files += glob.glob(os.path.join(root_dir, "*", csv_pattern))

    for fp in files:
        try:
            header = pd.read_csv(fp, nrows=0).columns.tolist()
        except Exception:
            continue

        use_cols = [c for c in cols_target if c in header]
        if not use_cols:
            continue

        try:
            for chunk in pd.read_csv(fp, usecols=use_cols, chunksize=chunksize):
                for c in use_cols:
                    vals = pd.to_numeric(chunk[c], errors="coerce").dropna()
                    if vals.empty:
                        continue
                    st = stats[c]
                    vmin = vals.min()
                    vmax = vals.max()
                    st["min"] = min(st["min"], vmin)
                    st["max"] = max(st["max"], vmax)

                    av = vals.abs()
                    st["abs_min"] = min(st["abs_min"], av.min())
                    st["abs_max"] = max(st["abs_max"], av.max())

                    st["count"] += int(av.size)
            # 若该文件中至少有一列产生了有效数值，则记覆盖文件数
            for c in use_cols:
                if stats[c]["count"] > 0:
                    stats[c]["files_covered"] += 1
        except Exception:
            continue

    # 整理为 DataFrame
    rows = []
    for c in cols_target:
        st = stats[c]
        if st["count"] == 0:
            rows.append({
                "col": c, "min": np.nan, "max": np.nan
            })
        else:
            rows.append({
                "col": c,
                "min": st["min"], "max": st["max"]
            })

    return pd.DataFrame(rows, columns=["col","min","max"])

if __name__ == '__main__':
    # matches = find_value_rows_in_csvs("/home/userdata/2024_hyn/dataset/SCAT/SCAT_csv/", column="am", value='TRUE')
    # for fp, idxs in matches:
    #     print(fp, idxs)

    df_stats = stats_value_range_multi("/home/userdata/2024_hyn/dataset/SCAT/SCAT_csv/", ["lat", "lon", "measured_flight_level", "vx", "vy","rocd"])
    print(df_stats)

"""
                     col           min           max
0                    lat     54.262682     69.038853
1                    lon     10.498563     24.127489
2  measured_flight_level     -3.000000    431.000000
3                     vx   -284.250000    313.500000
4                     vy   -307.500000    302.500000
5                   rocd -48831.250000  10718.750000
"""
