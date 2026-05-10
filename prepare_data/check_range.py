import os
import json
import pandas as pd


def detect_and_clean_outliers(directory, output_json):
    """
    遍历目录下所有CSV文件，删除异常值行，并记录全局最值。
    :param directory: 需要处理的目录
    :param output_json: 记录全局最值的JSON文件路径
    """
    feature_columns = ["lat", "lon", "measured_flight_level", "vx", "vy", "rocd"]
    global_min_max = {col: {"min": float("inf"), "max": float("-inf")} for col in feature_columns}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # 计算均值和标准差
                stats = df[feature_columns].describe()
                means = stats.loc["mean"]
                stds = stats.loc["std"]

                # 3σ异常值检测
                lower_bounds = means - 3 * stds
                upper_bounds = means + 3 * stds

                # 物理范围约束
                physical_limits = {
                    "lat": (-90, 90),
                    "lon": (-180, 180),
                }

                # 过滤异常值
                for col in feature_columns:
                    if col in physical_limits:
                        min_limit, max_limit = physical_limits[col]
                        df = df[(df[col] >= min_limit) & (df[col] <= max_limit)]
                        continue

                    df = df[(df[col] >= lower_bounds[col]) & (df[col] <= upper_bounds[col])]

                # 更新全局最小值和最大值
                for col in feature_columns:
                    global_min_max[col]["min"] = min(global_min_max[col]["min"], df[col].min())
                    global_min_max[col]["max"] = max(global_min_max[col]["max"], df[col].max())

                # 保存清理后的数据
                # df.to_csv(file_path, index=False)

    # 确保所有数据类型可以被 JSON 序列化
    for col in global_min_max:
        global_min_max[col]["min"] = float(global_min_max[col]["min"])
        global_min_max[col]["max"] = float(global_min_max[col]["max"])

    # 保存全局最值到 JSON
    with open(output_json, "w") as f:
        json.dump(global_min_max, f, indent=4)

    print(f"数据清理完成，最值已保存到 {output_json}")

if __name__ == "__main__":

    # detect_and_clean_outliers('/home/userdata/2024_hyn/dataset/SCAT/SCAT_csv/', 'max&min.json')

    detect_and_clean_outliers('/home/userdata/2024_hyn/dataset/SCAT_segments/', 'max&min_segments.json')
