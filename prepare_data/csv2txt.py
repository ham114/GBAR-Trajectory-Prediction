import os
import glob
import pandas as pd

def extract_and_save_columns(csv_dir, txt_dir):
    # 确保输出目录存在
    os.makedirs(txt_dir, exist_ok=True)

    # 指定字段
    columns_to_extract = ["E", "N", "RALT", "PTCH", "ROLL", "MH", "TRK", "GS", "ALTR", "LONG", "VRTG", "LATG", "PH"]

    # 初始化范围统计字典
    min_dict = {col: float('inf') for col in columns_to_extract if col != "PH"}
    max_dict = {col: float('-inf') for col in columns_to_extract if col != "PH"}
    ph_counter = {}

    csv_paths = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"目录 {csv_dir} 下未找到 *.csv 文件")

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        # 确保所需列存在
        if not set(columns_to_extract).issubset(df.columns):
            print(f"[!] 跳过文件 {os.path.basename(csv_path)}，缺少必要字段")
            continue

        df = df[columns_to_extract].copy()

        if "ALTR" in df.columns:
            df["ALTR"] = df["ALTR"] * 0.00508

        # 替换 PH == 3 为 4
        df["PH"] = df["PH"].apply(lambda x: 4 if x == 3 else x)

        # 更新统计范围
        for col in columns_to_extract:
            if col != "PH":
                min_dict[col] = min(min_dict[col], df[col].min())
                max_dict[col] = max(max_dict[col], df[col].max())

        # 统计 PH 频数
        ph_counts = df["PH"].value_counts().to_dict()
        for k, v in ph_counts.items():
            ph_counter[k] = ph_counter.get(k, 0) + v

        # 保存为 .txt
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        txt_path = os.path.join(txt_dir, f"{base_name}.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                row_str = "|".join(str(row[col]) for col in columns_to_extract)
                f.write(row_str + "\n")

        print(f"[✓] 已导出 {base_name}.txt")

    # 打印最终统计
    print("\n====== 属性变化范围（除 PH） ======")
    for col in columns_to_extract:
        if col != "PH":
            print(f"{col:<5}: min = {min_dict[col]:.6f}, max = {max_dict[col]:.6f}")

    print("\n====== 飞行阶段 PH 统计 ======")
    for ph in sorted(ph_counter):
        print(f"PH = {ph}: {ph_counter[ph]} 条记录")

if __name__ == "__main__":
    csv_input_dir = "/home/userdata/2024_hyn/dataset/nasa_dashlink/flidata/"       # ← 修改为CSV输入目录
    txt_output_dir = "/home/userdata/2024_hyn/dataset/nasa_dashlink/new_txt/"      # ← 修改为TXT输出目录
    extract_and_save_columns(csv_input_dir, txt_output_dir)

"""
====== 属性变化范围（除 PH） ======
E    : min = -1612.615442, max = 1626.615875
N    : min = -1045.914430, max = 1066.611382
RALT : min = -3.750000, max = 5506.000000
PTCH : min = -15.334996, max = 20.589383
ROLL : min = -44.940751, max = 44.906152
MH   : min = -180.003024, max = 179.995507
TRK  : min = -179.999132, max = 179.995334
GS   : min = -0.072538, max = 537.008500
ALTR : min = -37.428182, max = 38.032302
LONG : min = -1.083489, max = 0.345359
VRTG : min = -3.374457, max = 2.121780
LATG : min = -1.083003, max = 0.201586

====== 飞行阶段 PH 统计 ======
PH = 4: 2629160 条记录
PH = 5: 3610476 条记录
PH = 6: 2591616 条记录
PH = 7: 54012 条记录
"""