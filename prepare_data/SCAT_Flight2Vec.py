import os
import glob
import pandas as pd

# 单位换算
FT_TO_M = 0.3048
HFT_TO_M = 100 * FT_TO_M            # 百英尺 -> 米  = 30.48
HFTMIN_TO_MS = 100 * FT_TO_M / 60.0 # 百英尺/分钟 -> 米/秒 ≈ 0.508

def main(INPUT_DIR, OUTPUT_CSV):
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        print("未在目录中找到任何 csv 文件：", INPUT_DIR)
        return

    all_dfs = []

    for path in csv_files:
        print("处理文件:", path)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"读取失败，跳过 {path}，错误：{e}")
            continue

        # 取文件名作为 Callsign（去掉扩展名）
        callsign = os.path.splitext(os.path.basename(path))[0]

        # 检查必要列
        required_cols = ["time_of_track", "lat", "lon", "measured_flight_level", "vx", "vy"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"文件 {path} 缺失列 {missing}，跳过")
            continue

        # 构造统一的输出 DataFrame
        out = pd.DataFrame()
        out["Time"] = df["time_of_track"]
        out["Callsign"] = callsign
        out["longitude"] = df["lon"]
        out["latitude"] = df["lat"]
        # measured_flight_level: 百英尺 -> 米
        out["altitude"] = df["measured_flight_level"] * HFT_TO_M
        out["spdx"] = df["vx"]
        out["spdy"] = df["vy"]

        # 处理垂直速度 spdz：优先 rocd，其次 cocd
        vert_src = None
        if "rocd" in df.columns:
            vert_src = df["rocd"]
        elif "cocd" in df.columns:
            vert_src = df["cocd"]

        if vert_src is not None:
            out["spdz"] = vert_src * HFTMIN_TO_MS   # 百英尺/分钟 -> 米/秒
        else:
            # 如果既没有 rocd 也没有 cocd，就填 0 或者 NaN，看你需求
            out["spdz"] = 0.0

        all_dfs.append(out)

    if not all_dfs:
        print("没有任何文件成功转换，退出。")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print("合并完成，保存到：", OUTPUT_CSV)
    print("合并后的行数:", len(merged))

if __name__ == "__main__":
    # ======= 配置部分 =======
    # 输入目录：放所有单航迹 csv 的文件夹
    INPUT_DIR = r"/home/h3c/dataset/SCAT/train/1/"  # TODO: 改成你的目录
    # 输出合并后的 csv 路径
    OUTPUT_CSV = r"/home/h3c/project/python/Flight2Vec/data/aircraft.csv"

    main(INPUT_DIR, OUTPUT_CSV)
