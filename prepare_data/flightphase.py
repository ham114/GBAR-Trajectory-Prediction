import numpy as np
import skfuzzy as fuzz
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import os
import math

# margin of outputs rules determin if a state is a valid state
STATE_DIFF_MARGIN = 0.2

# logic states
# 1. 重新定义高度、爬升率、速度的数值范围
alt_range = np.arange(-3 * 100, 431 * 100 + 1, 100)  # 单位转换为英尺
roc_range = np.arange(-7500, 7500, 50)  # 取 ROCD 范围并简化
spd_range = np.arange(0, 390, 5)  # IAS 速度范围

states = np.arange(0, 6, 0.01)

alt_gnd = fuzz.zmf(alt_range, -300, 500)  # 地面高度范围 (-3 到 5 百英尺)
alt_lo = fuzz.gaussmf(alt_range, 10000, 5000)  # 低空 (10,000 英尺)
alt_hi = fuzz.gaussmf(alt_range, 35000, 20000)  # 高空 (35,000 英尺)


roc_zero = fuzz.gaussmf(roc_range, 0, 500)  # 接近 0 的爬升率
roc_plus = fuzz.smf(roc_range, 500, 5000)  # 爬升（500 到 5000 ft/min）
roc_minus = fuzz.zmf(roc_range, -5000, -500)  # 下降（-5000 到 -500 ft/min）


spd_lo = fuzz.zmf(spd_range, 0, 120)  # 低速（45~120 knots）
spd_md = fuzz.gaussmf(spd_range, 200, 80)  # 中速（200 knots）
spd_hi = fuzz.smf(spd_range, 250, 380)  # 高速（250~380 knots）


state_ground = fuzz.gaussmf(states, 1, 0.1)
state_climb = fuzz.gaussmf(states, 2, 0.1)
state_descent = fuzz.gaussmf(states, 3, 0.1)
state_cruise = fuzz.gaussmf(states, 4, 0.1)
state_level = fuzz.gaussmf(states, 5, 0.1)
state_take_off = fuzz.gaussmf(states, 6, 0.1)
state_approach = fuzz.gaussmf(states, 7, 0.1)

state_label_map = {
    1: "GND",  # 地面
    2: "CL",   # 爬升
    3: "DE",   # 下降
    4: "CR",   # 巡航
    5: "LVL",  # 平飞
    6: "TO",   # 起飞
    7: "APP",  # 进近
    8: "NA"    # 未知
}


def fuzzylabels(ts, alts, spds, rocs, twindow=60):
    """
    Fuzzy logic to determine the segments of the flight data
    segments are: ground [GND], climb [CL], descent [DE], cruise [CR].

    Default time window is 60 second.
    """

    if len(set([len(ts), len(alts), len(spds), len(rocs)])) > 1:
        raise RuntimeError("input ts and alts must have same length")

    n = len(ts)

    ts = np.array(ts)
    ts = ts - ts[0]
    idxs = np.arange(0, n)

    alts = UnivariateSpline(ts, alts)(ts)
    spds = UnivariateSpline(ts, spds)(ts)
    rocs = UnivariateSpline(ts, rocs)(ts)

    labels = ["NA"] * n
    twindows = ts // twindow

    for tw in range(0, int(max(twindows))):
        if tw not in twindows:
            continue

        mask = twindows == tw

        idxchk = idxs[mask]
        altchk = alts[mask]
        spdchk = spds[mask]
        rocchk = rocs[mask]

        # mean value or extream value as range
        alt = max(min(np.mean(altchk), alt_range[-1]), alt_range[0])
        spd = max(min(np.mean(spdchk), spd_range[-1]), spd_range[0])
        roc = max(min(np.mean(rocchk), roc_range[-1]), roc_range[0])

        # make sure values are within the boundaries
        alt = max(min(alt, alt_range[-1]), alt_range[0])
        spd = max(min(spd, spd_range[-1]), spd_range[0])
        roc = max(min(roc, roc_range[-1]), roc_range[0])

        alt_level_gnd = fuzz.interp_membership(alt_range, alt_gnd, alt)
        alt_level_lo = fuzz.interp_membership(alt_range, alt_lo, alt)
        alt_level_hi = fuzz.interp_membership(alt_range, alt_hi, alt)

        spd_level_hi = fuzz.interp_membership(spd_range, spd_hi, spd)
        spd_level_md = fuzz.interp_membership(spd_range, spd_md, spd)
        spd_level_lo = fuzz.interp_membership(spd_range, spd_lo, spd)

        roc_level_zero = fuzz.interp_membership(roc_range, roc_zero, roc)
        roc_level_plus = fuzz.interp_membership(roc_range, roc_plus, roc)
        roc_level_minus = fuzz.interp_membership(roc_range, roc_minus, roc)

        rule_ground = min(alt_level_gnd, roc_level_zero, spd_level_lo)
        state_activate_ground = np.fmin(rule_ground, state_ground)

        rule_climb = min(alt_level_lo, roc_level_plus, spd_level_md)
        state_activate_climb = np.fmin(rule_climb, state_climb)

        rule_descent = min(alt_level_lo, roc_level_minus, spd_level_md)
        state_activate_descent = np.fmin(rule_descent, state_descent)

        rule_cruise = min(alt_level_hi, roc_level_zero, spd_level_hi)
        state_activate_cruise = np.fmin(rule_cruise, state_cruise)

        rule_level = min(alt_level_lo, roc_level_zero, spd_level_md)
        state_activate_level = np.fmin(rule_level, state_level)

        rule_takeoff = min(alt_level_gnd, roc_level_plus, spd_level_md)
        state_activate_takeoff = np.fmin(rule_takeoff, state_take_off)

        rule_approaching = min(alt_level_lo, roc_level_minus, max(spd_level_lo, spd_level_md))
        state_activate_approaching = np.fmin(rule_approaching, state_approach)

        aggregated = np.max(
            np.vstack(
                [
                    state_activate_ground,  # 地面
                    state_activate_climb,  # 爬升
                    state_activate_descent,  # 下降
                    state_activate_cruise,  # 巡航
                    state_activate_level,  # 平飞
                    state_activate_takeoff,  # 起飞
                    state_activate_approaching  # 进近
                ]
            ),
            axis=0,
        )

        state_raw = fuzz.defuzz(states, aggregated, "lom")
        state = int(round(state_raw))
        if state > 8:
            state = 8
        if state < 1:
            state = 1

        if len(idxchk) > 0:
            label = state_label_map[state]
            labels[idxchk[0] : (idxchk[-1] + 1)] = [label] * len(idxchk)

    return labels

def process_file(file_path, output_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 确保所需列存在
    required_columns = ["timestamp", "altitude", "groundspeed", "vertical_rate"]
    if not all(col in df.columns for col in required_columns):
        print(f"文件 {file_path} 缺少必要的列，跳过...")
        return

    # 提取所需列的数据
    ts = df["timestamp"].values
    alts = df["altitude"].values
    spds = df["groundspeed"].values
    rocs = df["vertical_rate"].values

    # 调用 fuzzylabels 函数进行标注
    labels = fuzzylabels(ts, alts, spds, rocs)

    # 将标注结果添加到 DataFrame 中
    df["flight_phase"] = labels

    # 保存处理后的文件
    df.to_csv(output_path, index=False)
    print(f"文件 {output_path} 处理完成并保存。")

def process(file_path, output_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 确保所需列存在
    required_columns = ["time_of_track", "measured_flight_level", "vx", "vy", "rocd"]
    if not all(col in df.columns for col in required_columns):
        print(f"文件 {file_path} 缺少必要的列，跳过...")
        return

    # 转换时间戳为时间步
    try:
        # 将时间戳转换为 pandas 的 datetime 类型
        df["time_of_track"] = pd.to_datetime(df["time_of_track"])
        # 将时间戳转化为时间步（以第一个时间戳为起点，单位为秒）
        df["time_of_track"] = (df["time_of_track"] - df["time_of_track"].iloc[0]).dt.total_seconds().astype(int)
    except Exception as e:
        print(f"时间戳转换失败：{e}")
        return

    # 提取所需列的数据
    ts = df["time_of_track"].values  # 已经是整数时间步
    alts = df["measured_flight_level"].values * 100  # 转换飞行高度单位
    spdx = df["vx"].values
    spdy = df["vy"].values
    spds = np.sqrt(spdx ** 2 + spdy ** 2)  # 使用 numpy 进行矢量化计算
    rocs = df["rocd"].values

    # 调用 fuzzylabels 函数进行标注
    labels = fuzzylabels(ts, alts, spds, rocs)

    # 将标注结果添加到 DataFrame 中
    df["flight_phase"] = labels

    # 保存处理后的文件
    df.to_csv(output_path, index=False)
    print(f"文件 {output_path} 处理完成并保存。")

if __name__ == "__main__":
    # 文件夹路径，替换为你的实际路径
    folder = 'scat20161015_20161021'
    folder_path = "E:/datasets/SCAT_csv/" + folder
    output_folder = "E:/datasets/SCAT_marked/" + folder

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹中的所有 CSV 文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            input_file = os.path.join(folder_path, filename)
            output_file = os.path.join(output_folder, filename)  # 输出文件名与原始文件名相同
            process(input_file, output_file)
