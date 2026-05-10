import os
import glob
import math
import pandas as pd
from collections import Counter

# WGS-84 基准椭球参数
_A  = 6378137.0          # 长半轴 (m)
_F  = 1 / 298.257223563  # 扁率
_E2 = _F * (2 - _F)      # 第一偏心率平方

def geodetic_to_ecef(lat_rad: float, lon_rad: float, alt_m: float):
    """大地坐标 → ECEF"""
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    N = _A / math.sqrt(1 - _E2 * sin_lat**2)  # 卯酉圈曲率半径
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1 - _E2) + alt_m) * sin_lat
    return x, y, z

def ecef_to_enu(x, y, z, x0, y0, z0, lat0_rad, lon0_rad):
    """ECEF → ENU（以首个采样点为原点）"""
    dx, dy, dz = x - x0, y - y0, z - z0
    sin_lat0, cos_lat0 = math.sin(lat0_rad), math.cos(lat0_rad)
    sin_lon0, cos_lon0 = math.sin(lon0_rad), math.cos(lon0_rad)

    t = -sin_lon0 * dx +  cos_lon0 * dy      # East
    e = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz  # North
    u =  cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz  # Up
    return t, e, u

def process_directory(root_dir: str):
    # 统计量
    phase_counter = Counter()
    e_min = n_min = u_min = float("inf")
    e_max = n_max = u_max = float("-inf")

    csv_paths = glob.glob(os.path.join(root_dir, "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"目录 {root_dir} 下未找到 *.csv 文件")

    for path in csv_paths:
        df = pd.read_csv(path)

        # ----- 1. 选取轨迹起点作为本文件的 ENU 原点 -----
        lat0_deg, lon0_deg, alt0_m = df.loc[0, ["LATP", "LONP", "RALT"]]
        lat0_rad, lon0_rad = math.radians(lat0_deg), math.radians(lon0_deg)
        x0, y0, z0 = geodetic_to_ecef(lat0_rad, lon0_rad, alt0_m)

        # ----- 2. 逐行转换 -----
        e_list, n_list, u_list = [], [], []
        for lat, lon, alt in zip(df["LATP"], df["LONP"], df["RALT"]):
            lat_rad, lon_rad = math.radians(lat), math.radians(lon)
            x, y, z = geodetic_to_ecef(lat_rad, lon_rad, alt)
            east, north, up = ecef_to_enu(x, y, z, x0, y0, z0, lat0_rad, lon0_rad)
            # 单位换算：米 → 千米
            e_list.append(east / 1000.0)
            n_list.append(north / 1000.0)
            u_list.append(up    / 1000.0)

        df["E"] = e_list
        df["N"] = n_list
        df["U"] = u_list

        # ----- 3. 更新全局统计 -----
        phase_counter.update(df["PH"])
        e_min, n_min, u_min = min(e_min, df["E"].min()), min(n_min, df["N"].min()), min(u_min, df["U"].min())
        e_max, n_max, u_max = max(e_max, df["E"].max()), max(n_max, df["N"].max()), max(u_max, df["U"].max())

        # ----- 4. 覆盖保存 -----
        df.to_csv(path, index=False)
        print(f"[✓] 已处理并覆盖保存 {os.path.basename(path)}")

    # ----- 5. 汇总输出 -----
    print("\n====== 统计结果 ======")
    print("飞行阶段 PH 统计:")
    for ph, cnt in phase_counter.most_common():
        print(f"  PH = {ph}: {cnt} 条记录")

    print("\nENU 坐标变化范围（单位：千米）:")
    print(f"  E: {e_min:.3f}  ~ {e_max:.3f}")
    print(f"  N: {n_min:.3f}  ~ {n_max:.3f}")
    print(f"  U: {u_min:.3f}  ~ {u_max:.3f}")

# ===== 主调用 =====
if __name__ == "__main__":
    dir_path = "/home/userdata/2024_hyn/dataset/nasa_dashlink/flidata/"   #  ← 修改为实际路径
    process_directory(dir_path)


"""
====== 统计结果 ======
飞行阶段 PH 统计:
  PH = 5: 3610476 条记录
  PH = 4: 2629044 条记录
  PH = 6: 2591616 条记录
  PH = 7: 54012 条记录
  PH = 3: 116 条记录

ENU 坐标变化范围（单位：千米）:
  E: -1612.615  ~ 1626.616
  N: -1045.914  ~ 1066.611
  U: -220.396  ~ 5.372
"""

"""
E N RALT PTCH ROLL DA TRK GS ALTR LONG VRTG LATG
"""

"""
飞行阶段 PH 统计:
  PH = 5: 75596 条记录
  PH = 4: 31066 条记录
  PH = 6: 28562 条记录
  PH = 3: 692 条记录
  PH = 7: 460 条记录

ENU 坐标变化范围（单位：千米）:
  E: -1631.810  ~ 1608.806
  N: -1047.450  ~ 1064.408
  U: -210.285  ~ 5.508
"""
