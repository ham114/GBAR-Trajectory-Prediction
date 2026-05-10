#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将 CSV 中的经纬度 + measured_flight_level 转 ENU，并写回同一 CSV。
- 经纬度列：lat, lon
- 高度列：measured_flight_level（单位：百英尺）
- ENU 原点：传入 (lat0_deg, lon0_deg, h0_m)
- 单位：m 或 km
"""

import os
import glob
import math
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ==== WGS84 常量 ====
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)  # 第一偏心率平方

def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: np.ndarray):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat); cos_lat = np.cos(lat)
    sin_lon = np.sin(lon); cos_lon = np.cos(lon)
    N = _A / np.sqrt(1.0 - _E2 * sin_lat**2)
    X = (N + h_m) * cos_lat * cos_lon
    Y = (N + h_m) * cos_lat * sin_lon
    Z = (N * (1.0 - _E2) + h_m) * sin_lat
    return X, Y, Z

def ecef_to_enu(X, Y, Z, X0, Y0, Z0, lat0_deg: float, lon0_deg: float):
    lat0 = math.radians(lat0_deg); lon0 = math.radians(lon0_deg)
    sin_lat0 = math.sin(lat0);  cos_lat0 = math.cos(lat0)
    sin_lon0 = math.sin(lon0);  cos_lon0 = math.cos(lon0)
    dX = X - X0; dY = Y - Y0; dZ = Z - Z0
    # ECEF->ENU 旋转
    t = np.array([
        [-sin_lon0,             cos_lon0,            0.0],
        [-sin_lat0*cos_lon0,   -sin_lat0*sin_lon0,  cos_lat0],
        [ cos_lat0*cos_lon0,    cos_lat0*sin_lon0,  sin_lat0]
    ])
    enu = t @ np.vstack((dX, dY, dZ))
    return enu[0, :], enu[1, :], enu[2, :]

def convert_csv_to_enu(
    csv_path: str,
    origin_lat: float,
    origin_lon: float,
    origin_h_m: float,
    lat_col: str = "lat",
    lon_col: str = "lon",
    fl_col: str = "measured_flight_level",
    units: str = "m",   # "m" 或 "km"
    e_col: str = "e",
    n_col: str = "n",
    u_col: str = "u",
    backup: bool = False
) -> str:
    """
    将单个 CSV 转为 ENU，写回原文件；返回 csv_path。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    for c in (lat_col, lon_col, fl_col):
        if c not in df.columns:
            raise KeyError(f"缺少列: {c} in {csv_path}")

    # measured_flight_level: 百英尺 -> 英尺 -> 米
    fl = pd.to_numeric(df[fl_col], errors="coerce").fillna(0.0).to_numpy()
    alt_ft = fl * 100.0
    alt_m = alt_ft * 0.3048

    # 大地坐标 -> ECEF
    X, Y, Z = geodetic_to_ecef(
        pd.to_numeric(df[lat_col], errors="coerce").to_numpy(),
        pd.to_numeric(df[lon_col], errors="coerce").to_numpy(),
        alt_m
    )

    # 原点 ECEF
    X0, Y0, Z0 = geodetic_to_ecef(origin_lat, origin_lon, origin_h_m)

    # ECEF -> ENU
    e, n, u = ecef_to_enu(X, Y, Z, X0, Y0, Z0, origin_lat, origin_lon)

    if units.lower() == "km":
        scale = 1.0 / 1000.0
        e, n, u = e * scale, n * scale, u * scale
    elif units.lower() == "m":
        pass
    else:
        raise ValueError("units 仅支持 'm' 或 'km'")

    df[e_col] = e
    df[n_col] = n
    df[u_col] = u

    # 安全覆盖：写临时文件再替换
    tmp_path = csv_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    if backup:
        bak = csv_path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(csv_path, bak)
    os.replace(tmp_path, csv_path)
    return csv_path

def _worker_convert(args):
    """
    顶层 worker，供进程池/线程池调用，必须可被 pickle。
    """
    (p, origin_lat, origin_lon, origin_h_m,
     lat_col, lon_col, fl_col, units, e_col, n_col, u_col, backup) = args
    try:
        convert_csv_to_enu(
            p, origin_lat, origin_lon, origin_h_m,
            lat_col, lon_col, fl_col, units,
            e_col, n_col, u_col, backup
        )
        return (p, "OK", "")
    except Exception as ex:
        return (p, "FAIL", str(ex))

def convert_dir(
    root_dir: str,
    origin_lat: float,
    origin_lon: float,
    origin_h_m: float,
    lat_col: str = "lat",
    lon_col: str = "lon",
    fl_col: str = "measured_flight_level",
    units: str = "m",
    e_col: str = "e",
    n_col: str = "n",
    u_col: str = "u",
    max_workers: int = 8,
    backup: bool = False,
    use_processes: bool = True
):
    files = glob.glob(os.path.join(root_dir, "*", "*.csv"))
    if not files:
        print("未匹配到 CSV 文件")
        return

    tasks = [
        (p, origin_lat, origin_lon, origin_h_m,
         lat_col, lon_col, fl_col, units, e_col, n_col, u_col, backup)
        for p in files
    ]

    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    ok = fail = 0
    with Executor(max_workers=max_workers) as ex, \
         tqdm(total=len(files), desc="Convert ENU", unit="file") as bar:
        futs = {ex.submit(_worker_convert, t): t[0] for t in tasks}
        for f in as_completed(futs):
            p, status, msg = f.result()
            if status == "OK":
                ok += 1
            else:
                fail += 1
                print(f"[FAIL] {p}: {msg}")
            bar.update(1)
    print(f"完成: OK={ok}, FAIL={fail}")


if __name__ == "__main__":
    # 示例：ESSA 阿兰达机场。请按论文坐标替换为更精确的原点。
    # origin_lat, origin_lon = 59.6512, 17.9178
    # 场标高 137 ft -> 41.756 m
    # origin_h_m = 137.0 * 0.3048

    # ——示例 1：处理单文件——
    # convert_csv_to_enu(
    #     "/path/to/one.csv",
    #     origin_lat=59.6512, origin_lon=17.9178, origin_h_m=137.0*0.3048,
    #     lat_col="lat", lon_col="lon", fl_col="measured_flight_level",
    #     units="m",  # 或 "km"
    #     e_col="e", n_col="n", u_col="u",
    #     backup=False
    # )

    convert_dir(
        root_dir="/home/h3c/dataset/SCAT/SCAT_csv/",
        origin_lat=59.6512, origin_lon=17.9178, origin_h_m=137.0 * 0.3048,
        lat_col="lat", lon_col="lon", fl_col="measured_flight_level",
        units="km",
        e_col="e", n_col="n", u_col="u",
        max_workers=12,
        backup=False
    )
