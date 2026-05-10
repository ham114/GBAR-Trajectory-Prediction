import os
import numpy as np
import glob
import pandas as pd
from rdp import rdp
import plotly.graph_objects as go
import plotly.io as pio
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def load_trajectory_from_csv(
    csv_path: str,
    e_col: str = "e",
    n_col: str = "n",
    u_col: str = "measured_flight_level",
    units: str = "m"  # 或 "km"
) -> np.ndarray:
    df = pd.read_csv(csv_path)
    need = {e_col, n_col, u_col}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise KeyError(f"缺少列: {missing}")

    arr = df[[e_col, n_col, u_col]].to_numpy(dtype=float)


    if u_col == 'measured_flight_level':

        # u_col 是百英尺 → 英尺 → 米 or 千米
        arr[:, 2] = arr[:, 2] * 100.0 * 0.3048
        if units == "km":
            arr[:, 2] /= 1000.0

    return arr


def simplify_trajectory_rdp(
    coords: np.ndarray,
    epsilon: float,
    return_indices: bool = True
):
    """
    使用 RDP 算法对 coords (L,3) 简化。
    参数:
      coords: numpy array, shape (L,3)
      epsilon: 阈值（距线最大允许误差）
      return_indices: 若 True 返回被保留点在原数组中的索引；否则返回简化后的坐标数组。
    返回:
      若 return_indices=True: 返回索引列表
      否则: 返回简化后的 coords 数组
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords 必须形状為 (L,3)")
    # rdp 函数本身支持 3D 数组。参考文档说明。:contentReference[oaicite:1]{index=1}
    simplified = rdp(coords, epsilon=epsilon)
    simplified = np.asarray(simplified)
    if return_indices:
        # 由于 rdp 返回坐标，我们要在原 coords 中查对应索引
        # 简单做 ±匹配（注意若有重复点可能问题）
        # 更可靠的方法为 rdp(arr, return_mask=True)
        mask = rdp(coords, epsilon=epsilon, return_mask=True)
        indices = np.nonzero(mask)[0]
        return indices
    else:
        return simplified

def build_segments(coords: np.ndarray, keep_idx: np.ndarray) -> List[Tuple[int, int]]:
    """根据RDP保留点索引构造段 [i,j]，含两端点"""
    keep_idx = np.asarray(keep_idx, dtype=int)
    keep_idx.sort()
    # 保证首尾存在
    if keep_idx[0] != 0:
        keep_idx = np.r_[0, keep_idx]
    if keep_idx[-1] != (len(coords) - 1):
        keep_idx = np.r_[keep_idx, len(coords) - 1]
    segs = [(keep_idx[k], keep_idx[k+1]) for k in range(len(keep_idx)-1)]
    return segs

def plot_segments_3d(
    coords: np.ndarray,
    segments: List[Tuple[int, int]],
    keypoints: np.ndarray,
    out_html: str,
    title: str = "Trajectory with RDP Segments (ENU)",
    marker_size: int = 3,
    keypoint_size: int = 5
):
    """
    可视化所有点为散点，按分段着色；叠加全轨迹细线与RDP关键点。
    coords: (L,3) ENU
    segments: [(s,e), ...]，闭区间
    keypoints: RDP保留点索引数组
    out_html: 输出HTML路径
    """

    # 颜色表循环使用
    palette = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
    ]

    traces = []

    # 1) 全轨迹细线作为参考
    traces.append(go.Scatter3d(
        x=coords[:,0], y=coords[:,1], z=coords[:,2],
        mode="lines",
        line=dict(width=1, color="rgba(0,0,0,0.25)"),
        name="path"
    ))

    # 2) 分段散点，所有点都画，只是颜色按段变化
    for si, (s, e) in enumerate(segments):
        seg = coords[s:e+1]
        color = palette[si % len(palette)]
        traces.append(go.Scatter3d(
            x=seg[:,0], y=seg[:,1], z=seg[:,2],
            mode="markers",
            marker=dict(size=marker_size, color=color),
            name=f"seg {si} [{s}-{e}]",
            hoverinfo="text",
            text=[f"idx={s+i}" for i in range(len(seg))]
        ))

    # 3) RDP关键点标记
    keypoints = np.asarray(keypoints, dtype=int)
    kp = coords[keypoints]
    traces.append(go.Scatter3d(
        x=kp[:,0], y=kp[:,1], z=kp[:,2],
        mode="markers",
        marker=dict(size=keypoint_size, color="black", symbol="x"),
        name="RDP keypoints"
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="E", yaxis_title="N", zaxis_title="U",
            aspectmode="data"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    pio.write_html(fig, file=out_html, auto_open=False)
    return out_html


def rdp_segment_one(
    csv_path: str,
    epsilon: float,
    e_col="e", n_col="n", u_col="measured_flight_level",
    units="m",
    seg_col="segment_id"
):
    try:
        coords = load_trajectory_from_csv(
            csv_path, e_col=e_col, n_col=n_col, u_col=u_col, units=units
        )
        keep_idx = simplify_trajectory_rdp(coords, epsilon=epsilon, return_indices=True)

        keep_idx = np.sort(np.asarray(keep_idx, int))
        if keep_idx[0] != 0:
            keep_idx = np.r_[0, keep_idx]
        if keep_idx[-1] != len(coords) - 1:
            keep_idx = np.r_[keep_idx, len(coords) - 1]

        seg_id = np.zeros(len(coords), dtype=int)
        for i in range(len(keep_idx)-1):
            s, e = keep_idx[i], keep_idx[i+1]
            seg_id[s:e+1] = i

        df = pd.read_csv(csv_path)
        df[seg_col] = seg_id
        tmp = csv_path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, csv_path)
        return True, csv_path
    except Exception as ex:
        return False, f"{csv_path}: {ex}"

def rdp_segment_dir(
    root_dir: str,
    epsilon: float,
    units="m",
    max_workers=8,
    seg_col="segment_id"
):
    files = glob.glob(os.path.join(root_dir, "*", "*.csv"))
    if not files:
        print("未匹配到 CSV 文件")
        return

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex, \
         tqdm(total=len(files), desc="RDP segmenting", unit="file") as bar:

        futs = {
            ex.submit(rdp_segment_one, p, epsilon,
                      "e","n","measured_flight_level",
                      units, seg_col): p
            for p in files
        }

        for f in as_completed(futs):
            success, msg = f.result()
            if success:
                ok += 1
            else:
                fail += 1
                print(msg)
            bar.update(1)

    print(f"完成 OK={ok}, FAIL={fail}")


# ---------- 主入口示例 ----------

if __name__ == "__main__":
    rdp_segment_dir(
        root_dir="/home/h3c/dataset/SCAT/SCAT_csv/",
        epsilon=0.1,     # 单位 km 则约50 m
        units="km",
        max_workers=12,
        seg_col="segment_id"
    )

