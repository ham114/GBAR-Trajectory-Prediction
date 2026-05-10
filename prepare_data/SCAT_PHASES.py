import os, glob
import pandas as pd
from collections import Counter
from tqdm import tqdm

# ========== 配置：高度带与规则 ==========
# 以飞行层 FL 为单位（100 ft）
FL_MIN, FL_MAX = -3, 450

# 6 类标签
TAKEOFF_IC   = "takeoff_initial_climb"
CLIMB        = "climb"
HIGH_CRUISE  = "high_cruise"
MIDLOW_LEVEL = "midlow_level"
DESCENT      = "descent"
APPROACH     = "approach"
UNKNOWN = "unknown"

un = -1

# 分段众数的并列时优先级（可按需调整）
VERT_TIE_BREAK = [0, 1, 2]  # 1上升优先，其次平飞，再下降

def _mode_with_tie_priority(series, priority=VERT_TIE_BREAK):
    cnt = Counter(series.dropna().astype(int).tolist())
    if not cnt:
        return 0  # 无数据时当作平飞
    max_n = max(cnt.values())
    cands = [k for k, v in cnt.items() if v == max_n]
    # 按优先级挑选
    for p in priority:
        if p in cands:
            return p
    return cands[0]

def _label_phase(fl_val, seg_vert):
    """
    输入: fl_val=measured_flight_level(数值), seg_vert ∈ {0:平飞,1:上升,2:下降}
    输出: 六类之一
    规则互斥且覆盖:
      - 高空巡航: seg_vert==0 且 FL>=300
      - 中低空平飞: seg_vert==0 且 60<=FL<240
      - 进场: seg_vert in {0,2} 且 FL<40
      - 起飞爬升: seg_vert==1 且 FL<80
      - 爬升: seg_vert==1 且 80<=FL<300
      - 下降: seg_vert==2 且 FL>=40
    未命中分支的兜底：根据 seg_vert 映射到最接近的类
    """
    # 规范化 FL
    try:
        fl = float(fl_val)
    except Exception:
        fl = float("nan")
    if pd.isna(fl):
        # 缺失高度直接以 seg_vert 兜底
        return {0: MIDLOW_LEVEL, 1: CLIMB, 2: DESCENT}.get(int(seg_vert), MIDLOW_LEVEL)

    # 裁剪到全局范围
    if fl < 0:
        fl = 0.0
    if fl > FL_MAX:
        fl = FL_MAX

    seg_vert = int(seg_vert)

    # 互斥规则
    if seg_vert == 0:
        if fl >= 300:
            return HIGH_CRUISE
        return  MIDLOW_LEVEL

    if seg_vert == 1:
        if fl < 80:
            return TAKEOFF_IC
        return CLIMB

    if seg_vert == 2:
        if fl < 40:
            return APPROACH
        return DESCENT  # FL>=40

    un += 1

    return UNKNOWN

# ========== 文件处理 ==========
def label_phases_for_file(
    csv_path: str,
    lat_col="lat", lon_col="lon",
    fl_col="measured_flight_level",
    seg_col="segment_id",
    vert_col="vert",
    out_path=None
):
    """
    对单个 CSV 标注:
      - 计算每个 segment_id 的 vert 众数 → segment_vert_mode
      - 为每行生成 phase ∈ {6类}
    结果新增列: segment_vert_mode, phase
    """
    df = pd.read_csv(csv_path)
    if not {fl_col, seg_col, vert_col}.issubset(df.columns):
        missing = {fl_col, seg_col, vert_col} - set(df.columns)
        raise ValueError(f"缺少列: {missing} in {csv_path}")

    # 标准化 vert 为整数
    df[vert_col] = pd.to_numeric(df[vert_col], errors="coerce").fillna(0).astype(int)

    # 分段众数
    seg_mode = (
        df.groupby(seg_col)[vert_col]
          .apply(lambda s: _mode_with_tie_priority(s))
          .rename("segment_vert_mode")
    )
    df = df.merge(seg_mode, on=seg_col, how="left")

    # 标注 phase
    df["phase"] = [
        _label_phase(fl, vmode)
        for fl, vmode in zip(pd.to_numeric(df[fl_col], errors="coerce"), df["segment_vert_mode"])
    ]

    if out_path is None:
        root, ext = os.path.splitext(csv_path)
        out_path = root + "_labeled" + ext
    df.to_csv(out_path, index=False)
    return out_path

def label_phases_for_file_inplace(
    csv_path: str,
    fl_col="measured_flight_level",
    seg_col="segment_id",
    vert_col="vert",
):
    df = pd.read_csv(csv_path)

    if not {fl_col, seg_col, vert_col}.issubset(df.columns):
        return False

    df[vert_col] = pd.to_numeric(df[vert_col], errors="coerce").fillna(0).astype(int)

    seg_mode = (
        df.groupby(seg_col)[vert_col]
          .apply(lambda s: _mode_with_tie_priority(s))
          .rename("segment_vert_mode")
    )
    df = df.merge(seg_mode, on=seg_col, how="left")

    df["phase"] = [
        _label_phase(fl, vmode)
        for fl, vmode in zip(pd.to_numeric(df[fl_col], errors="coerce"),
                             df["segment_vert_mode"])
    ]

    df.to_csv(csv_path, index=False)
    return True


def batch_label_phases(
    root_dir: str,
    csv_pattern="*.csv",
    **kwargs
):
    files  = glob.glob(os.path.join(root_dir, csv_pattern))
    files += glob.glob(os.path.join(root_dir, "*", csv_pattern))

    results = []
    for fp in tqdm(files, desc="Labeling"):
        try:
            ok = label_phases_for_file_inplace(fp, **kwargs)
            if ok:
                results.append(fp)
        except Exception:
            pass
    return results


if __name__ == '__main__':

    res = batch_label_phases("/home/h3c/dataset/SCAT/SCAT_csv/",
                             fl_col="measured_flight_level",
                             seg_col="segment_id",
                             vert_col="vert")

    for fp in res:
        print(fp)

    print(un)
