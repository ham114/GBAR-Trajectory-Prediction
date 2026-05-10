# txt2pt.py
import os, glob, json, numpy as np, torch
from tqdm import tqdm
from collections import Counter, defaultdict
import argparse
from utils import load_config_from_json

# ----------------- 实用：小型在线百分位缓存 -----------------
class TinyQuantile:
    """维护一个有界缓存来近似估计分位数（无巨大内存开销）"""
    def __init__(self, cap=100_000):
        self.cap = cap
        self.buf = []     # 排序数组

    def push(self, arr):
        for v in arr:
            insort(self.buf, v)
            if len(self.buf) > self.cap:      # 超出容量时每 2 个保留 1 个
                self.buf = self.buf[::2]

    def percentiles(self, ps=(25, 50, 75)):
        if not self.buf:
            return [0.0] * len(ps)
        data = np.asarray(self.buf)
        return np.percentile(data, ps)

# ----------------- 主统计函数 -----------------
def delta_statistics(X, feat_slices, feat_names, thresholds=None):
    """
    thresholds: dict{name:(lo,hi)}  -> 若提供则过滤掉超阈样本
    return stats dict 与 保留样本索引 keep_idx
    """
    k = len(feat_names)
    # 初始化统计量
    min_delta = np.full(k,  1e9, dtype=np.int64)
    max_delta = np.full(k, -1e9, dtype=np.int64)
    count = np.zeros(k, dtype=np.int64)
    mean  = np.zeros(k, dtype=np.float64)
    M2    = np.zeros(k, dtype=np.float64)
    q_buf = [TinyQuantile() for _ in range(k)]

    keep_idx = []                         # 若过滤阈值，则存保留样本
    for idx in tqdm(range(X.shape[0]), desc="Scanning"):
        x_seq = X[idx]
        bad_sample = False

        for j, name in enumerate(feat_names):
            sl = feat_slices[j]
            idxs = x_seq[:, sl].numpy().dot(1 << np.arange(sl.stop-sl.start)[::-1])
            deltas = idxs[1:] - idxs[:-1]

            # --- 阈值过滤 ---
            if thresholds:
                lo, hi = thresholds[name]
                if (deltas < lo).any() or (deltas > hi).any():
                    bad_sample = True
                    break

            # --- 更新极值 ---
            d_min, d_max = deltas.min(), deltas.max()
            if d_min < min_delta[j]: min_delta[j] = d_min
            if d_max > max_delta[j]: max_delta[j] = d_max

            # --- Welford 均值方差 ---
            for d in deltas:
                count[j] += 1
                delta_mu = d - mean[j]
                mean[j] += delta_mu / count[j]
                M2[j]   += delta_mu * (d - mean[j])

            # --- 在线百分位缓存 ---
            q_buf[j].push(deltas)

        if not bad_sample:
            keep_idx.append(idx)

    variance = M2 / (count - 1)

    # 汇总结果
    stats = {}
    header = ("Δmin", "Q1", "Q2", "Q3", "Δmax", "mean", "var")
    for j, name in enumerate(feat_names):
        q1, q2, q3 = q_buf[j].percentiles()
        stats[name] = dict(zip(
            header,
            [int(min_delta[j]), int(q1), int(q2), int(q3),
             int(max_delta[j]), round(mean[j], 3), round(variance[j], 3)]
        ))
    return stats, keep_idx


# ----------------- 1. 二分查找二进制编码 -----------------
def convert_value_to_binary(value, bit_size, min_val, max_val):
    """连续数值 -> 二进制数组（np.uint8）"""
    value = max(min(value, max_val), min_val)          # 裁剪
    bins  = np.empty(bit_size, dtype=np.uint8)
    for i in range(bit_size):
        mid = (min_val + max_val) / 2
        if value >= mid:
            bins[i] = 1
            min_val = mid
        else:
            bins[i] = 0
            max_val = mid
    return bins

# ----------------- 2. 主处理函数 -----------------
def txt_dir_to_pt(txt_dir: str,
                  config_path: str,
                  pt_path: str,
                  inp_seq_len: int = 64,
                  horizon: int     = 4,
                  data_period: int = 1):
    # ---- 读取配置 ----
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    feat_names = ["E","N","RALT","PTCH","ROLL","MH","TRK","GS", "ALTR","LONG","VRTG","LATG"]
    bits_per_feat = [cfg[k]["bits"] for k in feat_names]
    mins_per_feat = [cfg[k]["min"]  for k in feat_names]
    maxs_per_feat = [cfg[k]["max"]  for k in feat_names]
    onehot_dict   = {4:0, 5:1, 6:2, 7:3}               # PH 映射

    # 全局范围统计
    global_min = defaultdict(lambda:  1e30)
    global_max = defaultdict(lambda: -1e30)

    X_list, y_list = [], []
    txt_paths = glob.glob(os.path.join(txt_dir, "*.txt"))
    if not txt_paths:
        raise FileNotFoundError(f"{txt_dir} 下未找到 *.txt")

    for txt_path in tqdm(txt_paths, desc="Processing txt", unit="file"):
        with open(txt_path, 'r') as fr:
            lines = [ln.strip().split('|') for ln in fr if ln.strip()]

        # ----------- data_period 拆分 -----------
        if data_period == 1:
            sub_lists = [lines]
        elif data_period == 2:
            sub_lists = [lines[::2], lines[1::2]]
        else:
            sub_lists = [lines[data_period - i::data_period]
                         for i in range(1, data_period)]

        for sub in sub_lists:
            # 滑窗采样
            win = inp_seq_len + horizon
            # win = inp_seq_len
            while len(sub) > win:
                seg = sub[:win]
                # sub = sub[inp_seq_len:]       # 滑动
                sub = sub[3:]
                # ----------- 取整体 y 标签 -----------
                ph_vals = [round(float(row[-1])) for row in seg]
                ph_major = Counter(ph_vals).most_common(1)[0][0]
                if ph_major not in onehot_dict:          # 忽略异常标签
                    continue
                y_vec = torch.zeros(4, dtype=torch.float32)
                y_vec[onehot_dict[ph_major]] = 1.0

                # ----------- 编码 X -----------
                bin_rows = []
                for row in seg:
                    features = list(map(float, row[:-1]))      # 前12列
                    bin_feat = [convert_value_to_binary(v, b, mn, mx)
                                for v, b, mn, mx in zip(features,
                                                       bits_per_feat,
                                                       mins_per_feat,
                                                       maxs_per_feat)]
                    bin_row = np.concatenate(bin_feat)         # (D,)
                    bin_rows.append(bin_row)

                    # 更新全局范围
                    for idx, name in enumerate(feat_names):
                        global_min[name] = min(global_min[name], features[idx])
                        global_max[name] = max(global_max[name], features[idx])

                X_tensor = torch.tensor(bin_rows, dtype=torch.uint8)  # (L, D)
                X_list.append(X_tensor)
                y_list.append(y_vec)



    # ----------- 打包保存 .pt -----------
    X_padded = torch.nn.utils.rnn.pad_sequence(X_list, batch_first=True, padding_value=0)
    y_tensor = torch.stack(y_list, dim=0)
    torch.save({'X': X_padded, 'y': y_tensor}, pt_path)
    print(f"\n✅ 已生成数据集 {pt_path}")
    print(f"   X shape: {X_padded.shape},  y shape: {y_tensor.shape}")

    # ----------- 打印全局范围 -----------
    print("\n====== 全局原始数值范围 ======")
    for n in feat_names:
        print(f"{n:<5}: {global_min[n]:.6f}  ~  {global_max[n]:.6f}")

# ----------------- 3. 命令行入口 -----------------
if __name__ == "__main__":

    config = load_config_from_json('/home/userdata/2024_hyn/project/python/FlightGPT/config_v12.json')


    # txt_dir_to_pt("/home/userdata/2024_hyn/dataset/nasa_dashlink/txt/",
    #               '/home/userdata/2024_hyn/project/python/FlightGPT/config_v12.json',
    #               "/home/userdata/2024_hyn/dataset/nasa_dashlink/cls.pt",
    #               inp_seq_len=config.inp_seq_len,
    #               horizon=config.horizon,
    #               data_period=config.data_period)

    data = torch.load("/home/userdata/2024_hyn/dataset/nasa_dashlink/cls.pt")
    X = data["X"]  # shape: (N, L, D)
    y = data["y"]  # shape: (N, 4)



    # ==== 1. 标签分布统计 ====
    y_labels = torch.argmax(y, dim=1) + 4  # 映射回 4~7
    label_counts = Counter(y_labels.tolist())

    print("====== 标签分布统计（PH=4~7）======")
    for ph in range(4, 8):
        print(f"PH = {ph}: {label_counts[ph]} 条")

    # ==== 2. 差分分析准备 ====
    # 维度信息
    N, L, D = X.shape
    feat_names = ["E", "N", "RALT", "PTCH", "ROLL", "MH", "TRK", "GS", "ALTR", "LONG", "VRTG", "LATG"]
    # ==== 从 config.json 动态读取位数信息 ====
    with open("/home/userdata/2024_hyn/project/python/FlightGPT/config_v12.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
        print("cfg loaded:", type(cfg))

    bit_lengths = [cfg[name]["bits"] for name in feat_names]
    # 手动阈值表：只收集写了 delta_lo/hi 的
    manual_thres = {
        n: (cfg[n]["delta_lo"], cfg[n]["delta_hi"])
        for n in feat_names
        if "delta_lo" in cfg[n] and "delta_hi" in cfg[n]
    }

    # ==== 构造切片索引 ====
    feat_slices = []
    start = 0
    for b in bit_lengths:
        feat_slices.append(slice(start, start + b))
        start += b

    #  ==== 3. 统计差分跨度范围 ====
    #  ---------- 收集所有 Δ 以便计算分位数 / 均值 σ ----------
    delta_pool = defaultdict(list)
    for seq in tqdm(X, desc="Collect Δ"):
        for j, name in enumerate(feat_names):
            idxs = seq[:, feat_slices[j]].numpy() \
                .dot(1 << np.arange(bit_lengths[j])[::-1])
            delta = idxs[1:] - idxs[:-1]
            delta_pool[name].extend(delta.tolist())

    # # ---------- 自动生成阈值 (IQR×3 + 备用 3σ) ----------
    # # --------------- 1. 设定要筛选的属性 ----------------
    # filter_attrs = {"E", "N", "RALT", "PTCH", "ROLL", "MH", "TRK", "GS", "ALTR", "LONG", "VRTG", "LATG", "PH"}  # 其余属性不做剔除
    #
    # # --------------- 2. 生成阈值：只给选中属性计算 ----------------
    # thresholds = {}
    #
    # for n, arr in delta_pool.items():
    #     if n in manual_thres:  # --- 直接用手动值 ---
    #         thresholds[n] = manual_thres[n]
    #         continue
    #
    #     # --- 否则统计学推断 (示例: IQR×3) ---
    #     arr = np.asarray(arr)
    #     q1, q3 = np.percentile(arr, [25, 75])
    #     iqr = q3 - q1
    #     thresholds[n] = (q1 - 3 * iqr, q3 + 3 * iqr)
    #
    # print(">>> 自动阈值（仅作用于选中属性）")
    # for n, (lo, hi) in thresholds.items():
    #     print(f"{n:<5}: {lo:.0f}  ~  {hi:.0f}")
    #
    # # --------------- 3. 过滤：只检查有阈值的属性 ----------------
    # keep_idx = []
    # for idx, seq in enumerate(tqdm(X, desc="Filter")):
    #     bad = False
    #     for j, name in enumerate(feat_names):
    #         if name not in thresholds:  # 不在筛选列表——直接跳过
    #             continue
    #
    #         lo, hi = thresholds[name]
    #         sl = feat_slices[j]
    #         idxs = seq[:, sl].numpy().dot(1 << np.arange(sl.stop - sl.start)[::-1])
    #         delta = idxs[1:] - idxs[:-1]
    #         if (delta < lo).any() or (delta > hi).any():
    #             bad = True
    #             break
    #     if not bad:
    #         keep_idx.append(idx)
    #
    # # --------------- 4. 保存新数据集 ----------------
    # X_clean, y_clean = X[keep_idx], y[keep_idx]
    # torch.save({'X': X_clean, 'y': y_clean}, "/home/userdata/2024_hyn/dataset/nasa_dashlink/cls_fixed.pt")
    # print(f"\n✅ 已保存 cls_fixed.pt，保留 {len(keep_idx)}/{len(X)}"
    #       f"（{len(keep_idx) / len(X):.1%}）样本")
    #
    # # ---- 1. 载入已筛选数据 ----
    # data = torch.load("/home/userdata/2024_hyn/dataset/nasa_dashlink/cls_fixed.pt")
    # X_f, y_f = data["X"], data["y"]  # X:(N,L,D)  y:(N,4)
    #
    # print(X_f.shape, y_f.shape)
    #
    # # ---- 2. 统计标签 ----
    # labels = torch.argmax(y_f, dim=1) + 4  # 0→4, 1→5, 2→6, 3→7
    # counter = Counter(labels.tolist())
    # total = len(labels)
    #
    # # ---- 3. 输出 ----
    # print("====== 过滤后标签分布 ======")
    # for ph in range(4, 8):
    #     cnt = counter.get(ph, 0)
    #     pct = 100 * cnt / total if total else 0
    #     print(f"PH={ph}: {cnt:>6}  ({pct:>5.1f}%)")
    #
    # print(f"\n总样本数: {total}")

"""
====== 跳变统计（极值 / 四分位 / 均值 / 方差）======
Attr     Δmin      Q1      Q2      Q3    Δmax      mean       var
E       -2604     -14      -1      14    2632     0.165   213.865
N       -5451     -12       2      15    5423     1.076   345.181
RALT   -64944       0       0       0   64944    -2.558 60576.558
PTCH      -40       0       0       0      30    -0.027     1.828
ROLL     -296      -1       0       1     315     0.001    53.154
MH      -1023       0       0       0    1023    -0.002   823.364
TRK     -1023       0       0       0    1023    -0.013   680.558
GS       -109       0       0       0      10    -0.040     1.514
ALTR     -449      -1       0       1     354    -0.021    23.308
LONG     -495       0       0       0     496    -0.041  3249.653
VRTG     -231       0       0       0     232     0.003  2111.333
LATG     -473      -1       0       1     468     0.005  3379.442

E    : -150  ~  150
N    : -150  ~  150
RALT : -500  ~  500
PTCH : -100  ~  100
ROLL : -100  ~  100
MH   : -120  ~  120
TRK  : -120  ~  120
GS   : -100  ~  100
ALTR : -400  ~  400
LONG : -400  ~  400
VRTG : -400  ~  400
LATG : -400  ~  400

====== 标签分布统计（PH=4~7）======
PH = 4: 208221 条
PH = 5: 288353 条
PH = 6: 207343 条
PH = 7: 803 条

torch.Size([603504, 32, 135]) torch.Size([603504, 4])


====== 过滤后标签分布 ======
PH=4: 389626  ( 27.3%)
PH=5: 647363  ( 45.4%)
PH=6: 385862  ( 27.1%)
PH=7:   1955  (  0.1%)

总样本数: 1424806

"""
