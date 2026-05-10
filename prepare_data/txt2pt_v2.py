# txt2pt.py
import os, glob, json, numpy as np, torch
from tqdm import tqdm
from collections import Counter, defaultdict
import argparse
# from utils import load_config_from_json

def dict_to_obj(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, dict_to_obj(j))
        elif isinstance(j, seqs):
            setattr(top, i,
                    type(j)(dict_to_obj(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top


def load_config_from_json(json_path='config.json'):
    with open(json_path, 'r') as fr:
        data = json.load(fr)
        # print(data)
    return dict_to_obj(data)


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


# -------- bit 向量 -> 区间索引 --------
def bitvec_to_index(mat_bits):
    bit_len = mat_bits.shape[1]
    weights = (1 << np.arange(bit_len)[::-1]).astype(np.int64)
    return mat_bits.astype(np.int64) @ weights

# -------- 主函数：预测数据集 --------
def txt_dir_to_pt_pred(txt_dir: str,
                       config_path: str,
                       pt_path: str,
                       inp_seq_len: int = 64,
                       horizon: int     = 4,
                       data_period: int = 1,
                       stride: int      = 3):
    if stride is None:            # 默认为 inp_seq_len, 你可传 1 或 3
        stride = inp_seq_len

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    feat_names = ["E","N","RALT","PTCH","ROLL","MH","TRK","GS", "ALTR","LONG","VRTG","LATG"]
    bits_per   = [cfg[k]["bits"] for k in feat_names]
    mins_per   = [cfg[k]["min"]  for k in feat_names]
    maxs_per   = [cfg[k]["max"]  for k in feat_names]
    onehot_map = {4:0, 5:1, 6:2, 7:3}

    # 全局 min / max
    g_min = defaultdict(lambda:  1e30)
    g_max = defaultdict(lambda: -1e30)

    # 收集列表
    X_bin_list, Y_bin_list = [], []
    raw_list, Δ_list, y_cls_list = [], [], []

    txt_paths = glob.glob(os.path.join(txt_dir, "*.txt"))
    if not txt_paths:
        raise FileNotFoundError(f"{txt_dir} 下无 .txt")

    win = inp_seq_len + horizon        # 总窗口长度

    for txt_path in tqdm(txt_paths, desc="Processing", unit="file"):
        with open(txt_path) as fr:
            lines = [ln.strip().split('|') for ln in fr if ln.strip()]

        # data_period 拆分
        if data_period == 1:
            sub_lists = [lines]
        elif data_period == 2:
            sub_lists = [lines[::2], lines[1::2]]
        else:
            sub_lists = [lines[data_period-i::data_period]
                         for i in range(1, data_period)]

        for sub in sub_lists:
            while len(sub) > win:
                seg = sub[:win]
                sub = sub[stride:]       # 自定义步长

                # -------- 原始 12 维矩阵 --------
                raw_mat = np.asarray([list(map(float, r[:-1])) for r in seg],
                                     dtype=np.float32)      # (win, 12)

                # -------- 二进制编码矩阵 --------
                bin_rows = [np.concatenate(
                                [convert_value_to_binary(v, b, mn, mx)
                                 for v, b, mn, mx in zip(row,
                                                          bits_per,
                                                          mins_per,
                                                          maxs_per)])
                            for row in raw_mat]
                bin_mat = np.vstack(bin_rows).astype(np.uint8)   # (win, D)

                # -------- Δ_bin 计算 --------
                delta_attrs = []
                col = 0
                for bit_len in bits_per:
                    slice_ = slice(col, col+bit_len)
                    idx_seq = bitvec_to_index(bin_mat[:, slice_])
                    delta_attrs.append(np.diff(idx_seq))         # (win-1,)
                    col += bit_len
                delta_mat = np.stack(delta_attrs, axis=1).astype(np.int16)  # (win-1,12)

                # -------- 分类标签 --------
                # ph_vals  = [round(float(r[-1])) for r in seg]
                # ph_major = Counter(ph_vals).most_common(1)[0][0]
                # if ph_major not in onehot_map:
                #     continue
                # y_vec = torch.zeros(4, dtype=torch.float32)
                # y_vec[onehot_map[ph_major]] = 1

                # -------- 分类标签（优先判断7是否占比超过20%）--------
                ph_vals = [round(float(r[-1])) for r in seg]
                count = Counter(ph_vals)
                total = len(ph_vals)

                # 优先判断：若标签7占比超过 1/5，则强制设为类别7
                # if count.get(7, 0) / total > 0.2:
                #     label = 7
                # else:
                #     label = count.most_common(1)[0][0]

                label = count.most_common(1)[0][0]

                # 若映射表中无该标签，则跳过
                if label not in onehot_map:
                    continue

                # 构建 one-hot 标签向量
                y_vec = torch.zeros(4, dtype=torch.float32)
                y_vec[onehot_map[label]] = 1

                # -------- 更新全局范围 --------
                for row in raw_mat:
                    for k, name in enumerate(feat_names):
                        g_min[name] = min(g_min[name], row[k])
                        g_max[name] = max(g_max[name], row[k])

                # -------- append to list --------
                X_bin_list.append(torch.from_numpy(bin_mat[:inp_seq_len]))
                Y_bin_list.append(torch.from_numpy(bin_mat[inp_seq_len:]))
                raw_list.append(torch.from_numpy(raw_mat))       # (L+H,12)
                Δ_list.append(torch.from_numpy(delta_mat))       # (L+H-1,12)
                y_cls_list.append(y_vec)

    # -------- 打包保存 --------
    torch.save({
        "X_bin":    torch.stack(X_bin_list),   # (N,L,D)
        # "Y_bin":    torch.stack(Y_bin_list),   # (N,H,D)
        "raw_data": torch.stack(raw_list),     # (N,L+H,12)
        "Δ_bin":    torch.stack(Δ_list),       # (N,L+H-1,12)
        "y_cls":    torch.stack(y_cls_list)    # (N,4)
    }, pt_path)

    print(f"\n✅ 已保存预测集 {pt_path}")
    print(f"   X_bin: {torch.stack(X_bin_list).shape}  Δ_list: {torch.stack(Δ_list).shape}")

    print("\n====== 全局原始数值范围 ======")
    for n in feat_names:
        print(f"{n:<5}: {g_min[n]:.6f}  ~  {g_max[n]:.6f}")

# ----------------- 3. 命令行入口 -----------------
if __name__ == "__main__":

    config_path = '/home/userdata/2024_hyn/project/python/FlightGPT/config_v15.json'
    config = load_config_from_json(config_path)

    path = '/home/userdata/2024_hyn/dataset/nasa_dashlink/pred.pt'
    fixed_path = '/home/userdata/2024_hyn/dataset/nasa_dashlink/pred.pt'


    txt_dir_to_pt_pred(
        txt_dir="/home/userdata/2024_hyn/dataset/nasa_dashlink/txt/",
        config_path=config_path,
        pt_path=path,
        inp_seq_len=config.inp_seq_len,
        horizon=config.horizon,
        data_period=config.data_period,
        stride=2
    )

    data = torch.load(path)

    # ---------- 1. 读取预测集 ----------
    Δ_bin = data["Δ_bin"]  # (N, L+H-1, 12)
    y_cls = data["y_cls"]

    # ---------- 2. 读取阈值 ----------
    with open(config_path) as f:
        cfg = json.load(f)

    feat_names = ["E", "N", "RALT", "PTCH", "ROLL", "MH", "TRK", "GS", "ALTR", "LONG", "VRTG", "LATG"]
    thresholds = [(cfg[n]["delta_lo"], cfg[n]["delta_hi"]) for n in feat_names]

    # ---------- 3. 过滤 ----------
    keep_idx = []
    for i in tqdm(range(Δ_bin.shape[0]), desc="Filter"):
        bad = False
        for j, (lo, hi) in enumerate(thresholds):
            col = Δ_bin[i, :, j]
            if (col < lo).any() or (col > hi).any():
                bad = True
                break
        if not bad:
            keep_idx.append(i)

    # ---------- 4. 保存 ----------
    filtered = {k: v[keep_idx] for k, v in data.items()}
    torch.save(filtered, fixed_path)
    print(f"保留 {len(keep_idx)}/{len(Δ_bin)} "
          f"({len(keep_idx) / len(Δ_bin):.1%}) 样本已写入 pred_fixed.pt")

    # ---------- 5. 标签分布 ----------
    labels = torch.argmax(filtered["y_cls"], dim=1) + 4
    total = len(labels)
    for ph in range(4, 8):
        c = (labels == ph).sum().item()
        print(f"PH={ph}: {c:>6} ({100 * c / total:>5.1f}%)")



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

保留 1215118/2334646 (52.0%) 样本已写入 pred_fixed.pt
PH=4: 329823 ( 27.1%)
PH=5: 552526 ( 45.5%)
PH=6: 332468 ( 27.4%)
PH=7:    301 (  0.0%)

"""
