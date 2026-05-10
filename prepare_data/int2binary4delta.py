# -*- coding: utf-8 -*-
import json, torch, numpy as np
from tqdm import tqdm


pt_in       = "/home/userdata/2024_hyn/dataset/nasa_dashlink/val.pt"
pt_out      = "/home/userdata/2024_hyn/dataset/nasa_dashlink/val_v2.pt"

pt_in       = "/home/userdata/2024_hyn/dataset/nasa_dashlink/train.pt"
pt_out      = "/home/userdata/2024_hyn/dataset/nasa_dashlink/train_v2.pt"
config_path = "/home/userdata/2024_hyn/project/python/FlightGPT/config_v15.json"

feat_names = ["E","N","RALT","PTCH","ROLL","MH","TRK","GS","ALTR","LONG","VRTG","LATG"]

def load_cfg(path):
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)

def encode_delta_bisect(vals, db, lo, hi):
    """将 Δ 整数值编码为二进制（db 位，区间范围 [lo,hi]）"""
    v = np.clip(vals.astype(np.float32), lo, hi)
    low  = np.full_like(v, lo, dtype=np.float32)
    high = np.full_like(v, hi, dtype=np.float32)
    out = np.empty((v.shape[0], db), dtype=np.uint8)
    for b in range(db):
        mid = (low + high) * 0.5
        ones = v >= mid
        out[:, b] = ones.astype(np.uint8)
        low  = np.where(ones, mid, low)
        high = np.where(ones, high, mid)
    return out

def main():
    cfg = load_cfg(config_path)
    d_bits_per = [cfg[k]["delta_bits"] for k in feat_names]
    d_los      = [cfg[k]["delta_lo"]   for k in feat_names]
    d_his      = [cfg[k]["delta_hi"]   for k in feat_names]
    D_dlt      = int(sum(d_bits_per))

    data = torch.load(pt_in, map_location="cpu")
    Δ_bin_old = data["Δ_bin"]  # (N, T, 12) 整数差分
    N, T, C = Δ_bin_old.shape
    assert C == len(feat_names)

    flat = Δ_bin_old.reshape(-1, C).numpy()
    new = np.empty((flat.shape[0], D_dlt), dtype=np.uint8)

    pbar = tqdm(range(C), desc="Encoding Δ_bin", unit="feat")
    col = 0
    for j in pbar:
        chunk = encode_delta_bisect(flat[:, j], d_bits_per[j], d_los[j], d_his[j])
        new[:, col:col+d_bits_per[j]] = chunk
        col += d_bits_per[j]

    Δ_bin_new = new.reshape(N, T, D_dlt)
    data["Δ_bin_old"] = data["Δ_bin"]  # 备份原始整型差分
    data["Δ_bin"] = torch.from_numpy(Δ_bin_new)  # 新的二进制编码

    torch.save(data, pt_out)
    print(f"✅ 已写入 {pt_out}")
    print(f"   Δ_bin_old: {Δ_bin_old.shape} (int)")
    print(f"   Δ_bin_new: {Δ_bin_new.shape} (uint8, 二分区间编码)")

if __name__ == "__main__":
    main()


"""
E     -> min:     -133 | max:      377
N     -> min:     -328 | max:      302
RALT  -> min:     -500 | max:      500
PTCH  -> min:      -39 | max:       27
ROLL  -> min:     -130 | max:      107
MH    -> min:      -32 | max:       33
TRK   -> min:      -34 | max:       38
GS    -> min:     -108 | max:       10
ALTR  -> min:     -424 | max:      334
LONG  -> min:     -478 | max:      478
VRTG  -> min:     -220 | max:      228
LATG  -> min:     -449 | max:      459
"""
