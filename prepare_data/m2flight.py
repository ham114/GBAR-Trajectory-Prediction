# -*- coding: utf-8 -*-
import torch
import math
import os

def build_m2flightnet_pt(
    in_path: str,
    out_path: str,
    feat_names = ("E","N","RALT","PTCH","ROLL","MH","TRK","GS","ALTR","LONG","VRTG","LATG"),
    gs_to_mps: float = 0.5144444444444444,  # 1 knot -> m/s
):
    """
    构造 6 特征数据并保存：
      - state:     [N,T,6] = [ΔE, ΔN, RALT, Vx, Vy, ALTR]
      - state_raw: [N,T,6] = [E(m), N(m), RALT, Vx, Vy, ALTR]
      - state_names: 通道名
      - stat: {每个特征的 min, max}
    """
    data = torch.load(in_path, map_location="cpu")
    X = data["raw_data"]  # [N,T,C]

    idx = {name:i for i,name in enumerate(feat_names)}

    E_m   = X[..., idx["E"]].float() * 1000.0  # km->m
    N_m   = X[..., idx["N"]].float() * 1000.0
    RALT  = X[..., idx["RALT"]].float()
    GS_mps= X[..., idx["GS"]].float() * gs_to_mps
    MH_rad= X[..., idx["MH"]].float() * math.pi / 180.0
    ALTR  = X[..., idx["ALTR"]].float()

    # Vx/Vy
    vx = GS_mps * torch.sin(MH_rad)
    vy = GS_mps * torch.cos(MH_rad)

    # 相邻差分
    def time_diff_first_zero(x):
        xd = torch.zeros_like(x)
        xd[:,1:] = x[:,1:] - x[:,:-1]
        return xd

    dE = time_diff_first_zero(E_m)
    dN = time_diff_first_zero(N_m)

    # 拼接
    state = torch.stack([dE, dN, RALT, vx, vy, ALTR], dim=-1)
    state_raw = torch.stack([E_m, N_m, RALT, vx, vy, ALTR], dim=-1)

    state_names = ["dE", "dN", "RALT", "Vx", "Vy", "ALTR"]

    # === 统计最值 ===
    feat_min = state.amin(dim=(0,1)).tolist()
    feat_max = state.amax(dim=(0,1)).tolist()
    stat = {name: {"min": float(lo), "max": float(hi)}
            for name, lo, hi in zip(state_names, feat_min, feat_max)}

    out = {
        "state": state,
        "state_raw": state_raw,
        "state_names": state_names,
        "stat": stat,
        "note": {
            "source_file": os.path.abspath(in_path),
            "heading_convention": "Vx=GS*sin(MH), Vy=GS*cos(MH), MH=0°北 顺时针",
            "gs_to_mps": gs_to_mps,
        }
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out, out_path)
    print(f"[OK] Saved to {out_path}")
    print("state      :", tuple(state.shape), state_names)
    print("最值统计：")
    for k,v in stat.items():
        print(f"  {k:8s} min={v['min']:.3f}, max={v['max']:.3f}")

if __name__ == "__main__":
    in_file  = "/home/userdata/2024_hyn/dataset/nasa_dashlink/train.pt"
    out_file = "/home/userdata/2024_hyn/dataset/nasa_dashlink/train_m2flight.pt"
    build_m2flightnet_pt(in_file, out_file)
