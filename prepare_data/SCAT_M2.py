import torch
import math

R = 6371000.0   # 地球半径 (m)

def latlon_diff_to_enu(lat1, lon1, lat2, lon2):
    """
    输入:
        lat1, lon1, lat2, lon2: 弧度(rad)，形状 (B, T-1)
    输出:
        dE, dN: 东向、北向位移 (m)，形状同上，允许正负
        - dE > 0: 向东
        - dE < 0: 向西
        - dN > 0: 向北
        - dN < 0: 向南
    使用局部平面近似:
        dN ≈ R * dφ
        dE ≈ R * cos(φ_mean) * dλ
    """
    dlat = lat2 - lat1     # dφ
    dlon = lon2 - lon1     # dλ

    lat_mean = (lat1 + lat2) / 2.0  # 使用相邻两点的平均纬度

    dN = R * dlat
    dE = R * torch.cos(lat_mean) * dlon

    return dE, dN


if __name__ == '__main__':
    pt_path = '/home/h3c/dataset/SCAT/valid.pt'
    out_pt_path = '/home/h3c/dataset/SCAT/valid_with_enu.pt'

    data = torch.load(pt_path)
    raw_data = data['windows']  # (N, T, C)
    N, T, C = raw_data.shape
    print("Loaded:", raw_data.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 在 CPU 上预先分配结果张量（避免频繁 cat）
    dE_all = torch.empty((N, T), dtype=torch.float32)
    dN_all = torch.empty((N, T), dtype=torch.float32)

    # 分块并行处理 N 维度
    batch_size = 8192  # 可根据显存/内存情况调整

    torch.set_grad_enabled(False)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            bN = end - start

            # 取出当前块 (bN, T)
            lat_deg = raw_data[start:end, :, 0]
            lon_deg = raw_data[start:end, :, 1]

            # 转 float32 + 弧度 + 移动到 GPU/CPU 计算设备
            lat = torch.deg2rad(lat_deg.to(torch.float32, non_blocking=True)).to(device)
            lon = torch.deg2rad(lon_deg.to(torch.float32, non_blocking=True)).to(device)

            # 相邻差分 (bN, T-1)
            lat_prev = lat[:, :-1]
            lat_next = lat[:, 1:]
            lon_prev = lon[:, :-1]
            lon_next = lon[:, 1:]

            # 计算当前块的 ENU 差分（在 device 上并行）
            dE, dN = latlon_diff_to_enu(lat_prev, lon_prev, lat_next, lon_next)  # (bN, T-1)

            # 前面补 0，表示相对于第一个点的位移为 0
            zero_pad = torch.zeros((bN, 1), dtype=torch.float32, device=device)
            dE_full = torch.cat([zero_pad, dE], dim=1)  # (bN, T)
            dN_full = torch.cat([zero_pad, dN], dim=1)

            # 拷回 CPU，写入总缓冲
            dE_all[start:end, :] = dE_full.cpu()
            dN_all[start:end, :] = dN_full.cpu()

            print(f"processed {end}/{N}")

    # 构造 diff 版数据：前两维改为 ENU 差分 (dE, dN)
    windows_diff = raw_data.clone()
    windows_diff[:, :, 0] = dE_all  # East (m)
    windows_diff[:, :, 1] = dN_all  # North (m)

    # 统计 min/max
    raw_min = raw_data.amin(dim=(0, 1))
    raw_max = raw_data.amax(dim=(0, 1))
    diff_min = windows_diff.amin(dim=(0, 1))
    diff_max = windows_diff.amax(dim=(0, 1))

    print("\n=== Raw Data Min ===")
    print(raw_min)
    print("=== Raw Data Max ===")
    print(raw_max)

    print("\n=== ENU Diff Data Min ===")
    print(diff_min)
    print("=== ENU Diff Data Max ===")
    print(diff_max)

    out_data = {
        "state_raw": raw_data,
        "state": windows_diff,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "diff_min": diff_min,
        "diff_max": diff_max,
    }
    torch.save(out_data, out_pt_path)
    print(f"\nSaved to: {out_pt_path}")
