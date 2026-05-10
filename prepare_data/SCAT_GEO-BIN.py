import os
import numpy as np
import torch, json


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_obs_len(cfg: dict, meta: dict | None, T: int) -> int:
    # 只在观测段加噪：优先从 json 里取，其次 meta，最后用 T
    for k in ("obs_len", "inp_seq_len", "seq_len", "T_obs", "T_in"):
        if k in cfg and isinstance(cfg[k], int) and cfg[k] > 0:
            return min(cfg[k], T)
    if meta:
        for k in ("obs_len", "inp_seq_len", "seq_len"):
            v = meta.get(k, None)
            if isinstance(v, int) and v > 0:
                return min(v, T)
    return T  # 兜底：全段（不建议，但不至于报错）

def gen_ar1_eps(shape, rho: float, eps_std: float, rng: np.random.Generator):
    """
    生成 AR(1): e[t] = rho*e[t-1] + sqrt(1-rho^2)*z[t], z~N(0, eps_std^2)
    输出方差稳定为 eps_std^2
    shape: (N, T_obs)
    """
    N, T = shape
    e = np.empty((N, T), dtype=np.float64)
    z = rng.normal(0.0, eps_std, size=(N, T))
    if T == 0:
        return np.zeros((N, 0), dtype=np.float64)

    e[:, 0] = z[:, 0]
    coef = np.sqrt(max(0.0, 1.0 - rho * rho))
    for t in range(1, T):
        e[:, t] = rho * e[:, t - 1] + coef * z[:, t]
    return e


def load_feature_config(json_path: str, feat_names: list[str]) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "features" not in cfg or not isinstance(cfg["features"], list):
        raise KeyError("JSON 中必须包含 features: [feat1, feat2, ...]")

    feat_cfg = {}
    for f_name in feat_names:
        if f_name not in cfg:
            raise KeyError(f"JSON 顶层缺少特征 {f_name} 的配置块")

        item = cfg[f_name]
        try:
            feat_cfg[f_name] = {
                "bits": int(item["bits"]),
                "min": float(item["min"]),
                "max": float(item["max"]),
            }
        except KeyError as e:
            raise KeyError(f"特征 {f_name} 缺少字段 {e}")

    return feat_cfg


def encode_pt_to_binary(
    pt_path: str,
    json_path: str,
    out_pt_path: str | None = None,
    noise_ratio: float = 0.0,
    noise_seed: int | None = None,
    obs_len: int | None = None,
    ar_rho: float = 0.98,
    rel_min_scale: float = 0.01,       # 现在解释为：std 的倍数（0附近兜底）
    noise_features: list[str] | None = None,
):
    if noise_ratio < 0:
        raise ValueError("noise_ratio 必须 >= 0")
    if obs_len is not None and obs_len < 0:
        raise ValueError("obs_len 必须 >= 0 或 None")

    rng = np.random.default_rng(noise_seed) if noise_seed is not None else np.random.default_rng()

    obj = torch.load(pt_path, map_location="cpu")
    windows = obj["windows"]  # [N,T,C]
    meta = obj.get("meta", {})
    feat_names = list(meta["columns"])
    N, T, C = windows.shape

    # 推断观测段长度
    if obs_len is None:
        v = meta.get("obs_len", None)
        if isinstance(v, int) and v > 0:
            obs_len_ = min(v, T)
        else:
            v = meta.get("inp_seq_len", None)
            obs_len_ = min(v, T) if isinstance(v, int) and v > 0 else T
    else:
        obs_len_ = min(obs_len, T)

    # 特征噪声mask
    if noise_features is None:
        noise_mask = np.ones(C, dtype=bool)
    else:
        s = set(noise_features)
        noise_mask = np.array([f in s for f in feat_names], dtype=bool)

    # 从 json 读取 bits/min/max
    feat_cfg = load_feature_config(json_path, feat_names)
    total_bits = sum(int(feat_cfg[f]["bits"]) for f in feat_names)

    win_np = windows.cpu().numpy().astype(np.float64)

    # ===== 新增：每个特征的 std（用于兜底尺度）=====
    # 用全体 windows 统计；若你想只用观测段统计，把 : 改成 :obs_len_
    feat_std = np.std(win_np[:, :obs_len_, :].reshape(-1, C), axis=0)
    feat_std = np.where(feat_std > 0, feat_std, 0.0)

    codes_bits = np.empty((N, T, total_bits), dtype=np.uint8)
    codes_int = np.empty((N, T, C), dtype=np.int32)

    bit_offset = 0
    for j, feat in enumerate(feat_names):
        bits = int(feat_cfg[feat]["bits"])
        vmin = float(feat_cfg[feat]["min"])
        vmax = float(feat_cfg[feat]["max"])
        levels = (1 << bits) - 1

        vals = win_np[:, :, j]
        vals_clip = np.clip(vals, vmin, vmax)

        # === 只在观测段前 obs_len_ 步加时间相关噪声 ===
        vals_noisy = vals_clip
        if noise_ratio > 0 and obs_len_ > 0 and (vmax > vmin) and noise_mask[j]:
            x = vals_clip[:, :obs_len_]  # [N, Tobs]

            std_j = float(feat_std[j])  # 你前面已算好的每特征 std
            scale = rel_min_scale * std_j  # [N, Tobs]

            # --- 时间相关 eps：AR(1) ---
            # eps_std 推荐两种选法：
            # A) eps_std = noise_ratio            (直观：noise_ratio 越大噪声越强)
            eps_std = noise_ratio/np.sqrt(3)
            # eps_std = noise_ratio  # 选 A；如需 B 改成 noise_ratio/np.sqrt(3)

            eps_ar = gen_ar1_eps((x.shape[0], x.shape[1]), rho=ar_rho, eps_std=eps_std, rng=rng)  # [N, Tobs]

            # 噪声：delta = scale * eps_ar
            x_noisy = np.clip(x + scale * eps_ar, vmin, vmax)

            vals_noisy = vals_clip.copy()
            vals_noisy[:, :obs_len_] = x_noisy

        # 编码
        if vmax > vmin:
            idx = np.rint((vals_noisy - vmin) / (vmax - vmin) * levels).astype(np.int32)
        else:
            idx = np.zeros_like(vals_noisy, dtype=np.int32)
        idx = np.clip(idx, 0, levels)

        codes_int[:, :, j] = idx

        shifts = np.arange(bits - 1, -1, -1, dtype=np.int32)
        feat_bits = ((idx[..., None] >> shifts) & 1).astype(np.uint8)
        codes_bits[:, :, bit_offset:bit_offset + bits] = feat_bits
        bit_offset += bits

    deltas = codes_int[:, 1:, :] - codes_int[:, :-1, :] if T > 1 else np.zeros((N, 0, C), dtype=np.int32)

    obj["geo_codes_bits"] = torch.from_numpy(codes_bits)
    obj["geo_deltas_int"] = torch.from_numpy(deltas)

    be = obj.setdefault("meta", {}).setdefault("binary_encoding", {})
    be.update({
        "total_bits": int(total_bits),
        "feature_bits": {f: int(feat_cfg[f]["bits"]) for f in feat_names},
        "binary_config_path": os.path.abspath(json_path),
        "noise_type": "relative_proportional_uniform_std_floor",
        "noise_ratio": float(noise_ratio),
        "rel_min_scale": float(rel_min_scale),
        "noise_seed": (int(noise_seed) if noise_seed is not None else None),
        "obs_len_noised": int(obs_len_),
        "noise_features": (noise_features if noise_features is not None else "ALL"),
        "feature_std_used_for_floor": feat_std.tolist(),
    })

    save_path = out_pt_path if out_pt_path is not None else pt_path
    torch.save(obj, save_path)
    print("Finished")
    return save_path, obj["meta"]



def encode_geo_deltas_to_binary(pt_path, json_path):
    """
    对 PT 文件中的 geo_deltas 做二分区间式二进制编码。

    输入:
        pt_path: 包含 geo_deltas 的 .pt 文件
            geo_deltas 形状 [N, T-1, C]
            meta['columns'] 给出特征名顺序
        json_path: 含有每个特征 delta_bits / delta_lo / delta_hi 的 JSON

    输出:
        在原 PT 中:
            geo_deltas: 从 [N,T-1,C] int32 替换为 [N,T-1,total_delta_bits] uint8
        同时在 meta 中记录 delta_binary_encoding
    """
    obj = torch.load(pt_path, map_location="cpu")

    # 优先从 geo_deltas_int 读取整型差分
    if "geo_deltas_int" not in obj:
        raise KeyError("PT 文件中找不到 'geo_deltas_int'，先运行 encode_pt_to_binary 并确保保存了整型差分。")

    deltas = obj["geo_deltas_int"]  # [N, T-1, C]，整数差分
    meta = obj["meta"]
    feat_names = list(meta["columns"])
    N, Tm1, C = deltas.shape

    print(f"加载 {pt_path}: geo_deltas 形状={deltas.shape}, 特征={feat_names}")

    # 1) 读取 delta 编码配置
    delta_cfg = load_delta_config(json_path, feat_names)
    for f in feat_names:
        cfg = delta_cfg[f]
        print(f"delta 配置 {f:24s} : bits={cfg['bits']}, range=[{cfg['lo']}, {cfg['hi']}]")

    # 2) 准备编码
    deltas_np = deltas.cpu().numpy().astype(np.float64)  # [N,T-1,C]
    total_delta_bits = sum(delta_cfg[f]["bits"] for f in feat_names)
    deltas_bits_all = np.empty((N, Tm1, total_delta_bits), dtype=np.uint8)

    bit_offset = 0
    for j, feat in enumerate(feat_names):
        cfg = delta_cfg[feat]
        bits = cfg["bits"]
        dlo, dhi = cfg["lo"], cfg["hi"]

        print(f"\n编码 delta({feat}): bits={bits}, range=[{dlo},{dhi}]")

        # vals: [N, T-1]
        vals = deltas_np[:, :, j]

        # 1) 裁剪到 [dlo, dhi]
        vals_clip = np.clip(vals, dlo, dhi)

        # 2) 映射到整数 idx ∈ [0, 2^bits-1]
        levels = (1 << bits) - 1
        if dhi > dlo:
            idx = np.rint((vals_clip - dlo) / (dhi - dlo) * levels).astype(np.int32)
        else:
            idx = np.zeros_like(vals_clip, dtype=np.int32)
        idx = np.clip(idx, 0, levels)  # [N,T-1]

        # 3) 一次性展开为 bit 矩阵 [N,T-1,bits]
        shifts = np.arange(bits - 1, -1, -1, dtype=np.int32)
        feat_bits = ((idx[..., None] >> shifts) & 1).astype(np.uint8)

        # 4) 写入整体 bit 张量
        deltas_bits_all[:, :, bit_offset: bit_offset + bits] = feat_bits
        bit_offset += bits

    # 3) 替换 geo_deltas 为二进制编码
    obj["geo_deltas"] = torch.from_numpy(deltas_bits_all)

    # 4) 更新 meta
    if "delta_binary_encoding" not in meta:
        meta["delta_binary_encoding"] = {}

    meta["delta_binary_encoding"]["total_bits"] = int(total_delta_bits)
    meta["delta_binary_encoding"]["feature_delta_bits"] = {
        f: int(delta_cfg[f]["bits"]) for f in feat_names
    }
    meta["delta_binary_encoding"]["delta_range"] = {
        f: [float(delta_cfg[f]["lo"]), float(delta_cfg[f]["hi"])] for f in feat_names
    }
    meta["delta_binary_config_path"] = os.path.abspath(json_path)
    obj["meta"] = meta

    # 覆盖保存原文件
    torch.save(obj, pt_path)

    print("\n=== geo_deltas 编码完成 ===")
    print(f"已更新原文件: {pt_path}")
    print(f"geo_deltas (bits) 形状: {deltas_bits_all.shape}")

    return pt_path, meta


def load_delta_config(json_path, feature_names):
    """
    从 JSON 读取每个特征的 delta 配置:
    {feat: {"delta_bits":..., "delta_lo":..., "delta_hi":...}, ...}
    """
    with open(json_path, "r") as f:
        cfg = json.load(f)

    delta_cfg = {}
    for feat in feature_names:
        if feat not in cfg:
            raise KeyError(f"配置文件中缺少特征段: {feat}")

        item = cfg[feat]
        delta_cfg[feat] = {
            "bits": int(item["delta_bits"]),
            "lo": float(item["delta_lo"]),
            "hi": float(item["delta_hi"]),
        }

    return delta_cfg

if __name__ == "__main__":
    pt_path = "/home/h3c/dataset/SCAT/valid_noise.pt"
    json_path = "/home/h3c/project/python/FlightGPT/config_v20.json"

    encode_pt_to_binary(
        pt_path, json_path,
        out_pt_path=None,
        rel_min_scale=0.2,
        noise_ratio=0.00,
        noise_seed=2025,
        obs_len=36
    )

    encode_geo_deltas_to_binary(pt_path, json_path)

"""

0.0, 0.05, 0.1, 0.2   （std 的比例）

0.005, 0.01, 0.02, 0.03, 0.05

"""

