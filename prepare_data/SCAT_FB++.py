# coding=utf-8
import torch, json, numpy as np
from tqdm import tqdm
from utils import load_config_from_json

def encode_signed_int(arr_int: np.ndarray, bits: int) -> np.ndarray:
    """
    arr_int: (B,) 已经缩放并取整的有符号整数
    bits   : 总位数 (含符号)
    return : (B, bits) 0/1 ndarray
    """
    mag_bits = bits - 1
    limit = (1 << mag_bits) - 1
    arr_int = np.clip(arr_int, -limit, limit)

    sign = (arr_int < 0).astype(np.int8)
    mag  = np.abs(arr_int).astype(np.int64)

    # 将幅值拆比特 (无循环版)
    bit_idx = np.arange(mag_bits-1, -1, -1, dtype=np.int64)
    mag_bits_arr = ((mag[:, None] >> bit_idx) & 1).astype(np.int8)
    return np.concatenate([sign[:, None], mag_bits_arr], axis=1)

if __name__ == "__main__":

    ###############################################################################
    # 3. 读取原始 .pt 数据
    ###############################################################################

    config = load_config_from_json('/home/h3c/project/python/FlightGPT/config_v20.json')
    feat_names = config.features
    data = torch.load('/home/h3c/dataset/SCAT/valid.pt')
    data_flt = data['windows']
    data_flt = data_flt.cpu().numpy()  # 转为 numpy
    N, L, C = data_flt.shape
    assert C == len(config.features), f"特征数必须为 {len(config.features)}"

    data_int = np.empty_like(data_flt, dtype=np.int64)   # [N,L,12] 缩放取整后的整数
    diff_int = np.empty((N, L-1, C), dtype=np.int64)     # [N,L-1,12]

    ###############################################################################
    # 4. 逐列完成：缩放取整、差分、二进制编码
    ###############################################################################
    orig_bits_list, diff_bits_list = [], []

    for col, name in enumerate(tqdm(feat_names, desc="Encoding features", unit="feat")):
        # cfg = feat_cfg[name]
        cfg = getattr(config, name)
        # 4-1. 缩放取整
        x_int = np.rint(data_flt[..., col] * cfg.scale).astype(np.int64)   # [N,L]
        data_int[..., col] = x_int

        # 4-2. 原值编码
        orig_bits_list.append(
            encode_signed_int(x_int.reshape(-1), cfg.bits)
        )

        # 4-3. 差分
        dx_int = x_int[:, 1:] - x_int[:, :-1]         # [N,L-1]
        diff_int[..., col] = dx_int

        # 4-4. 差分编码（位宽沿用 cfg['bits']；如需单独位宽，可改成 cfg['d_bits']）
        diff_bits_list.append(
            encode_signed_int(dx_int.reshape(-1), cfg.bits - 2)
        )

    # 4-5. 拼接特征 bit-stream
    orig_bits_flat = np.concatenate(orig_bits_list, axis=1)         # (N*L, B_orig)
    diff_bits_flat = np.concatenate(diff_bits_list, axis=1)         # (N*(L-1), B_diff)

    B_orig = orig_bits_flat.shape[1]
    B_diff = diff_bits_flat.shape[1]

    orig_bits = orig_bits_flat.reshape(N, L,   B_orig).astype(np.int8)
    diff_bits = diff_bits_flat.reshape(N, L-1, B_diff).astype(np.int8)

    print("orig_bits:", orig_bits.shape, " diff_bits:", diff_bits.shape)

    ###############################################################################
    # 5. 保存
    ###############################################################################
    torch.save(
        {
            "orig_bits": torch.from_numpy(orig_bits),
            "diff_bits": torch.from_numpy(diff_bits),
            "orig_int":  torch.from_numpy(data_int)
        },
        "/home/h3c/dataset/SCAT/valid_fb.pt"
    )

    # in_path = '/home/userdata/2024_hyn/dataset/nasa_dashlink/binary_train.pt'
    # out_dir = '/home/userdata/2024_hyn/dataset/nasa_dashlink/'
    #
    # split_pt_sequential(
    #     in_path   = in_path,
    #     train_path= out_dir + 'binary_train.pt',
    #     val_path  = out_dir + 'binary_val.pt',
    #     val_ratio = 0.1
    # )
