import torch


if __name__ == '__main__':

    # 替换为你自己的路径
    file = '/home/userdata/2024_hyn/dataset/nasa_dashlink/train.pt'


    feat_names = ["E", "N", "RALT", "PTCH", "ROLL", "MH", "TRK", "GS", "ALTR", "LONG", "VRTG", "LATG"]

    data = torch.load(file, map_location="cpu")

    print(data['raw_data'].shape) # [N,T,C]

    # if "Δ_bin" not in data:
    #     raise KeyError("pt 文件中未找到 'Δ_bin' 键。")
    #
    # Δ = data["Δ_bin"]  # 期望形状: (N, T, 12)，整型
    # if Δ.ndim != 3 or Δ.shape[-1] != len(feat_names):
    #     raise ValueError(f"Δ_bin 形状异常: {tuple(Δ.shape)}，应为 (N, T, 12)。")
    #
    # # 计算每个特征在全体 (N,T) 上的 min/max
    # mins = Δ.amin(dim=(0, 1))  # [12]
    # maxs = Δ.amax(dim=(0, 1))  # [12]
    #
    # print(f"Δ_bin dtype: {Δ.dtype}, shape: {tuple(Δ.shape)}\n")
    # print("=== 每个特征的区间差分最小/最大值 ===")
    # for i, name in enumerate(feat_names):
    #     print(f"{name:5s} -> min: {int(mins[i]) :>8d} | max: {int(maxs[i]) :>8d}")

    # print(data.keys())
    #
    # print(data['raw_data'].shape)
    #
    # print(data['X_bin'].shape)
    #
    # print(data['Δ_bin'].shape)
    #
    # print(data['Δ_bin'][0][46])

    # print(data['y_cls'].shape)

    # # 读取数据
    # train_data = torch.load('/home/userdata/2024_hyn/dataset/nasa_dashlink/train.pt')
    # val_data = torch.load('/home/userdata/2024_hyn/dataset/nasa_dashlink/val.pt')
    #
    # # y_cls 形状: [N, 4]，每一行是 one-hot
    # y_train = train_data['y_cls']
    # y_val = val_data['y_cls']
    #
    # # 转成标签索引（0~3）
    # labels_train = torch.argmax(y_train, dim=1)
    # labels_val = torch.argmax(y_val, dim=1)
    #
    # # 统计分布
    # train_counts = torch.bincount(labels_train, minlength=y_train.shape[1])
    # val_counts = torch.bincount(labels_val, minlength=y_val.shape[1])
    #
    # print("训练集标签分布：")
    # for i, c in enumerate(train_counts.tolist()):
    #     print(f"  类别 {i}: {c}")
    #
    # print("\n验证集标签分布：")
    # for i, c in enumerate(val_counts.tolist()):
    #     print(f"  类别 {i}: {c}")