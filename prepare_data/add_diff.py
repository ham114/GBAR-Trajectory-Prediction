import json
import torch
import torch.nn.functional as F



if __name__ == '__main__':

    data = torch.load('/home/userdata/2024_hyn/dataset/nasa_dashlink/pred_train.pt')
    for k, v in data.items():
        print(f"{k}: {v.shape}")

    # path = '/home/userdata/2024_hyn/dataset/nasa_dashlink/pred_train.pt'
    #
    # data = torch.load(path)
    #
    # print(data.keys())
    #
    # raw_data = data['raw_data'] # [N, L, C]
    #
    # diff_data = data['diff_data']
    #
    # print(raw_data.shape)
    # print(diff_data.shape)
    #
    # print(data['diff_stat'])

    #
    # # 1. 做相邻时间步差分
    # diff_data = raw_data[:, 1:, :] - raw_data[:, :-1, :]  # [N, L-1, C]
    #
    # # 2. 计算全局每个特征的最大最小差值
    # max_vals, _ = diff_data.view(-1, diff_data.shape[-1]).max(dim=0)  # [C]
    # min_vals, _ = diff_data.view(-1, diff_data.shape[-1]).min(dim=0)  # [C]
    # stat = torch.stack([min_vals, max_vals], dim=1)  # [C, 2]
    #
    # # 3. 保存回原文件
    # data['diff_data'] = diff_data  # 替换原始数据
    # # data['diff_stat'] = stat      # 新增一个统计信息
    #
    # torch.save(data, path)
    # print(f"Saved diff_data and stat to {path}")
    # print(f"Feature-wise diff min/max:\n{stat}")

    global_stat = torch.tensor([[ -1,   1],
        [ -1,   1],
        [-50,  50],
        [ -50,   50],
        [ -50,   50],
        [-50,  50],
        [-50,  50],
        [-53,  50],
        [-50,  53],
        [ -2,   2],
        [ -5,   5],
        [ -1,   1]])

    # 保存路径
    train_path = '/home/userdata/2024_hyn/dataset/nasa_dashlink/pred_train.pt'
    val_path = '/home/userdata/2024_hyn/dataset/nasa_dashlink/pred_val.pt'

    # 加载并添加字段
    for path in [train_path, val_path]:
        data = torch.load(path)
        data['diff_stat'] = global_stat
        torch.save(data, path)
        print(f"已将全局归一化因子保存至 {path}")

"""
tensor([[ -0.2902,   0.4522],
        [ -0.3179,   0.3105],
        [-42.0541,  42.0000],
        [ -5.6789,   3.9191],
        [ -8.6779,   8.6032],
        [-13.6226,  11.5893],
        [-12.0922,  13.6780],
        [-52.8859,   5.1169],
        [-40.7420,  52.2114],
        [ -1.1190,   1.1189],
        [ -4.8193,   4.9946],
        [ -0.9975,   0.9808]])
        
        [ -0.2769,   0.3152],
        [ -0.2946,   0.2780],
        [-42.0541,  42.0000],
        [ -2.9115,   3.5332],
        [ -8.1874,   6.4454],
        [-13.6226,  10.0152],
        [ -4.5350,   4.2594],
        [-52.8810,   3.3604],
        [-22.4153,  20.3570],
        [ -1.1174,   1.1186],
        [ -4.6253,   4.6095],
        [ -0.9975,   0.9808]
        
"""

