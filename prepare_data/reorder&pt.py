import json
import torch
import numpy as np
import math

if __name__ == '__main__':

    path = '/home/userdata/2024_hyn/project/python/FlightGPT/prepare_data/all_stats.json'

    # # 1. 加载原始 JSON 文件
    # with open(path, "r") as f:
    #     data = json.load(f)
    #
    # # 2. 给定特征排序顺序
    # desired_order = ["E", "N", "RALT", "PTCH", "ROLL", "MH", "TRK", "GS", "ALTR", "LONG", "VRTG", "LATG"]
    #
    # # 3. 重排序每个编号下的字段
    # reordered_data = {}
    # for key, stats in data.items():
    #     reordered_data[key] = {feat: stats[feat] for feat in desired_order if feat in stats}
    #
    # # 4. 保存为新的 JSON 文件
    # with open(path, "w") as f:
    #     json.dump(reordered_data, f, indent=2)



    with open(path, "r") as f:
        data = json.load(f)
    # 固定阶段顺序
    stage_keys = sorted(data.keys(), key=int)

    # 特征顺序
    feature_order = ["E", "N", "RALT", "PTCH", "ROLL", "MH", "TRK", "GS", "ALTR", "LONG", "VRTG", "LATG"]

    # 初始化张量 [4, 12, 6]
    num_stages = len(stage_keys)
    num_features = len(feature_order)
    max_stats = 6
    tensor = torch.zeros((num_stages, num_features, max_stats), dtype=torch.float32)

    # 逐个填充
    for i, stage in enumerate(stage_keys):
        stage_data = data[stage]
        for j, feat in enumerate(feature_order):
            feat_stats = stage_data.get(feat, {})
            values = []
            for stat_name, value in feat_stats.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    values.append(float(value))
                else:
                    values.append(0.0)
            values = values[:max_stats]  # 最多保留6个
            if len(values) < max_stats:
                values += [0.0] * (max_stats - len(values))  # 不足补0
            tensor[i, j, :] = torch.tensor(values, dtype=torch.float32)

    print(tensor[0])

    # 保存为 .pt 文件
    torch.save(tensor, "phases_stats_36.pt")
    print("✅ 保存成功：phases_stats_36.pt")

