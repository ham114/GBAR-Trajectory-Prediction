import torch
import random

# 定义一个函数把多个键对应的样本迁移
def move_samples(data_from, data_to, indices):
    for k in data_from.keys():
        data_to[k] = torch.cat([data_to[k], data_from[k][indices]], dim=0)
        mask = torch.ones(len(data_from[k]), dtype=torch.bool)
        mask[indices] = False
        data_from[k] = data_from[k][mask]


if __name__ == '__main__':

    # file = '/home/userdata/2024_hyn/dataset/nasa_dashlink/pred.pt'
    # data = torch.load(file)
    #
    # # 样本总数
    # n = data['X_bin'].shape[0]
    # split_idx = int(0.9 * n)  # 90%
    #
    # # 顺序切分并 clone，避免保存时引用整个大tensor
    # train_data = {
    #     'raw_data': data['raw_data'][:split_idx].clone(),
    #     'X_bin': data['X_bin'][:split_idx].clone(),
    #     'Δ_bin': data['Δ_bin'][:split_idx].clone(),
    #     'y_cls': data['y_cls'][:split_idx].clone()
    # }
    #
    # val_data = {
    #     'raw_data': data['raw_data'][split_idx:].clone(),
    #     'X_bin': data['X_bin'][split_idx:].clone(),
    #     'Δ_bin': data['Δ_bin'][split_idx:].clone(),
    #     'y_cls': data['y_cls'][split_idx:].clone()
    # }
    #
    # torch.save(train_data, '/home/userdata/2024_hyn/dataset/nasa_dashlink/train.pt')
    # torch.save(val_data, '/home/userdata/2024_hyn/dataset/nasa_dashlink/val.pt')



    # 路径
    train_path = '/home/userdata/2024_hyn/dataset/nasa_dashlink/train.pt'
    val_path = '/home/userdata/2024_hyn/dataset/nasa_dashlink/val.pt'

    # 加载
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)

    # 稀有类 = 3
    y_train = train_data['y_cls']
    rare_indices = (torch.argmax(y_train, dim=1) == 3).nonzero(as_tuple=True)[0].tolist()

    print(f"训练集中稀有类样本总数: {len(rare_indices)}")  # 应该是 46

    # 随机挑选 4 个
    selected = random.sample(rare_indices, 4)
    print("移到验证集的索引:", selected)


    # 迁移
    move_samples(train_data, val_data, selected)

    print("迁移后:")
    print("  训练集:", train_data['y_cls'].shape)
    print("  验证集:", val_data['y_cls'].shape)

    # 保存覆盖
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
