import os
import shutil
import random


def split_csv_files(data_dir, train_dir, val_dir, split_ratio=0.9, random_seed=None):
    """
    将目录下的CSV文件按照比例随机划分到训练集和验证集

    Args:
        data_dir (str): 包含CSV文件的源目录路径
        train_dir (str): 训练集目标目录路径
        val_dir (str): 验证集目标目录路径
        split_ratio (float): 训练集比例，默认0.9（90%）
        random_seed (int): 随机种子，用于可重复的结果
    """
    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)

    # 创建目标目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"在目录 {data_dir} 中没有找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 随机打乱文件列表
    random.shuffle(csv_files)

    # 计算分割点
    split_point = int(len(csv_files) * split_ratio)

    # 划分文件
    train_files = csv_files[:split_point]
    val_files = csv_files[split_point:]

    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")

    # 复制训练集文件
    for file in train_files:
        src_path = os.path.join(data_dir, file)
        dst_path = os.path.join(train_dir, file)
        shutil.copy2(src_path, dst_path)

    # 复制验证集文件
    for file in val_files:
        src_path = os.path.join(data_dir, file)
        dst_path = os.path.join(val_dir, file)
        shutil.copy2(src_path, dst_path)

    print("文件划分完成！")
    print(f"训练集文件保存在: {train_dir}")
    print(f"验证集文件保存在: {val_dir}")


# 使用示例
if __name__ == "__main__":
    # 设置路径
    source_directory = "/home/h3c/dataset/SCAT/PART/"  # 替换为你的CSV文件目录
    train_directory = "/home/h3c/dataset/SCAT/train/1/"  # 训练集目录
    val_directory = "/home/h3c/dataset/SCAT/valid/1/"  # 验证集目录

    # 调用函数
    split_csv_files(
        data_dir=source_directory,
        train_dir=train_directory,
        val_dir=val_directory,
        split_ratio=0.9,
        random_seed=42  # 设置随机种子确保结果可重复
    )