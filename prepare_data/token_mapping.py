import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import json


def generate_tokenizer(input_csv: str, output_tokenizer: str):
    """
    读取CSV文件中的哈希编码，使用tokenizer进行分词并保存。

    :param input_csv: 输入的CSV文件路径
    :param output_tokenizer: 输出的tokenizer JSON文件路径
    """
    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 确保列名正确
    if 'Geohash_4' not in df.columns:
        raise ValueError("CSV文件缺少 'Geohash_4' 列")

    # 获取唯一哈希编码
    hashes = df["Geohash_4"].dropna().unique().tolist()

    # 初始化tokenizer
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # 训练tokenizer
    trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]"])
    tokenizer.train_from_iterator(hashes, trainer)

    # 保存tokenizer
    tokenizer.save(output_tokenizer)

    print(f"Tokenizer 已保存至: {output_tokenizer}")


def generate_extended_geohash_tokenizer(input_csv: str, output_tokenizer: str):
    """
    读取4位地理哈希编码，并基于此拓展生成5位哈希编码，
    使用tokenizer进行分词并保存到JSON文件。

    :param input_csv: 输入的CSV文件路径
    :param output_tokenizer: 输出的tokenizer JSON文件路径
    """
    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 确保列名正确
    if 'Geohash_4' not in df.columns:
        raise ValueError("CSV文件缺少 'Geohash_4' 列")

    # 获取唯一的4位哈希编码
    base_hashes = df["Geohash_4"].dropna().unique().tolist()

    # Geohash 字符表（标准Geohash编码使用的32个字符）
    geohash_chars = "0123456789bcdefghjkmnpqrstuvwxyz"

    # 生成5位的地理哈希编码
    extended_hashes = sorted({base + char for base in base_hashes for char in geohash_chars})

    # 初始化tokenizer
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # 训练tokenizer，增加 vocab_size 以适应所有 Token
    trainer = trainers.WordLevelTrainer(vocab_size=len(extended_hashes) + 100, special_tokens=["[UNK]"])
    tokenizer.train_from_iterator(extended_hashes, trainer)

    # 保存tokenizer
    tokenizer.save(output_tokenizer)

    print(f"拓展后的5位地理哈希编码 Tokenizer 已保存至: {output_tokenizer}")


# 示例用法
if __name__ == "__main__":
    input_csv_path = "geohash_4.csv"  # 需替换为实际文件路径
    output_tokenizer_path = "tokenizer_1.json"
    output_extended_tokenizer_path = "tokenizer_2.json"

    # generate_tokenizer(input_csv_path, output_tokenizer_path)
    generate_extended_geohash_tokenizer(input_csv_path, output_extended_tokenizer_path)
