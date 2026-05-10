import os
import json
import csv
import math
import geohash2
from datetime import datetime
from collections import OrderedDict
import random
from torch.utils.data import Dataset
from tqdm import tqdm
import shutil

def process_json_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_path = os.path.join(input_folder, filename)

            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取 `plots` 和 `id`
            plots = data.get('plots', [])
            flight_id = data.get('id')

            # 检查文件是否包含至少一个 `I062/380`
            contains_i062_380 = any('I062/380' in plot for plot in plots)
            if not contains_i062_380:
                print(f"Skipped {filename}: No I062/380 found in plots.")
                continue

            # 如果航迹点数量超过 500
            if len(plots) > 100:
                output_file = os.path.join(output_folder, f"{flight_id}.csv")

                rows = []
                fieldnames = OrderedDict()  # 使用 OrderedDict 保证字段顺序

                for plot in plots:
                    row = OrderedDict()  # 确保每行数据的字段顺序一致

                    # 提取 `time_of_track`
                    time_of_track = plot.get('time_of_track')
                    if time_of_track:
                        formatted_time = datetime.fromisoformat(time_of_track).strftime('%Y-%m-%d %H:%M:%S')
                        row['time_of_track'] = formatted_time
                        fieldnames['time_of_track'] = None

                    # 遍历所有字段
                    for key, value in plot.items():
                        if key == 'I062/380':  # 特殊处理 I062/380
                            if isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    if isinstance(sub_value, dict):  # 如果还有嵌套，继续展开
                                        for sub_sub_key, sub_sub_value in sub_value.items():
                                            row[sub_sub_key] = sub_sub_value
                                            fieldnames[sub_sub_key] = None
                                    else:
                                        row[sub_key] = sub_value
                                        fieldnames[sub_key] = None
                        elif isinstance(value, dict):  # 处理其他嵌套字段
                            for sub_key, sub_value in value.items():
                                row[sub_key] = sub_value
                                fieldnames[sub_key] = None
                        elif key != 'time_of_track':  # 处理非嵌套字段
                            row[key] = value
                            fieldnames[key] = None

                    rows.append(row)

                # 写入 CSV 文件
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=list(fieldnames.keys()))
                    writer.writeheader()
                    writer.writerows(rows)


# 定义所需的列
REQUIRED_COLUMNS = [
    "time_of_track", "lat", "lon", "measured_flight_level", "vx", "vy",
    "adf", "long", "trans", "vert", "rocd", "baro_vert_rate",
    "ias", "mach", "mag_hdg", "altitude", "ah", "am", "mv"
]

# 必须检查完整性的列
CRITICAL_COLUMNS = ["lat", "lon", "vx", "vy", "rocd", "altitude", "measured_flight_level"]


def validate_and_clean_csv(output_folder, required_columns=REQUIRED_COLUMNS, critical_columns=CRITICAL_COLUMNS, min_rows=100):
    """
    验证和清理 CSV 文件：
    - 检查文件是否包含所有指定列，缺失则删除文件。
    - 检查每一行关键列是否缺失，或特定列值为 NA，缺失则删除行。
    - 如果文件清理后不足 500 行，删除文件。

    参数:
        output_folder (str): 输出文件夹路径。
        required_columns (list): 必须包含的列。
        critical_columns (list): 每行中不能缺失的关键列。
        min_rows (int): 文件中最小行数要求。
    """
    for csv_file in os.listdir(output_folder):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(output_folder, csv_file)

            try:
                # 检查列名
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    header = reader.fieldnames

                    # 如果列名不包含所有必要列，则删除文件
                    if not all(col in header for col in required_columns):
                        print(f"Deleting {csv_file}: Missing required columns.")
                        f.close()  # 确保文件关闭后删除
                        os.remove(csv_path)
                        continue

                # 检查每行关键列是否有缺失值，或 flight_phase 列为 NA
                cleaned_rows = []
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # 如果关键列中有缺失值，或者 flight_phase 列为 "NA"，则跳过该行
                        if any(row[col] in (None, '', 'NaN') for col in critical_columns):
                            continue
                        cleaned_rows.append(row)

                # 如果清理后行数不足 500，删除文件
                if len(cleaned_rows) < min_rows:
                    print(f"Deleting {csv_file}: Less than {min_rows} rows after cleaning.")
                    os.remove(csv_path)
                    continue

                # 写回清理后的文件
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(cleaned_rows)

            except PermissionError as e:
                print(f"Error deleting {csv_file}: {e}")
                continue



FLIGHT_PHASE_MAPPING = {
    "GND": "1",  # 地面
    "CL": "2",   # 爬升
    "DE": "3",   # 下降
    "CR": "4",   # 巡航
    "LVL": "5",  # 平飞
    "TO": "6",   # 起飞
    "APP": "7"   # 进近
}


def convert_csv_to_txt(output_folder, txt_folder):
    """
    将 CSV 文件转换为 TXT 文件，并根据指定字段和规则进行计算。
    如果同名 TXT 文件存在，则覆盖它。

    参数:
        output_folder (str): 存储 CSV 文件的文件夹路径。
        txt_folder (str): 转换后 TXT 文件的存储路径。
    """
    # 确保 TXT 输出文件夹存在
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    for csv_file in os.listdir(output_folder):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(output_folder, csv_file)
            txt_path = os.path.join(txt_folder, f"{os.path.splitext(csv_file)[0]}.txt")

            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                # 打开 TXT 文件进行写入（覆盖模式）
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    for row in reader:
                        # 提取字段并计算
                        ts = row.get('time_of_track', '0')
                        lon = float(row.get('lon', '0'))
                        lat = float(row.get('lat', '0'))
                        alt = str(float(row.get('measured_flight_level', '0')) * 100)  # 高度（×100）
                        vx = abs(float(row.get('vx', '0')))  # 速度分量 X
                        vy = abs(float(row.get('vy', '0')))  # 速度分量 Y
                        vz = row.get('rocd', '0')  # 爬升/下降率

                        # 计算 7 位 Geohash 并截取
                        geohash_7 = row.get('geohash_7', '0')
                        geohash_3 = geohash_7[:3]
                        geohash_5 = geohash_7[:5]

                        # 构造每行内容
                        txt_row = f"{ts}|{geohash_3}|{geohash_5}|{geohash_7}|{lon}|{lat}|{alt}|{vx}|{vy}|{vz}|0|0\n"
                        txt_file.write(txt_row)


def check_generated_txt(txt_folder, output_csv):
    """
    检查指定目录下所有 TXT 文件，提取 3 位地理哈希码的唯一集合，并保存到 CSV 文件。

    如果 CSV 文件已存在，则保留原有数据，并添加新数据，去重后写入。

    参数:
        txt_folder (str): 包含 TXT 文件的目录路径。
        output_csv (str): 输出的 CSV 文件路径。
    """
    geohash_4_set = set()

    # 读取已有的 CSV 文件内容
    if os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过表头
            for row in reader:
                if row:
                    geohash_4_set.add(row[0])

    # 遍历 TXT 文件并提取 4 位地理哈希码
    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(txt_folder, txt_file)

            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) > 1:
                        geohash_4_set.add(parts[2][:4])  # 假设第 2 列是 4 位哈希码

    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将去重后的集合写入 CSV 文件
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Geohash_4"])
        for geohash in sorted(geohash_4_set):
            writer.writerow([geohash])


def split_dataset(input_directory: str, output_directory: str, train_ratio: float = 0.8, seed: int = 42):
    """
    访问输入目录下的所有txt文件，按给定比例随机划分为训练集和测试集，
    并复制到目标目录的 Train 和 Valid 文件夹中。
    """
    random.seed(seed)

    # 获取所有 txt 文件
    file_list = [os.path.join(root, f) for root, _, files in os.walk(input_directory) for f in files if
                 f.endswith('.txt')]

    if not file_list:
        print(f"Error: No txt files found in {input_directory}!")
        return

    # 随机打乱文件列表
    random.shuffle(file_list)

    # 计算训练集和测试集的分界点
    split_idx = int(len(file_list) * train_ratio)
    train_files = file_list[:split_idx]
    valid_files = file_list[split_idx:]

    # 创建目标文件夹
    train_dir = os.path.join(output_directory, "Train")
    valid_dir = os.path.join(output_directory, "Valid")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # 复制文件到对应目录
    for file in train_files:
        shutil.copy(file, os.path.join(train_dir, os.path.basename(file)))
    for file in valid_files:
        shutil.copy(file, os.path.join(valid_dir, os.path.basename(file)))

    print(f"Dataset split completed! Train: {len(train_files)}, Valid: {len(valid_files)}")


if __name__ == "__main__":

    folders = ['scat20161015_20161021', 'scat20161112_20161118', 'scat20161210_20161216', 'scat20170107_20170113',
              'scat20170215_20170221', 'scat20170304_20170310', 'scat20170401_20170407', 'scat20170429_20170505',
              'scat20170527_20170602', 'scat20170624_20170630', 'scat20170722_20170728', 'scat20170819_20170825',
               'scat20170916_20170922'
              ]
    for folder in folders:
        input_folder = "E:/datasets/SCAT_dataset/" + folder
        output_folder = "/home/userdata/2024_hyn/dataset/SCAT/SCAT_csv/" + folder
        txt_folder = "/home/userdata/2024_hyn/dataset/SCAT/txt4FTP/" + folder
        # process_json_files(input_folder, output_folder)
        # validate_and_clean_csv(output_folder)

        # convert_csv_to_txt(output_folder, txt_folder)

    # check_generated_txt("/home/hyn/dataset/SCAT/Valid", "/home/hyn/dataset/SCAT/geohash_4.csv")

    convert_csv_to_txt('/home/userdata/2024_hyn/dataset/SCAT_segments/', '/home/userdata/2024_hyn/dataset/SCAT_segments_txt/')

    split_dataset('/home/userdata/2024_hyn/dataset/SCAT_segments_txt/','/home/userdata/2024_hyn/dataset/SCAT_segments4FTP/', 0.9)

