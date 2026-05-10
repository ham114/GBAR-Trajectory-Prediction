import os
import pandas as pd
import geohash2
import geohash

def add_geohash_to_csv(directory: str):
    """
    遍历指定目录及其所有子目录下的CSV文件，使用geohash2库计算lat和lon的七位地理哈希编码，
    并将其添加或覆盖至CSV文件的geohash_7列。
    """

    # 确保目录存在
    if not os.path.isdir(directory):
        raise ValueError(f"指定的目录不存在: {directory}")

    # 遍历目录及其子目录下所有CSV文件
    for root, _, files in os.walk(directory):

        for filename in files:
            if filename.endswith(".csv"):  # 仅处理CSV文件
                file_path = os.path.join(root, filename)

                try:
                    # 读取CSV文件
                    df = pd.read_csv(file_path)

                    # 检查是否存在lat和lon列
                    if 'lat' not in df.columns or 'lon' not in df.columns:
                        print(f"文件 {filename} 缺少 lat 或 lon 列，跳过处理。")
                        continue

                    # 计算geohash_7并添加到DataFrame
                    df['geohash_7'] = df.apply(lambda row: geohash2.encode(row['lat'], row['lon'], precision=7), axis=1)

                    # 保存回原文件
                    df.to_csv(file_path, index=False)
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")


def extract_segments(directory: str, output_directory: str):
    """
    检查已处理的CSV文件，提取 geohash_7 在指定区域内的轨迹片段，
    并将长度不少于 15 的片段单独保存到指定的输出目录。
    """
    valid_geohash_prefixes = {"u3ce", "u3c7", "u3cg", "u3cd", "u3c6", "u3cf", "u3cs", "u3ck", "u3cu"}

    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".csv"):  # 仅处理CSV文件
                file_path = os.path.join(root, filename)

                try:
                    # 读取CSV文件，保留原始索引
                    df = pd.read_csv(file_path)

                    # 检查是否存在 geohash_7 列
                    if 'geohash_7' not in df.columns:
                        print(f"文件 {filename} 缺少 geohash_7 列，跳过处理。")
                        continue

                    # 筛选 geohash_7 前四位在指定区域内的行，保留原始索引
                    df_filtered = df[df['geohash_7'].astype(str).str[:4].isin(valid_geohash_prefixes)].copy()

                    # 分割相邻的轨迹片段
                    segments = []
                    current_segment = []
                    prev_index = None

                    for idx, row in df_filtered.iterrows():
                        if prev_index is None or idx == prev_index + 1:
                            current_segment.append(row)
                        else:
                            if len(current_segment) >= 15:
                                segments.append(pd.DataFrame(current_segment))
                            current_segment = [row]
                        prev_index = idx

                    # 添加最后一个片段
                    if len(current_segment) >= 15:
                        segments.append(pd.DataFrame(current_segment))

                    # 保存每个片段
                    for idx, segment in enumerate(segments):
                        segment_filename = os.path.join(output_directory, f"{filename[:-4]}-segment-{idx + 1}.csv")
                        segment.to_csv(segment_filename, index=False)

                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")



if __name__ == "__main__":
    # 文件夹路径，替换为你的实际路径

    latitude = 55.53028
    longitude = 13.37167
    geohash_7 = geohash2.encode(latitude, longitude, precision=7)
    print(geohash_7)

    # add_geohash_to_csv('/home/userdata/2024_hyn/dataset/SCAT_csv/')

    base_geohash = "u3ce"


    # # 获取相邻的8个Geohash区域
    # neighbors = geohash.neighbors(base_geohash)
    #
    # print(neighbors)

    extract_segments('/home/userdata/2024_hyn/dataset/SCAT_csv/', '/home/userdata/2024_hyn/dataset/SCAT_segments/')


