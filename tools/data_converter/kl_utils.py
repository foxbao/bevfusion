import numpy as np



def generate_token():
    import uuid
    return str(uuid.uuid4())

# 预计算点云文件的时间戳
def precompute_timestamps(pointcloud_files):
    """
    预计算点云文件的时间戳。
    :param pointcloud_files: 点云文件列表（Path 对象）
    :return: 时间戳列表（整数列表）
    """
    return [float(f.stem) for f in pointcloud_files]



# 查找最近的点云文件
def find_nearest_data(timestamp, data_files, data_timestamps=None):
    """
    根据时间戳查找最近的传感器文件。
    :param timestamp: 标注文件的时间戳（整数或字符串）
    :param data_files: 传感器文件列表（Path 对象）
    :param data_timestamps: 预计算的时间戳列表（可选）
    :return: 最近的传感器文件路径（字符串）
    """
    timestamp = float(timestamp)  # 确保时间戳是整数

    # 如果没有预计算时间戳，则实时计算
    if data_timestamps is None:
        data_timestamps = [float(f.stem) for f in data_files]

    # 使用 NumPy 计算最小差值
    diffs = np.abs(np.array(data_timestamps) - timestamp)
    nearest_index = np.argmin(diffs)
    return str(data_files[nearest_index])

# 使用二分查找优化
def find_nearest_pointcloud_bisect(timestamp, pointcloud_files, pointcloud_timestamps=None):
    from bisect import bisect_left
    """
    根据时间戳查找最近的点云文件（使用二分查找优化）。
    :param timestamp: 标注文件的时间戳（整数或字符串）
    :param pointcloud_files: 点云文件列表（Path 对象）
    :param pointcloud_timestamps: 预计算的时间戳列表（可选）
    :return: 最近的点云文件路径（字符串）
    """
    timestamp = float(timestamp)  # 确保时间戳是整数

    # 如果没有预计算时间戳，则实时计算
    if pointcloud_timestamps is None:
        pointcloud_timestamps = [float(f.stem) for f in pointcloud_files]

    # 使用二分查找找到最近的索引
    pos = bisect_left(pointcloud_timestamps, timestamp)
    if pos == 0:
        nearest_index = 0
    elif pos == len(pointcloud_timestamps):
        nearest_index = len(pointcloud_timestamps) - 1
    else:
        # 比较左右两个时间戳，选择更接近的一个
        before = pointcloud_timestamps[pos - 1]
        after = pointcloud_timestamps[pos]
        nearest_index = pos - 1 if (timestamp - before) <= (after - timestamp) else pos

    return str(pointcloud_files[nearest_index])


def match_sensor_data(timestamp:str,sensor_files,sensor_timestamps):
    timestamp = float(timestamp)
    return find_nearest_data(timestamp, sensor_files, sensor_timestamps)

def match_multi_sensor_data(timestamp:str, multi_sensor_files:dict,multi_sensor_timestamps)->dict:
    data_dict = {}
    timestamp = float(timestamp)  # 确保时间戳是整数

    for sensor_name in multi_sensor_files:
        sensor_files=multi_sensor_files[sensor_name]
        sensor_timestamps=multi_sensor_timestamps[sensor_name]
        if not sensor_files:  # 如果列表为空，跳过这个 sensor
            continue
        nearest_file = find_nearest_data(timestamp, sensor_files, sensor_timestamps)
        data_dict[sensor_name] = str(nearest_file)
    return data_dict