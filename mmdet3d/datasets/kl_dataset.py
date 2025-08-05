import tempfile
from os import path as osp
from typing import Any, Dict
import copy
from pathlib import Path
import mmcv
import numpy as np
import pyquaternion
import torch
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS

from ..core.bbox import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset
from scipy.spatial.transform import Rotation as R

def check_nan_inf(arr):
    """
    检查数组中是否有 NaN 或 Inf，并打印其位置。
    
    :param arr: numpy 数组
    :return: True（如果有 NaN 或 Inf），否则 False
    """
    
    if arr.dtype.kind in {'U', 'S', 'O'}:
        # 判断是否是字符串类型
        if np.issubdtype(arr.dtype, np.str_) or np.issubdtype(arr.dtype, np.object_):
            # print("Array contains string data, skipping NaN/Inf check.")
            return False
        
    has_nan = np.isnan(arr)
    has_inf = np.isinf(arr)

    if np.any(has_nan):
        print("Found NaN at indices:", np.argwhere(has_nan))

    if np.any(has_inf):
        print("Found Inf at indices:", np.argwhere(has_inf))

    return np.any(has_nan) or np.any(has_inf)

def read_pcd_with_intensity(pcd_path):
    # 读取文件头
    with open(pcd_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line.startswith('DATA'):
                break

    # 解析字段、类型、大小
    fields, size, type_ = None, None, None
    for line in header:
        if line.startswith('FIELDS'):
            fields = line.split()[1:]
        elif line.startswith('SIZE'):
            size = list(map(int, line.split()[1:]))
        elif line.startswith('TYPE'):
            type_ = line.split()[1:]

    if fields is None or size is None or type_ is None:
        raise ValueError("Invalid PCD header: missing FIELDS/SIZE/TYPE")

    if not len(fields) == len(size) == len(type_):
        raise ValueError("FIELDS/SIZE/TYPE length mismatch")

    # 构建 dtype：根据 TYPE 和 SIZE 推断
    def get_numpy_dtype(t, s):
        if t == 'F':
            if s == 4:
                return np.float32
            elif s == 8:
                return np.float64
        elif t == 'U':
            if s == 1:
                return np.uint8
            elif s == 2:
                return np.uint16
            elif s == 4:
                return np.uint32
        elif t == 'I':
            if s == 1:
                return np.int8
            elif s == 2:
                return np.int16
            elif s == 4:
                return np.int32
        raise ValueError(f"Unsupported TYPE/SIZE combination: TYPE={t}, SIZE={s}")

    dtype = np.dtype([(f, get_numpy_dtype(t, s)) for f, t, s in zip(fields, type_, size)])

    # 计算数据起始位置
    data_offset = len('\n'.join(header)) + 1
    data = np.fromfile(pcd_path, dtype=dtype, offset=data_offset)

    # 检查字段存在
    required = {'x', 'y', 'z', 'intensity', 'ring'}
    if not required.issubset(data.dtype.names):
        raise ValueError(f"Missing required fields. Expected at least: {required}")

    # 构造输出数据（自动判断是否含有 timestamp_2us）
    base_fields = ['x', 'y', 'z', 'intensity', 'ring']
    base_fields = ['x', 'y', 'z', 'intensity']
    arrs = [data[f].astype(np.float32) for f in base_fields]

    # if 'timestamp_2us' in data.dtype.names:
    #     arrs.append(data['timestamp_2us'].astype(np.float32))

    all_data = np.vstack(arrs).T

    # 过滤含 NaN 的点
    valid_mask = ~np.isnan(all_data).any(axis=1)
    return all_data[valid_mask]


def read_pc(pc_file, verbose=False):
    """
    读取点云（支持.bin/.pcd），自动过滤NaN/Inf
    
    Args:
        pc_file: 文件路径（Path对象或字符串）
        verbose: 是否打印调试信息
        
    Returns:
        np.ndarray: (N, 4)的合法点云数据 [x, y, z, intensity]
    """
    pc_file = Path(pc_file)
    if not pc_file.exists():
        raise FileNotFoundError(f"Point cloud file not found: {pc_file}")

    try:
        if pc_file.suffix == '.bin':
            dtype = np.dtype([
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('intensity', np.float32), ('ring', np.float32),  # 根据实际格式调整
                ('timestamp_2us', np.float32)
            ])
            data = np.fromfile(pc_file, dtype=dtype)
            points = np.vstack((data['x'], data['y'], data['z'], data['intensity'])).T
            
        elif pc_file.suffix == '.pcd':
            points = read_pcd_with_intensity(pc_file)
            
        else:
            raise ValueError(f"Unsupported file format: {pc_file.suffix}")

        # 二次检查（防止上游未处理的情况）
        valid_mask = np.isfinite(points).all(axis=1)
        if np.any(~valid_mask):
            points = points[valid_mask]
            if verbose:
                print(f"Secondary filtering: Removed {np.sum(~valid_mask)} invalid points")

        # 空数据检查
        if len(points) == 0:
            raise ValueError(f"Empty point cloud after filtering: {pc_file}")

        points = points[np.max(np.abs(points[:, :3]), axis=1) < 1e3]  # 保留合理值,防止数值溢出
        return points

    except Exception as e:
        raise RuntimeError(f"Error reading {pc_file}: {str(e)}")

def transform_points(point_cloud, extrinsic):
    # 提取平移向量
    translation = np.array(extrinsic[:3])  # [Tx, Ty, Tz]

    # 提取四元数
    quaternion = np.array(extrinsic[3:])  # [qx, qy, qz, qw]
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    positions = point_cloud[:, :3]
    rotated_positions = np.dot(positions, rotation_matrix.T)
    transformed_positions = rotated_positions + translation
    point_cloud[:, :3] = transformed_positions
    return point_cloud


@DATASETS.register_module()
class KLDataset(Custom3DDataset):
    
    CLASSES = (
        "Pedestrian",
        "Car",
        "IGV-Full",
        "Truck",
        "Trailer-Empty",
        "Trailer-Full",
        "IGV-Empty",
        "Crane",
        "OtherVehicle",
        "Cone",
        "ContainerForklift",
        "Forklift",
        "Lorry",
        "ConstructionVehicle",
        "WheelCrane",
    )
    
    
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        custom_cfg=None,     # ✅ 接收自定义配置字典
    ) -> None:
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )
        self.map_classes = map_classes
        self.with_velocity = with_velocity
        self.eval_version = eval_version
        
        # ✅ 解析 custom_cfg 结构化字段
        self.custom_cfg = custom_cfg or {}

        self.filter_gt_by_points = self.custom_cfg.get('POINT_FILTER', {}).get('ENABLED', False)
        self.class_min_points_dict = self.custom_cfg.get('POINT_FILTER', {}).get('FILTER_MIN_POINTS_BY_CLASS', {})

        self.intensity_filter_cfg = self.custom_cfg.get("INTENSITY_FILTER", {})
        self.use_intensity_filter = self.intensity_filter_cfg.get("ENABLED", False)
        self.intensity_threshold = self.intensity_filter_cfg.get("THRESHOLD", 0.0)
        
        from nuscenes.eval.detection.config import config_factory

        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.use_camera=self.modality.get('use_camera', False)
        self.infos=self.data_infos
        
    def get_merged_lidar(self,index,use_extrinsic=True)->np.ndarray:
        
        point_clouds = []
        info = self.infos[index]
        # lidar_names=['helios_front_left','helios_rear_right']
        # lidar_extrinsic_names=['Tx_baselink_lidar_helios_front_left','Tx_baselink_lidar_helios_rear_right']

        extrinsic_names={}
        extrinsic_names['helios_front_left']='Tx_baselink_lidar_helios_front_left'
        extrinsic_names['helios_rear_right']='Tx_baselink_lidar_helios_rear_right'
        extrinsic_names['bp_front_left']='Tx_baselink_lidar_bp_front_left' # 向下补盲
        # extrinsic_names['bp_front_right']='Tx_baselink_lidar_bp_front_right' #向上补盲
        # extrinsic_names['bp_rear_left']='Tx_baselink_lidar_bp_rear_left' #向上补盲
        extrinsic_names['bp_rear_right']='Tx_baselink_lidar_bp_rear_right' #向下补盲
        
        # lidar_configurations=[]
        # lidar_configurations.append({"lidar_name": "helios_front_left","lidar_extrinsic_name": "Tx_baselink_lidar_helios_front_left"})
        # lidar_configurations.append({"lidar_name": "helios_rear_right","lidar_extrinsic_name": "Tx_baselink_lidar_helios_rear_right"})
        
        for lidar_name, lidar_extrinsic_name in extrinsic_names.items():
            # lidar_path = self.root_path / info['lidars'][lidar_name]
            lidar_path = info['lidars'][lidar_name]
            points=read_pc(lidar_path)
            
            # ⭐ 如果开启强度过滤
            if self.use_intensity_filter:
                intensity = points[:, 3]
                mask = intensity >= self.intensity_threshold
                points = points[mask]
            # times = np.zeros((points.shape[0], 1))
            # points = np.concatenate((points, times), axis=1)
            if use_extrinsic:
                lidar_extrinsic=info['sensor_extrinsics'][lidar_extrinsic_name]
                points=transform_points(points, lidar_extrinsic)
            point_clouds.append(points)
                
        if point_clouds:  # 如果列表不为空
            merged_point_cloud = np.concatenate(point_clouds, axis=0)

        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(merged_point_cloud[:, :3])
        # o3d.io.write_point_cloud("output_with_intensity.pcd", pcd, write_ascii=True)
        return merged_point_cloud
            

            
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos=list(data)
        data_infos.sort(key=lambda x: x['timestamp'])  # 升序排列
        # data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        # self.metadata = data["metadata"]
        # self.version = self.metadata["version"]
        return data_infos
    
    
    def get_data_info(self, index: int) -> Dict[str, Any]:
        
        info = copy.deepcopy(self.infos[index])
        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidars=info["lidars"],
            # sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            localization=info["localization"],
            sensor_extrinsics=info["sensor_extrinsics"],
        )
        
        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []
        # if self._merge_all_iters_to_one_epoch:
        #     index = index % len(self.infos)
        
        # info = copy.deepcopy(self.infos[index])

        # points=self.get_merged_lidar(index,True)
        # # check_nan_inf(points)
        # input_dict = {
        #     'points': points,
        #     'frame_id': Path(info['lidars']['helios_front_left']).stem,
        #     'metadata': {'token': info['token']}
        # }

        # if 'annos' in info:
        #     annos = info['annos']
        #     gt_names = annos['name']
        #     gt_boxes_lidar = annos['gt_boxes_lidar']
        #     gt_num_lidar_pts=annos['num_lidar_pts']
            
        #     # ⭐ 点数过滤逻辑开始 ⭐
        #     if getattr(self, 'filter_gt_by_points', False):
        #         keep_mask = np.ones(len(gt_names), dtype=bool)
        #         for i in range(len(gt_names)):
        #             cls = gt_names[i]
        #             min_pts = self.class_min_points_dict.get(cls, 0)
        #             if gt_num_lidar_pts[i] < min_pts:
        #                 keep_mask[i] = False

        #         gt_names = gt_names[keep_mask]
        #         gt_boxes_lidar = gt_boxes_lidar[keep_mask]
        #         gt_num_lidar_pts = gt_num_lidar_pts[keep_mask]
        #     # ⭐ 点数过滤逻辑结束 ⭐

        #     input_dict.update({
        #         'gt_names': gt_names,
        #         'gt_boxes': gt_boxes_lidar
        #         # 'gt_num_lidar_pts':gt_num_lidar_pts
        #     })

        # if self.use_camera:
        #     input_dict = self.load_camera_info(input_dict, info)

        # # data_dict = self.prepare_data(data_dict=input_dict)
        # data_dict=input_dict
        
        # if 'gt_boxes' in info:
        #     gt_boxes = data_dict['gt_boxes']
        #     gt_boxes[np.isnan(gt_boxes)] = 0
        #     data_dict['gt_boxes'] = gt_boxes
        
        
        # if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
        #     gt_boxes = data_dict['gt_boxes']
        #     gt_boxes[np.isnan(gt_boxes)] = 0
        #     data_dict['gt_boxes'] = gt_boxes

        # # if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
        # #     data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
        # data_dict['timestamp']=info['timestamp']
        # helios_front_left_path=info['lidars']['helios_front_left']
        # parts = helios_front_left_path.split('/')
        # sample_index = parts.index('sample')
        # folder = '/'.join(parts[sample_index+1:sample_index+3])
        # data_dict['folder']=folder
        
        annos = self.get_ann_info(index)
        data["ann_info"] = annos

        return data
    
    def get_ann_info(self, index):
        annos = self.infos[index]["annos"]
        return annos
