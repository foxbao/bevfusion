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
        with_velocity=False,
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
            
    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids
            
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
        
        
        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        # ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        # ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        # lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        # lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego
        
        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []
            
        else:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []
            
            
            # 造假的相机
            fake_image_path = "/path/to/fake_image.jpg"  # 可以放一张黑图占位
            data["image_paths"].append(fake_image_path)

            # 假设相机在激光雷达前 1 米，高度 1.5 米
            trans = np.array([1.0, 0.0, 1.5], dtype=np.float32)
            rot = np.eye(3, dtype=np.float32)

            # lidar -> camera
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = rot
            lidar2camera_rt[:3, 3] = -trans  # 反向平移
            data["lidar2camera"].append(lidar2camera_rt)

            # camera intrinsics（假设 1920x1080 图像）
            fx = fy = 1000.0
            cx, cy = 960.0, 540.0
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[0, 0] = fx
            camera_intrinsics[1, 1] = fy
            camera_intrinsics[0, 2] = cx
            camera_intrinsics[1, 2] = cy
            data["camera_intrinsics"].append(camera_intrinsics)

            # lidar -> image
            lidar2image = camera_intrinsics @ lidar2camera_rt
            data["lidar2image"].append(lidar2image)

            # camera -> ego（假设相机和激光雷达同在车体坐标系中）
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = rot
            camera2ego[:3, 3] = trans
            data["camera2ego"].append(camera2ego)

            # camera -> lidar（直接取反）
            camera2lidar = np.linalg.inv(lidar2camera_rt)
            data["camera2lidar"].append(camera2lidar)
            
        annos = self.get_ann_info(index)
        data["ann_info"] = annos

        return data
    
    def get_ann_info(self, index):
        
        
        info = self.data_infos[index]
        
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results
