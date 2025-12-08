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
from collections import defaultdict
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import BboxOverlaps3D,BboxOverlapsNearest3D, AxisAlignedBboxOverlaps3D


def iou3d(box_a, box_b):
    # 简化版 IoU 计算（AABB，不考虑旋转）
    # box: [x, y, z, l, w, h, ry]
    ax1, ay1, az1 = box_a[0]-box_a[3]/2, box_a[1]-box_a[4]/2, box_a[2]-box_a[5]/2
    ax2, ay2, az2 = box_a[0]+box_a[3]/2, box_a[1]+box_a[4]/2, box_a[2]+box_a[5]/2
    bx1, by1, bz1 = box_b[0]-box_b[3]/2, box_b[1]-box_b[4]/2, box_b[2]-box_b[5]/2
    bx2, by2, bz2 = box_b[0]+box_b[3]/2, box_b[1]+box_b[4]/2, box_b[2]+box_b[5]/2
    ix1, iy1, iz1 = max(ax1,bx1), max(ay1,by1), max(az1,bz1)
    ix2, iy2, iz2 = min(ax2,bx2), min(ay2,by2), min(az2,bz2)
    iw, ih, id_ = max(ix2-ix1,0), max(iy2-iy1,0), max(iz2-iz1,0)
    inter = iw*ih*id_
    vol_a = (ax2-ax1)*(ay2-ay1)*(az2-az1)
    vol_b = (bx2-bx1)*(by2-by1)*(bz2-bz1)
    return inter / (vol_a+vol_b-inter+1e-6)

def compute_ap(recall, precision):
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if np.any(recall >= t) else 0
        ap += p / 11
    return ap
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

def convert_yaw_to_mmdet3d(bboxes: np.ndarray) -> np.ndarray:
    """
    Convert 3D boxes yaw to mmdet3d LiDAR coordinate system.
    Does not modify z coordinate.

    Args:
        bboxes: np.ndarray, shape [N, 7+] (x, y, z, l, w, h, yaw, ...)

    Returns:
        bboxes with yaw converted
    """
    if bboxes is None or len(bboxes) == 0:
        return bboxes

    bboxes = bboxes.copy()  # 避免修改原数组
    yaw_orig = bboxes[..., 6]
    yaw_mmdet3d = - yaw_orig
    yaw_mmdet3d = (yaw_mmdet3d + np.pi) % (2 * np.pi) - np.pi
    bboxes[..., 6] = yaw_mmdet3d
    return bboxes


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
                use_camera=True,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.use_camera=self.modality.get('use_camera', False)
        self.infos=self.data_infos
        # self.fixed_cams = [
        #     # "h100f1a_front_left",
        #     # "h100f1a_rear_right",
        #     # "h120ua_front_left",
        #     "h120ua_front_mid",
        #     # "h120ua_front_right",
        #     # "h120ua_rear_left",
        #     # "h120ua_rear_mid",
        #     # "h120ua_rear_right"
        # ]
        self.fixed_cams = [
            # "h100f1a_front_left",
            # "h100f1a_rear_right",
            # "h120ua_front_left",
            "front_image",
            # "h120ua_front_right",
            # "h120ua_rear_left",
            # "h120ua_rear_mid",
            # "h120ua_rear_right"
        ]
        
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
        
        # info = copy.deepcopy(self.infos[index])
        info = self.data_infos[index]
        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidars=info["lidars"],
            # sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            localization=info["localization"],
            sensor_extrinsics=info["sensor_extrinsics"],
            label_path=info["label_path"],
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
            
            for cam_name in self.fixed_cams:
                if cam_name in info["cams"]:
                    camera_info = info["cams"][cam_name]

                    # --- 图像路径 ---
                    data["image_paths"].append(camera_info["data_path"])
                    
                    # lidar to camera transform
                    lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
                    lidar2camera_t = (
                        camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                    )
                    lidar2camera_rt = np.eye(4).astype(np.float32)
                    lidar2camera_rt[:3, :3] = lidar2camera_r.T
                    lidar2camera_rt[3, :3] = -lidar2camera_t
                    data["lidar2camera"].append(lidar2camera_rt.T)
                    # print("----------")
                    # print(data["lidar2camera"])
                    # camera intrinsics
                    camera_intrinsics = np.eye(4).astype(np.float32)
                    # --- 内参（鱼眼） ---
                    intrinsics = camera_info["camera_intrinsics"]
                    # 如果是字典，把 fx/fy/cx/cy 转成 4x4 矩阵
                    if isinstance(intrinsics, dict):
                        camera_intrinsics = np.eye(4, dtype=np.float32)
                        camera_intrinsics[0, 0] = intrinsics.get('fx', 1.0)
                        camera_intrinsics[1, 1] = intrinsics.get('fy', 1.0)
                        camera_intrinsics[0, 2] = intrinsics.get('cx', 0.0)
                        camera_intrinsics[1, 2] = intrinsics.get('cy', 0.0)
                        data["camera_intrinsics"].append(camera_intrinsics)
                    else:
                        # 已经是矩阵的话直接 append
                        data["camera_intrinsics"].append(intrinsics)

                    # lidar to image transform
                    lidar2image = camera_intrinsics @ lidar2camera_rt.T
                    data["lidar2image"].append(lidar2image)

                    # camera to ego transform
                    camera2ego = np.eye(4).astype(np.float32)
                    camera2ego[:3, :3] = Quaternion(
                        camera_info["sensor2ego_rotation"]
                    ).rotation_matrix
                    camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                    data["camera2ego"].append(camera2ego)

                    # camera to lidar transform
                    camera2lidar = np.eye(4).astype(np.float32)
                    camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                    camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                    data["camera2lidar"].append(camera2lidar)

                else:
                    # 缺失相机：用黑图和单位矩阵占位
                    H, W = 900, 1600  # 按照你的图像分辨率来
                    fake_path = "/home/baojiali/Downloads/public_code/bevfusion/fake_black.jpg"
                    data["image_paths"].append(fake_path)

                    data["lidar2camera"].append(np.eye(4, dtype=np.float32))
                    data["lidar2image"].append(np.eye(4, dtype=np.float32))
                    data["camera2ego"].append(np.eye(4, dtype=np.float32))
                    data["camera_intrinsics"].append(np.eye(4, dtype=np.float32))
                    data["camera2lidar"].append(np.eye(4, dtype=np.float32))

        else:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []
            
            
            # 造假的相机
            fake_image_path = "/home/baojiali/Downloads/public_code/bevfusion/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg"  # 可以放一张黑图占位
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
        gt_names = info["gt_names"][mask]
        num_lidar_pts=info["num_lidar_pts"][mask]
        
        # ⭐ 点数过滤逻辑开始 ⭐
        if getattr(self, 'filter_gt_by_points', False):
            keep_mask = np.ones(len(gt_names), dtype=bool)
            for i in range(len(gt_names)):
                cls = gt_names[i]
                min_pts = self.class_min_points_dict.get(cls, 0)
                if num_lidar_pts[i] < min_pts:
                    keep_mask[i] = False

            gt_names = gt_names[keep_mask]
            gt_bboxes_3d = gt_bboxes_3d[keep_mask]
            num_lidar_pts = num_lidar_pts[keep_mask]
            # gt_labels_3d= gt_labels_3d[keep_mask]
        # ⭐ 点数过滤逻辑结束 ⭐
        gt_labels_3d = []
        for cat in gt_names:
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
        # 即使真值里面没有速度，也不上vel两维，凑成9维度，配合检测端的维度
        else:
            zeros = np.zeros((gt_bboxes_3d.shape[0], 2), dtype=gt_bboxes_3d.dtype)
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, zeros], axis=-1)
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        

        # --------- yaw转换到 mmdet3d 坐标系 ---------
        gt_bboxes_3d = convert_yaw_to_mmdet3d(gt_bboxes_3d)
        # ------------------------------------------
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names,
            num_lidar_pts=num_lidar_pts
        )
        return anns_results
    
    def evaluate_slow(self, results, metric=None, iou_thr=0.5, logger=None, **kwargs):
        print("Evaluating...")
        preds, gts = defaultdict(list), defaultdict(list)

        # ==== 整理预测 ====
        print("整理预测")
        for i, res in enumerate(results):
            frame_id = self.data_infos[i]['token']
            boxes = res['boxes_3d'].tensor.cpu()   # (N, 7)
            labels = res['labels_3d'].cpu().numpy()
            scores = res['scores_3d'].cpu().numpy()
            for b, l, s in zip(boxes, labels, scores):
                preds[frame_id].append({
                    "class": self.CLASSES[l],
                    "box": b.unsqueeze(0),   # 保留7维
                    "score": float(s)
                })

        # ==== 整理GT ====
        print("整理GT")
        for i, info in enumerate(self.data_infos):
            frame_id = info['token']
            annos = self.get_ann_info(i)
            gt_boxes = annos['gt_bboxes_3d'].tensor.cpu()
            gt_labels = annos['gt_labels_3d']
            for b, l in zip(gt_boxes, gt_labels):
                if l >= 0:
                    gts[frame_id].append({
                        "class": self.CLASSES[l],
                        "box": b.unsqueeze(0)
                    })

        # IoU 计算器（3D）
        iou_calculators = {
            "3D": BboxOverlaps3D("lidar"),             # 支持旋转7维
            "BEV": BboxOverlapsNearest3D("lidar"),    # 至少7维
            "AxisAligned": AxisAlignedBboxOverlaps3D() # 只接受6维
        }

        # ==== 每类计算AP ====
        results_eval = {}

        for cls in self.CLASSES:
            pred_list = [obj for f in preds for obj in preds[f] if obj["class"] == cls]
            gt_list   = [obj for f in gts for obj in gts[f] if obj["class"] == cls]
            pred_list = sorted(pred_list, key=lambda x: x["score"], reverse=True)

            for iou_name, iou_calculator in iou_calculators.items():
                tp, fp = np.zeros(len(pred_list)), np.zeros(len(pred_list))
                gt_used = [False] * len(gt_list)

                for i, pred in enumerate(pred_list):
                    cand_gts = [gt for j, gt in enumerate(gt_list) if not gt_used[j]]
                    if not cand_gts:
                        fp[i] = 1
                        continue

                    # 根据 IoU 类型选择维度
                    if iou_name in ["3D", "BEV"]:
                        # 保留7维
                        pred_box = pred["box"].cuda()
                        gt_boxes = torch.cat([gt["box"] for gt in cand_gts]).cuda()
                    else:  # AxisAligned
                        # 取前6维
                        pred_box = pred["box"][:, :6].cuda()
                        gt_boxes = torch.cat([gt["box"][:, :6] for gt in cand_gts]).cuda()

                    ious = iou_calculator(pred_box, gt_boxes, mode="iou").cpu().numpy()[0]
                    max_iou_idx = np.argmax(ious)
                    if ious[max_iou_idx] >= iou_thr:
                        tp[i] = 1
                        gt_used[max_iou_idx] = True
                    else:
                        fp[i] = 1

                tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
                recall = tp_cum / (len(gt_list) + 1e-6)
                precision = tp_cum / (tp_cum + fp_cum + 1e-6)
                ap = compute_ap(recall, precision)
                results_eval[f"{cls}_AP_{iou_name}"] = ap

        # ==== 美化输出 ====
        print("\n===== KITTI-style Evaluation Results =====")
        print(f"IoU Threshold: {iou_thr}")
        print(f"{'Class':<20} {'3D AP':<10} {'BEV AP':<10} {'AxisAligned AP':<15}")
        for cls in self.CLASSES:
            ap3d  = results_eval.get(f"{cls}_AP_3D", 0)
            apbev = results_eval.get(f"{cls}_AP_BEV", 0)
            apax  = results_eval.get(f"{cls}_AP_AxisAligned", 0)
            print(f"{cls:<20} {ap3d:<10.4f} {apbev:<10.4f} {apax:<15.4f}")
        print("==========================================\n")

        return results_eval
    
    def evaluate(self, results, eval_metric='kitti', iou_thr=0.5, batch_size=5000, verbose=True, **kwargs):
        import torch
        from collections import defaultdict

        all_preds = defaultdict(lambda: {"boxes": [], "scores": []})
        all_gts   = defaultdict(lambda: {"boxes": []})

        # ==== 整理预测 ====
        for i, res in enumerate(results):
            for cls_idx, cls_name in enumerate(self.CLASSES):
                mask = res['labels_3d'].cpu().numpy() == cls_idx
                boxes = res['boxes_3d'].tensor.cpu()[mask]
                scores = res['scores_3d'].cpu()[mask]
                if len(boxes) > 0:
                    all_preds[cls_name]["boxes"].append(boxes)
                    all_preds[cls_name]["scores"].append(scores)

        # ==== 整理GT ====
        for i, info in enumerate(self.data_infos):
            annos = self.get_ann_info(i)
            for cls_idx, cls_name in enumerate(self.CLASSES):
                mask = annos['gt_labels_3d'] == cls_idx
                boxes = annos['gt_bboxes_3d'].tensor.cpu()[mask]
                if len(boxes) > 0:
                    all_gts[cls_name]["boxes"].append(boxes)

        results_eval = {}

        # ==== IoU 计算器 ====
        iou_calculators = {
            "3D": BboxOverlaps3D("lidar"),
            "BEV": BboxOverlapsNearest3D("lidar"),
            "AxisAligned": AxisAlignedBboxOverlaps3D()
        }

        for cls_name in self.CLASSES:
            if len(all_preds[cls_name]["boxes"]) == 0:
                continue

            # 合并帧
            pred_boxes = torch.cat(all_preds[cls_name]["boxes"], dim=0)
            pred_scores = torch.cat(all_preds[cls_name]["scores"], dim=0)
            gt_boxes = torch.cat(all_gts[cls_name]["boxes"], dim=0) if all_gts[cls_name]["boxes"] else torch.empty((0,7))

            # 按分数降序
            scores_sort = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[scores_sort]
            pred_scores = pred_scores[scores_sort]

            for iou_name, iou_calculator in iou_calculators.items():
                tp = torch.zeros(len(pred_boxes))
                fp = torch.zeros(len(pred_boxes))
                gt_used = torch.zeros(len(gt_boxes), dtype=torch.bool)

                # ==== 分批计算 IoU ====
                for start in range(0, len(pred_boxes), batch_size):
                    end = min(start + batch_size, len(pred_boxes))
                    batch_pred = pred_boxes[start:end].cuda()

                    if len(gt_boxes) > 0:
                        batch_gt = gt_boxes.cuda()
                        # AxisAligned 只用前6维
                        if iou_name == "AxisAligned":
                            batch_pred_ = batch_pred[:, :6]
                            batch_gt_ = batch_gt[:, :6]
                        else:
                            batch_pred_ = batch_pred
                            batch_gt_ = batch_gt

                        ious = iou_calculator(batch_pred_, batch_gt_, mode="iou").cpu()
                    else:
                        ious = torch.zeros((end-start, 0))

                    # ==== 匹配 TP/FP ====
                    for k in range(end-start):
                        if ious.shape[1] == 0 or (~gt_used).sum() == 0:
                            fp[start+k] = 1
                            continue

                        iou_row = ious[k].clone()
                        iou_row[gt_used] = -1
                        max_idx = torch.argmax(iou_row)
                        if iou_row[max_idx] >= iou_thr:
                            tp[start+k] = 1
                            gt_used[max_idx] = True
                        else:
                            fp[start+k] = 1

                # ==== 计算 AP ====
                tp_cum = torch.cumsum(tp, dim=0).numpy()
                fp_cum = torch.cumsum(fp, dim=0).numpy()
                recall = tp_cum / (len(gt_boxes) + 1e-6)
                precision = tp_cum / (tp_cum + fp_cum + 1e-6)
                ap = compute_ap(recall, precision)
                results_eval[f"{cls_name}_AP_{iou_name}"] = ap

        # ==== 美化输出 ====
        if verbose:
            print("\n===== KITTI-style Evaluation Results =====")
            print(f"IoU Threshold: {iou_thr}")
            print(f"{'Class':<20} {'3D AP':<10} {'BEV AP':<10} {'AxisAligned AP':<15}")
            for cls in self.CLASSES:
                ap3d  = results_eval.get(f"{cls}_AP_3D", 0)
                apbev = results_eval.get(f"{cls}_AP_BEV", 0)
                apax  = results_eval.get(f"{cls}_AP_AxisAligned", 0)
                print(f"{cls:<20} {ap3d:<10.4f} {apbev:<10.4f} {apax:<15.4f}")
            print("==========================================\n")

        return results_eval
