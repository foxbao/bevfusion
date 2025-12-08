"""
The NuScenes data pre-processing and evaluation is modified from
https://github.com/traveller59/second.pytorch and https://github.com/poodarchu/Det3D
"""

# import operator
# from functools import reduce
from pathlib import Path

import numpy as np
import tqdm
from nuscenes.utils.data_classes import Box
# from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from typing import List, Tuple
import json
# import os
# from bisect import bisect_left
from .kl import KL



def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw
    

def quaternion_to_yaw(rotation)->float:
    """
    将四元数转换为偏航角 (yaw)。
    :param rotation: 四元数，形状为 4 的 numpy 数组。
    :return: 偏航角
    """
    qx, qy, qz, qw = rotation[0], rotation[1], rotation[2], rotation[3]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return yaw



def convert_to_gt_boxes_7dof(xyz, lwh, rotation):
    """
    将 xyz, lwh 和 rotation 转换为 [x, y, z, l, w, h, yaw]
    支持 rotation 为四元数 [w, x, y, z] 或字典 {'x':..., 'y':..., 'z':...}
    """
    xyz = np.asarray(xyz)
    lwh = np.asarray(lwh)

    if isinstance(rotation, dict):
        # 如果是 dict，直接取 z 作为 yaw
        yaw = rotation.get('z', 0.0)
    else:
        # 如果是四元数，计算 yaw
        rotation = np.asarray(rotation)
        yaw = quaternion_to_yaw(rotation)

    gt_boxes = np.concatenate([xyz, lwh, [yaw]])
    return gt_boxes


def convert_json_to_annotations(json_data:List[dict]):
    annotations={}
    gt_boxes=[]
    gt_names=[]
    gt_subtype=[]
    gt_boxes_token=[]
    gt_track_ids=[]
    gt_num_lidar_pts=[]
    for data in json_data:
        if data['label']=='Container':
            continue
        if data['label']=='Vehicle':
            gt_names.append(data['subtype'])
        else:
            gt_names.append(data['label'])
        gt_boxes.append(convert_to_gt_boxes_7dof(data['xyz'],data['lwh'],data['rotation']))

        gt_subtype.append(data['subtype'])
        gt_boxes_token.append(data['track_id'])
        gt_track_ids.append(data['track_id'])
        gt_num_lidar_pts.append(data['num_lidar_pts'])
    gt_boxes = np.vstack(gt_boxes)
    gt_names = np.array(gt_names)
    gt_subtype= np.array(gt_subtype)
    gt_boxes_token = np.array(gt_boxes_token)
    gt_track_ids = np.array(gt_track_ids)
    gt_boxes_lidar=gt_boxes
    annotations['name'] = np.array(gt_names)
    annotations['num_lidar_pts']= np.array(gt_num_lidar_pts)
    
    num_gt = len(annotations['name'])
    # 获取标签截断程度
    annotations['location'] = np.array([[obj[0], obj[1], obj[2]] for obj in gt_boxes])  # xyz
    annotations['dimensions'] = np.array([[obj[3], obj[4], obj[5]] for obj in gt_boxes])  # lwh(camera) format
    annotations['rotation_y'] = np.array([obj[6] for obj in gt_boxes])
    annotations['score'] = np.zeros(num_gt, dtype=np.float32)
    annotations['difficulty'] = np.zeros(num_gt, dtype=np.float32)
    # annotations['gt_boxes_lidar'] = gt_boxes_lidar
    annotations['gt_bboxes_3d'] = gt_boxes_lidar
    return annotations


def convert_json_to_gt(json_data:List[dict]):
    gt_boxes=[]
    gt_names=[]
    gt_subtype=[]
    gt_boxes_token=[]
    gt_track_ids=[]
    for data in json_data:
        if data['label']=='Container':
            continue
        if data['label']=='Vehicle':
            gt_names.append(data['subtype'])
        else:
            gt_names.append(data['label'])
        gt_boxes.append(convert_to_gt_boxes_7dof(data['xyz'],data['lwh'],data['rotation']))

        gt_subtype.append(data['subtype'])
        gt_boxes_token.append(data['track_id'])
        gt_track_ids.append(data['track_id'])
    gt_boxes = np.vstack(gt_boxes)
    gt_names = np.array(gt_names)
    gt_subtype= np.array(gt_subtype)
    gt_boxes_token = np.array(gt_boxes_token)
    gt_track_ids = np.array(gt_track_ids)
    return gt_boxes,gt_names,gt_subtype,gt_boxes_token,gt_track_ids


def fill_trainval_infos(kl:KL,train_samples,val_samples,test_samples):
    train_kl_infos = []
    val_kl_infos = []
    test_kl_infos=[]
    # progress_bar = tqdm.tqdm(total=len(kl.samples), desc='create_info', dynamic_ncols=True)
    # for index, sample in enumerate(kl.samples):
    for sample in tqdm.tqdm(kl.samples, desc='create_info', dynamic_ncols=True):
        # progress_bar.update()
        with open(sample['label'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        # gt_boxes,gt_names,gt_subtypes,gt_boxes_token,gt_track_ids=convert_json_to_gt(data)
        annotations=convert_json_to_annotations(data)
        
        gt_boxes=annotations["gt_bboxes_3d"]
        gt_names=annotations["name"]
        num_lidar_pts=annotations["num_lidar_pts"]
        location=annotations["location"]
        with open(sample['extrinsics_path'], 'r', encoding='utf-8') as f:
            extrinsice_data = json.load(f)

        # ---------- 读取 intrinsics（若存在） ----------
        intr_path = sample.get('intrinsics_path')
        if intr_path is not None and Path(intr_path).exists():
            with open(intr_path, 'r', encoding='utf-8') as f:
                intrinsice_data = json.load(f)
        else:
            intrinsice_data = {}
            
        # ---------- 读取 camera_extrinsics（若存在） ----------
        cam_extr_path = sample.get('camera_extrinsics_path')
        if cam_extr_path is not None and Path(cam_extr_path).exists():
            with open(cam_extr_path, 'r', encoding='utf-8') as f:
                camera_extrinsics_data = json.load(f)
        else:
            camera_extrinsics_data = {}

        # ---------- 读取 localization（若有） ----------
        loc_path = sample.get('localization')
        if loc_path:                       # 既防 None，也防空字符串/Path
            with open(loc_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
        else:
            state = None           
            
        info = {
            'token': sample['token'],
            'timestamp': sample['timestamp'],
            # 'annos': annotations,
            'gt_boxes':gt_boxes,
            'gt_names':gt_names,
            'num_lidar_pts':num_lidar_pts,
            'location':location,
            # 'gt_subtypes':gt_subtypes,
            # 'gt_boxes_token':gt_boxes_token,
            # 'gt_track_ids':gt_track_ids,
            'lidars': sample['lidars'],
            'cams': dict(),
            'localization': sample['localization'],
            'state':state,
            'sensor_extrinsics': extrinsice_data,
            'sensor_intrinsics': intrinsice_data,
            'camera_extrinsics': camera_extrinsics_data,
            'label_path': sample['label'],
            "lidar2ego_translation": [0.0, 0.0, 0.0],
            "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "ego2global_translation": [0.0, 0.0, 0.0],
            "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
        }
        # 待补充真实外参数据
        
        # if not sample['cameras']:
        #     continue
        
        # ---------- cameras ----------
        
        # ===== 全局相机配置 =====
        CAMERA_WIDTH = 1920
        CAMERA_HEIGHT = 1536
        FOCAL_LENGTH = 1200.0  # 默认焦距，可按需修改

        # 默认 pinhole 内参（未提供真实标定时使用）
        DEFAULT_INTRINSICS = np.array([
            [FOCAL_LENGTH,       0.0, CAMERA_WIDTH / 2.0],
            [0.0,          FOCAL_LENGTH, CAMERA_HEIGHT / 2.0],
            [0.0,                0.0,              1.0]
        ], dtype=np.float32)

        for camera_type, data_path in sample['cameras'].items():
            # if camera_type not in VALID_CAMERA_TYPES:
            #     continue  # 跳过无效相机
            camera_info = dict()
            
            key_camera_extrinsic = "Tx_baselink_camera_" + camera_type.replace("_image", "")
            camera_extrinsic = camera_extrinsics_data[key_camera_extrinsic]
            key_camera_intrinsic = "camera_"+camera_type.replace("_image", "")
            camera_intrinsic= intrinsice_data[key_camera_intrinsic]
            

            # 解析 translation（np.array）
            translation = np.array(camera_extrinsic[:3], dtype=np.float32)

            # 解析 rotation（四元数 xyzw → numpy array）
            # rotation = np.array(camera_extrinsic[3:], dtype=np.float32)  
            # aaaa=camera_extrinsic[3:]
            # print(aaaa)
            rotation =Quaternion(np.array(camera_extrinsic[3:], dtype=np.float32)).rotation_matrix
            # print(rotation)
            # # rotation = [qx, qy, qz, qw]
            # from scipy.spatial.transform import Rotation as R
            
            
            # r = R.from_quat([np.array(camera_extrinsic[3:])])
            # euler_deg = r.as_euler('xyz', degrees=True)   # 转成角度
            # print(euler_deg)
            camera_info['data_path'] = data_path
            # camera_info["sensor2lidar_rotation"] = np.eye(3, dtype=np.float32)
            # camera_info["sensor2lidar_translation"] = np.array([0.5, 0.0, -1.5], dtype=np.float32)

            # 默认用全局 intrinsics，如果 sample 提供了，就覆盖
            camera_info["camera_intrinsics"] = intrinsice_data.get(
                key_camera_intrinsic, DEFAULT_INTRINSICS
            )
            
            camera_info["sensor2lidar_rotation"] = rotation              # np.array([qx, qy, qz, qw])
            camera_info["sensor2lidar_translation"] = translation        # np.array([x, y, z])
            
            camera_info["sensor2ego_rotation"] = np.array(camera_extrinsic[3:]) 
            camera_info["sensor2ego_translation"] = translation
            
            info["cams"].update({camera_type: camera_info})

        if sample['token'] in train_samples:
            train_kl_infos.append(info)
        elif sample['token'] in val_samples:
            val_kl_infos.append(info)
        else:
            test_kl_infos.append(info)
        

    # progress_bar.close()
    return train_kl_infos, val_kl_infos,test_kl_infos
             


def boxes_lidar_to_nusenes(det_info):
    boxes3d = det_info['boxes_lidar']
    scores = det_info['score']
    labels = det_info['pred_labels']

    box_list = []
    for k in range(boxes3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        velocity = (*boxes3d[k, 7:9], 0.0) if boxes3d.shape[1] == 9 else (0.0, 0.0, 0.0)
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, [4, 3, 5]],  # wlh
            quat, label=labels[k], score=scores[k], velocity=velocity,
        )
        box_list.append(box)
    return box_list



def format_nuscene_results(metrics, class_names, version='default'):
    result = '----------------Nuscene %s results-----------------\n' % version
    for name in class_names:
        threshs = ', '.join(list(metrics['label_aps'][name].keys()))
        ap_list = list(metrics['label_aps'][name].values())

        err_name =', '.join([x.split('_')[0] for x in list(metrics['label_tp_errors'][name].keys())])
        error_list = list(metrics['label_tp_errors'][name].values())

        result += f'***{name} error@{err_name} | AP@{threshs}\n'
        result += ', '.join(['%.2f' % x for x in error_list]) + ' | '
        result += ', '.join(['%.2f' % (x * 100) for x in ap_list])
        result += f" | mean AP: {metrics['mean_dist_aps'][name]}"
        result += '\n'

    result += '--------------average performance-------------\n'
    details = {}
    for key, val in metrics['tp_errors'].items():
        result += '%s:\t %.4f\n' % (key, val)
        details[key] = val

    result += 'mAP:\t %.4f\n' % metrics['mean_ap']
    result += 'NDS:\t %.4f\n' % metrics['nd_score']

    details.update({
        'mAP': metrics['mean_ap'],
        'NDS': metrics['nd_score'],
    })

    return result, details


def transform_points(point_cloud, extrinsic):
    from scipy.spatial.transform import Rotation as R
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
