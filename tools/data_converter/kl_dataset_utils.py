"""
The NuScenes data pre-processing and evaluation is modified from
https://github.com/traveller59/second.pytorch and https://github.com/poodarchu/Det3D
"""

import operator
from functools import reduce
from pathlib import Path

import numpy as np
import tqdm
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from typing import List, Tuple
import json
import os
from bisect import bisect_left
from .kl import KL

def get_available_scenes(nusc):
    """
    获取可用的场景。
    :param nusc: NuScenes 数据集类。
    :return: 可用的场景列表。
    """
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            # if not sd_rec['next'] == '':
            #     sd_rec = nusc.get('sample_data', sd_rec['next'])
            # else:
            #     has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes


def get_sample_data(nusc, sample_data_token, selected_anntokens=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    Args:
        nusc:
        sample_data_token: Sample_data token.
        selected_anntokens: If provided only return the selected annotation.

    Returns:

    """
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


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
    

def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    # if os.getcwd() in data_path:  # path from lyftdataset is absolute path
    #     data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    ).squeeze(0)
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep

# # 查找最近的点云文件
# def find_nearest_pointcloud(timestamp, pointcloud_files, pointcloud_timestamps=None):
#     """
#     根据时间戳查找最近的点云文件。
#     :param timestamp: 标注文件的时间戳（整数或字符串）
#     :param pointcloud_files: 点云文件列表（Path 对象）
#     :param pointcloud_timestamps: 预计算的时间戳列表（可选）
#     :return: 最近的点云文件路径（字符串）
#     """
#     timestamp = int(timestamp)  # 确保时间戳是整数

#     # 如果没有预计算时间戳，则实时计算
#     if pointcloud_timestamps is None:
#         pointcloud_timestamps = [int(f.stem) for f in pointcloud_files]

#     # 使用 NumPy 计算最小差值
#     diffs = np.abs(np.array(pointcloud_timestamps) - timestamp)
#     nearest_index = np.argmin(diffs)
#     return str(pointcloud_files[nearest_index])


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


# def convert_to_gt_boxes_7dof(xyz, lwh, rotation):
#     # 确保输入是 numpy 数组
#     xyz = np.asarray(xyz)
#     lwh = np.asarray(lwh)
#     rotation = np.asarray(rotation)
    
#     # 将四元数转换为偏航角
#     yaw = quaternion_to_yaw(rotation)
    
#     # 将 xyz, lwh, yaw 拼接成 gt_boxes
#     gt_boxes = np.concatenate([xyz, lwh, [yaw]])
    
#     return gt_boxes


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
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
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
        with open(sample['extrinsics_path'], 'r', encoding='utf-8') as f:
            extrinsice_data = json.load(f)

        # ---------- 读取 intrinsics（若存在） ----------
        intr_path = sample.get('intrinsics_path')
        if intr_path is not None and Path(intr_path).exists():
            with open(intr_path, 'r', encoding='utf-8') as f:
                intrinsice_data = json.load(f)
        else:
            intrinsice_data = {}

        # ---------- 读取 localization（若有） ----------
        loc_path = sample.get('localization')
        if loc_path:                       # 既防 None，也防空字符串/Path
            with open(loc_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
        else:
            state = None            

        # with open(sample['intrinsics_path'], 'r', encoding='utf-8') as f:
        #     intrinsice_data = json.load(f)
        # with open(sample['localization'], 'r', encoding='utf-8') as f:
        #     state=json.load(f)
        # 为每个样本添加 timestamp、token 和 pointcloud_path
        info = {
            'token': sample['token'],
            'timestamp': sample['timestamp'],
            'annos': annotations,
            # 'gt_boxes':gt_boxes,
            # 'gt_names':gt_names,
            # 'gt_subtypes':gt_subtypes,
            # 'gt_boxes_token':gt_boxes_token,
            # 'gt_track_ids':gt_track_ids,
            'lidars': sample['lidars'],
            'cameras': sample['cameras'],
            'localization': sample['localization'],
            'state':state,
            'sensor_extrinsics': extrinsice_data,
            'sensor_intrinsics': intrinsice_data
        }

        # gt_boxes增加速度
        # gt_boxes=info['gt_boxes']
        # locs = gt_boxes[:, :3]
        # dims = gt_boxes[:, 3:6]
        # rots = gt_boxes[:, 6].reshape(-1, 1)
        # velocity = np.zeros((gt_boxes.shape[0], 2))
        # info['gt_boxes'] = np.concatenate([locs, dims, rots, velocity], axis=1)
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


def lidar_nusc_box_to_global(nusc, boxes, sample_token):
    s_record = nusc.get('sample', sample_token)
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))
        box_list.append(box)
    return box_list


def transform_det_annos_to_nusc_annos(det_annos, nusc):
    nusc_annos = {
        'results': {},
        'meta': None,
    }

    for det in det_annos:
        annos = []
        box_list = boxes_lidar_to_nusenes(det)
        box_list = lidar_nusc_box_to_global(
            nusc=nusc, boxes=box_list, sample_token=det['metadata']['token']
        )

        for k, box in enumerate(box_list):
            name = det['name'][k]
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = None
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = None
            attr = attr if attr is not None else max(
                cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            nusc_anno = {
                'sample_token': det['metadata']['token'],
                'translation': box.center.tolist(),
                'size': box.wlh.tolist(),
                'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(),
                'detection_name': name,
                'detection_score': box.score,
                'attribute_name': attr
            }
            annos.append(nusc_anno)

        nusc_annos['results'].update({det["metadata"]["token"]: annos})

    return nusc_annos


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
