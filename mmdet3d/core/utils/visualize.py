import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from ..bbox import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]


# OBJECT_PALETTE = {
#     "car": (255, 158, 0),
#     "truck": (255, 99, 71),
#     "construction_vehicle": (233, 150, 70),
#     "bus": (255, 69, 0),
#     "trailer": (255, 140, 0),
#     "barrier": (112, 128, 144),
#     "motorcycle": (255, 61, 99),
#     "bicycle": (220, 20, 60),
#     "pedestrian": (0, 0, 230),
#     "traffic_cone": (47, 79, 79),
# }

OBJECT_PALETTE = {
    "Pedestrian": (0, 0, 230),           # 蓝色
    "Car": (255, 158, 0),                # 橙色
    "IGV-Full": (255, 99, 71),           # 番茄红
    "Truck": (233, 150, 70),             # 浅橙
    "Trailer-Empty": (255, 140, 0),      # 深橙
    "Trailer-Full": (255, 69, 0),        # 橙红
    "IGV-Empty": (112, 128, 144),        # 灰色
    "Crane": (160, 82, 45),              # 棕色
    "OtherVehicle": (128, 0, 128),       # 紫色
    "Cone": (47, 79, 79),                # 深灰绿
    "ContainerForklift": (220, 20, 60),  # 猩红
    "Forklift": (255, 61, 99),           # 粉红
    "Lorry": (0, 128, 0),                # 绿色
    "ConstructionVehicle": (0, 191, 255),# 深天蓝
    "WheelCrane": (255, 215, 0),         # 金色
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]

            # -------- 获取颜色，容错处理 --------
            if color is not None:
                color_bgr = tuple(color[::-1])  # 传进来的 RGB 转 BGR
            else:
                # OBJECT_PALETTE 里没有就用绿色
                color_bgr = tuple(OBJECT_PALETTE.get(name, (0, 255, 0))[::-1])
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int),
                    coords[index, end].astype(np.int),
                    color_bgr,
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
    show_axis: bool = True,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        # coords = bboxes.corners[:, [1, 2, 6, 5, 1], :2]
        centers = bboxes.gravity_center[:, :2]  # 中心点
        yaws = bboxes.yaw                       # 朝向角 (弧度)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            # color_rgb = np.array(color or OBJECT_PALETTE[name]) / 255
            # -------- 获取颜色，容错处理 --------
            if color is not None:
                color_rgb = np.array(color) / 255.0
            else:
                color_rgb = np.array(OBJECT_PALETTE.get(name, (0, 255, 0))) / 255.0
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )
            
            # ------- 朝向箭头 -------
            cx, cy = centers[index]
            yaw = -yaws[index]

            arrow_len = max(bboxes.tensor[index, 3:5]) * 0.5  # 箭头长度取长宽的一半
            dx = arrow_len * np.cos(yaw)
            dy = arrow_len * np.sin(yaw)

            plt.arrow(
                cx, cy, dx, dy,
                color=color_rgb,
                width=0.2,
                head_width=1.0,
                head_length=1.5,
                length_includes_head=True,
            )

    # ---------- 绘制坐标轴 ----------
    if show_axis:
        # 原点在 (0,0) 或点云中心
        origin = np.array([0.0, 0.0])
        ax_len = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.05  # 轴长度占画布 5%

        # x 轴 (红色)
        ax.arrow(
            origin[0], origin[1], ax_len, 0,
            color="red", width=0.2, head_width=ax_len*0.2, length_includes_head=True
        )
        ax.text(origin[0]+ax_len, origin[1], "X", color="red", fontsize=12, weight='bold')

        # y 轴 (绿色)
        ax.arrow(
            origin[0], origin[1], 0, ax_len,
            color="green", width=0.2, head_width=ax_len*0.2, length_includes_head=True
        )
        ax.text(origin[0], origin[1]+ax_len, "Y", color="green", fontsize=12, weight='bold')

        # z 轴 (蓝色) 注释
        ax.text(origin[0]-ax_len*0.3, origin[1]-ax_len*0.3, "Z ↑", color="blue", fontsize=12, weight='bold')


    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)
