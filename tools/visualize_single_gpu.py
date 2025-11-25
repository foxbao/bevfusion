import argparse
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def unwrap_data(obj):
    """递归解开 DataContainer"""
    if hasattr(obj, "data"):
        obj = obj.data
    if isinstance(obj, (list, tuple)):
        return [unwrap_data(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: unwrap_data(v) for k, v in obj.items()}
    else:
        return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    # 输出文件夹
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "camera"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "lidar"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "map"), exist_ok=True)

    # 数据
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # 模型
    model = None
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        model = model.to(device)
        model.eval()

    for data in tqdm(dataflow):
        # 解开 DataContainer
        data = unwrap_data(data)
        metas = data["metas"][0]  # batch=1
        metas = metas[0]  # 取第0个sample
        name = f"{metas['timestamp']}-{metas['token']}"

        # 处理 points: 保证是 list[Tensor] 并搬到 GPU
        if args.mode == "pred" and "points" in data:
            points_list = []
            for pc in data["points"]:  # batch 1
                if isinstance(pc, list):
                    # pc 是 list[Tensor]
                    points_list.extend([p.to(device) if isinstance(p, torch.Tensor) else torch.from_numpy(p).to(device) for p in pc])
                elif isinstance(pc, torch.Tensor):
                    points_list.append(pc.to(device))
                elif isinstance(pc, np.ndarray):
                    points_list.append(torch.from_numpy(pc).to(device))
            data["points"] = points_list

            # 移动其他 tensor 到 GPU
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)

            with torch.inference_mode():
                outputs = model(**data)

        # 处理 bbox
        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].tensor
            labels = data["gt_labels_3d"]
            if args.bbox_classes is not None:
                idx = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[idx]
                labels = labels[idx]
            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.cpu().numpy()
            scores = outputs[0]["scores_3d"].cpu().numpy()
            labels = outputs[0]["labels_3d"].cpu().numpy()
            if args.bbox_classes is not None:
                idx = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[idx]
                scores = scores[idx]
                labels = labels[idx]
            if args.bbox_score is not None:
                idx = scores >= args.bbox_score
                bboxes = bboxes[idx]
                scores = scores[idx]
                labels = labels[idx]
            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        # masks
        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].astype(bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].cpu().numpy() >= args.map_score
        else:
            masks = None

        # camera
        if "img" in data:
            for k, img_path in enumerate(metas["filename"]):
                image = mmcv.imread(img_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                )

        # lidar
        if "points" in data:
            lidar = data["points"][0]
            if isinstance(lidar, torch.Tensor):
                lidar = lidar.cpu().numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

        # map
        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()
