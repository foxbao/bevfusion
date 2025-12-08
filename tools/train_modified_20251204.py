import argparse
import os
import random
import time
import glob
import shutil
import torch
import torch.distributed as dist
import numpy as np
from mmcv import Config
from mmcv.runner import init_dist, get_dist_info, Hook, build_runner
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

def auto_resume_checkpoint(run_dir):
    """在运行目录中查找最新 checkpoint（优先 latest.pth，其次 epoch_*.pth）"""
    if not os.path.isdir(run_dir):
        return None

    latest_ckpt = None
    # 仅由 rank 0 查找，之后广播
    rank, _ = get_dist_info()
    if rank == 0:
        latest_path = os.path.join(run_dir, 'latest.pth')
        if os.path.isfile(latest_path):
            latest_ckpt = latest_path
        else:
            epoch_files = glob.glob(os.path.join(run_dir, 'epoch_*.pth'))
            if epoch_files:
                epoch_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))
                latest_ckpt = epoch_files[-1]

    ckpt_list = [latest_ckpt]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(ckpt_list, src=0)
    return ckpt_list[0]

class UpdateLatestHook(Hook):
    """每训练完一个 epoch，就保存 latest.pth 更新到当前最新 checkpoint"""
    def __init__(self, save_dir=None):
        self.save_dir = save_dir

    def after_train_epoch(self, runner):
        out_dir = self.save_dir if self.save_dir else runner.work_dir
        # 保存 checkpoint，模板文件名带 epoch 序号
        ckpt_path = runner.save_checkpoint(out_dir=out_dir, filename_tmpl='epoch_{}.pth'.format(runner.epoch + 1))
        latest_path = os.path.join(out_dir, 'latest.pth')
        shutil.copyfile(ckpt_path, latest_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file path")
    parser.add_argument("--run-dir", help="directory to save outputs")
    parser.add_argument("--resume-from", help="checkpoint file to resume from")
    parser.add_argument("--auto-resume", action="store_true", help="automatically resume from latest checkpoint")
    args, opts = parser.parse_known_args()

    # 配置加载
    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # 判断是否需要分布式初始化（有 RANK 则说明可能是多卡启动）
    is_dist = 'RANK' in os.environ
    if is_dist:
        init_dist('pytorch', **cfg.get('dist_params', dict(backend='nccl')))
        rank, world_size = get_dist_info()
    else:
        rank, world_size = 0, 1

    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)) if is_dist else 0)
    torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', True)

    # 自动创建/设置运行目录
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = auto_set_run_dir() if rank == 0 else None
        obj = [run_dir]
        if is_dist:
            dist.broadcast_object_list(obj, src=0)
        run_dir = obj[0]
    set_run_dir(run_dir)
    cfg.run_dir = run_dir

    # 自动 resume 逻辑
    resume_ckpt = args.resume_from or (auto_resume_checkpoint(run_dir) if args.auto_resume else None)
    if resume_ckpt:
        cfg.resume_from = resume_ckpt

    # 日志与配置保存
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if rank == 0:
        cfg.dump(os.path.join(run_dir, 'configs.yaml'))
    logger = get_root_logger(log_file=os.path.join(run_dir, f'{timestamp}.log') if rank == 0 else None)
    if rank == 0:
        logger.info(f'Configs:\n{cfg.pretty_text}')
        if cfg.get('resume_from', None):
            logger.info(f'Resuming from: {cfg.resume_from}')

    # 随机种子设置
    if cfg.get('seed', None) is not None:
        if rank == 0:
            logger.info(f"Set random seed to {cfg.seed}, deterministic: {cfg.get('deterministic', False)}")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 数据集与模型
    datasets = [build_dataset(cfg.data.train)]
    model = build_model(cfg.model)
    if cfg.get('init_weights', True):
        model.init_weights()
    if cfg.get('sync_bn', None):
        sync_cfg = cfg.sync_bn if isinstance(cfg.sync_bn, dict) else dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=sync_cfg.get('exclude', []))

    if rank == 0:
        logger.info(f'Model:\n{model}')

    # Runner 构建前需从 cfg 获取 max_epochs
    max_epochs = cfg.get('max_epochs', None)
    if max_epochs is None and cfg.get('train_cfg', None):
        max_epochs = cfg.train_cfg.get('max_epochs', None)
    if max_epochs is None:
        raise ValueError("无法获取 max_epochs！请在配置文件中设置 max_epochs 或 train_cfg.max_epochs")

    from mmcv.runner import build_runner
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr) if hasattr(cfg, 'optimizer') else None
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=run_dir,
            logger=logger,
            max_epochs=max_epochs
        )
    )
    runner.register_hook(UpdateLatestHook(save_dir=run_dir))

    train_model(
        model,
        datasets,
        cfg,
        distributed=is_dist,
        validate=True,
        timestamp=timestamp
    )

if __name__ == "__main__":
    main()
