# tools/hooks.py
import os
import shutil
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()   # 关键：把类注册到 HOOKS
class UpdateLatestHook(Hook):
    """每训练完一个 epoch，就保存 latest.pth 更新到当前最新 checkpoint"""
    def __init__(self, save_dir=None):
        self.save_dir = save_dir

    def after_train_epoch(self, runner):
        out_dir = self.save_dir if self.save_dir else runner.work_dir
        # 保存 checkpoint，模板文件名带 epoch 序号
        ckpt_path = runner.save_checkpoint(
            out_dir=out_dir,
            filename_tmpl='epoch_{}.pth'.format(runner.epoch + 1)
        )
        latest_path = os.path.join(out_dir, 'latest.pth')
        shutil.copyfile(ckpt_path, latest_path)
