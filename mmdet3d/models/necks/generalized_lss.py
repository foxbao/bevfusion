import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS

__all__ = ["GeneralizedLSSFPN"]

def check_nan_inf(obj):
    """递归检查 Tensor / tuple / list 是否包含 NaN / Inf 或空 tensor
    返回 True 表示正常，False 表示异常
    """
    if isinstance(obj, torch.Tensor):
        if not torch.isfinite(obj).all() or obj.numel() == 0:
            return False
        return True
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            if not check_nan_inf(item):
                return False
        return True
    else:
        # 其他类型不检查，认为正常
        return True
    
def find_nan_inf(obj, name="tensor", max_print=20):
    """
    递归检查 Tensor / tuple / list 中的 NaN 或 Inf。
    
    参数:
        obj: Tensor 或 tuple/list
        name: 当前对象名称，用于打印
        max_print: 最多打印前几个异常坐标
    
    返回:
        True: 没有 NaN/Inf
        False: 存在 NaN/Inf
    """
    if isinstance(obj, torch.Tensor):
        mask = ~torch.isfinite(obj)
        if mask.any():
            idx = mask.nonzero(as_tuple=False)
            print(f"❌ {name} 中 {mask.sum().item()} 个 NaN/Inf，前 {min(max_print, idx.size(0))} 个坐标及数值:")
            for i in range(min(max_print, idx.size(0))):
                coord = tuple(idx[i].tolist())
                value = obj[coord].item()
                print(f"  坐标 {coord} -> 值 {value}")
            return False
        return True
    elif isinstance(obj, (tuple, list)):
        all_ok = True
        for i, item in enumerate(obj):
            if not find_nan_inf(item, f"{name}[{i}]", max_print):
                all_ok = False
        return all_ok
    else:
        # 其他类型直接认为正常
        return True

@NECKS.register_module()
class GeneralizedLSSFPN(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN2d"),
        act_cfg=dict(type="ReLU"),
        upsample_cfg=dict(mode="bilinear", align_corners=True),
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i]
                + (
                    in_channels[i + 1]
                    if i == self.backbone_end_level - 1
                    else out_channels
                ),
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function with NaN/Inf checks."""
        assert len(inputs) == len(self.in_channels)

        # === 逐层检查输入 ===
        for idx, inp in enumerate(inputs):
            if not check_nan_inf(inp):
                print(f"❌ inputs[{idx}] 输入包含 NaN/Inf 或空 tensor")
                find_nan_inf(inp, f"inputs[{idx}] 输出")
            # else:
            #     print(f"✅ inputs[{idx}] 正常, shape={inp.shape}, min={inp.min().item():.4f}, max={inp.max().item():.4f}")

        # build laterals
        laterals = []
        for i in range(len(inputs) - self.start_level):
            feat = inputs[i + self.start_level]
            if not check_nan_inf(feat):
                print(f"❌ lateral init[{i}] 包含 NaN/Inf")
            laterals.append(feat)

        used_backbone_levels = len(laterals) - 1

        # build top-down path
        for i in range(used_backbone_levels - 1, -1, -1):
            # 1. 上采样
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            if not check_nan_inf(x):
                print(f"❌ lateral[{i}] upsample 后包含 NaN/Inf")
            # else:
            #     print(f"✅ lateral[{i}] upsample 正常, shape={x.shape}, min={x.min().item():.4f}, max={x.max().item():.4f}")

            # 2. concat
            concat = torch.cat([laterals[i], x], dim=1)
            if not check_nan_inf(concat):
                print(f"❌ lateral[{i}] concat 后包含 NaN/Inf")
            laterals[i] = concat

            # 3. lateral conv
            lat = self.lateral_convs[i](laterals[i])
            if not check_nan_inf(lat):
                print(f"❌ lateral[{i}] lateral_conv 后包含 NaN/Inf")
                find_nan_inf(lat, f"lateral[{i}] lateral_conv 输出")
            # else:
            #     print(f"✅ lateral[{i}] lateral_conv 正常, shape={lat.shape}, min={lat.min().item():.4f}, max={lat.max().item():.4f}")
            laterals[i] = lat

            # 4. fpn conv
            fpn = self.fpn_convs[i](laterals[i])
            if not check_nan_inf(fpn):
                print(f"❌ lateral[{i}] fpn_conv 后包含 NaN/Inf")
                find_nan_inf(fpn, f"lateral[{i}] fpn_conv 输出")
            # else:
            #     print(f"✅ lateral[{i}] fpn_conv 正常, shape={fpn.shape}, min={fpn.min().item():.4f}, max={fpn.max().item():.4f}")
            laterals[i] = fpn

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        if not check_nan_inf(outs):
            print("❌ FPN forward 输出包含 NaN/Inf")
        # else:
        #     for idx, out in enumerate(outs):
        #         print(f"✅ outs[{idx}] 正常, shape={out.shape}, min={out.min().item():.4f}, max={out.max().item():.4f}")

        return tuple(outs)

    # @auto_fp16()
    # def forward(self, inputs):
    #     """Forward function."""
    #     # upsample -> cat -> conv1x1 -> conv3x3
    #     assert len(inputs) == len(self.in_channels)

    #     # build laterals
    #     laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

    #     # build top-down path
    #     used_backbone_levels = len(laterals) - 1
    #     for i in range(used_backbone_levels - 1, -1, -1):
    #         x = F.interpolate(
    #             laterals[i + 1],
    #             size=laterals[i].shape[2:],
    #             **self.upsample_cfg,
    #         )
    #         # 检查 upsample 后
    #         if not check_nan_inf(x):
    #             print(f"❌ lateral[{i}] upsample 后包含 NaN/Inf 或空 tensor")
    #         laterals[i] = torch.cat([laterals[i], x], dim=1)
    #         if not check_nan_inf(laterals[i]):
    #             print(f"❌ lateral[{i}] concat 后包含 NaN/Inf 或空 tensor")
    #         laterals[i] = self.lateral_convs[i](laterals[i])
    #         if not check_nan_inf(laterals[i]):
    #             print(f"❌ lateral[{i}] lateral_conv 后包含 NaN/Inf 或空 tensor")
    #         if not find_nan_inf(laterals[i], f"lateral[{i}] lateral_conv 输出"):
    #             print(f"❌ lateral[{i}] lateral_conv 后包含 NaN/Inf 或空 tensor")
    #         laterals[i] = self.fpn_convs[i](laterals[i])
    #         if not check_nan_inf(laterals[i]):
    #             print(f"❌ lateral[{i}] fpn_conv 后包含 NaN/Inf 或空 tensor")

    #     # build outputs
    #     outs = [laterals[i] for i in range(used_backbone_levels)]
    #     # --- 检查返回值 ---
    #     if not check_nan_inf(outs):
    #         print("❌ FPN forward 输出包含 NaN/Inf 或空 tensor")
    #         # import pdb; pdb.set_trace()  # VSCode 会在这里停下来
    #     return tuple(outs)
