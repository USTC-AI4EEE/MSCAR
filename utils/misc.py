"""
Seeding, checkpoint loading and EMA BatchNorm re-estimation.
随机种子、权重加载与 EMA 模型的 BatchNorm 统计重估。
"""
import random

import numpy as np
import torch
import torch.distributed as dist


def seed_all(seed):
    """Seed python, numpy and torch.
    固定 python、numpy 和 torch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(model, path):
    """Strictly load a checkpoint; return the full dict.
    严格加载权重，返回完整字典。"""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    return checkpoint


@torch.no_grad()
def reestimate_batchnorm(model, loader, device):
    """
    Recompute FPN BatchNorm statistics of the EMA model on training images only.
    仅用训练集图像重新统计 EMA 模型中 FPN 的 BatchNorm 均值和方差。
    """
    model.eval()
    fpn = model.fpn
    norms = [m for m in fpn.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momentum = {m: m.momentum for m in norms}
    for m in norms:
        # momentum=None gives a cumulative average over all batches
        # momentum=None 表示对所有批次做累计平均
        m.reset_running_stats()
        m.momentum = None
        m.train()
    for batch in loader:
        ir = batch[0].flatten(0, 1).to(device, non_blocking=True)
        era5 = batch[1].flatten(0, 1).to(device, non_blocking=True)
        # Only the convolutional branches contain BatchNorm
        # 只有卷积分支包含 BatchNorm
        ir = fpn.ir_layer3(fpn.ir_layer2(fpn.ir_layer1(fpn.ir_bn1(fpn.ir_conv1(ir)))))
        fpn.era5_layer(fpn.era5_bn1(fpn.era5_conv1(era5)))
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world > 1:
        # Average statistics across ranks so every rank holds the same EMA model
        # 跨卡平均统计量，保证各卡 EMA 模型一致
        for m in norms:
            for value in (m.running_mean, m.running_var):
                dist.all_reduce(value)
                value.div_(world)
            dist.all_reduce(m.num_batches_tracked)
    for m in norms:
        m.momentum = momentum[m]
    model.eval()
