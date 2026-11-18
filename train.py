"""
Train MSCAR (24h or 120h) with MAE + L1, EMA and validation-based model selection.
训练 MSCAR（24h 或 120h）：MAE + L1 损失、EMA，并按验证集选择模型。

Single GPU / 单卡:  python train.py --config configs/mscar_24h.yaml
Multi GPU / 多卡:   torchrun --nproc_per_node=4 train.py --config configs/mscar_24h.yaml
"""
import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataset import TCDataset, EvaluationSampler, load_normalization
from models import MSCAR
from utils import seed_all, load_checkpoint, reestimate_batchnorm, evaluate


def build_scheduler(optimizer, tc, updates_per_epoch):
    """Linear decay per update (24h) or cosine decay per epoch (120h).
    线性衰减按每次参数更新（24h），余弦衰减按每轮（120h）。"""
    if tc.get('scheduler', 'cosine') == 'linear':
        total = tc['epochs'] * updates_per_epoch
        return torch.optim.lr_scheduler.LinearLR(optimizer, 1., tc['end_factor'], total), 'update'
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tc['epochs'], tc.get('min_lr', 1e-6)), 'epoch'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', help='resume from outputs/.../last.pth / 断点续训')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    tc, dc = cfg['training'], cfg['data']
    steps = cfg['model']['pred_len']

    # Distributed setup / 分布式初始化
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False  # FP32 as in the paper / 与论文一致使用 FP32
    output = Path(tc['output_dir'])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Data: train 1980-2017, valid 2018 (the test years are never used here)
    # 数据：训练 1980-2017，验证 2018（训练过程中不使用测试年份）
    stats = load_normalization()
    train_set = TCDataset(dc['train_cache'], 'train', stats, output_len=steps, augment=tc.get('augment', False))
    valid_set = TCDataset(dc['evaluation_cache'], 'valid', stats, output_len=steps)
    sampler = DistributedSampler(train_set, world, rank, seed=cfg['seed']) if world > 1 else None
    micro, accumulate, workers = tc['micro_batch_size'], tc['accumulate'], tc.get('workers', 4)
    loader = DataLoader(train_set, micro, shuffle=sampler is None, sampler=sampler, num_workers=workers,
                        persistent_workers=workers > 0, pin_memory=device.type == 'cuda')
    valid_loader = DataLoader(valid_set, micro, sampler=EvaluationSampler(valid_set, rank, world),
                              num_workers=workers, pin_memory=device.type == 'cuda')

    # Model, EMA, optimizer / 模型、EMA、优化器
    model = MSCAR(**cfg["model"])
    if tc.get('init_checkpoint'):
        # 120h starts from the 24h weights / 120h 以 24h 权重初始化
        load_checkpoint(model, tc['init_checkpoint'])
    model.to(device)
    if world > 1:
        model = DDP(model, device_ids=[local] if device.type == 'cuda' else None)
    raw = model.module if world > 1 else model
    ema = deepcopy(raw).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc['lr'], betas=tuple(tc['betas']),
                                  eps=tc['eps'], weight_decay=tc['weight_decay'])
    scheduler, schedule_unit = build_scheduler(optimizer, tc, math.ceil(len(loader) / accumulate))

    start_epoch, best, stale = 0, float('inf'), 0
    if args.resume:
        state = load_checkpoint(raw, args.resume)
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, best, stale = state['epoch'], state['best_metric'], state['stale']
    seed_all(cfg['seed'] + rank + start_epoch)

    for epoch in range(start_epoch, tc['epochs']):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, samples, t0 = 0., 0, time.time()
        for batch_index, (ir, era5, history, target, _) in enumerate(loader):
            ir, era5, history, target = [x.to(device, non_blocking=True) for x in (ir, era5, history, target)]
            last = (batch_index + 1) % accumulate == 0 or batch_index + 1 == len(loader)
            group = min(accumulate, len(loader) - (batch_index // accumulate) * accumulate)
            l1 = lambda: tc['l1_regularization'] * sum(p.abs().sum() for p in raw.parameters())

            if tc.get('per_lead_forward', False):
                # 24h (original training): for lead k, encode and roll out k AR steps;
                # loss averaged over the 4 leads
                # 24h（原始训练方式）：第 k 个时效重新编码并自回归 k 步，损失取 4 个时效平均
                batch_mae = 0.
                for lead in range(1, steps + 1):
                    sync = nullcontext() if world == 1 or (last and lead == steps) else model.no_sync()
                    with sync:
                        mae = (model(ir, era5, history, step=lead)[:, 0] - target[:, lead - 1]).abs().mean()
                        loss = mae / steps + (l1() if lead == steps else 0.)
                        (loss / group).backward()
                    batch_mae += float(mae.detach()) / steps
            else:
                # 120h: encode once and roll out 20 AR steps
                # 120h：编码一次后自回归 20 步
                sync = nullcontext() if world == 1 or last else model.no_sync()
                with sync:
                    pred = model(ir, era5, history, step=steps, return_sequence=True)
                    mae = (pred - target).abs().mean()
                    ((mae + l1()) / group).backward()
                batch_mae = float(mae.detach())
            loss_sum += batch_mae * len(ir)
            samples += len(ir)

            if last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc['clip_grad'])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if schedule_unit == 'update':
                    scheduler.step()
                with torch.no_grad():
                    # EMA of parameters; buffers are copied
                    # 参数做指数滑动平均；buffer 直接复制
                    for averaged, current in zip(ema.parameters(), raw.parameters()):
                        averaged.lerp_(current.detach(), 1 - tc['ema_decay'])
                    for averaged, current in zip(ema.buffers(), raw.buffers()):
                        averaged.copy_(current)
            if rank == 0 and batch_index % 50 == 0:
                print(f'epoch {epoch + 1} [{batch_index + 1}/{len(loader)}] '
                      f'mae {loss_sum / samples:.4f} lr {optimizer.param_groups[0]["lr"]:.2e}', flush=True)
        if schedule_unit == 'epoch':
            scheduler.step()

        # Validate the EMA model on 2018 / 在 2018 验证集上评估 EMA 模型
        if tc.get('ema_bn_reestimate', True):
            reestimate_batchnorm(ema, loader, device)
        elif world > 1:
            for buffer in ema.buffers():
                dist.broadcast(buffer, src=0)
        metric, _ = evaluate(ema, valid_loader, device)
        totals = torch.tensor([loss_sum, samples], device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(totals)
        improved = metric['normalized_mae'] < best
        best, stale = min(best, metric['normalized_mae']), (0 if improved else stale + 1)

        if rank == 0:
            state = dict(epoch=epoch + 1, model=raw.state_dict(), ema=ema.state_dict(),
                         optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         best_metric=best, stale=stale)
            torch.save(state, output / 'last.pth')
            if improved:
                # Saved format: EMA weights + normalization
                # 保存格式：EMA 权重 + 归一化参数
                torch.save(dict(model=ema.state_dict(), epoch=epoch + 1, pred_len=steps,
                                normalization=stats, validation=metric), output / 'best.pth')
            record = dict(epoch=epoch + 1, train_normalized_mae=float(totals[0] / totals[1]),
                          valid_normalized_mae=metric['normalized_mae'],
                          valid_wind_ms=metric['wind_ms'], valid_pressure_hpa=metric['pressure_hpa'],
                          improved=improved, seconds=time.time() - t0)
            with (output / 'history.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
        if world > 1:
            dist.barrier()
        if tc.get('patience') and stale >= tc['patience']:
            break
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
