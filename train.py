#!/usr/bin/env python3
"""
Train MSCAR model

Usage:
    python train.py --config configs/default.yaml
    torchrun --nproc_per_node=4 train.py --config configs/default.yaml
"""

import argparse
import os
import sys
import datetime
import yaml

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from models import MSCAR
from dataset import TCDataset
from utils import (MetricLogger, setup_seed, save_checkpoint, load_checkpoint,
                   is_main_process, get_rank, get_world_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--resume', default=None, help='checkpoint to resume')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output_dir', default=None)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def setup_distributed():
    """Initialize distributed training if available"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        local = int(os.environ['LOCAL_RANK'])
    else:
        rank, world, local = 0, 1, 0

    if world > 1:
        torch.cuda.set_device(local)
        dist.init_process_group('nccl', init_method='env://', world_size=world, rank=rank)
        dist.barrier()

    return rank, world, local


def build_dataset(cfg, split):
    """Create dataset from config"""
    years = {
        'train': range(cfg['data']['train_years'][0], cfg['data']['train_years'][1] + 1),
        'valid': range(cfg['data']['valid_years'][0], cfg['data']['valid_years'][1] + 1),
        'test': range(cfg['data']['test_years'][0], cfg['data']['test_years'][1] + 1),
    }
    return TCDataset(
        data_dir=cfg['data']['data_dir'],
        ibtracs_path=cfg['data']['ibtracs_path'],
        split=split,
        years=years,
        input_len=cfg['data']['input_len'],
        output_len=cfg['data']['output_len'],
        ir_size=cfg['data']['ir_size'],
        era5_size=cfg['data']['era5_size'],
        augment=cfg['training'].get('augment', False) and split == 'train',
        use_lds=cfg['training'].get('use_lds', False) and split == 'train',
        stats_dir=cfg['data'].get('stats_dir'),
    )


def build_model(cfg):
    """Create model from config"""
    m = cfg['model']
    return MSCAR(
        ir_size=(m['ir_size'], m['ir_size']),
        era5_size=(m['era5_size'], m['era5_size']),
        patch_size=(m['patch_size'], m['patch_size']),
        ir_ch=m['ir_channels'],
        era5_ch=m['era5_channels'],
        dim=m['dim'],
        seq_dim=m['seq_dim'],
        seq_len=m['seq_len'],
        pred_len=m['pred_len'],
        depth=m['depth'],
        heads=m['heads'],
        mlp_dim=m['mlp_dim'],
        dropout=m['dropout'],
        ar_hidden=m['ar_hidden'],
    )


def train_one_epoch(model, loader, optimizer, scheduler, warmup_steps, epoch, cfg, device, writer, step):
    """Train for one epoch"""
    model.train()
    logger = MetricLogger()
    freq = cfg['logging']['log_freq']
    lr = cfg['training']['lr']
    clip = cfg['training']['clip_grad']

    mean, std = loader.dataset.get_meanstd()
    mean, std = mean.to(device), std.to(device)

    criterion = nn.MSELoss(reduction='none')

    for batch in logger.log_every(loader, freq, f'Epoch [{epoch}]'):
        # unpack batch
        if len(batch) == 5:
            ir, era5, inp_label, label, weight = batch
            weight = weight.float().to(device)
        else:
            ir, era5, inp_label, label = batch
            weight = None

        ir = ir.float().to(device)
        era5 = era5.float().to(device)
        inp_label = inp_label.float().to(device)
        label = label.float().to(device)

        # warmup lr
        if step < warmup_steps:
            scale = min(1.0, (step + 1) / warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr * scale

        optimizer.zero_grad()

        # autoregressive forward with accumulated gradients
        T = label.shape[1]
        total_loss = 0.0
        for t in range(T):
            pred = model(ir, era5, inp_label, t + 1)
            loss_t = criterion(pred[:, -1], label[:, t])
            if weight is not None:
                loss_t = loss_t * weight[:, t]
            loss_t = loss_t.mean() / T
            loss_t.backward()
            total_loss += loss_t.item()

        if clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if scheduler and step >= warmup_steps:
            scheduler.step()

        logger.update(loss=total_loss, lr=optimizer.param_groups[0]['lr'])
        if writer and is_main_process():
            writer.add_scalar('train/loss', total_loss, step)
        step += 1

    return step, logger.meters['loss'].global_avg


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    """Evaluate model"""
    model.eval()
    mean, std = loader.dataset.get_meanstd()
    mean, std = mean.to(device), std.to(device)

    pred_len = cfg['model']['pred_len']
    n_labels = cfg['model']['seq_dim']
    mae_sum = torch.zeros(pred_len, n_labels, device=device)
    count = 0

    for batch in loader:
        if len(batch) == 5:
            ir, era5, inp_label, label, _ = batch
        else:
            ir, era5, inp_label, label = batch

        ir = ir.float().to(device)
        era5 = era5.float().to(device)
        inp_label = inp_label.float().to(device)
        label = label.float().to(device)

        B = ir.shape[0]
        T = label.shape[1]

        preds = []
        for t in range(T):
            pred = model(ir, era5, inp_label, t + 1)
            preds.append(pred[:, -1:])
        preds = torch.cat(preds, dim=1)

        # denormalize
        preds = preds * std + mean
        label = label * std + mean

        mae_sum += (preds - label).abs().sum(dim=0)
        count += B

    return mae_sum / count


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.seed:
        cfg['seed'] = args.seed
    if args.output_dir:
        cfg['logging']['output_dir'] = args.output_dir

    rank, world, local = setup_distributed()
    device = torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    setup_seed(cfg['seed'] + rank)

    # output dir
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(cfg['logging']['output_dir'], f'run_{timestamp}')
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(cfg, f)

    # datasets
    print(f"[Rank {rank}] Building datasets...")
    train_ds = build_dataset(cfg, 'train')
    valid_ds = build_dataset(cfg, 'valid')

    batch_size = cfg['training']['batch_size']
    if world > 1:
        batch_size = max(1, batch_size // world)

    train_sampler = DistributedSampler(train_ds) if world > 1 else None
    valid_sampler = DistributedSampler(valid_ds, shuffle=False) if world > 1 else None

    train_loader = DataLoader(train_ds, batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=cfg['training']['workers'],
                              pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size, shuffle=False, sampler=valid_sampler,
                              num_workers=cfg['training']['workers'], pin_memory=True)

    # model
    print(f"[Rank {rank}] Building model...")
    model = build_model(cfg).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local])

    # optimizer
    tcfg = cfg['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay'])
    steps_per_epoch = len(train_loader)
    total_steps = tcfg['epochs'] * steps_per_epoch
    warmup_steps = tcfg['warmup_epochs'] * steps_per_epoch

    if tcfg['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps - warmup_steps, tcfg['min_lr'])
    else:
        scheduler = None

    # resume
    start_epoch = 0
    best = float('inf')
    if args.resume:
        start_epoch, best = load_checkpoint(args.resume, model.module if world > 1 else model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}")

    writer = SummaryWriter(os.path.join(output_dir, 'tb')) if is_main_process() else None

    # eval only
    if args.eval:
        mae = evaluate(model, valid_loader, cfg, device)
        if is_main_process():
            print("Validation MAE:")
            for t in range(mae.shape[0]):
                print(f"  {(t+1)*6}h: WIND={mae[t,0]:.2f} knots, PRES={mae[t,1]:.2f} hPa")
        return

    # training loop
    print(f"[Rank {rank}] Starting training...")
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, tcfg['epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        global_step, train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, warmup_steps,
            epoch, cfg, device, writer, global_step
        )

        # evaluate
        if (epoch + 1) % cfg['logging']['eval_freq'] == 0:
            mae = evaluate(model, valid_loader, cfg, device)
            avg_mae = mae.mean().item()

            if is_main_process():
                print(f"Epoch {epoch} MAE:")
                for t in range(mae.shape[0]):
                    print(f"  {(t+1)*6}h: WIND={mae[t,0]:.2f}, PRES={mae[t,1]:.2f}")

                if writer:
                    writer.add_scalar('valid/mae_avg', avg_mae, epoch)

                is_best = avg_mae < best
                best = min(best, avg_mae)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': (model.module if world > 1 else model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'best_metric': best,
                }, is_best, output_dir, f'ckpt_epoch{epoch+1}.pth')

        if world > 1:
            dist.barrier()

    if writer:
        writer.close()
    print(f"[Rank {rank}] Done!")


if __name__ == '__main__':
    main()
