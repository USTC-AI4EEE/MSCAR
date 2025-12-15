#!/usr/bin/env python3
"""
Train baseline models (LSTM, Transformer, ConvGRU)

Usage:
    python baselines/train.py --model lstm --epochs 50
    python baselines/train.py --model transformer --epochs 50
    python baselines/train.py --model convgru --epochs 50
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import LSTM, TCTransformer, ConvGRU
from dataset import TCDataset


MODELS = {
    'lstm': LSTM,
    'transformer': TCTransformer,
    'convgru': ConvGRU,
}

DEFAULT_CONFIGS = {
    'lstm': {'hidden': 256, 'lstm_hidden': 128, 'n_layers': 2},
    'transformer': {'d_model': 256, 'n_heads': 4, 'n_layers': 3, 'd_ff': 512},
    'convgru': {'hidden': 64, 'gru_hidden': 64, 'n_layers': 2, 'spatial': 16},
}


def get_model(name, cfg):
    base = {'ir_ch': 1, 'era5_ch': 69, 'seq_dim': 2, 'input_len': 4, 'output_len': 4, 'dropout': 0.1}
    base.update(DEFAULT_CONFIGS.get(name, {}))
    base.update(cfg)
    return MODELS[name](**base)


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    n = 0

    for i, batch in enumerate(loader):
        if len(batch) == 5:
            ir, era5, inp_seq, label, _ = batch
        else:
            ir, era5, inp_seq, label = batch

        ir = ir.float().to(device)
        era5 = era5.float().to(device)
        inp_seq = inp_seq.float().to(device)
        label = label.float().to(device)

        optimizer.zero_grad()
        out = model(ir, era5, inp_seq)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

        if i % 20 == 0:
            print(f"  [{i}/{len(loader)}] loss: {loss.item():.4f}")

    return total_loss / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n = 0

    for batch in loader:
        if len(batch) == 5:
            ir, era5, inp_seq, label, _ = batch
        else:
            ir, era5, inp_seq, label = batch

        ir = ir.float().to(device)
        era5 = era5.float().to(device)
        inp_seq = inp_seq.float().to(device)
        label = label.float().to(device)

        out = model(ir, era5, inp_seq)
        loss = criterion(out, label)

        total_loss += loss.item()
        n += 1

    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lstm', choices=['lstm', 'transformer', 'convgru'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='outputs/baselines')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ibtracs_path', required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # output dir
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output_dir, f'{args.model}_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    # model
    model = get_model(args.model, {}).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, Params: {n_params:,}")

    # data
    years = {'train': range(1980, 2018), 'valid': range(2018, 2019), 'test': range(2019, 2021)}
    train_ds = TCDataset(args.data_dir, args.ibtracs_path, 'train', years)
    valid_ds = TCDataset(args.data_dir, args.ibtracs_path, 'valid', years)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, args.batch_size, shuffle=False, num_workers=4)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_loss = float('inf')
    history = {'train': [], 'valid': []}

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        valid_loss = validate(model, valid_loader, criterion, device)
        scheduler.step()

        history['train'].append(train_loss)
        history['valid'].append(valid_loss)

        print(f"Epoch {epoch+1}/{args.epochs} - train: {train_loss:.4f}, valid: {valid_loss:.4f}, "
              f"time: {time.time()-t0:.1f}s, lr: {scheduler.get_last_lr()[0]:.6f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': valid_loss,
            }, os.path.join(out_dir, 'best.pth'))
            print(f"  -> saved best model")

    # save final
    torch.save({'model': model.state_dict()}, os.path.join(out_dir, 'final.pth'))
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f)

    print(f"\nDone! Best loss: {best_loss:.4f}")
    print(f"Results: {out_dir}")


if __name__ == '__main__':
    main()
