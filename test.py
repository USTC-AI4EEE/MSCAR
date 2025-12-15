#!/usr/bin/env python3
"""
Evaluate MSCAR model

Usage:
    python test.py --checkpoint outputs/best.pth --data_dir /path/to/data
"""

import argparse
import os
import time
import random

import torch
import numpy as np
from torch.utils.data import DataLoader

from models import MSCAR
from dataset import TCDataset


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class MAETracker:
    """Track MAE per variable"""
    def __init__(self, n_vars, device):
        self.count = torch.zeros(n_vars, device=device)
        self.total = torch.zeros(n_vars, device=device)

    def update(self, **kwargs):
        for i, (k, v) in enumerate(kwargs.items()):
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.count[i] += 1
            self.total[i] += v

    def mean(self):
        return self.total / self.count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ibtracs_path', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # dataset
    print("Loading dataset...")
    ds = TCDataset(args.data_dir, args.ibtracs_path, split=args.split)
    loader = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.workers)

    mean, std = ds.get_meanstd()
    mean, std = mean.to(device), std.to(device)

    # model
    print("Building model...")
    model = MSCAR(
        ir_size=(140, 140), era5_size=(40, 40), patch_size=(5, 5),
        ir_ch=1, era5_ch=69, dim=128, seq_dim=2, seq_len=4, pred_len=4,
        depth=3, heads=4, mlp_dim=32, dropout=0.1, ar_hidden=256
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in ckpt:
            state = ckpt['model']
            if isinstance(state, dict) and 'mstar_vit_tc_gridsa_test' in state:
                model.load_state_dict(state['mstar_vit_tc_gridsa_test'])
            else:
                model.load_state_dict(state)
        else:
            model.load_state_dict(ckpt)
    else:
        print("Warning: using random weights")

    model.to(device)
    model.eval()

    # evaluate
    pred_len = 4
    trackers = [MAETracker(2, device) for _ in range(pred_len)]
    total_time = 0
    n_batches = 0

    print(f"\nEvaluating {len(ds)} samples...")
    print("=" * 60)

    for i, batch in enumerate(loader):
        if len(batch) == 5:
            ir, era5, inp_label, label, _ = batch
        else:
            ir, era5, inp_label, label = batch

        ir = ir.float().to(device)
        era5 = era5.float().to(device)
        inp_label = inp_label.float().to(device)
        label = label.float().to(device)

        with torch.no_grad():
            preds = []
            t0 = time.time()
            for t in range(label.shape[1]):
                pred = model(ir, era5, inp_label, t + 1)
                preds.append(pred)
            total_time += time.time() - t0
            n_batches += 1

            preds = torch.cat(preds, dim=1)

            # denormalize and compute MAE
            for t in range(pred_len):
                gt = label[:, t] * std + mean
                pr = preds[:, t] * std + mean
                mae = (pr - gt).abs().mean(dim=0)
                trackers[t].update(WIND=mae[0], PRES=mae[1])

        if i % 50 == 0:
            print(f"Step [{i+1}/{len(loader)}]")
            for t in range(pred_len):
                m = trackers[t].mean()
                print(f"  {(t+1)*6}h: WIND={m[0]:.2f} knots, PRES={m[1]:.2f} hPa")

    # final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for t in range(pred_len):
        m = trackers[t].mean()
        print(f"{(t+1)*6}h: WIND MAE = {m[0]:.2f} knots, PRES MAE = {m[1]:.2f} hPa")
    print(f"\nAvg inference time: {total_time/n_batches:.4f} s/batch")


if __name__ == '__main__':
    main()
