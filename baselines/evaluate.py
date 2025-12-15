#!/usr/bin/env python3
"""
Evaluate baseline models

Usage:
    python baselines/evaluate.py --model lstm --checkpoint outputs/baselines/lstm_xxx/best.pth
"""

import os
import sys
import argparse
import json
import numpy as np

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import LSTM, TCTransformer, ConvGRU
from baselines.train import get_model
from dataset import TCDataset


def compute_metrics(pred, target, mean=None, std=None):
    """Compute MAE, RMSE, R2 for each timestep and variable"""
    if mean is not None and std is not None:
        pred = pred * std + mean
        target = target * std + mean

    N, T, V = pred.shape
    var_names = ['WIND', 'PRES']
    metrics = {}

    for t in range(T):
        interval = (t + 1) * 6
        metrics[f'{interval}h'] = {}

        for v in range(V):
            p = pred[:, t, v]
            g = target[:, t, v]

            mae = np.mean(np.abs(p - g))
            rmse = np.sqrt(np.mean((p - g) ** 2))
            ss_res = np.sum((g - p) ** 2)
            ss_tot = np.sum((g - g.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)

            metrics[f'{interval}h'][var_names[v]] = {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2': float(r2),
            }

    # overall
    metrics['overall'] = {}
    for v in range(V):
        p = pred[:, :, v].flatten()
        g = target[:, :, v].flatten()
        metrics['overall'][var_names[v]] = {
            'MAE': float(np.mean(np.abs(p - g))),
            'RMSE': float(np.sqrt(np.mean((p - g) ** 2))),
        }

    return metrics


@torch.no_grad()
def evaluate(model, loader, device, mean=None, std=None):
    model.eval()
    all_pred, all_target = [], []

    for batch in loader:
        if len(batch) == 5:
            ir, era5, inp_seq, label, _ = batch
        else:
            ir, era5, inp_seq, label = batch

        ir = ir.float().to(device)
        era5 = era5.float().to(device)
        inp_seq = inp_seq.float().to(device)

        out = model(ir, era5, inp_seq)

        all_pred.append(out.cpu().numpy())
        all_target.append(label.numpy())

    pred = np.concatenate(all_pred)
    target = np.concatenate(all_target)

    return compute_metrics(pred, target, mean, std)


def print_metrics(metrics, name):
    print(f"\n{'='*50}")
    print(f"Results for {name}")
    print(f"{'='*50}")

    print("\nMAE per interval:")
    print(f"{'Interval':<10} {'WIND (kt)':<12} {'PRES (hPa)':<12}")
    print("-" * 35)
    for h in ['6h', '12h', '18h', '24h']:
        if h in metrics:
            print(f"{h:<10} {metrics[h]['WIND']['MAE']:<12.2f} {metrics[h]['PRES']['MAE']:<12.2f}")

    print("\nOverall:")
    print(f"  WIND MAE: {metrics['overall']['WIND']['MAE']:.2f}, RMSE: {metrics['overall']['WIND']['RMSE']:.2f}")
    print(f"  PRES MAE: {metrics['overall']['PRES']['MAE']:.2f}, RMSE: {metrics['overall']['PRES']['RMSE']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lstm', choices=['lstm', 'transformer', 'convgru'])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ibtracs_path', required=True)
    parser.add_argument('--output', default=None, help='save metrics to file')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # model
    model = get_model(args.model, {})
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        print(f"Loaded {args.checkpoint}")
    else:
        print("Warning: random weights")

    model.to(device)

    # data
    years = {'train': range(1980, 2018), 'valid': range(2018, 2019), 'test': range(2019, 2021)}
    test_ds = TCDataset(args.data_dir, args.ibtracs_path, 'test', years)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=4)

    mean, std = test_ds.get_meanstd()
    mean, std = mean.numpy(), std.numpy()

    metrics = evaluate(model, test_loader, device, mean, std)
    print_metrics(metrics, args.model.upper())

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
