"""
Per-lead MAE in physical units (wind m/s, pressure hPa).
按预报时效计算物理单位 MAE（风速 m/s，气压 hPa）。
"""
import time

import numpy as np
import torch
import torch.distributed as dist

from dataset.tc_dataset import KNOT_TO_MS


@torch.inference_mode()
def evaluate(model, loader, device, save_predictions=False):
    """
    Evaluate one rollout per sample; missing targets (NaN) are ignored per lead.
    每个样本做一次自回归预报；缺失目标 (NaN) 在对应时效上不计入。
    """
    model.eval()
    mean, std = [x.to(device) for x in loader.dataset.get_meanstd()]
    horizon = loader.dataset.output_len
    sums = torch.zeros(horizon, 2, device=device, dtype=torch.float64)
    counts = torch.zeros(horizon, 2, device=device, dtype=torch.long)
    normalized = torch.zeros((), device=device, dtype=torch.float64)
    samples, preds, targets, ids, start = 0, [], [], [], time.time()
    for ir, era5, history, target, index in loader:
        ir, era5, history, target = [x.to(device, non_blocking=True) for x in (ir, era5, history, target)]
        forecast = model(ir, era5, history, step=horizon, return_sequence=True)
        valid = torch.isfinite(target)
        error = torch.where(valid, (forecast - target).abs(), 0.)
        # De-normalize the error: |a-b| * std
        # 误差反归一化：|a-b| * std
        sums += (error * std).sum(0).double()
        counts += valid.sum(0)
        normalized += error.double().sum()
        samples += len(ir)
        if save_predictions:
            preds.append((forecast * std + mean).cpu().numpy())
            targets.append((target * std + mean).cpu().numpy())
            ids.append(index.numpy())
    if dist.is_initialized():
        total = torch.tensor(samples, device=device)
        for tensor in (sums, counts, normalized, total):
            dist.all_reduce(tensor)
        samples = int(total)
    counts_np = counts.cpu().numpy()
    mae = np.divide(sums.cpu().numpy(), counts_np, out=np.full((horizon, 2), np.nan), where=counts_np > 0)
    to_list = lambda values: [float(v) if np.isfinite(v) else None for v in values]
    result = dict(samples=samples, lead_hours=[6 * (i + 1) for i in range(horizon)],
                  samples_per_lead=counts_np.min(1).tolist(),
                  target_policy=loader.dataset.target_policy,
                  wind_ms=to_list(mae[:, 0] * KNOT_TO_MS), pressure_hpa=to_list(mae[:, 1]),
                  normalized_mae=float(normalized) / max(int(counts_np.sum()), 1),
                  seconds=time.time() - start)
    predictions = None
    if save_predictions:
        predictions = (np.concatenate(ids), np.concatenate(preds), np.concatenate(targets))
        if dist.is_initialized():
            pieces = [None] * dist.get_world_size()
            dist.all_gather_object(pieces, predictions)
            predictions = tuple(np.concatenate([p[i] for p in pieces]) for i in range(3))
            order = np.argsort(predictions[0])
            predictions = tuple(x[order] for x in predictions)
    return result, predictions
