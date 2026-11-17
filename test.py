"""
Run inference with trained MSCAR weights on the 2019-2020 test set (or another split).
在 2019-2020 测试集（或其他划分）上用训练好的 MSCAR 权重推理。

python test.py --config configs/mscar_24h.yaml
python test.py --config configs/mscar_120h.yaml
python test.py --config configs/mscar_120h.yaml --split valid --target-policy available
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import TCDataset, EvaluationSampler
from models import MSCAR
from utils import load_checkpoint, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', help='override the config / 覆盖配置中的权重路径')
    p.add_argument('--cache', help='override data.evaluation_cache / 覆盖评测数据路径')
    p.add_argument('--split', choices=['train', 'valid', 'test'])
    p.add_argument('--target-policy', choices=['complete', 'available'])
    p.add_argument('--years', type=int, nargs='+', help='e.g. 2019 / 例如 2019')
    p.add_argument('--basins', nargs='+', help='e.g. WP / 例如 WP')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output')
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    ec = cfg['evaluation']
    checkpoint = a.checkpoint or cfg['checkpoint']
    cache = a.cache or cfg['data']['evaluation_cache']
    split = a.split or ec['split']
    policy = a.target_policy or ec['target_policy']
    output = Path(a.output or ec['output'])

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(local)
    if world > 1:
        torch.distributed.init_process_group('nccl' if device.type == 'cuda' else 'gloo')

    model = MSCAR(**cfg["model"])
    state = load_checkpoint(model, checkpoint)
    model.to(device)
    # Each checkpoint carries its own normalization / 每个权重自带归一化参数
    dataset = TCDataset(cache, split, state['normalization'], output_len=cfg['model']['pred_len'],
                        target_policy=policy, years=a.years, basins=a.basins)
    loader = DataLoader(dataset, a.batch_size, sampler=EvaluationSampler(dataset, rank, world),
                        num_workers=a.workers, pin_memory=device.type == 'cuda')
    result, (ids, preds, targets) = evaluate(model, loader, device, save_predictions=True)
    result.update(checkpoint=str(checkpoint), split=split, years=a.years, basins=a.basins)

    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        # Raw predictions (wind kt, pressure hPa) with storm metadata
        # 保存原始预测值（风速 kt，气压 hPa）及对应台风信息
        rows = dataset.selection[ids]
        meta = {k: np.asarray(dataset.meta[k])[rows] for k in ('years', 'basins', 'sids', 'initial_times')}
        np.savez_compressed(output.with_suffix('.npz'), predictions=preds, targets=targets, **meta)
        for h, w, pr in zip(result['lead_hours'], result['wind_ms'], result['pressure_hpa']):
            print(f'{h:>4}h  wind {w:.4f} m/s  pressure {pr:.4f} hPa')
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
