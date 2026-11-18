"""
Packed TC dataset for MSCAR (24h / 120h).
MSCAR 打包后的热带气旋数据集（24h / 120h 通用）。

Split / 数据划分:  train 1980-2017, valid 2018, test 2019-2020
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

KNOT_TO_MS = 0.514444
SPLITS = {'train': (1980, 2017), 'valid': (2018, 2018), 'test': (2019, 2020)}
ERA5_SINGLE = ['u10', 'v10', 't2m', 'msl']
ERA5_MULTI = ['z', 'q', 'u', 'v', 't']
ERA5_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def load_normalization():
    """Build the mean/std used by the trained checkpoints.
    构建训练好的权重所用的均值/标准差。"""
    root = Path(__file__).parent / 'stats'
    read = lambda name: json.loads((root / name).read_text())
    ir = read('GRIDSAT_TC_mean_std.json')
    single, multi = read('ERA5_single_TC_mean_std.json'), read('ERA5_TC_mean_std.json')
    # 4 single-level + 5 variables x 13 pressure levels = 69 ERA5 channels
    # 4 个单层变量 + 5 个变量 x 13 个气压层 = 69 个 ERA5 通道
    era5_mean = [single['mean'][v] for v in ERA5_SINGLE]
    era5_std = [single['std'][v] for v in ERA5_SINGLE]
    for v in ERA5_MULTI:
        for level in ERA5_LEVELS:
            era5_mean.append(multi['mean'][v][str(level)])
            era5_std.append(multi['std'][v][str(level)])
    # Intensity labels: [wind (kt), pressure (hPa)]
    # 强度标签：[风速 (kt), 气压 (hPa)]
    label = read('legacy/TCIR_mean_std.json')['label']
    return dict(ir_mean=[ir['mean']['irwin_cdr']], ir_std=[ir['std']['irwin_cdr']],
                era5_mean=era5_mean, era5_std=era5_std,
                label_mean=[label['mean'][i] for i in (3, 5)],
                label_std=[label['std'][i] for i in (3, 5)], profile='legacy')


class TCDataset(Dataset):
    """
    Returns (ir, era5, history, target, index):
      ir      (T, 1, 140, 140)   past IR images
      era5    (T, 69, 40, 40)    past ERA5 fields
      history (T, 2)             past intensity (normalized)
      target  (L, 2)             future intensity (normalized, NaN if missing)
    返回 (ir, era5, history, target, index)：
      ir      过去 T 步红外图像；era5 过去 T 步 ERA5 场；
      history 过去强度（归一化）；target 未来强度（归一化，缺失为 NaN）。
    """

    def __init__(self, cache, split, stats, output_len=None, target_policy='complete',
                 augment=False, nan_threshold=0.01, years=None, basins=None):
        root = Path(cache)
        if not (root / 'COMPLETE.json').is_file():
            raise FileNotFoundError(f'Incomplete cache, run tools/prepare_data.py first: {root}')
        self.meta = json.loads((root / 'metadata.json').read_text())
        frame_cache = Path(self.meta.get('frame_cache', '.'))
        self.frame_root = frame_cache if frame_cache.is_absolute() else (root / frame_cache).resolve()
        self.input_len = int(self.meta['input_len'])
        self.output_len = int(output_len or self.meta['output_len'])
        if not 1 <= self.output_len <= int(self.meta['output_len']):
            raise ValueError('output_len exceeds the prepared horizon')
        if target_policy not in ('complete', 'available'):
            raise ValueError('target_policy must be complete or available')
        self.split, self.target_policy = split, target_policy

        # Sample selection: year split, IR quality, finite ERA5, legacy cohort
        # 样本筛选：年份划分、红外缺测率、ERA5 有限值、原始样本集
        sample_years = np.array(self.meta['years'])
        lo, hi = SPLITS[split]
        indices = np.load(root / 'indices.npy')
        rates = np.load(self.frame_root / 'ir_nan_rate.npy')
        finite = np.load(self.frame_root / 'era5_finite.npy')
        valid = ((sample_years >= lo) & (sample_years <= hi)
                 & (rates[indices] <= nan_threshold).all(1) & finite[indices].all(1)
                 & np.load(root / 'legacy_eligible.npy'))
        if years is not None:
            valid &= np.isin(sample_years, years)
        if basins is not None:
            valid &= np.isin(self.meta['basins'], basins)

        labels = np.load(root / 'labels.npy')[:, :self.input_len + self.output_len]
        future = np.isfinite(labels[:, self.input_len:])
        # complete: every lead observed; available: at least one lead observed
        # complete：所有时效都有观测；available：至少一个时效有观测
        valid &= future.all((1, 2)) if target_policy == 'complete' else future.any((1, 2))
        self.selection = np.flatnonzero(valid)
        if not len(self.selection):
            raise ValueError(f'No {split} samples in {cache}')
        self.indices, self.labels = indices[self.selection], labels[self.selection]
        self.augment = augment and split == 'train'

        self.ir = self.era5 = None
        self.ir_mean = np.array(stats['ir_mean'], np.float32)[:, None, None]
        self.ir_std = np.array(stats['ir_std'], np.float32)[:, None, None]
        self.era5_mean = np.array(stats['era5_mean'], np.float32)[:, None, None]
        self.era5_std = np.array(stats['era5_std'], np.float32)[:, None, None]
        self.label_mean = torch.tensor(stats['label_mean'], dtype=torch.float32)
        self.label_std = torch.tensor(stats['label_std'], dtype=torch.float32)
        print(f'{split}: {len(self.selection)} windows, horizon={self.output_len}', flush=True)

    def __len__(self):
        # Augmentation adds 90/180/270-degree rotations
        # 数据增强额外加入 90/180/270 度旋转
        return len(self.indices) * (4 if self.augment else 1)

    def __getstate__(self):
        # Memory maps are reopened inside each worker
        # 内存映射在每个 worker 中重新打开
        state = self.__dict__.copy()
        state['ir'] = state['era5'] = None
        return state

    def __getitem__(self, index):
        n = len(self.indices)
        rotation, index = divmod(index, n)
        if self.ir is None:
            self.ir = np.load(self.frame_root / 'ir.npy', mmap_mode='r')
            self.era5 = np.load(self.frame_root / 'era5.npy', mmap_mode='r')
        ids = self.indices[index]
        ir = np.nan_to_num((np.array(self.ir[ids]) - self.ir_mean) / self.ir_std, nan=0., posinf=0., neginf=0.)
        era5 = (np.array(self.era5[ids]) - self.era5_mean) / self.era5_std
        if rotation:
            ir = np.rot90(ir, rotation, axes=(-2, -1)).copy()
            era5 = np.rot90(era5, rotation, axes=(-2, -1)).copy()
        labels = (torch.from_numpy(self.labels[index]) - self.label_mean) / self.label_std
        t = self.input_len
        return torch.from_numpy(ir), torch.from_numpy(era5), labels[:t], labels[t:], index

    def get_meanstd(self):
        return self.label_mean, self.label_std


class EvaluationSampler(Sampler):
    """Shard evaluation data across ranks without repeating samples.
    在多卡间切分评测数据，不重复采样。"""

    def __init__(self, dataset, rank=0, world=1):
        self.length, self.rank, self.world = len(dataset), rank, world

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world))

    def __len__(self):
        return len(range(self.rank, self.length, self.world))
