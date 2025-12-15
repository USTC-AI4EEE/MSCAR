"""
Dataset for Tropical Cyclone Intensity Prediction

Loads GRIDSAT IR satellite images and ERA5 reanalysis data with IBTrACS labels.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from datetime import datetime
import random

from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d


# ERA5 pressure levels
PRESSURE_LEVELS = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
                   225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
                   775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]


class RotateTransform:
    """Rotate by one of the given angles"""
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class TCDataset(Dataset):
    """
    TC Dataset loading satellite IR and ERA5 data.

    Expected data structure:
        data_dir/
            {year}/
                {tc_id}/
                    {timestamp}/
                        GRIDSAT_data.npy  # (3, H, W) satellite channels
                        ERA5_data.npy     # (C, H, W) atmospheric variables

    Args:
        data_dir: root directory containing TC data
        ibtracs_path: path to IBTrACS CSV file
        split: 'train', 'valid', or 'test'
        years: dict with year ranges for each split
        input_len: number of input time steps
        output_len: number of output time steps
        ir_size: GRIDSAT crop size
        era5_size: ERA5 crop size
        augment: whether to use rotation augmentation
    """

    def __init__(self, data_dir, ibtracs_path, split='train', years=None,
                 input_len=4, output_len=4, ir_size=140, era5_size=40,
                 augment=False, use_lds=False, stats_dir=None, **kwargs):

        self.data_dir = data_dir
        self.split = split
        self.input_len = input_len
        self.output_len = output_len
        self.window_size = input_len + output_len
        self.ir_size = ir_size
        self.era5_size = era5_size
        self.use_lds = use_lds

        # default year ranges
        if years is None:
            years = {
                'train': range(1980, 2018),
                'valid': range(2018, 2019),
                'test': range(2019, 2021),
            }
        self.years = years[split]

        # variable configuration
        self.single_vars = kwargs.get('single_vars', ['u10', 'v10', 't2m', 'msl'])
        self.multi_vars = kwargs.get('multi_vars', ['z', 'q', 'u', 'v', 't'])
        self.levels = kwargs.get('levels', [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
        self.label_vars = kwargs.get('label_vars', ['USA_WIND', 'USA_PRES'])
        self.basins = kwargs.get('basins', ['EP', 'NA', 'NI', 'SA', 'SI', 'SP', 'WP'])

        # build variable index mapping
        self.var_idx = self._build_var_index()

        # load normalization stats
        self.stats_dir = stats_dir or os.path.join(os.path.dirname(__file__), 'stats')
        self._load_stats()

        # load IBTrACS and build sample list
        self.samples = self._build_samples(ibtracs_path)
        print(f"[{split}] loaded {len(self.samples)} samples")

        # data augmentation (4x rotation for training)
        self.augment = augment and split == 'train'
        self._setup_transforms()
        if self.augment:
            self._apply_augmentation()

        # label distribution smoothing
        if use_lds and split == 'train':
            self._setup_lds(kwargs.get('lds_config', {}))

    def _build_var_index(self):
        """Build mapping from variable name to channel index"""
        idx = {}
        i = 0
        for v in self.single_vars:
            idx[v] = i
            i += 1
        for v in self.multi_vars:
            idx[v] = {}
            for h in self.levels:
                idx[v][h] = i
                i += 1
        return idx

    def _load_stats(self):
        """Load mean/std for normalization"""
        # defaults in case files don't exist
        self.ir_mean = torch.tensor([260.0])
        self.ir_std = torch.tensor([30.0])
        self.era5_mean = torch.zeros(len(self.single_vars) + len(self.multi_vars) * len(self.levels))
        self.era5_std = torch.ones_like(self.era5_mean)
        self.label_mean = torch.tensor([50.0, 990.0])
        self.label_std = torch.tensor([28.0, 21.0])

        # try loading from files
        ir_path = os.path.join(self.stats_dir, 'GRIDSAT_TC_mean_std.json')
        if os.path.exists(ir_path):
            with open(ir_path) as f:
                stats = json.load(f)
            self.ir_mean = torch.tensor([stats['mean'].get('irwin_cdr', 260.0)])
            self.ir_std = torch.tensor([stats['std'].get('irwin_cdr', 30.0)])

        label_path = os.path.join(self.stats_dir, 'IBTrACS_intensity_TC_mean_std.json')
        if os.path.exists(label_path):
            with open(label_path) as f:
                stats = json.load(f)
            means = [stats['mean'].get(v, 50.0) for v in self.label_vars]
            stds = [stats['std'].get(v, 28.0) for v in self.label_vars]
            self.label_mean = torch.tensor(means)
            self.label_std = torch.tensor(stds)

        single_path = os.path.join(self.stats_dir, 'ERA5_single_TC_mean_std.json')
        multi_path = os.path.join(self.stats_dir, 'ERA5_TC_mean_std.json')
        if os.path.exists(single_path) and os.path.exists(multi_path):
            with open(single_path) as f:
                single_stats = json.load(f)
            with open(multi_path) as f:
                multi_stats = json.load(f)
            means, stds = [], []
            for v in self.single_vars:
                means.append(single_stats['mean'].get(v, 0))
                stds.append(single_stats['std'].get(v, 1))
            for v in self.multi_vars:
                for h in self.levels:
                    means.append(multi_stats['mean'].get(v, {}).get(str(h), 0))
                    stds.append(multi_stats['std'].get(v, {}).get(str(h), 1))
            self.era5_mean = torch.tensor(means)
            self.era5_std = torch.tensor(stds)

    def _build_samples(self, ibtracs_path):
        """Build list of valid samples from IBTrACS"""
        df = pd.read_csv(ibtracs_path, low_memory=False)
        samples = []

        for year in self.years:
            year_df = df[df['SEASON'] == str(year)]
            tc_ids = year_df['SID'].unique()

            for tc_id in tc_ids:
                tc_df = year_df[year_df['SID'] == tc_id]
                basin = tc_df['BASIN'].iloc[0]
                if pd.isna(basin):
                    basin = 'NA'
                if basin not in self.basins:
                    continue

                # collect valid time points
                times, labels = [], []
                for _, row in tc_df.iterrows():
                    iso_time = row['ISO_TIME']
                    if iso_time[11:] not in ['00:00:00', '06:00:00', '12:00:00', '18:00:00']:
                        continue

                    # check data exists
                    time_str = iso_time.replace(':', '_')
                    data_path = os.path.join(self.data_dir, str(year), tc_id, time_str)
                    if not os.path.exists(os.path.join(data_path, 'GRIDSAT_data.npy')):
                        continue
                    if not os.path.exists(os.path.join(data_path, 'ERA5_data.npy')):
                        continue

                    # check labels are valid
                    label = []
                    valid = True
                    for v in self.label_vars:
                        val = row[v]
                        if pd.isna(val) or val == ' ':
                            valid = False
                            break
                        try:
                            label.append(float(val))
                        except:
                            valid = False
                            break
                    if not valid:
                        continue

                    times.append((iso_time, data_path))
                    labels.append(label)

                # create sliding windows
                for i in range(len(times) - self.window_size + 1):
                    window = times[i:i + self.window_size]
                    window_labels = labels[i:i + self.window_size]

                    # check 6h consecutive
                    if not self._check_consecutive(window):
                        continue

                    inp_times = window[:self.input_len]
                    out_times = window[self.input_len:]
                    inp_labels = window_labels[:self.input_len]
                    out_labels = window_labels[self.input_len:]

                    samples.append({
                        'input_paths': [t[1] for t in inp_times],
                        'input_labels': inp_labels,
                        'output_labels': out_labels,
                    })

        return samples

    def _check_consecutive(self, times):
        """Check if times are 6h apart"""
        for i in range(len(times) - 1):
            t1 = datetime.strptime(times[i][0], '%Y-%m-%d %H:%M:%S')
            t2 = datetime.strptime(times[i + 1][0], '%Y-%m-%d %H:%M:%S')
            if (t2 - t1).total_seconds() != 6 * 3600:
                return False
        return True

    def _setup_transforms(self):
        """Setup image transforms"""
        self.ir_transform = transforms.CenterCrop(self.ir_size)
        self.era5_transform = transforms.CenterCrop(self.era5_size)

        if self.augment:
            self.ir_transforms = {
                0: transforms.Compose([transforms.CenterCrop(self.ir_size), RotateTransform([0])]),
                90: transforms.Compose([transforms.CenterCrop(self.ir_size), RotateTransform([90])]),
                180: transforms.Compose([transforms.CenterCrop(self.ir_size), RotateTransform([180])]),
                270: transforms.Compose([transforms.CenterCrop(self.ir_size), RotateTransform([270])]),
            }
            self.era5_transforms = {
                0: transforms.Compose([transforms.CenterCrop(self.era5_size), RotateTransform([0])]),
                90: transforms.Compose([transforms.CenterCrop(self.era5_size), RotateTransform([90])]),
                180: transforms.Compose([transforms.CenterCrop(self.era5_size), RotateTransform([180])]),
                270: transforms.Compose([transforms.CenterCrop(self.era5_size), RotateTransform([270])]),
            }

    def _apply_augmentation(self):
        """Quadruple dataset with rotations"""
        n = len(self.samples)
        self.angles = [0] * n + [90] * n + [180] * n + [270] * n
        self.samples = self.samples * 4

    def _setup_lds(self, config):
        """Setup label distribution smoothing weights"""
        reweight = config.get('reweight', 'sqrt_inv')
        use_smooth = config.get('lds', False)
        kernel = config.get('lds_kernel', 'gaussian')
        ks = config.get('lds_ks', 5)
        sigma = config.get('lds_sigma', 2)
        min_label = config.get('min_label', [0, 850])
        max_label = config.get('max_label', [200, 1030])

        n_samples = len(self.samples) // (4 if self.augment else 1)
        labels = np.array([s['output_labels'] for s in self.samples[:n_samples]])

        self.weights = np.zeros((n_samples, self.output_len, len(self.label_vars)))
        for t in range(self.output_len):
            for v in range(len(self.label_vars)):
                self.weights[:, t, v] = self._compute_weights(
                    labels[:, t, v], reweight, min_label[v], max_label[v],
                    use_smooth, kernel, ks, sigma
                )

        if self.augment:
            self.weights = np.tile(self.weights, (4, 1, 1))

    def _compute_weights(self, labels, reweight, min_val, max_val, smooth, kernel, ks, sigma):
        """Compute sample weights for imbalanced regression"""
        shifted = labels - min_val
        max_bin = int(max_val - min_val)

        counts = {i: 0 for i in range(max_bin)}
        for l in shifted:
            counts[min(max_bin - 1, int(l))] += 1

        if reweight == 'sqrt_inv':
            counts = {k: np.sqrt(v) for k, v in counts.items()}

        per_label = [counts[min(max_bin - 1, int(l))] for l in shifted]

        if smooth:
            half_ks = (ks - 1) // 2
            if kernel == 'gaussian':
                base = [0.] * half_ks + [1.] + [0.] * half_ks
                win = gaussian_filter1d(base, sigma)
                win = win / win.max()
            elif kernel == 'triang':
                win = triang(ks)
            else:
                laplace = lambda x: np.exp(-abs(x) / sigma) / (2 * sigma)
                win = [laplace(i) for i in range(-half_ks, half_ks + 1)]
                win = np.array(win) / max(win)

            smoothed = convolve1d([counts[i] for i in range(max_bin)], win, mode='constant')
            per_label = [smoothed[min(max_bin - 1, int(l))] for l in shifted]

        weights = [1.0 / x if x > 0 else 0 for x in per_label]
        scale = len(weights) / sum(weights) if sum(weights) > 0 else 1
        return [w * scale for w in weights]

    def _load_ir(self, path):
        """Load and normalize IR data"""
        data = np.load(os.path.join(path, 'GRIDSAT_data.npy'))
        data = data[[0]]  # use only first channel (irwin_cdr)
        data = torch.from_numpy(data).float()
        data = (data - self.ir_mean.view(-1, 1, 1)) / self.ir_std.view(-1, 1, 1)
        data = torch.nan_to_num(data, nan=0.0)
        return data

    def _load_era5(self, path):
        """Load and normalize ERA5 data"""
        data = np.load(os.path.join(path, 'ERA5_data.npy'))

        # select channels
        idx = []
        for v in self.single_vars:
            idx.append(self.var_idx[v])
        for v in self.multi_vars:
            for h in self.levels:
                idx.append(self.var_idx[v][h])
        data = data[idx]

        data = torch.from_numpy(data).float()
        data = (data - self.era5_mean.view(-1, 1, 1)) / self.era5_std.view(-1, 1, 1)
        return data

    def get_meanstd(self):
        """Return label mean/std for denormalization"""
        return self.label_mean, self.label_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # load input data
        ir_list, era5_list = [], []
        for path in sample['input_paths']:
            ir = self._load_ir(path)
            era5 = self._load_era5(path)

            # apply transforms
            if self.augment:
                angle = self.angles[idx]
                ir = self.ir_transforms[angle](ir)
                era5 = self.era5_transforms[angle](era5)
            else:
                ir = self.ir_transform(ir)
                era5 = self.era5_transform(era5)

            ir_list.append(ir)
            era5_list.append(era5)

        ir_data = torch.stack(ir_list)
        era5_data = torch.stack(era5_list)

        # labels
        inp_label = torch.tensor(sample['input_labels']).float()
        out_label = torch.tensor(sample['output_labels']).float()

        # normalize labels
        inp_label = (inp_label - self.label_mean) / self.label_std
        out_label = (out_label - self.label_mean) / self.label_std

        if self.use_lds and self.split == 'train':
            weight = torch.tensor(self.weights[idx]).float()
            return ir_data, era5_data, inp_label, out_label, weight

        return ir_data, era5_data, inp_label, out_label


if __name__ == '__main__':
    # test loading
    ds = TCDataset(
        data_dir='/storage/yuxiaoning/ERA5/TC_ERA5',
        ibtracs_path='/storage/yuxiaoning/ERA5/ibtracs.ALL.list.v04r00.csv',
        split='train',
        years={'train': range(2011, 2012), 'valid': range(2018, 2019), 'test': range(2019, 2020)},
    )
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        print(f"IR shape: {sample[0].shape}")
        print(f"ERA5 shape: {sample[1].shape}")
        print(f"Input label shape: {sample[2].shape}")
        print(f"Output label shape: {sample[3].shape}")
