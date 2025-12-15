"""
Training utilities
"""

import os
import time
import datetime
import random
from collections import defaultdict, deque

import torch
import torch.distributed as dist
import numpy as np


class SmoothedValue:
    """Track values and provide smoothed average"""

    def __init__(self, window=100):
        self.deque = deque(maxlen=window)
        self.total = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.deque.append(val)
        self.count += n
        self.total += val * n

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque)).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    def sync(self):
        """Synchronize across distributed processes"""
        if not is_dist_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.all_reduce(t)
        t = t / dist.get_world_size()
        self.count = int(t[0].item())
        self.total = t[1].item()


class MetricLogger:
    """Log training metrics"""

    def __init__(self, delimiter="  "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        parts = [f"{k}: {m.median:.4f} ({m.global_avg:.4f})" for k, m in self.meters.items()]
        return self.delimiter.join(parts)

    def log_every(self, iterable, freq, header=""):
        i = 0
        start = time.time()
        end = time.time()
        iter_time = SmoothedValue()
        data_time = SmoothedValue()

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if i % freq == 0:
                eta = iter_time.global_avg * (len(iterable) - i)
                eta_str = str(datetime.timedelta(seconds=int(eta)))
                mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                print(f"{header} [{i}/{len(iterable)}] eta: {eta_str}  {self}  "
                      f"time: {iter_time.avg:.3f}  data: {data_time.avg:.3f}  mem: {mem:.0f}MB")

            i += 1
            end = time.time()

        total = time.time() - start
        print(f"{header} Total: {str(datetime.timedelta(seconds=int(total)))} ({total/len(iterable):.3f} s/it)")


class AverageMeter:
    """Simple average meter"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_initialized() else 0


def is_main_process():
    return get_rank() == 0


def setup_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    """Save training checkpoint"""
    if not is_main_process():
        return
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(save_dir, 'best.pth')
        torch.save(state, best_path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load checkpoint and return epoch, best metric"""
    ckpt = torch.load(path, map_location='cpu')

    # handle different checkpoint formats
    if 'model' in ckpt:
        state = ckpt['model']
        if 'mstar_vit_tc_gridsa_test' in state:
            model.load_state_dict(state['mstar_vit_tc_gridsa_test'])
        else:
            model.load_state_dict(state)
    else:
        model.load_state_dict(ckpt)

    epoch = ckpt.get('epoch', 0)
    best = ckpt.get('best_metric', float('inf'))

    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])

    return epoch, best


def get_grad_norm(parameters, norm_type=2):
    """Compute gradient norm"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm = float(norm_type)
    total = sum(p.grad.data.norm(norm).item() ** norm for p in parameters)
    return total ** (1.0 / norm)
