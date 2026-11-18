"""
Rebuild the sample eligibility of the original MSCAR loader (future IR quality).
重建原始 MSCAR 数据加载器的样本筛选规则（未来时刻红外质量）。

Future frames are only checked for selection and are never fed to the model.
未来时刻数据仅用于样本筛选，不输入模型。
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def load_rate(source, rel):
    """IR missing rate of one frame; 1.0 if the frame does not exist.
    单帧红外缺测率；若该帧不存在则为 1.0。"""
    path = source / rel
    if not (path / 'ERA5_data.npy').is_file() or not (path / 'GRIDSAT_data.npy').is_file():
        return rel, 1.
    return rel, float((~np.isfinite(np.load(path / 'GRIDSAT_data.npy')[0])).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    root = Path(a.cache)
    meta = json.loads((root / 'metadata.json').read_text())
    source = Path(meta['source'])
    lookup = dict(zip(meta['records'], np.load(root / 'ir_nan_rate.npy')))

    # Future frames of every window / 每个窗口对应的未来帧
    future, required = [], set()
    for year, sid, t in zip(meta['years'], meta['sids'], meta['initial_times']):
        initial = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        paths = [f'{year}/{sid}/' + (initial + timedelta(hours=6 * lead)).strftime('%Y-%m-%d %H_%M_%S')
                 for lead in range(1, meta['output_len'] + 1)]
        future.append(paths)
        required.update(path for path in paths if path not in lookup)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        lookup.update(pool.map(lambda rel: load_rate(source, rel), sorted(required)))

    # Keep windows whose future IR frames are all <= 1% missing
    # 保留未来红外帧缺测率均不超过 1% 的窗口
    valid = np.array([all(lookup[path] <= .01 for path in paths) for paths in future], dtype=bool)
    np.save(root / 'legacy_eligible.npy', valid)
    print(json.dumps(dict(windows=len(valid), eligible=int(valid.sum()))), flush=True)


if __name__ == '__main__':
    main()
