"""
Build a 120h (20-step) label view on top of a 24h cache, without copying images.
在 24h 缓存基础上构建 120h（20 步）标签视图，不复制图像。

Missing future labels are NaN; target_policy=complete keeps full 120h windows only.
缺失的未来标签记为 NaN；target_policy=complete 只保留完整 120h 窗口。

python tools/prepare_horizon.py --base-cache data/train --output data/train_120h --output-len 20
"""
import argparse
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def prepare(base_cache, output, output_len=20):
    base, dest = Path(base_cache).resolve(), Path(output).resolve()
    if dest == base:
        raise ValueError('The horizon view must not overwrite the base cache')
    if (dest / 'COMPLETE.json').exists():
        print('Already complete:', dest, flush=True)
        return
    if not (base / 'COMPLETE.json').is_file():
        raise FileNotFoundError(base)
    meta = json.loads((base / 'metadata.json').read_text())
    input_len, old_labels = meta['input_len'], np.load(base / 'labels.npy')

    # Look up future IBTrACS labels by (year, SID, time)
    # 按 (年份, SID, 时间) 查找未来 IBTrACS 标签
    table = pd.read_csv(meta['ibtracs'], usecols=['SID', 'SEASON', 'ISO_TIME', 'USA_WIND', 'USA_PRES'],
                        dtype=str, keep_default_na=False)
    table['year'] = pd.to_numeric(table.SEASON, errors='coerce')
    table = table[table.year.isin(set(meta['years']))]
    table = table[table.ISO_TIME.str[11:].isin(['00:00:00', '06:00:00', '12:00:00', '18:00:00'])]
    table['wind'] = pd.to_numeric(table.USA_WIND, errors='coerce')
    table['pres'] = pd.to_numeric(table.USA_PRES, errors='coerce')
    table = table.sort_values(['year', 'SID', 'ISO_TIME']).drop_duplicates(['year', 'SID', 'ISO_TIME'])
    lookup = {}
    for row in table.itertuples():
        values = np.array([row.wind, row.pres], dtype=np.float32)
        values[~np.isfinite(values) | (values <= 0)] = np.nan
        lookup[(int(row.year), row.SID, row.ISO_TIME)] = values

    labels = np.full((len(old_labels), input_len + output_len, 2), np.nan, dtype=np.float32)
    for i, (year, sid, initial) in enumerate(zip(meta['years'], meta['sids'], meta['initial_times'])):
        start = datetime.strptime(initial, '%Y-%m-%d %H:%M:%S')
        labels[i, :input_len] = old_labels[i, :input_len]
        for lead in range(1, output_len + 1):
            key = (int(year), sid, (start + timedelta(hours=6 * lead)).strftime('%Y-%m-%d %H:%M:%S'))
            if key in lookup:
                labels[i, input_len + lead - 1] = lookup[key]
    # The first 24h must equal the base cache / 前 24h 必须与原缓存一致
    common = min(old_labels.shape[1], labels.shape[1])
    np.testing.assert_array_equal(labels[:, :common], old_labels[:, :common])

    dest.mkdir(parents=True, exist_ok=True)
    for name in ['indices.npy', 'legacy_eligible.npy']:
        shutil.copy2(base / name, dest / name)
    np.save(dest / 'labels.npy', labels)
    # Point to the base image arrays instead of copying them
    # 指向原缓存的图像数组，而不是复制
    frame_cache = Path(meta.get('frame_cache', '.'))
    frame_cache = frame_cache if frame_cache.is_absolute() else (base / frame_cache).resolve()
    meta.update(output_len=output_len, frame_cache=os.path.relpath(frame_cache, dest),
                base_cache=os.path.relpath(base, dest))
    (dest / 'metadata.json').write_text(json.dumps(meta), encoding='utf-8')
    complete = dict(windows=len(labels), input_len=input_len, output_len=output_len,
                    available_targets_per_lead=np.isfinite(labels[:, input_len:]).all(2).sum(0).tolist(),
                    complete_windows=int(np.isfinite(labels).all((1, 2)).sum()))
    (dest / 'COMPLETE.json').write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base-cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--output-len', type=int, default=20)
    args = p.parse_args()
    prepare(args.base_cache, args.output, args.output_len)
