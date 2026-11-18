"""
Pack historical center crops of TC_datasets into memory-mapped arrays.
将 TC_datasets 的历史时刻中心裁剪打包为内存映射数组。

IR (3,286,286) -> (1,140,140); ERA5 (69,80,80) -> (69,40,40).
Only the past 4 six-hourly frames enter the model; future fields are never read.
模型只使用过去 4 个 6 小时时刻；不读取未来场。

python tools/prepare_data.py --data-dir /path/to/TC_datasets --ibtracs /path/to/ibtracs.csv \
    --begin-year 1980 --end-year 2017 --output data/train
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BASINS = ['EP', 'NA', 'NI', 'SA', 'SI', 'SP', 'WP', '']
SYNOPTIC = ['00:00:00', '06:00:00', '12:00:00', '18:00:00']


def crop(array, size):
    """Center crop with the same rounding as torchvision CenterCrop.
    中心裁剪，取整方式与 torchvision CenterCrop 一致。"""
    h, w = array.shape[-2:]
    if h < size or w < size:
        raise ValueError(f'Cannot crop {array.shape} to {size}')
    top, left = int(round((h - size) / 2)), int(round((w - size) / 2))
    return array[..., top:top + size, left:left + size]


def prepare(args):
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'COMPLETE.json').exists():
        print('Already complete:', dest, flush=True)
        return

    # Read IBTrACS, keep 6-hourly records with valid wind and pressure
    # 读取 IBTrACS，保留 6 小时间隔且风速、气压有效的记录
    columns = ['SID', 'SEASON', 'ISO_TIME', 'BASIN', 'USA_WIND', 'USA_PRES']
    frame = pd.read_csv(args.ibtracs, usecols=columns, dtype=str, keep_default_na=False)
    frame['year'] = pd.to_numeric(frame.SEASON, errors='coerce')
    frame['wind'] = pd.to_numeric(frame.USA_WIND, errors='coerce')
    frame['pres'] = pd.to_numeric(frame.USA_PRES, errors='coerce')
    frame = frame[frame.year.between(args.begin_year, args.end_year) & frame.wind.gt(0) & frame.pres.gt(0)]
    frame = frame[frame.ISO_TIME.str[11:].isin(SYNOPTIC) & frame.BASIN.isin(BASINS)]
    frame = frame.sort_values(['year', 'SID', 'ISO_TIME']).drop_duplicates(['year', 'SID', 'ISO_TIME'])

    # Build sliding windows: input_len history + output_len future, contiguous 6h steps
    # 构建滑动窗口：input_len 个历史 + output_len 个未来，时间连续（6 小时）
    records, windows, labels, years, basins, sids, initial_times = [], [], [], [], [], [], []
    exists, root = {}, Path(args.data_dir)
    for (year, sid), group in frame.groupby(['year', 'SID'], sort=True):
        group = group.sort_values('ISO_TIME')
        rows = list(group.itertuples())
        time_ns = pd.to_datetime(group.ISO_TIME).astype('int64').to_numpy()
        for start in range(len(rows) - args.input_len - args.output_len + 1):
            end = start + args.input_len + args.output_len
            if not np.all(np.diff(time_ns[start:end]) == 6 * 3600 * 10 ** 9):
                continue
            inp = rows[start:start + args.input_len]
            paths = [root / str(int(year)) / sid / r.ISO_TIME.replace(':', '_') for r in inp]
            for path in paths:
                if path not in exists:
                    exists[path] = (path / 'GRIDSAT_data.npy').is_file() and (path / 'ERA5_data.npy').is_file()
            if not all(exists[path] for path in paths):
                continue
            rel = [str(p.relative_to(root)).replace('\\', '/') for p in paths]
            records.extend(rel)
            windows.append(rel)
            labels.append([[r.wind, r.pres] for r in rows[start:end]])
            years.append(int(year))
            sids.append(sid)
            basins.append(str(rows[start].BASIN).strip() or 'NA')
            initial_times.append(inp[-1].ISO_TIME)
    records = sorted(set(records))
    if not records:
        raise ValueError('No valid historical input frames')
    record_idx = {path: i for i, path in enumerate(records)}
    indices = np.array([[record_idx[p] for p in window] for window in windows], dtype=np.int64)
    metadata = dict(records=records, years=years, basins=basins, sids=sids, initial_times=initial_times,
                    source=str(root), ibtracs=str(Path(args.ibtracs)),
                    input_len=args.input_len, output_len=args.output_len)
    (dest / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    np.save(dest / 'indices.npy', indices)
    np.save(dest / 'labels.npy', np.array(labels, dtype=np.float32))

    # Pack unique frames into two large arrays
    # 将去重后的帧打包为两个大数组
    print(f'Packing {len(records)} frames / {len(windows)} windows', flush=True)
    ir = np.lib.format.open_memmap(dest / 'ir.npy', mode='w+', dtype='float32', shape=(len(records), 1, 140, 140))
    era5 = np.lib.format.open_memmap(dest / 'era5.npy', mode='w+', dtype='float32', shape=(len(records), 69, 40, 40))
    nan_rate = np.zeros(len(records), dtype=np.float32)
    good_era5 = np.ones(len(records), dtype=bool)

    def load(item):
        i, rel = item
        path = root / rel
        sat, atm = np.load(path / 'GRIDSAT_data.npy'), np.load(path / 'ERA5_data.npy')
        if sat.shape != (3, 286, 286) or atm.shape != (69, 80, 80):
            raise ValueError(f'Unexpected TC_datasets shape at {path}: {sat.shape}, {atm.shape}')
        sat_crop = np.asarray(crop(sat[:1], 140), dtype=np.float32).copy()  # IR channel only / 仅红外通道
        atm_crop = np.asarray(crop(atm, 40), dtype=np.float32).copy()
        return i, sat_crop, atm_crop, float((~np.isfinite(sat[0])).mean())

    start_time, chunk = time.time(), args.workers * 4
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Submit in chunks to bound memory use / 分块提交以限制内存占用
        for chunk_start in range(0, len(records), chunk):
            items = [(i, records[i]) for i in range(chunk_start, min(chunk_start + chunk, len(records)))]
            for i, sat, atm, missing in pool.map(load, items):
                ir[i], era5[i], nan_rate[i], good_era5[i] = sat, atm, missing, np.isfinite(atm).all()
            if chunk_start % 512 < chunk:
                print(f'{chunk_start + len(items)}/{len(records)} frames, {time.time() - start_time:.1f}s', flush=True)
    ir.flush()
    era5.flush()
    np.save(dest / 'ir_nan_rate.npy', nan_rate)
    np.save(dest / 'era5_finite.npy', good_era5)
    complete = dict(frames=len(records), windows=len(windows), seconds=time.time() - start_time,
                    input_len=args.input_len, output_len=args.output_len)
    (dest / 'COMPLETE.json').write_text(json.dumps(complete, indent=2), encoding='utf-8')
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--ibtracs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--begin-year', type=int, default=1980)
    p.add_argument('--end-year', type=int, default=2020)
    p.add_argument('--input-len', type=int, default=4)
    p.add_argument('--output-len', type=int, default=4)
    args = p.parse_args()
    prepare(args)
    # Rebuild the original sample eligibility / 重建原始样本筛选
    if not (Path(args.output) / 'legacy_eligible.npy').is_file():
        subprocess.run([sys.executable, str(Path(__file__).with_name('prepare_legacy_cohort.py')),
                        '--cache', args.output, '--workers', str(args.workers)], check=True)
