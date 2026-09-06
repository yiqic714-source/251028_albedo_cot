from pathlib import Path

import pandas as pd

from util_ocean_season_division import split_by_ocean_season


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / 'uniform_fov_product'
OUTPUT_DIR = INPUT_DIR / 'ocean_season'


def main():
    csv_files = sorted(INPUT_DIR.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {INPUT_DIR}')

    frames = []
    for csv_file in csv_files:
        print(f'Reading {csv_file.name}')
        frames.append(pd.read_csv(csv_file))

    data = pd.concat(frames, ignore_index=True)
    required_columns = {'time', 'lat', 'lon'}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f'Missing required columns: {sorted(missing_columns)}')

    data['time'] = pd.to_datetime(data['time'], format='mixed')
    print(f'Total rows: {len(data)}')
    split_by_ocean_season(data, OUTPUT_DIR, time_col='time')
    print(f'Output directory: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
