import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import (
    albedo_to_y, cot_to_albedo, cot_to_x, oceans, season_dict,
    format_panel_tag,
)

BASE_DIR = Path(__file__).resolve().parent
FOV_INPUT_DIR = BASE_DIR / 'RFOV_product' / 'ocean_season'
L3_INPUT_DIR = BASE_DIR / 'L3_product'
OUTPUT_PATH = BASE_DIR / 'figs' / 'fig3_sza_lncotstd_impacts.png'
MIN_COT = 2.5
MIN_CF = 0.1
MIN_GROUP_SIZE = 250
SZA_EDGES = np.arange(0, 95, 5)
STDEV_EDGES = np.arange(0, 1 + 1 / 30, 1 / 30)


def add_frequency_statistics(df):
    bins = []
    pattern = re.compile(r'^cot_freq_(\d+)_(\d+)$')
    for column in df.columns:
        match = pattern.match(column)
        if match:
            lower, upper = map(int, match.groups())
            bins.append((lower, upper, column))
    bins.sort()
    columns = [column for _, _, column in bins]
    midpoints = np.array([(lower + upper) / 2 for lower, upper, _ in bins])
    frequencies = df[columns].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0).to_numpy(float)
    weights = frequencies.sum(axis=1)
    log_midpoints = np.log(midpoints)
    mean = np.divide(frequencies @ log_midpoints, weights, out=np.full(len(df), np.nan), where=weights > 0)
    log10_midpoints = np.log10(midpoints)
    mean_log10 = np.divide(frequencies @ log10_midpoints, weights, out=np.full(len(df), np.nan), where=weights > 0)
    variance = np.divide(frequencies @ (log10_midpoints ** 2), weights, out=np.full(len(df), np.nan), where=weights > 0) - mean_log10 ** 2
    result = df.copy()
    result['cot_fov'] = np.exp(mean)
    result['logcot_std'] = np.sqrt(np.maximum(variance, 0))
    return result


def load_fov_df():
    frames = []
    for path in sorted(FOV_INPUT_DIR.glob('*.csv')):
        ocean, season = path.stem.rsplit('_', 1)
        data = add_frequency_statistics(pd.read_csv(path))
        data['ret_albedo'] = pd.to_numeric(data['ret_albedo'], errors='coerce')
        data['ocean'], data['season'] = ocean, season
        frames.append(data)
    if not frames:
        raise FileNotFoundError(f'No RFOV CSV files found in {FOV_INPUT_DIR}')
    data = pd.concat(frames, ignore_index=True).dropna(subset=['cot_fov', 'logcot_std', 'ret_albedo', 'solar_zenith'])
    return data.query('cot_fov >= @MIN_COT and ret_albedo >= 0 and ret_albedo <= 1').copy()


def load_os_df():
    frames = []
    for ocean in oceans:
        for season in season_dict:
            path = L3_INPUT_DIR / f'{ocean}_{season}.csv'
            if path.exists():
                data = pd.read_csv(path)
                data['ocean'], data['season'] = ocean, season
                frames.append(data)
    if not frames:
        raise FileNotFoundError(f'No L3 CSV files found in {L3_INPUT_DIR}')
    data = pd.concat(frames, ignore_index=True)
    data['albedo'] = ((data['sw_all'] - data['sw_clr'] * (1 - data['cf_ceres'])) / data['cf_ceres'] / data['solar_incoming'])
    mask = ((data['cf_ceres'] > MIN_CF) & (data['cf_ret_tot'] > MIN_CF) & (data['cf_liq_ceres'] / data['cf_ceres'] > .99) & (data['cot'] > MIN_COT) & data['albedo'].between(0, 1) & (data['cttmin'] >= 270))
    return data[mask].dropna(subset=['cot', 'albedo', 'logcot_std', 'sza']).copy()


def add_sbdart_columns(fov_df):
    result = fov_df.copy()
    result['sbd_cot'] = result['cot_fov']
    result['sbd_albedo'] = np.nan
    for (ocean, season), indices in result.groupby(['ocean', 'season']).groups.items():
        rows = result.loc[indices]
        result.loc[indices, 'sbd_albedo'] = cot_to_albedo(
            rows['sbd_cot'].to_numpy(), 'sbdart',
            sza=rows['solar_zenith'].to_numpy(), table_folder='cp',
            ocean=ocean, season=season
        )
    return result.dropna(subset=['sbd_cot', 'sbd_albedo'])

def fit_k(cot, albedo):
    cot, albedo = np.asarray(cot, float), np.asarray(albedo, float)
    mask = np.isfinite(cot) & np.isfinite(albedo) & (cot > 0) & (albedo > 0) & (albedo < 1)
    if mask.sum() < MIN_GROUP_SIZE:
        return np.nan
    return np.polyfit(cot_to_x(cot[mask]), albedo_to_y(albedo[mask]), 1)[0]


def grouped_k(data, x_edges, y_edges, x_col, y_col, cot_col, albedo_col):
    x_bin = pd.cut(data[x_col], x_edges, labels=False, include_lowest=True)
    y_bin = pd.cut(data[y_col], y_edges, labels=False, include_lowest=True)
    x_values, y_values, k_values = [], [], []
    for y_index in range(len(y_edges) - 1):
        for x_index in range(len(x_edges) - 1):
            subset = data[(x_bin == x_index) & (y_bin == y_index)]
            if len(subset) < MIN_GROUP_SIZE:
                continue
            k = fit_k(subset[cot_col], subset[albedo_col])
            if np.isfinite(k):
                x_values.append((x_edges[x_index] + x_edges[x_index + 1]) / 2)
                y_values.append((y_edges[y_index] + y_edges[y_index + 1]) / 2)
                k_values.append(k)
    return np.asarray(x_values), np.asarray(y_values), np.asarray(k_values)


def grouped_k_by_sza(data, x_edges, x_col, y_col, cot_col, albedo_col):
    x_bin = pd.cut(data[x_col], x_edges, labels=False, include_lowest=True)
    x_values, y_values, k_values = [], [], []
    for x_index in range(len(x_edges) - 1):
        subset = data[x_bin == x_index]
        if len(subset) < MIN_GROUP_SIZE:
            continue
        k = fit_k(subset[cot_col], subset[albedo_col])
        if np.isfinite(k):
            x_values.append((x_edges[x_index] + x_edges[x_index + 1]) / 2)
            y_values.append(subset[y_col].mean())
            k_values.append(k)
    return np.asarray(x_values), np.asarray(y_values), np.asarray(k_values)


def draw_k_scatter(ax, grouped, xlabel, ylabel, title, norm):
    x_values, y_values, k_values = grouped
    scatter = ax.scatter(
        x_values, y_values, c=k_values, cmap='viridis', norm=norm,
        s=28, marker='s', edgecolors='none',
    )
    ax.set(
        xlabel=xlabel, ylabel=ylabel, title=title,
        xlim=(SZA_EDGES[0], SZA_EDGES[-1]),
        ylim=(STDEV_EDGES[0], STDEV_EDGES[-1]),
    )
    ax.set_xticks(SZA_EDGES)
    ax.set_yticks(STDEV_EDGES[::3])
    ax.grid(color='0.85', linewidth=0.5)
    return scatter


def main():
    fov_df = add_sbdart_columns(load_fov_df())
    os_df = load_os_df()

    panel_a = grouped_k_by_sza(
        fov_df, SZA_EDGES, 'solar_zenith', 'logcot_std',
        'cot_fov', 'sbd_albedo'
    )
    panel_b = grouped_k(
        fov_df, SZA_EDGES, STDEV_EDGES, 'solar_zenith', 'logcot_std',
        'cot_fov', 'ret_albedo'
    )
    panel_c = grouped_k(
        os_df, SZA_EDGES, STDEV_EDGES, 'sza', 'logcot_std',
        'cot', 'albedo'
    )

    all_k = np.concatenate([
        values[2][np.isfinite(values[2])]
        for values in (panel_a, panel_b, panel_c)
    ])
    norm = plt.Normalize(vmin=all_k.min(), vmax=all_k.max())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    scatters = [
        draw_k_scatter(
            axes[0], panel_a, 'SZA (degree)', 'mean std(log10(COT))',
            'RFOV COT vs SBDART albedo', norm
        ),
        draw_k_scatter(
            axes[1], panel_b, 'SZA (degree)', 'std(log10(COT))',
            'RFOV COT vs RFOV albedo', norm
        ),
        draw_k_scatter(
            axes[2], panel_c, 'SZA (degree)', 'std(log10(COT))',
            'COT vs albedo', norm
        ),
    ]
    axes[0].text(
        -0.04, 1.02, format_panel_tag(0, 'science'),
        transform=axes[0].transAxes, fontsize=13,
        va='bottom', ha='left',
    )
    axes[1].text(
        -0.04, 1.02, format_panel_tag(1, 'science'),
        transform=axes[1].transAxes, fontsize=13,
        va='bottom', ha='left',
    )
    axes[2].text(
        -0.04, 1.02, format_panel_tag(2, 'science'),
        transform=axes[2].transAxes, fontsize=13,
        va='bottom', ha='left',
    )
    axes[0].tick_params(axis='y', labelleft=False)
    fig.colorbar(scatters[0], ax=axes, label='k')
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_PATH}')
    print(f'fov rows: {len(fov_df)}, os rows: {len(os_df)}')
    print(f'valid cells: {[len(values[2]) for values in (panel_a, panel_b, panel_c)]}')


if __name__ == '__main__':
    main()
