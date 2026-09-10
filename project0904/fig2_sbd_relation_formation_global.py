import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import (
    cot_to_albedo, cot_to_x, albedo_to_y, mc_fit,
)

BASE_DIR = Path(__file__).resolve().parent
RFOV_DIR = BASE_DIR / 'RFOV_product' / 'ocean_season'
MIN_COT = 2.5
MIN_GROUP_SIZE = 5

COLORS = {
    'visible': '#606581',
    'shortwave': '#F354F3',
    'surface': '#31B704',
    'gas': '#16a085',
    'sza': '#574cff',
}


def add_rfov_cot(data):
    bins = []
    pattern = re.compile(r'^cot_freq_(\d+)_(\d+)$')
    for column in data.columns:
        match = pattern.match(column)
        if match:
            lower, upper = map(int, match.groups())
            bins.append((lower, upper, column))
    bins.sort()
    columns = [column for _, _, column in bins]
    midpoints = np.array([(lower + upper) / 2 for lower, upper, _ in bins])
    weights = data[columns].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0).to_numpy(float)
    total = weights.sum(axis=1)
    mean_log_cot = np.divide(weights @ np.log(midpoints), total, out=np.full(len(data), np.nan), where=total > 0)
    result = data.copy()
    result['cot_rfov'] = np.exp(mean_log_cot)
    return result


def load_rfov():
    frames = []
    for path in sorted(RFOV_DIR.glob('*.csv')):
        ocean, season = path.stem.rsplit('_', 1)
        data = add_rfov_cot(pd.read_csv(path))
        data['ret_albedo'] = pd.to_numeric(data['ret_albedo'], errors='coerce')
        data['ocean'], data['season'] = ocean, season
        frames.append(data[['cot_rfov', 'ret_albedo', 'solar_zenith', 'ocean', 'season']])
    data = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    return data[(data['cot_rfov'] >= MIN_COT) & data['ret_albedo'].between(0, 1)].reset_index(drop=True).copy()


def bin_relation(data, cot_col, albedo_col, edges):
    labels = pd.cut(data[cot_col], edges, labels=False, include_lowest=True)
    cot_bins, albedo_bins, albedo_std = [], [], []
    for index in range(len(edges) - 1):
        subset = data[labels == index]
        if len(subset) < MIN_GROUP_SIZE:
            continue
        cot_bins.append(subset[cot_col].mean())
        albedo_bins.append(subset[albedo_col].mean())
        albedo_std.append(subset[albedo_col].std())
    return np.asarray(cot_bins), np.asarray(albedo_bins), np.asarray(albedo_std)


def calculate_sbdart(data, folder, sza_mode):
    result = np.full(len(data), np.nan)
    for (ocean, season), indices in data.groupby(['ocean', 'season']).groups.items():
        rows = data.loc[indices]
        if sza_mode == 'fixed':
            sza = 54.74
        else:
            sza = rows['solar_zenith'].to_numpy()
        result[indices] = cot_to_albedo(
            rows['cot_rfov'].to_numpy(), 'sbdart', sza=sza,
            table_folder=folder, ocean=ocean, season=season,
        )
    return result


def fit_and_plot(ax, data, albedo_col, label, color, edges, marker=True, linestyle='-'):
    cot_bins, albedo_bins, albedo_std = bin_relation(data, 'cot_rfov', albedo_col, edges)
    if len(cot_bins) < 3:
        return
    k, b, _, _ = mc_fit(
        cot_bins, albedo_bins,
        cot_std=0.0, albedo_std=0.03, n_mc=300, bootstrap=True,
    )
    cot_fit = np.geomspace(MIN_COT, 76, 200)
    fit_albedo = 1 / (1 + np.exp(-(k * cot_to_x(cot_fit) + b)))
    ax.plot(
        cot_fit, fit_albedo, color=color, lw=1.8, ls=linestyle,
        label=rf'{label}: $k$={k:.2f}', alpha=.9,
    )
    if marker:
        ax.errorbar(
            cot_bins, albedo_bins, yerr=albedo_std,
            color=color, fmt='o', lw=0.8, ms=3, capsize=2.5, alpha=.9,
        )


def draw_global(ax, data):
    edges = np.geomspace(MIN_COT, 76, 17)
    data = data.copy()
    data['visible'] = calculate_sbdart(data, 'dcp_0p4to0p7', 'fixed')
    data['shortwave'] = calculate_sbdart(data, 'dcp', 'fixed')
    data['surface'] = calculate_sbdart(data, 'gasdcp_surcp', 'fixed')
    data['gas'] = calculate_sbdart(data, 'cp', 'fixed')
    data['sza'] = calculate_sbdart(data, 'cp', 'per_point')

    cot_fit = np.geomspace(MIN_COT, 76, 200)
    lh74_albedo = cot_to_albedo(cot_fit, 'l74')
    ax.plot(cot_fit, lh74_albedo, color='#222222', lw=1.8,
            label=r'LH74: $k$=1.00')

    fit_and_plot(ax, data, 'visible', 'SBDART Reproduce', COLORS['visible'], edges)
    fit_and_plot(ax, data, 'shortwave', 'Shortwave', COLORS['shortwave'], edges)
    fit_and_plot(ax, data, 'surface', r'+ Real $A_{\mathrm{sfc}}$', COLORS['surface'], edges, linestyle='--')
    fit_and_plot(ax, data, 'gas', '+ Real Gas', COLORS['gas'], edges)
    fit_and_plot(ax, data, 'sza', r'+ SZA$_{\mathrm{1030}}$', COLORS['sza'], edges)

    ax.set(
        xlim=(0, 60), ylim=(0.05, 0.95), xlabel='COT', ylabel=r'$A_{\mathrm{c}}$'
    )
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.grid(alpha=.25)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=.85)


def main():
    data = load_rfov()
    fig, ax = plt.subplots(figsize=(5, 4.1))
    draw_global(ax, data)
    fig.tight_layout()
    output_path = BASE_DIR / 'figs' / 'fig2_sbd_relation_formation_global.png'
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')
    print(f'Global RFOV rows: {len(data)}')


if __name__ == '__main__':
    main()
