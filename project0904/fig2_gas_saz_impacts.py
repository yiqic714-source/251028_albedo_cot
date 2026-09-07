import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import cot_to_albedo, cot_to_x, albedo_to_y, mc_fit, oceans, season_dict

BASE_DIR = Path(__file__).resolve().parent
RFOV_DIR = BASE_DIR / 'RFOV_product' / 'ocean_season'
OUTPUT_PATH = BASE_DIR / 'figs' / 'fig3_gas_saz_impacts.png'
MIN_COT = 2.5
MIN_GROUP_SIZE = 5

COLORS = {
    'visible': '#606581',
    'shortwave': '#00bfff',
    'surface': '#F354F3',
    'gas': '#31B704',
    'sza': '#025D37',
    'coupled': '#574cff',
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


def fit_and_plot(ax, data, albedo_col, label, color, edges, marker=True):
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
        cot_fit, fit_albedo, color=color, lw=1.8, ls='-',
        label=rf'{label}: $k$={k:.2f}',
    )
    if marker:
        ax.errorbar(
            cot_bins, albedo_bins, yerr=albedo_std,
            color=color, fmt='o', ms=4.5, capsize=3, alpha=.65,
        )


def main():
    data = load_rfov()
    edges = np.geomspace(MIN_COT, 76, 17)

    data['visible'] = calculate_sbdart(data, 'dcp_0p4to0p7', 'fixed')
    data['shortwave'] = calculate_sbdart(data, 'dcp', 'fixed')
    data['surface'] = calculate_sbdart(data, 'gasdcp_surcp', 'fixed')
    data['gas'] = calculate_sbdart(data, 'cp', 'fixed') # 'surdcp_gascp'
    data['sza'] = calculate_sbdart(data, 'cp', 'per_point')
    data['coupled'] = calculate_sbdart(data, 'cp', 'per_point')

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    cot_fit = np.geomspace(MIN_COT, 76, 200)
    LH74_albedo = cot_to_albedo(cot_fit, 'quadrature', sza=54.74)
    ax.plot(cot_fit, LH74_albedo, color='#222222', lw=1.8, label=r'LH74: $k$=1.00')

    fit_and_plot(ax, data, 'visible', 'SBDART Reproduce', COLORS['visible'], edges)
    fit_and_plot(ax, data, 'shortwave', 'Shortwave', COLORS['shortwave'], edges)
    fit_and_plot(ax, data, 'surface', r'+ real $A_{\mathrm{sfc}}$', COLORS['surface'], edges)
    fit_and_plot(ax, data, 'gas', '+ real Gas', COLORS['gas'], edges)
    fit_and_plot(ax, data, 'sza', r'+ SZA$_{\mathrm{1030}}$', COLORS['sza'], edges)
    fit_and_plot(ax, data, 'coupled', 'Shortwave, All Coupled', COLORS['coupled'], edges)

    ax.set(xlim=(0, 60), ylim=(0, 1), xlabel='COT', ylabel=r'$A_{\mathrm{c}}$', title='SBDART gas and SZA impacts')
    ax.grid(alpha=.25)
    ax.legend(loc='lower right', fontsize=9, framealpha=.85)
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_PATH}')
    print(f'RFOV rows: {len(data)}')


if __name__ == '__main__':
    main()
