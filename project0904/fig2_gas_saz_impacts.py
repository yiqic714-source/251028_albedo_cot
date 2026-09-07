import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import (
    cot_to_albedo, cot_to_x, albedo_to_y, mc_fit, oceans, season_dict,
    format_panel_tag,
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
        cot_fit, fit_albedo, color=color, lw=1.5, ls=linestyle,
        label=rf'{label}: $k$={k:.2f}', alpha=.75,
    )
    if marker:
        ax.errorbar(
            cot_bins, albedo_bins, yerr=albedo_std,
            color=color, fmt='o', lw=0.5, ms=2.2, capsize=2, alpha=.75,
        )


def draw_ocean(ax, ocean, data):
    edges = np.geomspace(MIN_COT, 76, 17)
    data = data.copy()
    data['visible'] = calculate_sbdart(data, 'dcp_0p4to0p7', 'fixed')
    data['shortwave'] = calculate_sbdart(data, 'dcp', 'fixed')
    data['surface'] = calculate_sbdart(data, 'gasdcp_surcp', 'fixed')
    data['gas'] = calculate_sbdart(data, 'cp', 'fixed')
    data['sza'] = calculate_sbdart(data, 'cp', 'per_point')

    cot_fit = np.geomspace(MIN_COT, 76, 200)
    lh74_albedo = cot_to_albedo(cot_fit, 'quadrature', sza=54.74)
    ax.plot(cot_fit, lh74_albedo, color='#222222', lw=1.5,
            label=r'LH74: $k$=1.00')

    fit_and_plot(ax, data, 'visible', 'SBDART Reproduce', COLORS['visible'], edges)
    fit_and_plot(ax, data, 'shortwave', '→ Shortwave', COLORS['shortwave'], edges)
    fit_and_plot(ax, data, 'surface', r'+ real $A_{\mathrm{sfc}}$', COLORS['surface'], edges, linestyle='--')
    fit_and_plot(ax, data, 'gas', '+ real Gas', COLORS['gas'], edges)
    fit_and_plot(ax, data, 'sza', r'+ SZA$_{\mathrm{1030}}$', COLORS['sza'], edges)

    ax.set(
        xlim=(0, 60), ylim=(0, 1), xlabel='COT', ylabel=r'$A_{\mathrm{c}}$',
        title=ocean,
    )
    ax.grid(alpha=.25)
    ax.tick_params(labelsize=7)
    ax.legend(loc='lower right', fontsize=6.5, framealpha=.85)


def main():
    data = load_rfov()
    layout = [
        ['NPO', 'NAO', None],
        ['TPO', 'TAO', 'TIO'],
        ['SPO', 'SAO', 'SIO'],
    ]
    fig, axes = plt.subplots(
        3, 3, figsize=(9, 8), sharex=True, sharey=True,
        squeeze=False,
    )
    for row, ocean_row in enumerate(layout):
        for column, ocean in enumerate(ocean_row):
            ax = axes[row, column]
            if ocean is None:
                ax.axis('off')
                continue
            ocean_data = data[data['ocean'] == ocean].reset_index(drop=True)
            if len(ocean_data) < MIN_GROUP_SIZE:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8)
                continue
            draw_ocean(ax, ocean, ocean_data)
            panel_index = row * 3 + column
            ax.text(
                -0.03, 1.01, format_panel_tag(panel_index, 'science'),
                transform=ax.transAxes, fontsize=12,
                va='bottom', ha='left',
            )
            if row == 2:
                ax.set_xlabel('COT', fontsize=11)
            if column == 0:
                ax.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=11)

    fig.tight_layout()
    output_path = BASE_DIR / 'figs' / 'fig2_gas_saz_impacts.png'
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
