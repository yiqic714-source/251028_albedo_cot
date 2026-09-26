# -*- coding: utf-8 -*-
"""
figsupp_sbd_relation_formation_8oceans.py

Eight panels (one per ocean).  Each panel is the same as panel (b) of
fig2_sbd_relation_formation_global.py -- the five SBDART COT->albedo relations
(Real SZA, Shortwave, Real Gas, Real AOD, Real A_sfc) -- but computed
from a single ocean's RFOV footprints only.

Folders (3D (cot, sza, cer) SBDART LUTs):
    Real SZA   : gasdcp_aoddcp_sfcdcp_vis
    Shortwave  : gasdcp_aoddcp_sfcdcp_sw
    Real Gas   : gascp_aoddcp_sfcdcp_sw
    Real AOD   : gascp_aodcp_sfcdcp_sw
    Real A_sfc : gascp_aodcp_sfccp_sw
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import (
    cot_to_albedo, cot_to_x, mc_fit, format_panel_tag,
)

BASE_DIR = Path(__file__).resolve().parent
RFOV_DIR = BASE_DIR / 'RFOV_product' / 'ocean_season'
MIN_COT = 2.5
MIN_GROUP_SIZE = 5

# Same colours as fig2_sbd_relation_formation_global.py
COLORS = {
    'visible': '#606581',
    'sza': "#3AE102",
    'shortwave': "#157B59",
    'gas': "#FF02F2",
    'aod': "#BCBD22",
    'surface': '#574cff',
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
        frames.append(data[['cot_rfov', 'ret_albedo', 'cer_ret_mean',
                            'solar_zenith', 'ocean', 'season']])
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
        if isinstance(sza_mode, (int, float, np.number)):
            sza = sza_mode
        else:
            sza = rows['solar_zenith'].to_numpy()
        result[indices] = cot_to_albedo(
            rows['cot_rfov'].to_numpy(), 'sbdart', sza=sza,
            cer=rows['cer_ret_mean'].to_numpy(),
            table_folder=folder, ocean=ocean, season=season,
        )
    return result


def fit_and_plot(ax, data, albedo_col, label, color, edges, linestyle='-', linewidth=1.8):
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
        cot_fit, fit_albedo, color=color, lw=linewidth, ls=linestyle,
        label=rf'{label}: $k$={k:.2f}', alpha=.9,
    )


def draw_ocean(ax, ocean, data):
    """One panel == fig2 panel (b), from a single ocean's data."""
    edges = np.geomspace(MIN_COT, 76, 17)
    data = data.copy()
    data['sza'] = calculate_sbdart(data, 'gasdcp_aoddcp_sfcdcp_vis', 'per_point')
    data['shortwave'] = calculate_sbdart(data, 'gasdcp_aoddcp_sfcdcp_sw', 'per_point')
    data['gas'] = calculate_sbdart(data, 'gascp_aoddcp_sfcdcp_sw', 'per_point')
    data['aod'] = calculate_sbdart(data, 'gascp_aodcp_sfcdcp_sw', 'per_point')
    data['surface'] = calculate_sbdart(data, 'gascp_aodcp_sfccp_sw', 'per_point')

    fit_and_plot(ax, data, 'sza', r'Real SZA$_{\mathrm{1030}}$', COLORS['sza'], edges)
    fit_and_plot(ax, data, 'shortwave', 'Shortwave', COLORS['shortwave'], edges)
    fit_and_plot(ax, data, 'gas', 'Real Gas', COLORS['gas'], edges)
    fit_and_plot(ax, data, 'aod', 'Real AOD', COLORS['aod'], edges)
    fit_and_plot(ax, data, 'surface', r'Real $A_{\mathrm{sfc}}$', COLORS['surface'], edges,
                 linestyle=':', linewidth=2.5)

    ax.set(
        xlim=(0, 60), ylim=(0.1, 0.95), title=ocean,
    )
    ax.grid(alpha=.25)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=.85)


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
    output_path = BASE_DIR / 'figs' / 'figsupp_sbd_relation_formation_8oceans.png'
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()

