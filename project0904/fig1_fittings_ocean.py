# -*- coding: utf-8 -*-
"""Ocean-level COT-albedo relationships with M14 function points."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import cot_k_b_to_albedo, cot_to_albedo, cot_to_x, mc_fit, oceans

BASE_DIR = Path(__file__).resolve().parent
L3_DIR = BASE_DIR / 'L3_product'
RFOV_DIR = BASE_DIR / 'RFOV_product' / 'ocean_season'
FIG_DIR = BASE_DIR / 'figs'
OUTPUT_PATH = FIG_DIR / 'fig1_fittings_ocean.png'
MIN_COT = 2.5
MIN_CF = 0.1
COT_EDGES = np.geomspace(MIN_COT, 76, 17)
COT_FIT = np.geomspace(MIN_COT, 60, 300)

LH74_COLOR = '#222222'
SBD_COLOR = '#574cff'
RFOV_COLOR = '#16a085'
GRID_COLOR = '#f20d38'
M14_COLOR = '#ff852e'

M14_PARAMS = {
    'NPO': [0.00163, 0.0052, 0.337],
    'NAO': [0.00101, 0.0052, 0.325],
    'TPO': [0.00017, 0.0064, 0.405],
    'TAO': [0.00027, 0.0071, 0.423],
    'TIO': [0.00016, 0.0069, 0.425],
    'SPO': [0.00013, 0.0062, 0.342],
    'SAO': [0.00024, 0.0057, 0.333],
    'SIO': [0.00028, 0.0053, 0.324],
}


def bin_data(data, cot_column, albedo_column):
    labels = pd.cut(data[cot_column], COT_EDGES, labels=False, include_lowest=True)
    cot_values, albedo_values, albedo_stds = [], [], []
    for index in range(len(COT_EDGES) - 1):
        subset = data[labels == index]
        if len(subset) < 5:
            continue
        cot_values.append(subset[cot_column].mean())
        albedo_values.append(subset[albedo_column].mean())
        albedo_stds.append(subset[albedo_column].std())
    return (np.asarray(cot_values), np.asarray(albedo_values),
            np.asarray(albedo_stds))


def fit_line(cot, albedo, cot_std, albedo_std, calculate_uncertainty=True):
    if len(cot) < 3:
        return np.nan, np.nan
    return mc_fit(
        cot, albedo, cot_std=cot_std, albedo_std=albedo_std,
        n_mc=300, bootstrap=True,
        calculate_uncertainty=calculate_uncertainty,
    )[:2]


def load_l3_data():
    frames = []
    for path in sorted(L3_DIR.glob('*.csv')):
        parts = path.stem.rsplit('_', 1)
        if len(parts) != 2:
            continue
        data = pd.read_csv(path)
        data['ocean'] = parts[0]
        data['season'] = parts[1]
        frames.append(data)
    data = pd.concat(frames, ignore_index=True)
    data['albedo'] = ((data['sw_all'] - data['sw_clr'] * (1 - data['cf_ceres'])) /
                      data['cf_ceres'] / data['solar_incoming'])
    mask = (
        (data['cf_ceres'] > MIN_CF) &
        (data['cf_ret_tot'] > MIN_CF) &
        (data['cf_liq_ceres'] / data['cf_ceres'] > 0.99) &
        (data['cot'] > MIN_COT) &
        data['albedo'].between(0, 1) &
        (data['cttmin'] >= 270)
    )
    return data[mask].dropna(subset=['cot', 'albedo', 'sza']).copy()


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


def load_rfov_data():
    frames = []
    for path in sorted(RFOV_DIR.glob('*.csv')):
        ocean, season = path.stem.rsplit('_', 1)
        data = add_rfov_cot(pd.read_csv(path))
        data['ret_albedo'] = pd.to_numeric(data['ret_albedo'], errors='coerce')
        data['ocean'], data['season'] = ocean, season
        frames.append(data[['cot_rfov', 'ret_albedo', 'solar_zenith', 'ocean', 'season']])
    data = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    return data[(data['cot_rfov'] >= MIN_COT) & data['ret_albedo'].between(0, 1)].copy()


def add_sbdart_albedo(data):
    result = data.copy()
    result['sbd_albedo'] = np.nan
    for (ocean, season), indices in result.groupby(['ocean', 'season']).groups.items():
        rows = result.loc[indices]
        result.loc[indices, 'sbd_albedo'] = cot_to_albedo(
            rows['cot_rfov'].to_numpy(), 'sbdart',
            sza=rows['solar_zenith'].to_numpy(), table_folder='cp',
            ocean=ocean, season=season,
        )
    return result.dropna(subset=['sbd_albedo'])


def m14_values(ocean):
    a3, a4, a6 = M14_PARAMS[ocean]
    return (a3 + a4 * 0.5 * COT_FIT) ** a6


def draw_ocean(ax, ocean, l3_data, rfov_data):
    l3 = l3_data[l3_data['ocean'] == ocean]
    rfov = add_sbdart_albedo(rfov_data[rfov_data['ocean'] == ocean])
    plotted = {}

    # LH74 theoretical relation.
    lh74 = cot_to_albedo(COT_FIT, 'quadrature', sza=54.74)
    plotted['LH74'] = (LH74_COLOR, r'LH74: $k$=1.00')
    ax.plot(COT_FIT, lh74, color=LH74_COLOR, lw=1.5)

    if len(rfov) >= 5:
        rfov_cot, rfov_albedo, rfov_std = bin_data(
            rfov, 'cot_rfov', 'ret_albedo'
        )
        k_rfov, b_rfov = fit_line(rfov_cot, rfov_albedo, 0.10, 0.20)
        plotted['RFOV'] = (RFOV_COLOR, rf'RFOV: $k$={k_rfov:.2f}')
        ax.errorbar(
            rfov_cot, rfov_albedo, yerr=rfov_std, color=RFOV_COLOR,
            fmt='o', lw=1, ms=2.5, capsize=2, capthick=0.6,
        )
        ax.plot(
            COT_FIT, cot_k_b_to_albedo(COT_FIT, k_rfov, np.exp(b_rfov)),
            color=RFOV_COLOR, lw=1.5,
        )

        sbd_cot, sbd_albedo, sbd_std = bin_data(
            rfov, 'cot_rfov', 'sbd_albedo'
        )
        k_sbd, b_sbd = fit_line(sbd_cot, sbd_albedo, 0.0, 0.03)
        plotted['SBDART'] = (SBD_COLOR, rf'SBDART: $k$={k_sbd:.2f}')
        ax.errorbar(
            sbd_cot, sbd_albedo, yerr=sbd_std, color=SBD_COLOR,
            fmt='D', lw=1, ms=2.5, capsize=2, capthick=0.6,
        )
        ax.plot(
            COT_FIT, cot_k_b_to_albedo(COT_FIT, k_sbd, np.exp(b_sbd)),
            color=SBD_COLOR, lw=1.5,
        )

    if len(l3) >= 5:
        grid_cot, grid_albedo, grid_std = bin_data(l3, 'cot', 'albedo')
        k_grid, b_grid = fit_line(grid_cot, grid_albedo, 0.10, 0.20)
        plotted['Grid'] = (GRID_COLOR, rf'Grid: $k$={k_grid:.2f}')
        ax.errorbar(
            grid_cot, grid_albedo, yerr=grid_std, color=GRID_COLOR,
            fmt='s', lw=1, ms=2.5, capsize=2, capthick=0.6,
        )
        ax.plot(
            COT_FIT, cot_k_b_to_albedo(COT_FIT, k_grid, np.exp(b_grid)),
            color=GRID_COLOR, lw=1.5,
        )

    # M14 function points and its fitted solid line, without uncertainty sampling.
    m14_cot = np.linspace(MIN_COT, 60, 16)
    a3, a4, a6 = M14_PARAMS[ocean]
    m14_albedo = (a3 + a4 * 0.5 * m14_cot) ** a6
    k_m14, b_m14 = fit_line(
        m14_cot, m14_albedo, 0.0, 0.0,
        calculate_uncertainty=False,
    )
    plotted['M14'] = (M14_COLOR, rf'M14: $k$={k_m14:.2f}')
    ax.scatter(m14_cot, m14_albedo, color=M14_COLOR, s=13, marker='o', zorder=4)
    ax.plot(
        COT_FIT, cot_k_b_to_albedo(COT_FIT, k_m14, np.exp(b_m14)),
        color=M14_COLOR, lw=1.5,
    )

    ax.set(xlim=(0, 60), ylim=(0, 1), title=ocean)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    # Keep SBDART first, followed by RFOV, Grid, and M14.
    legend_order = ['LH74', 'SBDART', 'RFOV', 'Grid', 'M14']
    handles = [
        plt.Line2D([0], [0], color=plotted[name][0], lw=1.8,
                   label=plotted[name][1])
        for name in legend_order if name in plotted
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=6.5, framealpha=0.8)


def main():
    l3_data = load_l3_data()
    rfov_data = load_rfov_data()
    layout = [['NPO', 'NAO', None], ['TPO', 'TAO', 'TIO'], ['SPO', 'SAO', 'SIO']]
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
    for row, ocean_row in enumerate(layout):
        for column, ocean in enumerate(ocean_row):
            ax = axes[row, column]
            if ocean is None:
                ax.axis('off')
                continue
            draw_ocean(ax, ocean, l3_data, rfov_data)
            if row == 2:
                ax.set_xlabel('COT', fontsize=11)
            if column == 0:
                ax.set_ylabel(r'$A_c$', fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
