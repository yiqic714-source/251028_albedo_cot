# -*- coding: utf-8 -*-
"""Ocean-level COT-albedo relationships with M14 function points."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import (
    albedo_to_y, cot_k_b_to_albedo, cot_to_albedo, cot_to_x, mc_fit, oceans,
    format_panel_tag,
)

BASE_DIR = Path(__file__).resolve().parent
L3_DIR = BASE_DIR / 'L3_product'
RFOV_DIR = BASE_DIR / 'RFOV_product' / 'ocean_season'
FIG_DIR = BASE_DIR / 'figs'
OUTPUT_PATH = FIG_DIR / 'fig1_fittings_ocean.png'
LINEAR_OUTPUT_PATH = FIG_DIR / 'figsupp_fittings_ocean_linear.png'
SENSITIVITY_CSV_PATH = BASE_DIR / 'processed_data' / 'sensitivity_albedo_vs_cot_ocean.csv'
MIN_COT = 2.5
MIN_CF = 0.1
COT_EDGES = np.geomspace(MIN_COT, 76, 17)
COT_FIT = np.geomspace(MIN_COT, 60, 300)

LH74_COLOR = '#222222'
SBD_COLOR = '#574cff'
RFOV_COLOR = '#00bfff'
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


def logit_yerr(albedo, std):
    albedo = np.asarray(albedo, dtype=float)
    std = np.asarray(std, dtype=float)
    center = albedo_to_y(np.clip(albedo, 1e-6, 1 - 1e-6))
    low = albedo_to_y(np.clip(albedo - std, 1e-6, 1 - 1e-6))
    high = albedo_to_y(np.clip(albedo + std, 1e-6, 1 - 1e-6))
    return np.vstack([center - low, high - center])


def draw_ocean(ax, ocean, l3_data, rfov_data, linear=False):
    l3 = l3_data[l3_data['ocean'] == ocean]
    rfov = add_sbdart_albedo(rfov_data[rfov_data['ocean'] == ocean])
    plotted = {}
    k_m14 = b_m14 = k_m14_unc = lnb_m14_unc = np.nan

    # LH74 theoretical relation.
    lh74 = cot_to_albedo(COT_FIT, 'quadrature', sza=54.74)
    plotted['LH74'] = (LH74_COLOR, r'LH74: $k$=1.00')
    if linear:
        ax.plot(cot_to_x(COT_FIT), albedo_to_y(lh74), color=LH74_COLOR, lw=1.5)
    else:
        ax.plot(COT_FIT, lh74, color=LH74_COLOR, lw=1.5)

    if len(rfov) >= 5:
        rfov_cot, rfov_albedo, rfov_std = bin_data(
            rfov, 'cot_rfov', 'ret_albedo'
        )
        k_rfov, b_rfov = fit_line(rfov_cot, rfov_albedo, 0.10, 0.20)
        plotted['RFOV'] = (RFOV_COLOR, rf'RFOV: $k$={k_rfov:.2f}')
        rfov_fit = cot_k_b_to_albedo(COT_FIT, k_rfov, np.exp(b_rfov))
        if linear:
            ax.errorbar(cot_to_x(rfov_cot), albedo_to_y(rfov_albedo),
                        yerr=logit_yerr(rfov_albedo, rfov_std), color=RFOV_COLOR,
                        fmt='*', lw=1, ms=2.4, capsize=2, capthick=0.6)
            ax.plot(cot_to_x(COT_FIT), albedo_to_y(rfov_fit), color=RFOV_COLOR, lw=1.5)
        else:
            ax.errorbar(rfov_cot, rfov_albedo, yerr=rfov_std, color=RFOV_COLOR,
                        fmt='*', lw=1, ms=2.4, capsize=2, capthick=0.6)
            ax.plot(COT_FIT, rfov_fit, color=RFOV_COLOR, lw=1.5)

        sbd_cot, sbd_albedo, sbd_std = bin_data(
            rfov, 'cot_rfov', 'sbd_albedo'
        )
        k_sbd, b_sbd = fit_line(sbd_cot, sbd_albedo, 0.0, 0.03)
        plotted['SBDART'] = (SBD_COLOR, rf'SBDART: $k$={k_sbd:.2f}')
        sbd_fit = cot_k_b_to_albedo(COT_FIT, k_sbd, np.exp(b_sbd))
        if linear:
            ax.errorbar(cot_to_x(sbd_cot), albedo_to_y(sbd_albedo),
                        yerr=logit_yerr(sbd_albedo, sbd_std), color=SBD_COLOR,
                        fmt='o', lw=1, ms=2.4, capsize=2, capthick=0.6)
            ax.plot(cot_to_x(COT_FIT), albedo_to_y(sbd_fit), color=SBD_COLOR, lw=1.5)
        else:
            ax.errorbar(sbd_cot, sbd_albedo, yerr=sbd_std, color=SBD_COLOR,
                        fmt='o', lw=1, ms=2.4, capsize=2, capthick=0.6)
            ax.plot(COT_FIT, sbd_fit, color=SBD_COLOR, lw=1.5)

    if len(l3) >= 5:
        grid_cot, grid_albedo, grid_std = bin_data(l3, 'cot', 'albedo')
        k_grid, b_grid = fit_line(grid_cot, grid_albedo, 0.10, 0.20)
        plotted['Grid'] = (GRID_COLOR, rf'Grid: $k$={k_grid:.2f}')
        grid_fit = cot_k_b_to_albedo(COT_FIT, k_grid, np.exp(b_grid))
        if linear:
            ax.errorbar(cot_to_x(grid_cot), albedo_to_y(grid_albedo),
                        yerr=logit_yerr(grid_albedo, grid_std), color=GRID_COLOR,
                        fmt='s', lw=1, ms=2.4, capsize=2, capthick=0.6)
            ax.plot(cot_to_x(COT_FIT), albedo_to_y(grid_fit), color=GRID_COLOR, lw=1.5)
        else:
            ax.errorbar(grid_cot, grid_albedo, yerr=grid_std, color=GRID_COLOR,
                        fmt='s', lw=1, ms=2.4, capsize=2, capthick=0.6)
            ax.plot(COT_FIT, grid_fit, color=GRID_COLOR, lw=1.5)

    # M14 uses the same Grid rows, binning, error bars, and fit workflow.
    if len(l3) >= 5:
        a3, a4, a6 = M14_PARAMS[ocean]
        m14_data = l3.copy()
        m14_data['m14_albedo'] = (
            a3 + a4 * m14_data['cf_ceres'].to_numpy() *
            m14_data['cot'].to_numpy()
        ) ** a6
        m14_cot, m14_albedo, m14_std = bin_data(
            m14_data, 'cot', 'm14_albedo'
        )
        k_m14, b_m14, k_m14_unc, lnb_m14_unc = mc_fit(
            m14_cot, m14_albedo,
            cot_std=0.10, albedo_std=0.20,
            n_mc=300, bootstrap=True,
        )
        plotted['M14'] = (M14_COLOR, rf'M14: $k$={k_m14:.2f}')
        m14_fit = cot_k_b_to_albedo(COT_FIT, k_m14, np.exp(b_m14))
        if linear:
            ax.errorbar(cot_to_x(m14_cot), albedo_to_y(m14_albedo),
                        yerr=logit_yerr(m14_albedo, m14_std), color=M14_COLOR,
                        fmt='D', lw=1, ms=2.4, capsize=2, capthick=0.6)
            ax.plot(cot_to_x(COT_FIT), albedo_to_y(m14_fit), color=M14_COLOR, lw=1.5)
        else:
            ax.errorbar(m14_cot, m14_albedo, yerr=m14_std, color=M14_COLOR,
                        fmt='D', lw=1, ms=2.4, capsize=2, capthick=0.6)
            ax.plot(COT_FIT, m14_fit, color=M14_COLOR, lw=1.5)

    if linear:
        ax.set(title=ocean)
    else:
        ax.set(xlim=(0, 60), ylim=(0, 0.9), title=ocean)
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
    return {
        'Ocean': ocean,
        'k_m14': k_m14,
        'lnb_m14': b_m14,
        'k_m14_unc': k_m14_unc,
        'lnb_m14_unc': lnb_m14_unc,
    }


def make_figure(l3_data, rfov_data, linear=False):
    layout = [['NPO', 'NAO', None], ['TPO', 'TAO', 'TIO'], ['SPO', 'SAO', 'SIO']]
    fig, axes = plt.subplots(3, 3, figsize=(9, 8), sharex=True, sharey=True)
    records = []
    for row, ocean_row in enumerate(layout):
        for column, ocean in enumerate(ocean_row):
            ax = axes[row, column]
            if ocean is None:
                ax.axis('off')
                continue
            record = draw_ocean(ax, ocean, l3_data, rfov_data, linear=linear)
            records.append(record)
            panel_index = row * 3 + column
            ax.text(-0.03, 1.01, format_panel_tag(panel_index, 'science'),
                    transform=ax.transAxes, fontsize=12, va='bottom', ha='left')
            if row == 2:
                ax.set_xlabel(r'$\ln(\mathrm{COT})$' if linear else 'COT', fontsize=11)
            if column == 0:
                ax.set_ylabel(r'$\ln[A_{\mathrm{c}}/(1-A_{\mathrm{c}})]$' if linear else r'$A_{\mathrm{c}}$', fontsize=11)
    fig.tight_layout()
    output_path = LINEAR_OUTPUT_PATH if linear else OUTPUT_PATH
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return records, output_path


def main():
    l3_data = load_l3_data()
    rfov_data = load_rfov_data()
    records, output_path = make_figure(l3_data, rfov_data)
    FIG_DIR.mkdir(exist_ok=True)
    SENSITIVITY_CSV_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(records).sort_values('Ocean').to_csv(SENSITIVITY_CSV_PATH, index=False)
    _, linear_output_path = make_figure(l3_data, rfov_data, linear=True)
    print(f'Saved: {output_path}')
    print(f'Saved: {linear_output_path}')


if __name__ == '__main__':
    main()
