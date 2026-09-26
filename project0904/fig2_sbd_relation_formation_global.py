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
        frames.append(data[['cot_rfov', 'ret_albedo', 'cer_ret_mean', 'solar_zenith', 'ocean', 'season']])
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


def prepare_data(data):
    data = data.copy()
    # SBD-Reproduce (fixed SZA), Real SZA: default (gasdcp/aoddcp/sfcdcp) vis LUT
    data['visible'] = calculate_sbdart(data, 'gasdcp_aoddcp_sfcdcp_vis', 0)
    # extra reference: SBDART visible at the diffuse SZA theta = arccos(1/sqrt(3))
    data['visible_mu13'] = calculate_sbdart(
        data, 'gasdcp_aoddcp_sfcdcp_vis', np.degrees(np.arccos(3 ** (-0.5)))
    )
    data['sza'] = calculate_sbdart(data, 'gasdcp_aoddcp_sfcdcp_vis', 'per_point')
    data['shortwave'] = calculate_sbdart(data, 'gasdcp_aoddcp_sfcdcp_sw', 'per_point')
    data['gas'] = calculate_sbdart(data, 'gascp_aoddcp_sfcdcp_sw', 'per_point')
    data['aod'] = calculate_sbdart(data, 'gascp_aodcp_sfcdcp_sw', 'per_point')
    data['surface'] = calculate_sbdart(data, 'gascp_aodcp_sfccp_sw', 'per_point')
    return data


def draw_global(ax, data, curves):
    """Plot the requested subset of curves (order: analy_mu13, analy,
    visible_mu13, visible, sza, shortwave, gas, aod, surface)."""
    edges = np.geomspace(MIN_COT, 76, 17)
    cot_fit = np.geomspace(MIN_COT, 76, 200)

    if 'analy_mu13' in curves:
        ax.plot(cot_fit, cot_to_albedo(cot_fit, 'analy', miu=3 ** (-0.5)),
                color='k', lw=1.8, label=r'Analytical (54.74°): $k$=1.00')
    if 'visible_mu13' in curves:
        fit_and_plot(ax, data, 'visible_mu13', r'SBD-Reproduce (54.74°)', '0.5', edges)
    if 'analy' in curves:
        ax.plot(cot_fit, cot_to_albedo(cot_fit, 'analy', miu=1),
                color='k', ls='--', lw=1.8, label=r'Analytical (0°): $k$=1.00')
    if 'visible' in curves:
        fit_and_plot(ax, data, 'visible', r'SBD-Reproduce (0°)', COLORS['visible'],
                     edges, linestyle='--')
    if 'sza' in curves:
        fit_and_plot(ax, data, 'sza', r'Real SZA$_{\mathrm{1030}}$', COLORS['sza'], edges)
    if 'shortwave' in curves:
        fit_and_plot(ax, data, 'shortwave', 'Shortwave', COLORS['shortwave'], edges)
    if 'gas' in curves:
        fit_and_plot(ax, data, 'gas', 'Real Gas', COLORS['gas'], edges)
    if 'aod' in curves:
        fit_and_plot(ax, data, 'aod', 'Real AOD', COLORS['aod'], edges)
    if 'surface' in curves:
        fit_and_plot(ax, data, 'surface', r'Real $A_{\mathrm{sfc}}$', COLORS['surface'],
                     edges, linestyle=':', linewidth=2.5)

    ax.set(
        xlim=(0, 60), ylim=(0.1, 0.95), xlabel='COT', ylabel=r'$A_{\mathrm{c}}$'
    )
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.grid(alpha=.25)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=.85)


def main():
    data = prepare_data(load_rfov())
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8, 4.1))
    # lines 1-5 in panel a; lines 5-8 in panel b (line 5 = sza in both)
    draw_global(
        ax_a, data,
        ('analy_mu13', 'analy', 'visible_mu13', 'visible', 'sza'),
    )
    draw_global(
        ax_b, data,
        ('sza', 'shortwave', 'gas', 'aod', 'surface'),
    )

    ax_a.text(-0.03, 1.02, format_panel_tag(0, 'science'),
              transform=ax_a.transAxes, fontsize=14, va='bottom', ha='left')
    ax_b.text(-0.03, 1.02, format_panel_tag(1, 'science'),
              transform=ax_b.transAxes, fontsize=14, va='bottom', ha='left')

    fig.tight_layout()
    output_path = BASE_DIR / 'figs' / 'fig2_sbd_relation_formation_global.png'
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')
    print(f'Global RFOV rows: {len(data)}')


if __name__ == '__main__':
    main()
