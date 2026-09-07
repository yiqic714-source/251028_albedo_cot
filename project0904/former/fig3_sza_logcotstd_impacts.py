import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import (
    albedo_to_y, cot_to_albedo, cot_to_x, mc_fit, oceans, season_dict,
    format_panel_tag,
)

BASE_DIR = Path(__file__).resolve().parent
FOV_INPUT_DIR = BASE_DIR / 'RFOV_product' / 'ocean_season'
L3_INPUT_DIR = BASE_DIR / 'L3_product'
OUTPUT_PATH = BASE_DIR / 'figs' / 'fig3_sza_lncotstd_impacts.png'
MIN_COT = 2.5
MIN_CF = 0.1
MIN_GROUP_SIZE = 500
SZA_EDGES = np.arange(0, 101, 5)
STDEV_EDGES = np.arange(0, 1 + 1 / 10, 1 / 10)


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
    return data[mask].dropna(
        subset=['cot', 'albedo', 'logcot_std', 'sza']
    ).copy()


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

def add_l3_sbdart_albedo(data):
    result = data.copy()
    result['sbd_albedo'] = np.nan
    for (ocean, season), indices in result.groupby(['ocean', 'season']).groups.items():
        rows = result.loc[indices]
        result.loc[indices, 'sbd_albedo'] = cot_to_albedo(
            rows['cot'].to_numpy(), 'sbdart',
            sza=rows['sza'].to_numpy(), table_folder='cp',
            ocean=ocean, season=season,
        )
    return result.dropna(subset=['sbd_albedo'])


def grouped_k(data, x_edges, y_edges, x_col, y_col, cot_col, albedo_col):
    x_bin = pd.cut(data[x_col], x_edges, labels=False, include_lowest=True)
    y_bin = pd.cut(data[y_col], y_edges, labels=False, include_lowest=True)
    x_values, y_values, k_values = [], [], []
    for y_index in range(len(y_edges) - 1):
        for x_index in range(len(x_edges) - 1):
            subset = data[(x_bin == x_index) & (y_bin == y_index)]
            if len(subset) < MIN_GROUP_SIZE:
                continue
            k, _, _, _ = mc_fit(
                subset[cot_col].to_numpy(),
                subset[albedo_col].to_numpy(),
                cot_std=0.0,
                albedo_std=0.0,
                n_mc=300,
                bootstrap=True,
                calculate_uncertainty=False,
            )
            if np.isfinite(k):
                x_values.append((x_edges[x_index] + x_edges[x_index + 1]) / 2)
                y_values.append((y_edges[y_index] + y_edges[y_index + 1]) / 2)
                k_values.append(k)
    return np.asarray(x_values), np.asarray(y_values), np.asarray(k_values)


def ratio_grouped_k(raw_panel, fitted_panel):
    raw_x, raw_y, raw_k = raw_panel
    fitted_x, fitted_y, fitted_k = fitted_panel
    fitted = {
        (round(x, 8), round(y, 8)): k
        for x, y, k in zip(fitted_x, fitted_y, fitted_k)
    }
    x_values, y_values, differences = [], [], []
    for x, y, k in zip(raw_x, raw_y, raw_k):
        fitted_k_value = fitted.get((round(x, 8), round(y, 8)))
        if fitted_k_value is not None and np.isfinite(fitted_k_value):
            x_values.append(x)
            y_values.append(y)
            differences.append(k - fitted_k_value)
    return np.asarray(x_values), np.asarray(y_values), np.asarray(differences)


def sza_range_label(sza):
    index = np.searchsorted(SZA_EDGES, sza, side='right') - 1
    index = np.clip(index, 0, len(SZA_EDGES) - 2)
    lower = SZA_EDGES[index]
    upper = SZA_EDGES[index + 1]
    return f'{lower:g}°<SZA≤{upper:g}°'


def sza_color(sza):
    index = np.searchsorted(SZA_EDGES, sza, side='right') - 1
    return plt.get_cmap('tab10')(index % 10)


def draw_delta_lines(ax, rfov_delta, l3_delta):
    handles = []
    for values, name, linestyle, marker in (
        (rfov_delta, 'RFOV', '-', 'o'),
        (l3_delta, 'Grid', '--', 's'),
    ):
        sza_values, x_values, delta_values = values
        for sza in np.unique(sza_values):
            mask = np.isclose(sza_values, sza)
            order = np.argsort(x_values[mask])
            color = sza_color(sza)
            label = f'{name}, {sza_range_label(sza)}'
            line, = ax.plot(
                x_values[mask][order], delta_values[mask][order],
                color=color, linestyle=linestyle, marker=marker,
                markersize=3.5, lw=1.2, alpha=0.8, label=label,
            )
            handles.append(line)
    ax.set(
        xlabel='Standard deviation of lg(COT)', ylabel=r'$\Delta k = k_{\mathrm{obs}}-k_{\mathrm{SBDART}}$',
        title='RFOV/Grid $\Delta k$ by SZA bin', xlim=(0, 0.5),
        # ylim=(-0.45, 0.0),
    )
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.grid(color='0.85', linewidth=0.5)
    ax.legend(handles=handles, fontsize=8.5, ncol=1, framealpha=0.85)


def draw_delta_fit_curves(ax, rfov_delta, l3_delta):
    handles = []
    rfov_sza, rfov_x, rfov_y = rfov_delta
    l3_sza, l3_x, l3_y = l3_delta
    for sza in np.unique(np.concatenate([rfov_sza, l3_sza])):
        rfov_mask = np.isclose(rfov_sza, sza)
        l3_mask = np.isclose(l3_sza, sza)
        color = sza_color(sza)

        x = np.concatenate([rfov_x[rfov_mask], l3_x[l3_mask]])
        y = np.concatenate([rfov_y[rfov_mask], l3_y[l3_mask]])
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2 or np.unique(x[valid]).size < 2:
            continue
        coefficients = np.polyfit(x[valid], y[valid], 2)
        x_fit = np.linspace(x[valid].min(), x[valid].max(), 100)
        line, = ax.plot(
            x_fit, np.polyval(coefficients, x_fit),
            color=color, lw=1.8, label=sza_range_label(sza),
        )
        handles.append(line)

    ax.set(
        xlabel='Standard deviation of lg(COT)', ylabel='',
        title='$\Delta k$ fits by SZA bin', xlim=(0, 0.5),
        # ylim=(-0.45, 0.0),
    )
    ax.xaxis.label.set_size(14)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.grid(color='0.85', linewidth=0.5)
    ax.legend(handles=handles, fontsize=8.5, ncol=1, framealpha=0.85)


def main():
    fov_df = add_sbdart_columns(load_fov_df())
    os_df = load_os_df()

    panel_a = grouped_k(
        fov_df, SZA_EDGES, STDEV_EDGES, 'solar_zenith', 'logcot_std',
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
    os_sbd_df = add_l3_sbdart_albedo(os_df)
    panel_d = grouped_k(
        os_sbd_df, SZA_EDGES, STDEV_EDGES, 'sza', 'logcot_std',
        'cot', 'sbd_albedo'
    )

    rfov_delta = ratio_grouped_k(panel_b, panel_a)
    l3_delta = ratio_grouped_k(panel_c, panel_d)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    draw_delta_lines(axes[0], rfov_delta, l3_delta)
    draw_delta_fit_curves(axes[1], rfov_delta, l3_delta)
    for index, ax in enumerate(axes):
        ax.text(
            -0.04, 1.02, format_panel_tag(index, 'science'),
            transform=ax.transAxes, fontsize=15,
            va='bottom', ha='left',
        )
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_PATH}')
    print(f'fov rows: {len(fov_df)}, os rows: {len(os_df)}')
    print(f'delta cells: {len(rfov_delta[2])}, {len(l3_delta[2])}')


if __name__ == '__main__':
    main()
