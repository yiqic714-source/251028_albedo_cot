import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from utils_fitting import (
    oceans, season_dict, cot_range, albedo_to_y,
    cot_to_x, mc_fit, format_panel_tag
)


BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIG_SAVE_PATH = f'{BASE_PATH}/figs/fig3_bias_attribution.png'
os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)

MIN_COT = 2.5
MIN_CF = 0.1

LINECOLOR = ['steelblue', 'orange', 'coral', 'red', 'purple']
LINESTYLE = ['-', '-', '--', ':', '-']
BOX_COLORS = ['red', 'blue']


def cot_to_albedo(cot, method, sza=None, table_folder='dcp'):
    cot = np.asarray(cot, dtype=float)

    if method == 'sbdart':
        file_path = (
            f'{BASE_PATH}/build_sbdart_lookup_table/'
            f'cot_sza_to_albedo_lookup_table_{table_folder}/'
            f'cot_sza_to_albedo_lookup_table_TPO_MAM.csv'
        )

        if not os.path.exists(file_path):
            return np.full(cot.shape, np.nan)

        df = pd.read_csv(file_path, index_col=0)
        sza_grid = np.array(df.index, dtype=float)
        cot_grid = np.array(df.columns, dtype=float)
        albedo_grid = df.values

        sza_mesh, cot_mesh = np.meshgrid(sza_grid, cot_grid, indexing='ij')
        points = np.column_stack([sza_mesh.ravel(), cot_mesh.ravel()])
        values = albedo_grid.ravel()

        valid = np.isfinite(values)

        cot_arr = np.atleast_1d(cot)
        if np.ndim(sza) == 0:
            sza_arr = np.full_like(cot_arr, sza, dtype=float)
        else:
            sza_arr = np.asarray(sza, dtype=float)

        target = np.column_stack([sza_arr, cot_arr])

        albedo = griddata(
            points[valid],
            values[valid],
            target,
            method='linear',
            fill_value=np.nan
        )

        return albedo.reshape(cot_arr.shape)

    if method == 'l74':
        g = 0.85
        b = 1 - g
        return b * cot / (1 + b * cot)

    if method == 'quadrature':
        g = 0.85
        mu = np.cos(np.radians(sza))
        b = np.sqrt(3) / 2 * (1 - g)
        return (
            b * cot + (1 / 2 - np.sqrt(3) / 2 * mu) * (1 - np.exp(-cot / mu))
        ) / (1 + b * cot)

    if method == 'eddington':
        g = 0.85
        mu = np.cos(np.radians(sza))
        return (
            (1 - g) * cot + (2 / 3 - mu) * (1 - np.exp(-cot / mu))
        ) / (4 / 3 + (1 - g) * cot)

    raise ValueError(f'Unsupported method: {method}')


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights)
    if not np.any(mask) or np.sum(weights[mask]) <= 0:
        return np.nan

    return np.sum(values[mask] * weights[mask]) / np.sum(weights[mask])


def calc_global_slope_from_raw(
    cot, albedo, season, x2,
    cot_std=0.0, albedo_std=0.0, n_mc=100
):
    cot = np.asarray(cot, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    season = np.asarray(season)

    valid = (
        np.isfinite(cot) &
        np.isfinite(albedo) &
        (cot > 0) &
        (albedo > 0) &
        (albedo < 1)
    )

    slopes, intercepts, weights = [], [], []

    for s in season_dict:
        mask = valid & (season == s)
        if np.sum(mask) < 5:
            continue

        k, b, _, _ = mc_fit(
            cot[mask],
            albedo[mask],
            cot_std=cot_std,
            albedo_std=albedo_std,
            n_mc=n_mc,
            bootstrap=True,
            random_seed=42
        )

        if np.isfinite(k):
            slopes.append(k)
            intercepts.append(b)
            weights.append(np.sum(mask))

    if not weights:
        return np.nan, np.nan, np.full_like(x2, np.nan, dtype=float)

    k_global = weighted_mean(slopes, weights)
    b_global = weighted_mean(intercepts, weights)

    return k_global, b_global, k_global * x2 + b_global


def split_data_by_percentile(df, col_name, n_bins=2):
    edges = np.unique(np.percentile(df[col_name].dropna(), np.linspace(0, 100, n_bins + 1)))
    labels = pd.cut(df[col_name], bins=edges, labels=False, include_lowest=True)
    return labels.values, edges


def prepare_dataframe(ocean, season_name):
    file_path = f'{BASE_PATH}/processed_data/merged_data/{ocean}_{season_name}.csv'
    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)
    df['season'] = season_name

    df['albedo'] = (
        (df['sw_all'] - df['sw_clr'] * (1 - df['cf_ceres'])) /
        df['cf_ceres'] / df['solar_incoming']
    )

    df['cot_disp'] = df['ret_cotstd_cer'] / df['ret_cot_cer']
    df['unr_fra'] = 1 - df['cf_ret_liq_mod08'] - df['clr_fra']

    mask = (
        (df['cf_ceres'] > MIN_CF) &
        (df['cf_liq_ceres'] / df['cf_ceres'] > 0.99) &
        (df['cot_mod08'] > MIN_COT) &
        (df['ret_cot_cer'] > MIN_COT) &
        (df['ret_albedo'].between(0, 1)) &
        (df['albedo'].between(0, 1)) &
        df['aod_mod08'].notna()
    )

    return df[mask].dropna()


def calc_slope_diff_for_bin(bin_df, mode, x2):
    season_vals = bin_df['season'].values

    if mode == 'cot_disp':
        ret_cot = bin_df['ret_cot_cer'].values
        ret_albedo = bin_df['ret_albedo'].values
        albedo_sbd = cot_to_albedo(ret_cot, 'sbdart', sza=bin_df['sza'].values, table_folder='cp')

        k_ret, _, _ = calc_global_slope_from_raw(
            ret_cot, ret_albedo, season_vals, x2,
            cot_std=0.0, albedo_std=0.03, n_mc=1000
        )
        k_sbd, _, _ = calc_global_slope_from_raw(
            ret_cot, albedo_sbd, season_vals, x2,
            cot_std=0.0, albedo_std=0.03, n_mc=1000
        )

        return k_sbd - k_ret

    if mode == 'unr_fra':
        msk_cot = bin_df['cot_mod08'].values
        msk_albedo = bin_df['albedo'].values
        ret_cot = bin_df['ret_cot_cer'].values
        ret_albedo = bin_df['ret_albedo'].values

        k_msk, _, _ = calc_global_slope_from_raw(
            msk_cot, msk_albedo, season_vals, x2,
            cot_std=0.10, albedo_std=0.13, n_mc=1000
        )
        k_ret, _, _ = calc_global_slope_from_raw(
            ret_cot, ret_albedo, season_vals, x2,
            cot_std=0.10, albedo_std=0.10, n_mc=1000
        )

        return k_ret - k_msk

    raise ValueError(f'Unsupported mode: {mode}')


def calc_high_low_ratio(df, split_col, mode, x2, n_bins=2):
    labels, edges = split_data_by_percentile(df.copy(), split_col, n_bins)

    bin_values = {}
    for idx in range(len(edges) - 1):
        bin_df = df[labels == idx].copy()
        if len(bin_df) < 5:
            continue

        bin_values[idx] = calc_slope_diff_for_bin(bin_df, mode, x2)

    low = bin_values.get(0, np.nan)
    high = bin_values.get(1, np.nan)

    if np.isfinite(low) and np.isfinite(high) and low > 0:
        return high / low

    return np.nan


def process_all_oceans_by_season(n_bins=2):
    cot_ref = np.exp(np.linspace(np.log(3), 4.5, 15))
    x2 = cot_to_x(cot_ref)

    records = []

    for ocean in oceans:
        for season_name in season_dict:
            df = prepare_dataframe(ocean, season_name)
            if df is None or len(df) < 10:
                continue

            records.append({
                'Ocean': ocean,
                'Season': season_name,
                'cot_disp_ratio': calc_high_low_ratio(df, 'cot_disp', 'cot_disp', x2, n_bins),
                'unr_fra_ratio': calc_high_low_ratio(df, 'unr_fra', 'unr_fra', x2, n_bins),
                'aod_cot_ratio': calc_high_low_ratio(df, 'aod_mod08', 'cot_disp', x2, n_bins),
                'aod_unr_ratio': calc_high_low_ratio(df, 'aod_mod08', 'unr_fra', x2, n_bins),
            })

    return records


def draw_two_boxplot(ax, data, labels, ylabel):
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.4,
        showfliers=False
    )

    for box, color in zip(bp['boxes'], BOX_COLORS):
        box.set_facecolor('none')
        box.set_edgecolor(color)
        box.set_linewidth(2)

    for median, color in zip(bp['medians'], BOX_COLORS):
        median.set_color(color)
        median.set_linewidth(2)

    for i, whisker in enumerate(bp['whiskers']):
        whisker.set_color(BOX_COLORS[i // 2])
        whisker.set_linewidth(1.5)

    for i, cap in enumerate(bp['caps']):
        cap.set_color(BOX_COLORS[i // 2])
        cap.set_linewidth(1.5)

    rng = np.random.default_rng(42)
    for i, y in enumerate(data, start=1):
        y = np.asarray(y, dtype=float)
        jitter = rng.uniform(-0.08, 0.08, size=len(y))
        ax.scatter(
            np.full(len(y), i) + jitter,
            y,
            color='k',
            alpha=0.3,
            s=20,
            zorder=5
        )

    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', linestyle='--', alpha=0.3)


def plot_combined_4panels_v2(icon_style='nature'):
    season_records = process_all_oceans_by_season(n_bins=2)

    cot_disp_ratios = [r['cot_disp_ratio'] for r in season_records if np.isfinite(r['cot_disp_ratio'])]
    aod_cot_ratios = [r['aod_cot_ratio'] for r in season_records if np.isfinite(r['aod_cot_ratio'])]
    unr_fra_ratios = [r['unr_fra_ratio'] for r in season_records if np.isfinite(r['unr_fra_ratio'])]
    aod_unr_ratios = [r['aod_unr_ratio'] for r in season_records if np.isfinite(r['aod_unr_ratio'])]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6), dpi=300)

    # Panel a
    # Compute k values in ln(COT) vs ln(Ac/(1-Ac)) space
    x_fit = cot_to_x(cot_range)

    alb_sbd = cot_to_albedo(cot_range, 'sbdart', sza=54.5, table_folder='dcp')
    y_fit = albedo_to_y(alb_sbd)
    mask = np.isfinite(y_fit)
    k_sbd, _ = np.polyfit(x_fit[mask], y_fit[mask], 1)

    alb_mono = cot_to_albedo(cot_range, 'sbdart', sza=54.5, table_folder='dcp_mono')
    y_fit = albedo_to_y(alb_mono)
    mask = np.isfinite(y_fit)
    k_mono, _ = np.polyfit(x_fit[mask], y_fit[mask], 1)

    alb_quad = cot_to_albedo(cot_range, 'quadrature', sza=54.5)
    y_fit = albedo_to_y(alb_quad)
    mask = np.isfinite(y_fit)
    k_quad, _ = np.polyfit(x_fit[mask], y_fit[mask], 1)

    ax1.plot(
        cot_range, alb_sbd,
        color=LINECOLOR[1], lw=2,
        label=rf'Sbdart, SW ($k_\mathrm{{dcp}}$={k_sbd:.2f})'
    )
    ax1.plot(
        cot_range, alb_mono,
        color=LINECOLOR[3], lw=2, linestyle='--',
        label=rf'Sbdart, VS ($k_\mathrm{{dcp}}$={k_mono:.2f})'
    )
    ax1.plot(
        cot_range, alb_quad,
        color=LINECOLOR[0], lw=2,
        label=rf'Quadrature ($k_\mathrm{{T91}}$={k_quad:.2f})'
    )

    ax1.set_xlabel('COT', fontsize=14, fontweight='medium')
    ax1.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=14, fontweight='medium')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_title(format_panel_tag(1, icon_style), fontsize=15, loc='left')
    ax1.legend(loc='lower right', fontsize=9.5, framealpha=0.9)

    # Panel b
    lookup_folders = ['dcp', 'surdcp_gascp', 'gasdcp_surcp', 'dcp', 'cp']
    lookup_labels = [
        r'Decoupled ($k_{\mathrm{dcp}}=$',
        r'$A_{\mathrm{sfc}}$ Coupled ($k_{\mathrm{cp}}=$',
        r'Gas Coupled ($k_{\mathrm{cp}}=$',
        r'SZA Coupled ($k_{\mathrm{cp}}=$',
        r'All Coupled ($k_{\mathrm{cp}}=$'
    ]

    for idx, folder in enumerate(lookup_folders[:3]):
        alb_vals = cot_to_albedo(cot_range, 'sbdart', sza=54.4, table_folder=folder)
        y_fit = albedo_to_y(alb_vals)
        mask = np.isfinite(y_fit)
        k_val, _ = np.polyfit(x_fit[mask], y_fit[mask], 1)
        ax2.plot(
            cot_range, alb_vals,
            color=LINECOLOR[idx],
            lw=2,
            linestyle=LINESTYLE[idx],
            label=f'{lookup_labels[idx]}{k_val:.2f})'
        )

    all_sza = []
    for ocean in oceans:
        for season_name in season_dict:
            path = f'{BASE_PATH}/processed_data/merged_data/{ocean}_{season_name}.csv'
            if os.path.exists(path):
                all_sza.extend(pd.read_csv(path)['sza'].dropna().values)

    mean_sza = np.mean(all_sza)
    print(f'Global mean SZA: {mean_sza:.2f}')
    for idx in [3, 4]:
        folder = lookup_folders[idx]
        alb_vals = cot_to_albedo(cot_range, 'sbdart', sza=mean_sza, table_folder=folder)
        y_fit = albedo_to_y(alb_vals)
        mask = np.isfinite(y_fit)
        k_val, _ = np.polyfit(x_fit[mask], y_fit[mask], 1)
        ax2.plot(
            cot_range, alb_vals,
            color=LINECOLOR[idx],
            lw=2,
            linestyle=LINESTYLE[idx],
            label=f'{lookup_labels[idx]}{k_val:.2f})'
        )

    ax2.set_xlabel('COT', fontsize=14, fontweight='medium')
    ax2.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=14, fontweight='medium')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_title(format_panel_tag(2, icon_style), fontsize=15, loc='left')
    ax2.legend(loc='lower right', fontsize=9.5, framealpha=0.5)

    # Panel c
    draw_two_boxplot(
        ax3,
        data=[cot_disp_ratios, aod_cot_ratios],
        labels=[r'High $d_{\mathrm{COT}}$ / Low $d_{\mathrm{COT}}$', 'High AOD / Low AOD'],
        ylabel=r'Ratio of $k_{\mathrm{cp}}-k_{\mathrm{ret}}$'
    )
    ax3.set_title(format_panel_tag(3, icon_style), fontsize=15, loc='left')

    # Panel d
    draw_two_boxplot(
        ax4,
        data=[unr_fra_ratios, aod_unr_ratios],
        labels=['High TZF / Low TZF', 'High AOD / Low AOD'],
        ylabel=r'Ratio of $k_{\mathrm{ret}}-k_{\mathrm{msk}}$'
    )
    ax4.set_title(format_panel_tag(4, icon_style), fontsize=15, loc='left')

    plt.subplots_adjust(
        left=0.08,
        right=0.97,
        top=0.95,
        bottom=0.08,
        wspace=0.3,
        hspace=0.32
    )

    plt.savefig(FIG_SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Final combined figure saved to: {os.path.abspath(FIG_SAVE_PATH)}')


if __name__ == '__main__':
    plot_combined_4panels_v2(icon_style='nature')