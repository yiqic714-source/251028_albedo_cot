import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_fitting import (
    oceans, season_dict, cot_range, albedo_to_y,
    cot_to_x, cot_to_albedo, mc_fit, format_panel_tag
)


BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIG_SAVE_PATH = f'{BASE_PATH}/figs/fig3_bias_attribution.png'
os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)

MIN_COT = 2.5
MIN_CF = 0.1

LINECOLOR = ['steelblue', 'orange', 'coral', 'red', 'purple']
LINESTYLE = ['-', '-', '--', ':', '-']
BOX_COLORS = ['red', 'blue']


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
            cot_std=0.0, albedo_std=0.03, n_mc=300
        )
        k_sbd, _, _ = calc_global_slope_from_raw(
            ret_cot, albedo_sbd, season_vals, x2,
            cot_std=0.0, albedo_std=0.03, n_mc=300
        )

        return k_sbd - k_ret

    if mode == 'unr_fra':
        msk_cot = bin_df['cot_mod08'].values
        msk_albedo = bin_df['albedo'].values
        ret_cot = bin_df['ret_cot_cer'].values
        ret_albedo = bin_df['ret_albedo'].values

        k_msk, _, _ = calc_global_slope_from_raw(
            msk_cot, msk_albedo, season_vals, x2,
            cot_std=0.10, albedo_std=0.13, n_mc=300
        )
        k_ret, _, _ = calc_global_slope_from_raw(
            ret_cot, ret_albedo, season_vals, x2,
            cot_std=0.10, albedo_std=0.20, n_mc=300
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


def load_global_data():
    """Load and merge all ocean x season data (same as fig2_global_5curves.py)."""
    dfs = []
    for ocean in oceans:
        for season_name in season_dict:
            file_path = f'{BASE_PATH}/processed_data/merged_data/{ocean}_{season_name}.csv'
            if not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)
            df['season'] = season_name
            df['ocean'] = ocean
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError('No data files found.')

    df = pd.concat(dfs, ignore_index=True)

    df['albedo'] = (
        (df['sw_all'] - df['sw_clr'] * (1 - df['cf_ceres'])) /
        df['cf_ceres'] / df['solar_incoming']
    )

    mask = (
        (df['cf_ceres'] > MIN_CF) &
        (df['cf_liq_ceres'] / df['cf_ceres'] > 0.99) &
        (df['cot_mod08'] > MIN_COT) &
        (df['ret_cot_cer'] > MIN_COT) &
        (df['ret_albedo'].between(0, 1)) &
        (df['albedo'].between(0, 1))
    )

    return df[mask].dropna()


def bin_data_by_cot(df, cot_col, albedo_col, bin_edges):
    """Bin data by COT and compute mean/std (same as fig2_global_5curves.py)."""
    labels = pd.cut(df[cot_col], bins=bin_edges, labels=False, include_lowest=True)

    bin_means_cot = []
    bin_means_alb = []
    bin_stds_alb = []

    for i in range(len(bin_edges) - 1):
        mask = labels == i
        if mask.sum() < 5:
            continue

        cot_vals = df.loc[mask, cot_col].values
        alb_vals = df.loc[mask, albedo_col].values

        bin_means_cot.append(np.mean(cot_vals))
        bin_means_alb.append(np.mean(alb_vals))
        bin_stds_alb.append(np.std(alb_vals))

    return np.array(bin_means_cot), np.array(bin_means_alb), np.array(bin_stds_alb)


def compute_sbdart_albedo_per_point(df, table_folder, sza_col='sza'):
    """Compute SBDART albedo per data point using per-point SZA (same as fig2_global_5curves.py)."""
    result = np.full(len(df), np.nan)

    for ocean in oceans:
        for season_name in season_dict:
            mask = (df['ocean'] == ocean) & (df['season'] == season_name)
            if mask.sum() == 0:
                continue

            result[mask.values] = cot_to_albedo(
                df.loc[mask, 'ret_cot_cer'].values,
                'sbdart',
                sza=df.loc[mask, sza_col].values,
                table_folder=table_folder,
                ocean=ocean,
                season=season_name
            )

    return result


def compute_sbdart_albedo_fixed_sza(df, table_folder, sza=54.4):
    """Compute SBDART albedo per data point using a fixed SZA."""
    result = np.full(len(df), np.nan)

    for ocean in oceans:
        for season_name in season_dict:
            mask = (df['ocean'] == ocean) & (df['season'] == season_name)
            if mask.sum() == 0:
                continue

            result[mask.values] = cot_to_albedo(
                df.loc[mask, 'ret_cot_cer'].values,
                'sbdart',
                sza=sza,
                table_folder=table_folder,
                ocean=ocean,
                season=season_name
            )

    return result


def plot_4panels(icon_style='nature'):
    season_records = process_all_oceans_by_season(n_bins=2)

    cot_disp_ratios = [r['cot_disp_ratio'] for r in season_records if np.isfinite(r['cot_disp_ratio'])]
    aod_cot_ratios = [r['aod_cot_ratio'] for r in season_records if np.isfinite(r['aod_cot_ratio'])]
    unr_fra_ratios = [r['unr_fra_ratio'] for r in season_records if np.isfinite(r['unr_fra_ratio'])]
    aod_unr_ratios = [r['aod_unr_ratio'] for r in season_records if np.isfinite(r['aod_unr_ratio'])]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6), dpi=300)

    # Panel a
    # Load global data
    df = load_global_data()
    print(f'Total data points: {len(df)}')

    # Compute k values in ln(COT) vs ln(Ac/(1-Ac)) space
    x_fit = cot_to_x(df['ret_cot_cer'])

    alb_sbd = cot_to_albedo(df['ret_cot_cer'], 'sbdart', sza=54.5, table_folder='dcp')
    y_fit = albedo_to_y(alb_sbd)
    mask = np.isfinite(y_fit)
    k_sbd, _ = np.polyfit(x_fit[mask], y_fit[mask], 1)

    alb_mono = cot_to_albedo(df['ret_cot_cer'], 'sbdart', sza=54.5, table_folder='dcp_mono')
    y_fit = albedo_to_y(alb_mono)
    mask = np.isfinite(y_fit)
    k_mono, _ = np.polyfit(x_fit[mask], y_fit[mask], 1)

    alb_quad = cot_to_albedo(df['ret_cot_cer'], 'quadrature', sza=54.5)
    y_fit = albedo_to_y(alb_quad)
    mask = np.isfinite(y_fit)
    k_quad, _ = np.polyfit(x_fit[mask], y_fit[mask], 1)

    sorted_idx = np.argsort(df['ret_cot_cer'])
    ax1.plot(
        df['ret_cot_cer'].values[sorted_idx], alb_sbd[sorted_idx],
        color=LINECOLOR[1], lw=2,
        label=rf'Sbdart, SW ($k_\mathrm{{dcp}}$={k_sbd:.2f})'
    )
    ax1.plot(
        df['ret_cot_cer'].values[sorted_idx], alb_mono[sorted_idx],
        color=LINECOLOR[3], lw=2, linestyle='--',
        label=rf'Sbdart, VS ($k_\mathrm{{dcp}}$={k_mono:.2f})'
    )
    ax1.plot(
        df['ret_cot_cer'].values[sorted_idx], alb_quad[sorted_idx],
        color=LINECOLOR[0], lw=2,
        label=rf'Quadrature ($k_\mathrm{{T91}}$={k_quad:.2f})'
    )

    ax1.set_xlabel('COT', fontsize=14, fontweight='medium')
    ax1.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=14, fontweight='medium')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_title(format_panel_tag(1, icon_style), fontsize=15, loc='left')
    ax1.legend(loc='lower right', fontsize=9.5, framealpha=0.9)

    # Panel b
    bin_edges = cot_range

    # --- Line 1: Decoupled SBDART (dcp) with fixed sza=54.4 (unchanged) ---
    ax2.plot(
        df['ret_cot_cer'].values[sorted_idx], alb_sbd[sorted_idx],
        color=LINECOLOR[0], lw=2,
        linestyle=LINESTYLE[0],
        label=rf'Decoupled ($k_{{\mathrm{{dcp}}}}$={k_sbd:.2f})'
    )

    # --- Lines 2-3: Fixed sza=54.4, per-point cot, with errorbar ---
    lookup_folders_fixed_sza = ['gasdcp_surcp', 'surdcp_gascp']
    lookup_labels_fixed_sza = [
        r'$A_{\mathrm{sfc}}$ Coupled ($k_{\mathrm{cp}}=$',
        r'Gas Coupled ($k_{\mathrm{cp}}=$'
    ]

    for idx_offset, folder in enumerate(lookup_folders_fixed_sza):
        idx = idx_offset + 1  # line index 1, 2
        print(f'  Computing {folder} with fixed sza=54.4...')
        alb_vals = compute_sbdart_albedo_fixed_sza(df, folder, sza=54.4)
        df[f'alb_{folder}'] = alb_vals

        cot_bins, alb_bins, alb_std = bin_data_by_cot(
            df, 'ret_cot_cer', f'alb_{folder}', bin_edges
        )

        # mc_fit for k
        k_val, b_val, _, _ = mc_fit(
            df['ret_cot_cer'].values,
            alb_vals,
            cot_std=0.0,
            albedo_std=0.03,
            n_mc=300,
            bootstrap=True
        )

        ax2.errorbar(
            cot_bins, alb_bins, yerr=alb_std,
            color=LINECOLOR[idx], fmt='o-', lw=2, capsize=3, capthick=1,
            label=f'{lookup_labels_fixed_sza[idx_offset]}{k_val:.2f})'
        )

    # --- Lines 4-5: Per-point sza, with errorbar ---
    lookup_folders_sza = ['dcp', 'cp']
    lookup_labels_sza = [
        r'SZA Coupled ($k_{\mathrm{cp}}=$',
        r'All Coupled ($k_{\mathrm{cp}}=$'
    ]

    for idx_offset, folder in enumerate(lookup_folders_sza):
        idx = idx_offset + 3  # line index 3, 4
        print(f'  Computing {folder} with per-point sza...')
        alb_vals = compute_sbdart_albedo_per_point(df, folder, sza_col='sza')
        df[f'alb_{folder}_persza'] = alb_vals

        cot_bins, alb_bins, alb_std = bin_data_by_cot(
            df, 'ret_cot_cer', f'alb_{folder}_persza', bin_edges
        )

        # mc_fit for k
        k_val, b_val, _, _ = mc_fit(
            df['ret_cot_cer'].values,
            alb_vals,
            cot_std=0.0,
            albedo_std=0.03,
            n_mc=300,
            bootstrap=True
        )

        ax2.errorbar(
            cot_bins, alb_bins, yerr=alb_std,
            color=LINECOLOR[idx], fmt='o-', lw=2, capsize=3, capthick=1,
            label=f'{lookup_labels_sza[idx_offset]}{k_val:.2f})'
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
    plot_4panels(icon_style='nature')
