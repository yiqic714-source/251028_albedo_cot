# -*- coding: utf-8 -*-
"""
figsupp_fittings_ocean_season.py

For each ocean-season, draw a Fig. 2a-style plot (5 curves: T91, DCP, CP, RET, MSK)
and save per-ocean-season sensitivity coefficients to CSV.

Layout: 4 rows (oceans) × 4 columns (seasons) per figure, two figures total.
Each subplot has its own legend showing solid lines with k= values.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils_fitting import (
    oceans, season_dict, cot_range, albedo_to_y,
    cot_to_x, cot_to_albedo, mc_fit, format_panel_tag, cot_k_b_to_albedo
)

BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIG_DIR = f'{BASE_PATH}/figs'
SENSITIVITY_CSV_PATH = f'{BASE_PATH}/processed_data/sensitivity_albedo_vs_cot_1030.csv'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SENSITIVITY_CSV_PATH), exist_ok=True)

MIN_COT = 2.5
MIN_CF = 0.1

# Colors (same as fig2_fittings_global_and_reasons.py)
T91_COLOR = '#222222'
DCP_COLOR = '#00bfff'
CP_COLOR = '#574cff'
RET_COLOR = '#ff852e'
MSK_COLOR = '#f20d38'

season_keys = list(season_dict.keys())


def load_global_data():
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


def fit_k_b_in_logit_space(cot, albedo):
    x = cot_to_x(np.asarray(cot, dtype=float))
    y = albedo_to_y(np.asarray(albedo, dtype=float))

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan

    k, b = np.polyfit(x[mask], y[mask], 1)
    return k, b


def compute_sbdart_albedo_per_point(df, table_folder):
    result = np.full(len(df), np.nan)

    for ocean in oceans:
        for season_name in season_dict:
            mask = (df['ocean'] == ocean) & (df['season'] == season_name)
            if mask.sum() == 0:
                continue

            result[mask.values] = cot_to_albedo(
                df.loc[mask, 'ret_cot_cer'].values,
                'sbdart',
                sza=df.loc[mask, 'sza'].values,
                table_folder=table_folder,
                ocean=ocean,
                season=season_name
            )

    return result


def compute_per_ocean_season_fits(df):
    """
    Compute k, lnb for each method (dcp, cp, ret, msk) per ocean-season.
    Returns a list of dicts for building a wide-format CSV.
    """
    print('Computing per-ocean-season fits...')
    records = []

    for ocean in oceans:
        for season_name in season_dict.keys():
            mask = (df['ocean'] == ocean) & (df['season'] == season_name)
            sub = df[mask]
            n_pts = len(sub)
            if n_pts < 5:
                records.append({
                    'Ocean': ocean, 'Season': season_name,
                    'k_dcp': np.nan, 'lnb_dcp': np.nan,
                    'k_dcp_unc': np.nan, 'lnb_dcp_unc': np.nan,
                    'k_cp': np.nan, 'lnb_cp': np.nan,
                    'k_cp_unc': np.nan, 'lnb_cp_unc': np.nan,
                    'k_ret': np.nan, 'lnb_ret': np.nan,
                    'k_ret_unc': np.nan, 'lnb_ret_unc': np.nan,
                    'k_msk': np.nan, 'lnb_msk': np.nan,
                    'k_msk_unc': np.nan, 'lnb_msk_unc': np.nan,
                })
                continue

            # dcp: fixed SZA=54.4
            alb_dcp_os = cot_to_albedo(
                sub['ret_cot_cer'].values, 'sbdart',
                sza=54.4, table_folder='dcp'
            )
            k_dcp_os, lnb_dcp_os = fit_k_b_in_logit_space(sub['ret_cot_cer'].values, alb_dcp_os)

            # cp: per-point SZA
            alb_cp_os = cot_to_albedo(
                sub['ret_cot_cer'].values, 'sbdart',
                sza=sub['sza'].values, table_folder='cp',
                ocean=ocean, season=season_name
            )
            k_cp_os, lnb_cp_os, k_cp_unc, lnb_cp_unc = mc_fit(
                sub['ret_cot_cer'].values, alb_cp_os,
                cot_std=0.0, albedo_std=0.03, n_mc=300, bootstrap=True
            )

            # ret
            k_ret_os, lnb_ret_os, k_ret_unc, lnb_ret_unc = mc_fit(
                sub['ret_cot_cer'].values, sub['ret_albedo'].values,
                cot_std=0.10, albedo_std=0.13, n_mc=300, bootstrap=True
            )

            # msk
            k_msk_os, lnb_msk_os, k_msk_unc, lnb_msk_unc = mc_fit(
                sub['cot_mod08'].values, sub['albedo'].values,
                cot_std=0.10, albedo_std=0.20, n_mc=300, bootstrap=True
            )

            records.append({
                'Ocean': ocean, 'Season': season_name,
                'k_dcp': k_dcp_os, 'lnb_dcp': lnb_dcp_os,
                'k_dcp_unc': np.nan, 'lnb_dcp_unc': np.nan,
                'k_cp': k_cp_os, 'lnb_cp': lnb_cp_os,
                'k_cp_unc': k_cp_unc, 'lnb_cp_unc': lnb_cp_unc,
                'k_ret': k_ret_os, 'lnb_ret': lnb_ret_os,
                'k_ret_unc': k_ret_unc, 'lnb_ret_unc': lnb_ret_unc,
                'k_msk': k_msk_os, 'lnb_msk': lnb_msk_os,
                'k_msk_unc': k_msk_unc, 'lnb_msk_unc': lnb_msk_unc,
            })

    return records


def draw_ocean_season_panel(ax, sub, ocean, season_name, bin_edges):
    """Draw a Fig. 2a-style panel for one ocean-season.
    Adds legend with solid lines showing k values for this subplot."""
    n_pts = len(sub)
    if n_pts < 5:
        ax.text(0.5, 0.5, 'Insufficient data',
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        return

    # T91
    alb_t91 = cot_to_albedo(cot_range, 'quadrature', sza=54.4)
    k_t91, lnb_t91 = fit_k_b_in_logit_space(cot_range, alb_t91)
    alb_t91_fit = cot_k_b_to_albedo(cot_range, k_t91, np.exp(lnb_t91))

    # DCP: fixed SZA=54.4
    alb_dcp_os = cot_to_albedo(
        sub['ret_cot_cer'].values, 'sbdart',
        sza=54.4, table_folder='dcp'
    )
    k_dcp_os, lnb_dcp_os = fit_k_b_in_logit_space(sub['ret_cot_cer'].values, alb_dcp_os)
    alb_dcp_fit = cot_k_b_to_albedo(cot_range, k_dcp_os, np.exp(lnb_dcp_os))

    # CP: per-point SZA
    alb_cp_os = cot_to_albedo(
        sub['ret_cot_cer'].values, 'sbdart',
        sza=sub['sza'].values, table_folder='cp',
        ocean=ocean, season=season_name
    )
    k_cp_os, lnb_cp_os, _, _ = mc_fit(
        sub['ret_cot_cer'].values, alb_cp_os,
        cot_std=0.0, albedo_std=0.03, n_mc=300, bootstrap=True
    )
    alb_cp_fit = cot_k_b_to_albedo(cot_range, k_cp_os, np.exp(lnb_cp_os))
    cp_cot_bins, cp_alb_bins, cp_alb_std = bin_data_by_cot(
        sub, 'ret_cot_cer', 'cp_albedo', bin_edges
    )

    # RET
    k_ret_os, lnb_ret_os, _, _ = mc_fit(
        sub['ret_cot_cer'].values, sub['ret_albedo'].values,
        cot_std=0.10, albedo_std=0.13, n_mc=300, bootstrap=True
    )
    alb_ret_fit = cot_k_b_to_albedo(cot_range, k_ret_os, np.exp(lnb_ret_os))
    ret_cot_bins, ret_alb_bins, ret_alb_std = bin_data_by_cot(
        sub, 'ret_cot_cer', 'ret_albedo', bin_edges
    )

    # MSK
    k_msk_os, lnb_msk_os, _, _ = mc_fit(
        sub['cot_mod08'].values, sub['albedo'].values,
        cot_std=0.10, albedo_std=0.20, n_mc=300, bootstrap=True
    )
    alb_msk_fit = cot_k_b_to_albedo(cot_range, k_msk_os, np.exp(lnb_msk_os))
    msk_cot_bins, msk_alb_bins, msk_alb_std = bin_data_by_cot(
        sub, 'cot_mod08', 'albedo', bin_edges
    )

    # Plot T91
    ax.plot(cot_range, alb_t91, color=T91_COLOR, lw=1.2, ls='-')
    ax.plot(cot_range, alb_t91_fit, color=T91_COLOR, lw=1, ls='--', alpha=0.7)

    # Plot DCP
    sorted_idx = np.argsort(sub['ret_cot_cer'])
    ax.plot(sub['ret_cot_cer'].values[sorted_idx], alb_dcp_os[sorted_idx],
            color=DCP_COLOR, lw=1.2, ls='-')
    ax.plot(cot_range, alb_dcp_fit, color=DCP_COLOR, lw=1, ls='--', alpha=0.7)

    # Plot CP
    ax.errorbar(cp_cot_bins, cp_alb_bins, yerr=cp_alb_std,
                color=CP_COLOR, fmt='o-', lw=1, ms=2.5, capsize=2, capthick=0.6)
    ax.plot(cot_range, alb_cp_fit, color=CP_COLOR, lw=1, ls='--', alpha=0.7)

    # Plot RET
    ax.errorbar(ret_cot_bins, ret_alb_bins, yerr=ret_alb_std,
                color=RET_COLOR, fmt='o-', lw=1, ms=2.5, capsize=2, capthick=0.6)
    ax.plot(cot_range, alb_ret_fit, color=RET_COLOR, lw=1, ls='--', alpha=0.7)

    # Plot MSK
    ax.errorbar(msk_cot_bins, msk_alb_bins, yerr=msk_alb_std,
                color=MSK_COLOR, fmt='s-', lw=1, ms=2.5, capsize=2, capthick=0.6)
    ax.plot(cot_range, alb_msk_fit, color=MSK_COLOR, lw=1, ls='--', alpha=0.7)

    ax.set_xlim(0, 60)
    ax.tick_params(axis='both', labelsize=7)

    # Legend with solid lines showing this subplot's k values
    legend_elements = [
        Line2D([0], [0], color=T91_COLOR, lw=2, ls='-',
               label=rf'T91: $k$={k_t91:.2f}'),
        Line2D([0], [0], color=DCP_COLOR, lw=2, ls='-',
               label=rf'DCP: $k$={k_dcp_os:.2f}'),
        Line2D([0], [0], color=CP_COLOR, lw=2, ls='-',
               label=rf'CP: $k$={k_cp_os:.2f}'),
        Line2D([0], [0], color=RET_COLOR, lw=2, ls='-',
               label=rf'RET: $k$={k_ret_os:.2f}'),
        Line2D([0], [0], color=MSK_COLOR, lw=2, ls='-',
               label=rf'MSK: $k$={k_msk_os:.2f}'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7,
              framealpha=0.5, handlelength=1.2)


def make_figure(df, bin_edges):
    """Create a single figure with 8 rows (oceans) × 4 columns (seasons), no gaps."""
    n_rows = len(oceans)
    n_cols = len(season_keys)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.7 * n_cols, 2.0 * n_rows),
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(wspace=0, hspace=0, left=0.04, right=0.98, bottom=0.04, top=0.96)

    for i, ocean in enumerate(oceans):
        for j, season_name in enumerate(season_keys):
            ax = axes[i, j]
            mask = (df['ocean'] == ocean) & (df['season'] == season_name)
            sub = df[mask]

            draw_ocean_season_panel(ax, sub, ocean, season_name, bin_edges)

            # Y-axis label on the leftmost column
            if j == 0:
                ax.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=9)

            # Column label (season) on the top row
            if i == 0:
                ax.set_title(season_name, fontsize=10, fontweight='bold')

            # X-axis label only on bottom row
            if i == n_rows - 1:
                ax.set_xlabel('COT', fontsize=8)
            else:
                ax.set_xlabel('')

    # Ocean names vertically on the left side, aligned to each row's center
    fig.canvas.draw()
    for i, ocean in enumerate(oceans):
        ax_pos = axes[i, 0].get_position()
        y_center = (ax_pos.y0 + ax_pos.y1) / 2
        fig.text(
            -0.007, y_center,
            ocean,
            fontsize=10, fontweight='bold',
            rotation=90, va='center', ha='center'
        )

    out_path = os.path.join(FIG_DIR, 'figsupp_ocean_season_fittings.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    print('Loading global data...')
    df = load_global_data()
    print(f'Total data points: {len(df)}')

    # Compute CP albedo for all points (needed for binning in per-ocean-season plots)
    print('Computing coupled SBDART albedo...')
    df['cp_albedo'] = compute_sbdart_albedo_per_point(df, 'cp')

    bin_edges = cot_range

    # ---- Compute per-ocean-season fits and save to CSV ----
    os_records = compute_per_ocean_season_fits(df)
    os_df = pd.DataFrame(os_records)
    os_df = os_df.sort_values(['Ocean', 'Season']).reset_index(drop=True)
    os_df.to_csv(SENSITIVITY_CSV_PATH, index=False)
    print(f'Saved per-ocean-season fits to: {SENSITIVITY_CSV_PATH}')

    # ---- Create single figure: 8 rows × 4 columns, no gaps ----
    make_figure(df, bin_edges)

    print('All done.')


if __name__ == '__main__':
    main()
