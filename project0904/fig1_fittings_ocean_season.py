# -*- coding: utf-8 -*-
"""
figsupp_fittings_ocean_season.py

For each ocean-season, draw a Fig. 2a-style plot with T91, CP, Grid, and RFOV curves
and save per-ocean-season sensitivity coefficients to CSV.

Layout: 4 rows (oceans) × 4 columns (seasons) per figure, two figures total.
Each subplot has its own legend showing solid lines with k= values.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils_fitting import (
    oceans, season_dict, cot_range, albedo_to_y,
    cot_to_x, cot_to_albedo, mc_fit, cot_k_b_to_albedo
)

BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIG_DIR = f'./figs'
SENSITIVITY_CSV_PATH = f'./processed_data/sensitivity_albedo_vs_cot_1030.csv'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SENSITIVITY_CSV_PATH), exist_ok=True)

MIN_COT = 2.5
MIN_CF = 0.25

# Colors (same as fig2_fittings_global_and_reasons.py)
T91_COLOR = '#222222'
SBD_COLOR = '#574cff'
RET_COLOR = '#ff852e'
GRID_COLOR = '#f20d38'
RFOV_COLOR = '#16a085'
FOV_INPUT_DIR = f'./RFOV_product/ocean_season'

season_keys = list(season_dict.keys())


def load_global_data():
    dfs = []
    for ocean in oceans:
        for season_name in season_dict:
            file_path = f'./L3_product/{ocean}_{season_name}.csv'
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
        (df['cf_ret_tot'] > MIN_CF) &
        (df['cf_liq_ceres'] / df['cf_ceres'] > 0.99) &
        (df['cot'] > MIN_COT) &
        (df['albedo'].between(0, 1)) &
        (df['cttmin'] >= 270)
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

def compute_sbdart_albedo_per_point(df, table_folder):
    result = np.full(len(df), np.nan)

    for ocean in oceans:
        for season_name in season_dict:
            mask = (df['ocean'] == ocean) & (df['season'] == season_name)
            if mask.sum() == 0:
                continue

            result[mask.values] = cot_to_albedo(
                df.loc[mask, 'cot'].values,
                'sbdart',
                sza=df.loc[mask, 'sza'].values,
                table_folder=table_folder,
                ocean=ocean,
                season=season_name
            )

    return result


def compute_per_ocean_season_fits(df, fov_df):
    """
    Compute k, lnb and uncertainties for each method per ocean-season.
    Returns a list of dicts for building a wide-format CSV.
    """
    print('Computing per-ocean-season fits...')
    records = []

    for ocean in oceans:
        for season_name in season_dict.keys():
            mask = (df['ocean'] == ocean) & (df['season'] == season_name)
            sub = df[mask]
            fov_sub = fov_df[
                (fov_df['ocean'] == ocean) &
                (fov_df['season'] == season_name)
            ]
            n_pts = len(sub)
            if n_pts < 5:
                records.append({
                    'Ocean': ocean, 'Season': season_name,
                    'k_sbd': np.nan, 'lnb_sbd': np.nan,
                    'k_sbd_unc': np.nan, 'lnb_sbd_unc': np.nan,
                    'k_grid': np.nan, 'lnb_grid': np.nan,
                    'k_grid_unc': np.nan, 'lnb_grid_unc': np.nan,
                    'k_rfov': np.nan, 'lnb_rfov': np.nan,
                    'k_rfov_unc': np.nan, 'lnb_rfov_unc': np.nan,
                })
                continue


            # SBDART: use RFOV COT rather than Grid COT.
            if len(fov_sub) >= 5:
                fov_sbd = fov_sub.copy()
                fov_sbd['sbd_albedo'] = cot_to_albedo(
                    fov_sbd['cot_fov'].values, 'sbdart',
                    sza=fov_sbd['solar_zenith'].values,
                    table_folder='cp', ocean=ocean, season=season_name
                )
                sbd_cot, sbd_albedo, _ = bin_data_by_cot(
                    fov_sbd, 'cot_fov', 'sbd_albedo', cot_range
                )
                k_sbd_os, lnb_sbd_os, k_sbd_unc, lnb_sbd_unc = mc_fit(
                    sbd_cot, sbd_albedo,
                    cot_std=0.0, albedo_std=0.03,
                    n_mc=300, bootstrap=True
                )
            else:
                k_sbd_os = lnb_sbd_os = np.nan
                k_sbd_unc = lnb_sbd_unc = np.nan

            # Grid
            k_grid_os, lnb_grid_os, k_grid_unc, lnb_grid_unc = mc_fit(
                sub['cot'].values, sub['albedo'].values,
                cot_std=0.10, albedo_std=0.20, n_mc=300, bootstrap=True
            )

            # RFOV: use the same binned data and mc_fit settings as the plot
            if len(fov_sub) >= 5:
                rfov_cot, rfov_albedo, _ = bin_data_by_cot(
                    fov_sub, 'cot_fov', 'ret_albedo', cot_range
                )
                k_rfov, lnb_rfov, k_rfov_unc, lnb_rfov_unc = mc_fit(
                    rfov_cot, rfov_albedo,
                    cot_std=0.10, albedo_std=0.20,
                    n_mc=300, bootstrap=True
                )
            else:
                k_rfov = lnb_rfov = np.nan
                k_rfov_unc = lnb_rfov_unc = np.nan

            records.append({
                'Ocean': ocean, 'Season': season_name,
                'k_sbd': k_sbd_os, 'lnb_sbd': lnb_sbd_os,
                'k_sbd_unc': k_sbd_unc, 'lnb_sbd_unc': lnb_sbd_unc,
                'k_grid': k_grid_os, 'lnb_grid': lnb_grid_os,
                'k_grid_unc': k_grid_unc, 'lnb_grid_unc': lnb_grid_unc,
                'k_rfov': k_rfov, 'lnb_rfov': lnb_rfov,
                'k_rfov_unc': k_rfov_unc, 'lnb_rfov_unc': lnb_rfov_unc,
            })

    return records


def add_frequency_weighted_cot(df):
    """Calculate exp(weighted mean(log(COT))) from COT frequency bins."""
    bins = []
    pattern = re.compile(r'^cot_freq_(\d+)_(\d+)$')
    for column in df.columns:
        match = pattern.match(column)
        if match:
            lower, upper = map(int, match.groups())
            bins.append((lower, upper, column))
    if not bins:
        raise ValueError('No cot_freq_<lower>_<upper> columns found')

    bins.sort()
    frequency_columns = [column for _, _, column in bins]
    cot_midpoints = np.array([(lower + upper) / 2 for lower, upper, _ in bins])
    frequencies = df[frequency_columns].apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0).clip(lower=0).to_numpy(dtype=float)
    weights = frequencies.sum(axis=1)
    weighted_log_cot = np.divide(
        frequencies @ np.log(cot_midpoints),
        weights,
        out=np.full(len(df), np.nan),
        where=weights > 0,
    )
    result = df.copy()
    result['cot_fov'] = np.exp(weighted_log_cot)
    return result


def load_fov_ocean_season_data():
    frames = []
    for file_path in sorted(os.listdir(FOV_INPUT_DIR)):
        if not file_path.endswith('.csv'):
            continue
        ocean, season_name = file_path[:-4].rsplit('_', 1)
        df = add_frequency_weighted_cot(
            pd.read_csv(os.path.join(FOV_INPUT_DIR, file_path))
        )
        df['ret_albedo'] = pd.to_numeric(df['ret_albedo'], errors='coerce')
        df['ocean'] = ocean
        df['season'] = season_name
        frames.append(df[['cot_fov', 'ret_albedo', 'solar_zenith', 'ocean', 'season']])

    if not frames:
        raise FileNotFoundError(f'No FOV CSV files found in {FOV_INPUT_DIR}')
    data = pd.concat(frames, ignore_index=True)
    return data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['cot_fov', 'ret_albedo']
    ).query('cot_fov >= @MIN_COT and ret_albedo >= 0 and ret_albedo <= 1')


def draw_ocean_season_panel(ax, sub, ocean, season_name, bin_edges, fov_sub=None):
    """Draw a Fig. 2a-style panel for one ocean-season.
    Adds legend with solid lines showing k values for this subplot."""
    n_pts = len(sub)
    if n_pts < 5:
        ax.text(0.5, 0.5, 'Insufficient data',
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        return

    # T91
    alb_t91 = cot_to_albedo(cot_range, 'quadrature', sza=54.74)
    k_t91, lnb_t91, _, _ = mc_fit(
        cot_range, alb_t91, calculate_uncertainty=False
    )
    alb_t91_fit = cot_k_b_to_albedo(cot_range, k_t91, np.exp(lnb_t91))


    # SBDART and RFOV use RFOV COT bins.
    if fov_sub is not None and len(fov_sub) >= 5:
        rfov_cot_bins, rfov_albedo_bins, rfov_albedo_std = bin_data_by_cot(
            fov_sub, 'cot_fov', 'ret_albedo', bin_edges
        )
        fov_sbd = fov_sub.copy()
        fov_sbd['sbd_albedo'] = cot_to_albedo(
            fov_sbd['cot_fov'].values, 'sbdart',
            sza=fov_sbd['solar_zenith'].values,
            table_folder='cp', ocean=ocean, season=season_name
        )
        sbd_cot_bins, sbd_alb_bins, sbd_alb_std = bin_data_by_cot(
            fov_sbd, 'cot_fov', 'sbd_albedo', bin_edges
        )
        k_sbd_os, lnb_sbd_os, _, _ = mc_fit(
            sbd_cot_bins, sbd_alb_bins,
            cot_std=0.0, albedo_std=0.03, n_mc=300, bootstrap=True
        )
        k_rfov, lnb_rfov, _, _ = mc_fit(
            rfov_cot_bins, rfov_albedo_bins,
            cot_std=0.10, albedo_std=0.20, n_mc=300, bootstrap=True
        )
    else:
        rfov_cot_bins = rfov_albedo_bins = rfov_albedo_std = np.array([])
        sbd_cot_bins = sbd_alb_bins = sbd_alb_std = np.array([])
        k_sbd_os = lnb_sbd_os = k_rfov = lnb_rfov = np.nan

    alb_sbd_fit = cot_k_b_to_albedo(cot_range, k_sbd_os, np.exp(lnb_sbd_os))

    # Grid
    k_grid_os, lnb_grid_os, _, _ = mc_fit(
        sub['cot'].values, sub['albedo'].values,
        cot_std=0.10, albedo_std=0.20, n_mc=300, bootstrap=True
    )
    alb_grid_fit = cot_k_b_to_albedo(cot_range, k_grid_os, np.exp(lnb_grid_os))
    grid_cot_bins, grid_alb_bins, grid_alb_std = bin_data_by_cot(
        sub, 'cot', 'albedo', bin_edges
    )

    # Plot T91
    ax.plot(cot_range, alb_t91, color=T91_COLOR, lw=1.2, ls='-')
    ax.plot(cot_range, alb_t91_fit, color=T91_COLOR, lw=1, ls='--', alpha=0.7)


    # Plot SBDART
    ax.errorbar(sbd_cot_bins, sbd_alb_bins, yerr=sbd_alb_std,
                color=SBD_COLOR, fmt='o-', lw=1, ms=2.5, capsize=2, capthick=0.6)
    ax.plot(cot_range, alb_sbd_fit, color=SBD_COLOR, lw=1, ls='--', alpha=0.7)



    # Plot RFOV
    ax.errorbar(rfov_cot_bins, rfov_albedo_bins, yerr=rfov_albedo_std,
                color=RFOV_COLOR, fmt='o-', lw=1, ms=2.5, capsize=2, capthick=0.6)
    ax.plot(cot_range, cot_k_b_to_albedo(cot_range, k_rfov, np.exp(lnb_rfov)),
            color=RFOV_COLOR, lw=1, ls='--')

    # Plot Grid
    ax.errorbar(grid_cot_bins, grid_alb_bins, yerr=grid_alb_std,
                color=GRID_COLOR, fmt='s-', lw=1, ms=2.5, capsize=2, capthick=0.6)
    ax.plot(cot_range, alb_grid_fit, color=GRID_COLOR, lw=1, ls='--', alpha=0.7)

    ax.set_xlim(0, 60)
    ax.tick_params(axis='both', labelsize=7)

    # Legend with solid lines showing this subplot's k values
    legend_elements = [
        Line2D([0], [0], color=T91_COLOR, lw=2, ls='-',
               label=rf'T91: $k$={k_t91:.2f}'),
        Line2D([0], [0], color=SBD_COLOR, lw=2, ls='-', label=rf'SBDART: $k$={k_sbd_os:.2f}'),
        Line2D([0], [0], color=RFOV_COLOR, lw=2, ls='-',
               label=rf'RFOV: $k$={k_rfov:.2f}'),
        Line2D([0], [0], color=GRID_COLOR, lw=2, ls='-',
               label=rf'Grid: $k$={k_grid_os:.2f}'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7,
              framealpha=0.5, handlelength=1.2)


def make_asymmetric_logit_yerr(alb_mean, alb_std):
    """Convert Ac-space std to asymmetric logit-space yerr."""
    alb_mean = np.asarray(alb_mean, dtype=float)
    alb_std = np.asarray(alb_std, dtype=float)

    alb_low = np.clip(alb_mean - alb_std, 1e-6, 1 - 1e-6)
    alb_high = np.clip(alb_mean + alb_std, 1e-6, 1 - 1e-6)
    alb_mean_clip = np.clip(alb_mean, 1e-6, 1 - 1e-6)

    y_mean = albedo_to_y(alb_mean_clip)
    y_low = albedo_to_y(alb_low)
    y_high = albedo_to_y(alb_high)

    return np.vstack([y_mean - y_low, y_high - y_mean])


def draw_ocean_season_linear_panel(ax, sub, ocean, season_name, bin_edges, fov_sub=None):
    """Draw the same curves in linearized space:
    x = ln(COT), y = ln[Ac/(1-Ac)].
    """
    n_pts = len(sub)
    if n_pts < 5:
        ax.text(0.5, 0.5, 'Insufficient data',
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        return

    x_range = cot_to_x(cot_range)

    # T91
    alb_t91 = cot_to_albedo(cot_range, 'quadrature', sza=54.4)
    k_t91, lnb_t91, _, _ = mc_fit(
        cot_range, alb_t91, calculate_uncertainty=False
    )
    y_t91 = albedo_to_y(alb_t91)
    y_t91_fit = k_t91 * x_range + lnb_t91


    # SBDART and RFOV use RFOV COT bins.
    if fov_sub is not None and len(fov_sub) >= 5:
        rfov_cot_bins, rfov_albedo_bins, rfov_albedo_std = bin_data_by_cot(
            fov_sub, 'cot_fov', 'ret_albedo', bin_edges
        )
        fov_sbd = fov_sub.copy()
        fov_sbd['sbd_albedo'] = cot_to_albedo(
            fov_sbd['cot_fov'].values, 'sbdart',
            sza=fov_sbd['solar_zenith'].values,
            table_folder='cp', ocean=ocean, season=season_name
        )
        sbd_cot_bins, sbd_alb_bins, sbd_alb_std = bin_data_by_cot(
            fov_sbd, 'cot_fov', 'sbd_albedo', bin_edges
        )
        k_sbd_os, lnb_sbd_os, _, _ = mc_fit(
            sbd_cot_bins, sbd_alb_bins,
            cot_std=0.0, albedo_std=0.03, n_mc=300, bootstrap=True
        )
        k_rfov, lnb_rfov, _, _ = mc_fit(
            rfov_cot_bins, rfov_albedo_bins,
            cot_std=0.10, albedo_std=0.20, n_mc=300, bootstrap=True
        )
    else:
        rfov_cot_bins = rfov_albedo_bins = rfov_albedo_std = np.array([])
        sbd_cot_bins = sbd_alb_bins = sbd_alb_std = np.array([])
        k_sbd_os = lnb_sbd_os = k_rfov = lnb_rfov = np.nan

    y_sbd_fit = k_sbd_os * x_range + lnb_sbd_os

    # Grid
    k_grid_os, lnb_grid_os, _, _ = mc_fit(
        sub['cot'].values, sub['albedo'].values,
        cot_std=0.10, albedo_std=0.20, n_mc=300, bootstrap=True
    )
    y_grid_fit = k_grid_os * x_range + lnb_grid_os
    grid_cot_bins, grid_alb_bins, grid_alb_std = bin_data_by_cot(
        sub, 'cot', 'albedo', bin_edges
    )

    # Plot T91
    ax.plot(x_range, y_t91, color=T91_COLOR, lw=1.2, ls='-')
    ax.plot(x_range, y_t91_fit, color=T91_COLOR, lw=1, ls='--', alpha=0.7)

    # Plot SBDART
    ax.errorbar(
        cot_to_x(sbd_cot_bins),
        albedo_to_y(sbd_alb_bins),
        yerr=make_asymmetric_logit_yerr(sbd_alb_bins, sbd_alb_std),
        color=SBD_COLOR, fmt='o-', lw=1, ms=2.5, capsize=2, capthick=0.6
    )
    ax.plot(x_range, y_sbd_fit, color=SBD_COLOR, lw=1, ls='--', alpha=0.7)

    # Plot RFOV
    ax.errorbar(
        cot_to_x(rfov_cot_bins), albedo_to_y(rfov_albedo_bins),
        yerr=make_asymmetric_logit_yerr(rfov_albedo_bins, rfov_albedo_std),
        color=RFOV_COLOR, fmt='o-', lw=1, ms=2.5, capsize=2, capthick=0.6
    )
    ax.plot(x_range, k_rfov * x_range + lnb_rfov,
            color=RFOV_COLOR, lw=1, ls='--')

    # Plot Grid
    ax.errorbar(
        cot_to_x(grid_cot_bins),
        albedo_to_y(grid_alb_bins),
        yerr=make_asymmetric_logit_yerr(grid_alb_bins, grid_alb_std),
        color=GRID_COLOR, fmt='s-', lw=1, ms=2.5, capsize=2, capthick=0.6
    )
    ax.plot(x_range, y_grid_fit, color=GRID_COLOR, lw=1, ls='--', alpha=0.7)

    ax.tick_params(axis='both', labelsize=7)

    legend_elements = [
        Line2D([0], [0], color=T91_COLOR, lw=2, ls='-',
               label=rf'T91: $k$={k_t91:.2f}'),
        Line2D([0], [0], color=SBD_COLOR, lw=2, ls='-', label=rf'SBDART: $k$={k_sbd_os:.2f}'),
        Line2D([0], [0], color=RFOV_COLOR, lw=2, ls='-',
               label=rf'RFOV: $k$={k_rfov:.2f}'),
        Line2D([0], [0], color=GRID_COLOR, lw=2, ls='-',
               label=rf'Grid: $k$={k_grid_os:.2f}'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7,
              framealpha=0.5, handlelength=1.2)


def make_linear_figure(df, bin_edges, fov_df):
    """Create linearized figure: x = ln(COT), y = ln[Ac/(1-Ac)]."""
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

            fov_sub = fov_df[(fov_df['ocean'] == ocean) & (fov_df['season'] == season_name)]
            draw_ocean_season_linear_panel(ax, sub, ocean, season_name, bin_edges, fov_sub)

            if j == 0:
                ax.set_ylabel(r'$\ln[A_{\mathrm{c}}/(1-A_{\mathrm{c}})]$', fontsize=9)

            if i == 0:
                ax.set_title(season_name, fontsize=10, fontweight='bold')

            if i == n_rows - 1:
                ax.set_xlabel(r'$\ln(\mathrm{COT})$', fontsize=8)
            else:
                ax.set_xlabel('')

    fig.canvas.draw()
    for i, ocean in enumerate(oceans):
        ax_pos = axes[i, 0].get_position()
        y_center = (ax_pos.y0 + ax_pos.y1) / 2
        fig.text(
            -0.014, y_center,
            ocean,
            fontsize=10, fontweight='bold',
            rotation=90, va='center', ha='center'
        )

    out_path = os.path.join(FIG_DIR, 'figsupp_ocean_season_linear.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')

def make_figure(df, bin_edges, fov_df):
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

            fov_sub = fov_df[(fov_df['ocean'] == ocean) & (fov_df['season'] == season_name)]
            draw_ocean_season_panel(ax, sub, ocean, season_name, bin_edges, fov_sub)

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
    df['sbd_albedo'] = compute_sbdart_albedo_per_point(df, 'cp')

    bin_edges = cot_range
    fov_df = load_fov_ocean_season_data()

    # ---- Compute per-ocean-season fits and save to CSV ----
    os_records = compute_per_ocean_season_fits(df, fov_df)
    os_df = pd.DataFrame(os_records)
    os_df = os_df.sort_values(['Ocean', 'Season']).reset_index(drop=True)
    os_df.to_csv(SENSITIVITY_CSV_PATH, index=False)
    print(f'Saved per-ocean-season fits to: {SENSITIVITY_CSV_PATH}')

    # ---- Create single figure: 8 rows × 4 columns, no gaps ----
    make_figure(df, bin_edges, fov_df)

    # ---- Create linearized figure: ln(COT) vs ln[Ac/(1-Ac)] ----
    make_linear_figure(df, bin_edges, fov_df)

    print('All done.')


if __name__ == '__main__':
    main()
