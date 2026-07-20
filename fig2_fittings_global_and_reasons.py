# -*- coding: utf-8 -*-
"""
fig2_reason.py

Layout:
  Row 1 (ax1, spans 2 cols): fig2_global_5curves plot (5 curves), legend on right outside
  Row 2 (ax1, ax2): fig3_bias_attribution panels a and b (COT vs albedo lines)
  Row 3 (ax3, ax4): fig3_bias_attribution panels c and d (boxplots)
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from utils_fitting import (
    oceans, season_dict, cot_range, albedo_to_y,
    cot_to_x, cot_to_albedo, mc_fit, format_panel_tag, cot_k_b_to_albedo
)


BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIG_SAVE_PATH = f'{BASE_PATH}/figs/fig2_reason.png'
os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)

MIN_COT = 2.5
MIN_CF = 0.1

# ============================================================
# Unified color scheme
# Panel (a) five main colors:
#   T91: black
#   DCP: green  (swapped with CP as requested)
#   CP : blue   (swapped with DCP as requested)
#   RET: orange-yellow
#   MSK: red-brown
# Panels (b)/(c) reuse the same colors whenever the same physical line appears.
# Auxiliary decomposition lines use high-contrast colors not used by panel (a).
# ============================================================
T91_COLOR = '#222222'
DCP_COLOR = '#00bfff'
CP_COLOR  = '#574cff'
RET_COLOR = '#ff852e'
MSK_COLOR = '#f20d38'

AUX_VS_COLOR = "#606581"
AUX_SURFACE_COLOR = "#F354F3"
AUX_GAS_COLOR = "#31B704"
AUX_SZA_COLOR = "#025D37"

# ============================================================
# Shared helper functions (from fig2_global_5curves.py)
# ============================================================

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


def compute_sbdart_albedo_fixed_sza(df, table_folder, sza=54.74):
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


# ============================================================
# Boxplot helpers (from fig3_bias_attribution.py)
# ============================================================

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
    df['ocean'] = ocean

    df['albedo'] = (
        (df['sw_all'] - df['sw_clr'] * (1 - df['cf_ceres'])) /
        df['cf_ceres'] / df['solar_incoming']
    )

    df['cot_disp'] = df['ret_cotstd_cer'] / df['ret_cot_cer']
    df['unr_fra'] = df['cf_ceres'] - df['cf_ret_liq_mod08']

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
        ocean_val = bin_df['ocean'].iloc[0]
        season_val = bin_df['season'].iloc[0]
        albedo_sbd = cot_to_albedo(
            ret_cot, 'sbdart', sza=bin_df['sza'].values, table_folder='cp',
            ocean=ocean_val, season=season_val
        )

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


def calc_low_high_delta_k(df, split_col, mode, x2, n_bins=2):
    labels, edges = split_data_by_percentile(df.copy(), split_col, n_bins)

    bin_values = {}
    for idx in range(len(edges) - 1):
        bin_df = df[labels == idx].copy()
        if len(bin_df) < 5:
            continue

        bin_values[idx] = calc_slope_diff_for_bin(bin_df, mode, x2)

    low = bin_values.get(0, np.nan)
    high = bin_values.get(1, np.nan)

    return low, high


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
                'cot_disp_delta_k': calc_low_high_delta_k(df, 'cot_disp', 'cot_disp', x2, n_bins),
                'unr_fra_delta_k': calc_low_high_delta_k(df, 'unr_fra', 'unr_fra', x2, n_bins),
                'aod_cot_delta_k': calc_low_high_delta_k(df, 'aod_mod08', 'cot_disp', x2, n_bins),
                'aod_unr_delta_k': calc_low_high_delta_k(df, 'aod_mod08', 'unr_fra', x2, n_bins),
            })

    return records


def draw_delta_k_boxplot(ax, data, labels, ylabel):
    box_edge_colors = ['#8B0000', '#8B0000', '#003366', '#003366']
    box_face_colors = ['#F6D1D1', '#F08A8A', '#D8E8F7', '#79AEDD']

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.45,
        showfliers=False
    )

    for box, edge_color, face_color in zip(bp['boxes'], box_edge_colors, box_face_colors):
        box.set_facecolor(face_color)
        box.set_edgecolor(edge_color)
        box.set_linewidth(2)
        box.set_alpha(0.72)

    for median, color in zip(bp['medians'], box_edge_colors):
        median.set_color(color)
        median.set_linewidth(2)

    for i, whisker in enumerate(bp['whiskers']):
        whisker.set_color(box_edge_colors[i // 2])
        whisker.set_linewidth(1.5)

    for i, cap in enumerate(bp['caps']):
        cap.set_color(box_edge_colors[i // 2])
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

    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='x', labelsize=11.5)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.45)


def get_paired_delta_k_values(records, key):
    low_vals = []
    high_vals = []

    for record in records:
        low, high = record[key]
        if np.isfinite(low) and np.isfinite(high):
            low_vals.append(low)
            high_vals.append(high)

    return np.asarray(low_vals, dtype=float), np.asarray(high_vals, dtype=float)


def calc_paired_wilcoxon_p(low_vals, high_vals):
    if len(low_vals) < 2:
        return np.nan

    diff = high_vals - low_vals
    if np.allclose(diff, 0.0, equal_nan=False):
        return 1.0

    try:
        _, p_value = wilcoxon(low_vals, high_vals, zero_method='wilcox', alternative='two-sided')
    except ValueError:
        return np.nan

    return p_value


def format_p_value(p_value):
    if not np.isfinite(p_value):
        return 'p=n/a'
    if p_value < 0.001:
        return 'p<0.001'
    return f'p={p_value:.3f}'


def add_p_value_annotation(ax, x1, x2, data1, data2, p_value):
    values = np.concatenate([
        np.asarray(data1, dtype=float),
        np.asarray(data2, dtype=float)
    ])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    if y_range <= 0:
        y_range = 1.0

    y = max(np.nanmax(values), y_max - 0.18 * y_range) + 0.04 * y_range
    h = 0.025 * y_range
    text_y = y + h + 0.015 * y_range

    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        color='0.25',
        lw=1.0,
        clip_on=False
    )
    ax.text(
        (x1 + x2) / 2,
        text_y,
        format_p_value(p_value),
        ha='center',
        va='bottom',
        fontsize=9.5,
        color='0.15'
    )
    ax.set_ylim(y_min, max(y_max, text_y + 0.06 * y_range))


def prepare_global_5curves_data(verbose=True, include_simulations=True):
    if verbose:
        print('Loading global data for 5-curve panel...')
    df = load_global_data()
    bin_edges = cot_range

    if verbose:
        print('Binning observational data for relationship panel...')
    ret_cot_bins, ret_alb_bins, ret_alb_std = bin_data_by_cot(
        df, 'ret_cot_cer', 'ret_albedo', bin_edges
    )

    msk_cot_bins, msk_alb_bins, msk_alb_std = bin_data_by_cot(
        df, 'cot_mod08', 'albedo', bin_edges
    )

    alb_t91 = cot_to_albedo(cot_range, 'quadrature', sza=54.74)

    if verbose:
        print('Fitting k values for relationship panel...')
    k_t91, lnb_t91 = fit_k_b_in_logit_space(cot_range, alb_t91)

    k_ret, lnb_ret, _, _ = mc_fit(
        df['ret_cot_cer'].values,
        df['ret_albedo'].values,
        cot_std=0.10,
        albedo_std=0.13,
        n_mc=300,
        bootstrap=True
    )

    k_msk, lnb_msk, _, _ = mc_fit(
        df['cot_mod08'].values,
        df['albedo'].values,
        cot_std=0.10,
        albedo_std=0.20,
        n_mc=300,
        bootstrap=True
    )

    panel_data = {
        'df': df,
        'alb_t91': alb_t91,
        'ret_cot_bins': ret_cot_bins,
        'ret_alb_bins': ret_alb_bins,
        'ret_alb_std': ret_alb_std,
        'msk_cot_bins': msk_cot_bins,
        'msk_alb_bins': msk_alb_bins,
        'msk_alb_std': msk_alb_std,
        'alb_t91_fit': cot_k_b_to_albedo(cot_range, k_t91, np.exp(lnb_t91)),
        'alb_ret_fit': cot_k_b_to_albedo(cot_range, k_ret, np.exp(lnb_ret)),
        'alb_msk_fit': cot_k_b_to_albedo(cot_range, k_msk, np.exp(lnb_msk)),
        'k_t91': k_t91,
        'k_ret': k_ret,
        'k_msk': k_msk,
    }

    if include_simulations:
        if verbose:
            print('Computing SBDART simulation data for relationship panel...')
        alb_dcp = cot_to_albedo(df['ret_cot_cer'], 'sbdart', sza=54.74, table_folder='dcp')
        df['cp_albedo'] = compute_sbdart_albedo_per_point(df, 'cp')

        cp_cot_bins, cp_alb_bins, cp_alb_std = bin_data_by_cot(
            df, 'ret_cot_cer', 'cp_albedo', bin_edges
        )

        k_dcp, lnb_dcp = fit_k_b_in_logit_space(df['ret_cot_cer'], alb_dcp)
        k_cp, lnb_cp, _, _ = mc_fit(
            df['ret_cot_cer'].values,
            df['cp_albedo'].values,
            cot_std=0.0,
            albedo_std=0.03,
            n_mc=300,
            bootstrap=True
        )

        panel_data.update({
            'alb_dcp': alb_dcp,
            'cp_cot_bins': cp_cot_bins,
            'cp_alb_bins': cp_alb_bins,
            'cp_alb_std': cp_alb_std,
            'alb_dcp_fit': cot_k_b_to_albedo(cot_range, k_dcp, np.exp(lnb_dcp)),
            'alb_cp_fit': cot_k_b_to_albedo(cot_range, k_cp, np.exp(lnb_cp)),
            'k_dcp': k_dcp,
            'k_cp': k_cp,
        })

    return panel_data


def draw_global_5curves_panel(
    ax,
    panel_data,
    icon_style='nature',
    tag_index=0,
    tag_fontsize=17,
    axis_label_fontsize=15,
    tick_labelsize=12,
    legend_fontsize=10.5,
    legend_anchor=(1.15, 0.5),
    include_simulations=True,
):
    df = panel_data['df']

    solid_labels = [
        'T91',
        'Ret Obs.',
        'Msk Obs.'
    ]

    dashed_labels = [
        rf'$k_{{\mathrm{{T91}}}}$={panel_data["k_t91"]:.2f}',
        rf'$k_{{\mathrm{{ret}}}}$={panel_data["k_ret"]:.2f}',
        rf'$k_{{\mathrm{{msk}}}}$={panel_data["k_msk"]:.2f}'
    ]

    if include_simulations:
        solid_labels[1:1] = ['Dcp Simu.', 'Cp Simu.']
        dashed_labels[1:1] = [
            rf'$k_{{\mathrm{{dcp}}}}$={panel_data["k_dcp"]:.2f}',
            rf'$k_{{\mathrm{{cp}}}}$={panel_data["k_cp"]:.2f}'
        ]

    solid_handles = []
    dashed_handles = []

    h, = ax.plot(cot_range, panel_data['alb_t91'], color=T91_COLOR, lw=2, ls='-')
    solid_handles.append(h)
    h, = ax.plot(cot_range, panel_data['alb_t91_fit'], color=T91_COLOR, lw=1.5, ls='--', alpha=0.7)
    dashed_handles.append(h)

    if include_simulations:
        sorted_idx_a = np.argsort(df['ret_cot_cer'])
        alb_dcp = panel_data['alb_dcp']
        h, = ax.plot(df['ret_cot_cer'].values[sorted_idx_a], alb_dcp[sorted_idx_a], color=DCP_COLOR, lw=2, ls='-')
        solid_handles.append(h)
        h, = ax.plot(cot_range, panel_data['alb_dcp_fit'], color=DCP_COLOR, lw=1.5, ls='--', alpha=0.7)
        dashed_handles.append(h)

        h = ax.errorbar(
            panel_data['cp_cot_bins'], panel_data['cp_alb_bins'], yerr=panel_data['cp_alb_std'],
            color=CP_COLOR, fmt='o-', lw=1.5, ms=3.5, capsize=3, capthick=0.8
        )
        solid_handles.append(h)
        h, = ax.plot(cot_range, panel_data['alb_cp_fit'], color=CP_COLOR, lw=1.5, ls='--', alpha=0.7)
        dashed_handles.append(h)

    h = ax.errorbar(
        panel_data['ret_cot_bins'], panel_data['ret_alb_bins'], yerr=panel_data['ret_alb_std'],
        color=RET_COLOR, fmt='o-', lw=1.5, ms=3.5, capsize=3, capthick=0.8
    )
    solid_handles.append(h)
    h, = ax.plot(cot_range, panel_data['alb_ret_fit'], color=RET_COLOR, lw=1.5, ls='--', alpha=0.7)
    dashed_handles.append(h)

    h = ax.errorbar(
        panel_data['msk_cot_bins'], panel_data['msk_alb_bins'], yerr=panel_data['msk_alb_std'],
        color=MSK_COLOR, fmt='s-', lw=1.5, ms=3.5, capsize=3, capthick=0.8
    )
    solid_handles.append(h)
    h, = ax.plot(cot_range, panel_data['alb_msk_fit'], color=MSK_COLOR, lw=1.5, ls='--', alpha=0.7)
    dashed_handles.append(h)

    ax.set_xlabel('COT', fontsize=axis_label_fontsize, fontweight='medium')
    ax.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=axis_label_fontsize, fontweight='medium')
    ax.tick_params(axis='both', labelsize=tick_labelsize)
    ax.set_xlim(0, 60)

    ax.legend(
        solid_handles + dashed_handles,
        solid_labels + dashed_labels,
        loc='center left',
        bbox_to_anchor=legend_anchor,
        fontsize=legend_fontsize,
        framealpha=0.9,
        ncol=2,
        columnspacing=0.4,
        labelspacing=1.5
    )

    ax.text(-0.03, 1.01, f'{format_panel_tag(tag_index, icon_style)}',
            transform=ax.transAxes, fontsize=tag_fontsize, va='bottom', ha='left')


# ============================================================
# Main plotting function
# ============================================================

def main(icon_style='nature'):
    print('Loading global data...')
    df = load_global_data()
    print(f'Total data points: {len(df)}')

    bin_edges = cot_range

    # ---- Compute data for panels (a)-(d): bias attribution (from fig3_bias_attribution) ----
    print('Computing bias attribution data...')
    season_records = process_all_oceans_by_season(n_bins=2)

    cot_disp_low = [r['cot_disp_delta_k'][0] for r in season_records if np.isfinite(r['cot_disp_delta_k'][0])]
    cot_disp_high = [r['cot_disp_delta_k'][1] for r in season_records if np.isfinite(r['cot_disp_delta_k'][1])]
    aod_cot_low = [r['aod_cot_delta_k'][0] for r in season_records if np.isfinite(r['aod_cot_delta_k'][0])]
    aod_cot_high = [r['aod_cot_delta_k'][1] for r in season_records if np.isfinite(r['aod_cot_delta_k'][1])]

    unr_fra_low = [r['unr_fra_delta_k'][0] for r in season_records if np.isfinite(r['unr_fra_delta_k'][0])]
    unr_fra_high = [r['unr_fra_delta_k'][1] for r in season_records if np.isfinite(r['unr_fra_delta_k'][1])]
    aod_unr_low = [r['aod_unr_delta_k'][0] for r in season_records if np.isfinite(r['aod_unr_delta_k'][0])]
    aod_unr_high = [r['aod_unr_delta_k'][1] for r in season_records if np.isfinite(r['aod_unr_delta_k'][1])]

    cot_disp_pair_low, cot_disp_pair_high = get_paired_delta_k_values(season_records, 'cot_disp_delta_k')
    aod_cot_pair_low, aod_cot_pair_high = get_paired_delta_k_values(season_records, 'aod_cot_delta_k')
    unr_fra_pair_low, unr_fra_pair_high = get_paired_delta_k_values(season_records, 'unr_fra_delta_k')
    aod_unr_pair_low, aod_unr_pair_high = get_paired_delta_k_values(season_records, 'aod_unr_delta_k')

    p_cot_disp = calc_paired_wilcoxon_p(cot_disp_pair_low, cot_disp_pair_high)
    p_aod_cot = calc_paired_wilcoxon_p(aod_cot_pair_low, aod_cot_pair_high)
    p_unr_fra = calc_paired_wilcoxon_p(unr_fra_pair_low, unr_fra_pair_high)
    p_aod_unr = calc_paired_wilcoxon_p(aod_unr_pair_low, aod_unr_pair_high)

    # ---- Compute data for panel (b): fig3 panel a (3 lines) ----
    print('Computing SBDART comparison data for panel (b)...')
    alb_sbd = cot_to_albedo(df['ret_cot_cer'], 'sbdart', sza=54.5, table_folder='dcp')
    y_fit = albedo_to_y(alb_sbd)
    mask = np.isfinite(y_fit)
    k_sbd, _ = np.polyfit(cot_to_x(df['ret_cot_cer'])[mask], y_fit[mask], 1)

    alb_mono = cot_to_albedo(df['ret_cot_cer'], 'sbdart', sza=54.5, table_folder='dcp_0p4to0p7')
    y_fit = albedo_to_y(alb_mono)
    mask = np.isfinite(y_fit)
    k_mono, _ = np.polyfit(cot_to_x(df['ret_cot_cer'])[mask], y_fit[mask], 1)

    alb_quad = cot_to_albedo(df['ret_cot_cer'], 'quadrature', sza=54.5)
    y_fit = albedo_to_y(alb_quad)
    mask = np.isfinite(y_fit)
    k_quad, _ = np.polyfit(cot_to_x(df['ret_cot_cer'])[mask], y_fit[mask], 1)

    # ---- Compute data for panel (c): fig3 panel b (multiple lines with errorbars) ----
    print('Computing coupling decomposition data for panel (c)...')
    sorted_idx = np.argsort(df['ret_cot_cer'])

    # --- Lines 2-3: Fixed sza=54.74, per-point cot, with errorbar ---
    lookup_folders_fixed_sza = ['gasdcp_surcp', 'surdcp_gascp']
    lookup_labels_fixed_sza = [
        r'$A_{\mathrm{sfc}}$ Coupled: $k$=',
        r'Gas Coupled: $k$='
    ]
    lookup_colors_fixed_sza = [AUX_SURFACE_COLOR, AUX_GAS_COLOR]

    # Store results for panel (c)
    panel_c_lines = []

    for idx_offset, folder in enumerate(lookup_folders_fixed_sza):
        print(f'  Computing {folder} with fixed sza=54.74...')
        alb_vals = compute_sbdart_albedo_fixed_sza(df, folder, sza=54.74)
        df[f'alb_{folder}'] = alb_vals

        cot_bins, alb_bins, alb_std = bin_data_by_cot(
            df, 'ret_cot_cer', f'alb_{folder}', bin_edges
        )

        k_val, b_val, _, _ = mc_fit(
            df['ret_cot_cer'].values,
            alb_vals,
            cot_std=0.0,
            albedo_std=0.03,
            n_mc=300,
            bootstrap=True
        )

        panel_c_lines.append({
            'cot_bins': cot_bins,
            'alb_bins': alb_bins,
            'alb_std': alb_std,
            'color': lookup_colors_fixed_sza[idx_offset],
            'label': f'{lookup_labels_fixed_sza[idx_offset]}{k_val:.2f}'
        })

    # --- Lines 4-5: Per-point sza, with errorbar ---
    lookup_folders_sza = ['dcp', 'cp']
    lookup_labels_sza = [
        r'SZA Coupled: $k$=',
        r'Coupled: $k_{\mathrm{cp}}=$'
    ]
    lookup_colors_sza = [AUX_SZA_COLOR, CP_COLOR]

    for idx_offset, folder in enumerate(lookup_folders_sza):
        print(f'  Computing {folder} with per-point sza...')
        alb_vals = compute_sbdart_albedo_per_point(df, folder)
        df[f'alb_{folder}_persza'] = alb_vals

        cot_bins, alb_bins, alb_std = bin_data_by_cot(
            df, 'ret_cot_cer', f'alb_{folder}_persza', bin_edges
        )

        k_val, b_val, _, _ = mc_fit(
            df['ret_cot_cer'].values,
            alb_vals,
            cot_std=0.0,
            albedo_std=0.03,
            n_mc=300,
            bootstrap=True
        )

        panel_c_lines.append({
            'cot_bins': cot_bins,
            'alb_bins': alb_bins,
            'alb_std': alb_std,
            'color': lookup_colors_sza[idx_offset],
            'label': f'{lookup_labels_sza[idx_offset]}{k_val:.2f}'
        })

    # ================================================================
    # Create figure: 2 rows, 2 columns.
    # Original panels (b)-(e) are now panels (a)-(d).
    # ================================================================
    print('Plotting...')
    fig = plt.figure(figsize=(10, 7.5), dpi=300)

    gs = fig.add_gridspec(
        2, 2,
        hspace=0.30, wspace=0.30,
        bottom=0.08, top=0.95,
        left=0.07, right=0.95
    )

    ax1 = fig.add_subplot(gs[0, 0])  # Panel (a): fig3 panel a (3 lines)
    ax2 = fig.add_subplot(gs[0, 1])  # Panel (b): fig3 panel b (coupling decomposition)
    ax3 = fig.add_subplot(gs[1, 0])  # Panel (c): boxplot 1
    ax4 = fig.add_subplot(gs[1, 1])  # Panel (d): boxplot 2

    # ================================================================
    # Panel (a): fig3 panel a (3 lines: Quadrature, Sbdart VS, Sbdart SW)
    # ================================================================
    ax1.plot(
        df['ret_cot_cer'].values[sorted_idx], alb_quad[sorted_idx],
        color=T91_COLOR, lw=2,
        label=rf'Analytical: $k_\mathrm{{T91}}$={k_quad:.2f}'
    )
    ax1.plot(
        df['ret_cot_cer'].values[sorted_idx], alb_mono[sorted_idx],
        color=AUX_VS_COLOR, lw=2, linestyle='--',
        label='Simu. VIS'
    )
    ax1.plot(
        df['ret_cot_cer'].values[sorted_idx], alb_sbd[sorted_idx],
        color=DCP_COLOR, lw=2,
        label=rf'Simu. SW: $k_\mathrm{{dcp}}$={k_sbd:.2f}'
    )
    ax1.set_xlim(0, 60)
    ax1.set_xlabel('COT', fontsize=14, fontweight='medium')
    ax1.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=14, fontweight='medium')
    ax1.tick_params(axis='both', labelsize=11)
    ax1.text(-0.01, 1.01, f'{format_panel_tag(0, icon_style)}',
             transform=ax1.transAxes, fontsize=17, va='bottom', ha='left')
    ax1.set_title('T91 vs. Dcp Simu.', fontsize=14, loc='center', pad=4.5)
    ax1.legend(loc='lower right', fontsize=10.5, framealpha=0.9)

    # ================================================================
    # Panel (b): fig3 panel b (coupling decomposition lines)
    # ================================================================
    # Line 1: Decoupled SBDART (dcp) with fixed sza=54.74
    ax2.plot(
        df['ret_cot_cer'].values[sorted_idx], alb_sbd[sorted_idx],
        color=DCP_COLOR, lw=2,
        label=rf'Decoupled: $k_{{\mathrm{{dcp}}}}$={k_sbd:.2f}'
    )

    for line_data in panel_c_lines[0:3]:
        ax2.errorbar(
            line_data['cot_bins'], line_data['alb_bins'], yerr=line_data['alb_std'],
            color=line_data['color'], fmt='o-', lw=1.3, ms=3.5, capsize=2.6, capthick=0.8, alpha=0.7,
            label=line_data['label']
        )

    line_data = panel_c_lines[3] 
    ax2.errorbar(
        line_data['cot_bins'], line_data['alb_bins'], yerr=line_data['alb_std'],
        color=line_data['color'], fmt='o-', lw=1.3, ms=3.5, capsize=2.6, capthick=0.8,
        label=line_data['label']
    )

    ax2.set_xlim(0, 60)
    ax2.set_xlabel('COT', fontsize=14, fontweight='medium')
    ax2.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=14, fontweight='medium')
    ax2.tick_params(axis='both', labelsize=11)
    ax2.text(-0.01, 1.01, f'{format_panel_tag(1, icon_style)}',
             transform=ax2.transAxes, fontsize=17, va='bottom', ha='left')
    ax2.set_title('Dcp Simu. vs. Cp Simu.', fontsize=14, loc='center', pad=4.5)
    ax2.legend(loc='lower right', fontsize=10.5, framealpha=0.5)

    # ================================================================
    # Panel (c): fig3 panel c (boxplot: cot_disp vs aod_cot)
    # ================================================================
    draw_delta_k_boxplot(
        ax3,
        data=[cot_disp_low, cot_disp_high, aod_cot_low, aod_cot_high],
        labels=[
            r'Low $d_{\mathrm{COT}}$',
            r'High $d_{\mathrm{COT}}$',
            'Low AOD',
            'High AOD'
        ],
        ylabel=r'$k_{\mathrm{cp}}-k_{\mathrm{ret}}$'
    )
    add_p_value_annotation(ax3, 1, 2, cot_disp_low, cot_disp_high, p_cot_disp)
    add_p_value_annotation(ax3, 3, 4, aod_cot_low, aod_cot_high, p_aod_cot)
    ax3.text(-0.01, 1.01, f'{format_panel_tag(2, icon_style)}',
             transform=ax3.transAxes, fontsize=17, va='bottom', ha='left')
    ax3.set_title('Cp Simu. vs. Ret Obs.', fontsize=14, loc='center', pad=4.5)

    # ================================================================
    # Panel (d): fig3 panel d (boxplot: unr_fra vs aod_unr)
    # ================================================================
    draw_delta_k_boxplot(
        ax4,
        data=[unr_fra_low, unr_fra_high, aod_unr_low, aod_unr_high],
        labels=[
            'Low URF',
            'High URF',
            'Low AOD',
            'High AOD'
        ],
        ylabel=r'$k_{\mathrm{ret}}-k_{\mathrm{msk}}$'
    )
    add_p_value_annotation(ax4, 1, 2, unr_fra_low, unr_fra_high, p_unr_fra)
    add_p_value_annotation(ax4, 3, 4, aod_unr_low, aod_unr_high, p_aod_unr)
    ax4.text(-0.01, 1.01, f'{format_panel_tag(3, icon_style)}',
             transform=ax4.transAxes, fontsize=17, va='bottom', ha='left')
    ax4.set_title('Ret Obs. vs. Msk Obs.', fontsize=14, loc='center', pad=4.5)

    # Save figure
    plt.savefig(FIG_SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Figure saved to: {os.path.abspath(FIG_SAVE_PATH)}')


if __name__ == '__main__':
    main(icon_style='nature')

