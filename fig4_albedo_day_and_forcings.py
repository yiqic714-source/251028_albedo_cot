import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from utils_fitting import (
    oceans, season_dict, cot_range, cot_to_albedo,
    cot_to_x, albedo_to_y, mc_fit, format_panel_tag
)
from utils_sza_adjustments import compute_daytime_fit_data, logit_fit_to_albedo

# Paths
BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
TABLE_FOLDER = 'cp'  # coupled SBDART lookup tables (per ocean-season)
TABLE_DIR = f'{BASE_PATH}/build_sbdart_lookup_table/cot_sza_to_albedo_lookup_table_{TABLE_FOLDER}'
FIG_DIR = f'{BASE_PATH}/figs'
FIT_DATA_PATH = f'{BASE_PATH}/processed_data/fig4_panel_b_fit_data.npz'
os.makedirs(FIG_DIR, exist_ok=True)

MIN_COT = 2.5
MIN_CF = 0.1

# Colors for the 5 lines (order: t91, ret, ret_day, msk, msk_day)
colors = plt.cm.tab10(np.linspace(0, 1, 5))
LINE_COLORS = [colors[0], colors[3], colors[3], colors[4], colors[4]]
LINE_STYLES = ['-', '--', '-', '--', '-']
LINE_LABELS = [
    r'$k_{\mathrm{T91}}$=',
    r'$k_{\mathrm{ret,1030}}$=',
    r'$k_{\mathrm{ret,day}}$=',
    r'$k_{\mathrm{msk,1030}}$=',
    r'$k_{\mathrm{msk,day}}$=',
]


# ============================================================
# Helper functions (from fig2_reason.py)
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


# ============================================================
# Panel (a): contourf of mean lookup table
# ============================================================

def read_lookup_table(ocean, season):
    """Read the cot-sza-to-albedo lookup table for a given ocean and season."""
    file_name = f'cot_sza_to_albedo_lookup_table_{ocean}_{season}.csv'
    file_path = os.path.join(TABLE_DIR, file_name)

    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path, index_col=0)
    sza_grid = np.array(df.index, dtype=float)
    cot_grid = np.array(df.columns, dtype=float)
    albedo_grid = df.values

    return sza_grid, cot_grid, albedo_grid


def compute_mean_lookup_table():
    """Average lookup tables across all ocean-season combinations."""
    albedo_sum = None
    count = 0
    common_sza_grid = None
    common_cot_grid = None

    for ocean in oceans:
        for season in season_dict.keys():
            result = read_lookup_table(ocean, season)
            if result is None:
                continue

            sza_grid, cot_grid, albedo_grid = result

            if common_sza_grid is None:
                common_sza_grid = sza_grid
                common_cot_grid = cot_grid
                albedo_sum = np.zeros_like(albedo_grid)
            elif not (np.array_equal(sza_grid, common_sza_grid) and
                      np.array_equal(cot_grid, common_cot_grid)):
                continue

            albedo_sum += albedo_grid
            count += 1

    if count == 0:
        return None, None, None, 0

    albedo_mean = albedo_sum / count
    return common_sza_grid, common_cot_grid, albedo_mean, count


def draw_contourf(ax, sza_grid, cot_grid, albedo_mean, count):
    """Draw contourf plot on given axes."""
    levels = np.linspace(0, 1, 21)
    pcm = ax.contourf(
        cot_grid, sza_grid, albedo_mean,
        levels=levels,
        cmap='viridis',
        extend='both',
    )
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 70)
    ax.set_xlabel('COT', fontsize=13)
    ax.set_ylabel('SZA (deg)', fontsize=13)
    return pcm


# ============================================================
# Panel (b): 5 fit lines (t91, ret, ret_day, msk, msk_day)
# ============================================================

def draw_fit_lines(ax, recompute=False):
    """Draw t91, ret, msk, and daytime-adjusted ret/msk fit lines."""
    print('Loading global data for panel (b)...')
    df = load_global_data()
    print(f'Total data points: {len(df)}')

    # --- T91 (quadrature) ---
    alb_t91 = cot_to_albedo(cot_range, 'quadrature', sza=54.4)
    x_t91 = cot_to_x(cot_range)
    y_t91 = albedo_to_y(alb_t91)
    mask_t91 = np.isfinite(x_t91) & np.isfinite(y_t91)
    k_t91, b_t91 = np.polyfit(x_t91[mask_t91], y_t91[mask_t91], 1)
    alb_t91_fit = logit_fit_to_albedo(cot_range, k_t91, b_t91)

    # --- ret (retrieval-domain obs.) ---
    k_ret, b_ret, _, _ = mc_fit(
        df['ret_cot_cer'].values,
        df['ret_albedo'].values,
        cot_std=0.10,
        albedo_std=0.13,
        n_mc=300,
        bootstrap=True
    )
    alb_ret_fit = logit_fit_to_albedo(cot_range, k_ret, b_ret)

    # --- msk (mask-domain obs.) ---
    k_msk, b_msk, _, _ = mc_fit(
        df['cot_mod08'].values,
        df['albedo'].values,
        cot_std=0.10,
        albedo_std=0.20,
        n_mc=300,
        bootstrap=True
    )
    alb_msk_fit = logit_fit_to_albedo(cot_range, k_msk, b_msk)

    # --- Daytime-adjusted ret and msk ---
    if recompute or not os.path.exists(FIT_DATA_PATH):
        alb_ret_day_fit, alb_msk_day_fit, k_ret_day, k_msk_day = compute_daytime_fit_data(df)
    else:
        print(f'Loading saved fit data from {FIT_DATA_PATH}')
        data = np.load(FIT_DATA_PATH)
        alb_ret_day_fit = data['alb_ret_day_fit']
        alb_msk_day_fit = data['alb_msk_day_fit']
        k_ret_day = float(data['k_ret_day'])
        k_msk_day = float(data['k_msk_day'])

    # --- Plot all 5 lines ---
    fit_curves = [alb_t91_fit, alb_ret_fit, alb_ret_day_fit, alb_msk_fit, alb_msk_day_fit]
    k_values = [k_t91, k_ret, k_ret_day, k_msk, k_msk_day]

    for i in range(5):
        ax.plot(cot_range, fit_curves[i],
                color=LINE_COLORS[i], lw=2, ls=LINE_STYLES[i],
                label=rf'{LINE_LABELS[i]}{k_values[i]:.2f}')

    ax.set_xlim(0, 60)
    ax.set_xlabel('COT', fontsize=13)
    ax.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=13)
    ax.legend(loc='lower right', fontsize=10.5, framealpha=0.9)


# ============================================================
# Main
# ============================================================

def main(recompute=False):
    # ---- Panel (a): contourf ----
    print('Computing mean lookup table...')
    sza_grid, cot_grid, albedo_mean, count = compute_mean_lookup_table()
    if count == 0:
        print('No lookup tables found!')
        return
    print(f'Averaged {count} ocean-season lookup tables.')

    # ---- Create figure ----
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(wspace=0.35)

    # Panel (a): contourf
    pcm = draw_contourf(ax_a, sza_grid, cot_grid, albedo_mean, count)
    cbar = fig.colorbar(pcm, ax=ax_a, label='$A_\mathrm{c}$')
    ax_a.text(-0.01, 1.01, format_panel_tag(0, 'nature'),
              transform=ax_a.transAxes, fontsize=17, va='bottom', ha='left')

    # Panel (b): fit lines
    draw_fit_lines(ax_b, recompute=recompute)
    ax_b.text(-0.01, 1.01, format_panel_tag(1, 'nature'),
              transform=ax_b.transAxes, fontsize=17, va='bottom', ha='left')

    out_path = os.path.join(FIG_DIR, 'fig4_albedo_day_and_forcings.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    import sys
    recompute = '--recompute' in sys.argv
    main(recompute=recompute)
