import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_fitting import (
    oceans, season_dict, cot_range,
    mc_fit, format_panel_tag
)
from utils_solar import (
    compute_daytime_fit_data, cot_k_b_to_albedo
)

# Paths
BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
TABLE_FOLDER = 'cp'  # coupled SBDART lookup tables (per ocean-season)
TABLE_DIR = f'{BASE_PATH}/build_sbdart_lookup_table/cot_sza_to_albedo_lookup_table_{TABLE_FOLDER}'
FIG_DIR = f'{BASE_PATH}/figs'
FIT_DATA_PATH = f'{BASE_PATH}/processed_data/fig4_panel_b_fit_data.npz'
os.makedirs(FIG_DIR, exist_ok=True)

MIN_COT = 2.5
MIN_CF = 0.1
MAIN_FACE_COLOR = (1, 1, 1, 1.0)

# Colors for panel (b): order = T91, ret_1030, ret_day, msk_1030, msk_day
T91_COLOR = '#222222'
RET_DAY_COLOR = '#D49102'
MSK_DAY_COLOR = '#8B1E3F'
RET_1030_COLOR = '#ff852e'
MSK_1030_COLOR = '#f20d38'
LINE_COLORS = [T91_COLOR, RET_1030_COLOR, RET_DAY_COLOR, MSK_1030_COLOR, MSK_DAY_COLOR]
LINE_STYLES = ['-', '-', '--', '-', '--']
LINE_LABELS = [
    r'T91: $k$=',
    r'Ret ($A_{\mathrm{c,1030}}): k$=',
    r'Ret (COT$_{\mathrm{1030}}): k$=',
    r'Msk ($A_{\mathrm{c,1030}}): k$=',
    r'Msk (COT$_{\mathrm{1030}}): k$=',
]

# T91 / uncorrected parameters
k_t91 = 1.0
lnb_t91 = np.log(0.13)


def apply_main_background(fig, axes=None):
    fig.patch.set_facecolor(MAIN_FACE_COLOR)
    fig.patch.set_alpha(MAIN_FACE_COLOR[-1])

    if axes is None:
        axes = fig.axes
    elif not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    for ax in axes:
        ax.patch.set_facecolor(MAIN_FACE_COLOR)
        ax.patch.set_alpha(MAIN_FACE_COLOR[-1])


def save_png(fig, out_path, dpi=300, bbox_inches='tight'):
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        facecolor=fig.get_facecolor(),
        edgecolor='none',
        transparent=False
    )


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


def draw_contourf(ax, sza_grid, cot_grid, albedo_mean):
    levels = np.arange(0, 0.85, 0.05)
    pcm = ax.contourf(
        cot_grid, sza_grid, albedo_mean,
        levels=levels,
        cmap='Blues',
    )
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 70)
    ax.set_xlabel('COT', fontsize=13)
    ax.set_ylabel('SZA (deg)', fontsize=13)
    return pcm


# ============================================================
# Panel (b): five Ac-COT relationships
# ============================================================

def draw_fit_lines(ax, recompute=False):
    print('Loading global data for panel (b)...')
    df = load_global_data()
    print(f'Total data points: {len(df)}')

    # T91 / uncorrected
    alb_t91_fit = cot_k_b_to_albedo(cot_range, k_t91, np.exp(lnb_t91))

    # Retrieval-domain observations
    k_ret, lnb_ret, _, _ = mc_fit(
        df['ret_cot_cer'].values,
        df['ret_albedo'].values,
        cot_std=0.10,
        albedo_std=0.13,
        n_mc=300,
        bootstrap=True
    )
    alb_ret_fit = cot_k_b_to_albedo(cot_range, k_ret, np.exp(lnb_ret))

    # Mask-domain observations
    k_msk, lnb_msk, _, _ = mc_fit(
        df['cot_mod08'].values,
        df['albedo'].values,
        cot_std=0.10,
        albedo_std=0.20,
        n_mc=300,
        bootstrap=True
    )
    alb_msk_fit = cot_k_b_to_albedo(cot_range, k_msk, np.exp(lnb_msk))

    # Daytime-adjusted retrieval-domain and mask-domain relationships
    if recompute or not os.path.exists(FIT_DATA_PATH):
        alb_ret_day_fit, alb_msk_day_fit, k_ret_day, k_msk_day = compute_daytime_fit_data(df)
    else:
        print(f'Loading saved fit data from {FIT_DATA_PATH}')
        data = np.load(FIT_DATA_PATH)
        alb_ret_day_fit = data['alb_ret_day_fit']
        alb_msk_day_fit = data['alb_msk_day_fit']
        k_ret_day = float(data['k_ret_day'])
        k_msk_day = float(data['k_msk_day'])

    fit_curves = [alb_t91_fit, alb_ret_fit, alb_ret_day_fit, alb_msk_fit, alb_msk_day_fit]
    k_values = [k_t91, k_ret, k_ret_day, k_msk, k_msk_day]

    for i in range(5):
        ax.plot(
            cot_range, fit_curves[i],
            color=LINE_COLORS[i], lw=2, ls=LINE_STYLES[i],
            label=rf'{LINE_LABELS[i]}{k_values[i]:.2f}'
        )

    ax.set_xlim(0, 60)
    ax.set_xlabel('COT', fontsize=13)
    ax.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=13)
    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.9)


def main(recompute=False):
    print('Computing mean lookup table...')
    sza_grid, cot_grid, albedo_mean, count = compute_mean_lookup_table()
    if count == 0:
        print('No lookup tables found!')
        return
    print(f'Averaged {count} ocean-season lookup tables.')

    fig = plt.figure(figsize=(12, 4.8))
    apply_main_background(fig)

    gs = fig.add_gridspec(
        1, 2,
        wspace=0.35,
        bottom=0.16,
        top=0.92,
        left=0.07,
        right=0.97
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    apply_main_background(fig, [ax_a, ax_b])

    pcm = draw_contourf(ax_a, sza_grid, cot_grid, albedo_mean)
    cbar = fig.colorbar(pcm, ax=ax_a)
    cbar.set_label(r'$A_\mathrm{c,dcp}$', fontsize=13)
    ax_a.text(-0.01, 1.01, format_panel_tag(0, 'nature'),
              transform=ax_a.transAxes, fontsize=17, va='bottom', ha='left')

    draw_fit_lines(ax_b, recompute=recompute)
    ax_b.text(-0.01, 1.01, format_panel_tag(1, 'nature'),
              transform=ax_b.transAxes, fontsize=17, va='bottom', ha='left')

    out_path = os.path.join(FIG_DIR, 'figs_ac_cot_relationships.png')
    save_png(fig, out_path, dpi=300)
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    import sys
    recompute = '--recompute' in sys.argv
    main(recompute=recompute)
