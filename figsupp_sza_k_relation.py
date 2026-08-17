import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_fitting import (
    oceans, season_dict, cot_range,
    mc_fit
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
# Panel (a): ln[Ac(1-Ac)] vs. lnCOT at fixed SZA
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


def interp_albedo_at_sza(sza_grid, albedo_mean, target_sza):
    vals = np.full(albedo_mean.shape[1], np.nan)
    for j in range(albedo_mean.shape[1]):
        col = albedo_mean[:, j].astype(float)
        good = np.isfinite(sza_grid) & np.isfinite(col)
        if np.sum(good) >= 2:
            vals[j] = np.interp(target_sza, sza_grid[good], col[good])
    return vals


def draw_ac_cot_curves(ax, sza_grid, cot_grid, albedo_mean):
    sza_start = np.ceil(np.nanmin(sza_grid) / 15.0) * 15.0
    sza_stop = np.floor(np.nanmax(sza_grid) / 15.0) * 15.0
    sza_targets = np.arange(sza_start, sza_stop + 0.1, 15.0)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(sza_targets)))
    legend_labels = []

    for target_sza, color in zip(sza_targets, colors):
        albedo_vals = interp_albedo_at_sza(sza_grid, albedo_mean, target_sza)
        good = (
            np.isfinite(cot_grid) &
            np.isfinite(albedo_vals) &
            (cot_grid >= 2.5) &
            (cot_grid <= 62.5) &
            (albedo_vals > 0) &
            (albedo_vals < 1)
        )
        if not np.any(good):
            continue

        x = cot_grid[good]
        y = albedo_vals[good]
        k_val, _, _, _ = mc_fit(
            x, y,
            cot_std=0.0,
            albedo_std=0.0,
            n_mc=300,
            bootstrap=True
        )
        legend_labels.append(rf'SZA={target_sza:.0f}°: $k$={k_val:.2f}')
        ax.plot(x, y, lw=2, color=color, label=f'SZA={target_sza:.0f}°')

    for line, label in zip(ax.lines, legend_labels):
        line.set_label(label)

    ax.set_xlim(2.5, 60)
    ax.set_xlabel('COT', fontsize=13)
    ax.set_ylabel(r'$A_{\mathrm{c,cp}}$', fontsize=13)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)


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


def main():
    print('Computing mean lookup table...')
    sza_grid, cot_grid, albedo_mean, count = compute_mean_lookup_table()
    if count == 0:
        print('No lookup tables found!')
        return
    print(f'Averaged {count} ocean-season lookup tables.')

    fig = plt.figure(figsize=(5.0, 4.5))
    apply_main_background(fig)
    ax = fig.add_subplot(1, 1, 1)
    apply_main_background(fig, ax)

    draw_ac_cot_curves(ax, sza_grid, cot_grid, albedo_mean)

    out_path = os.path.join(FIG_DIR, 'figsupp_sza_k_relation.png')
    save_png(fig, out_path, dpi=300)
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
