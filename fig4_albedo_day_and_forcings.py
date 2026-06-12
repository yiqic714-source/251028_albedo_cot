import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from utils_fitting import (
    oceans, season_dict, cot_range, cot_to_albedo,
    cot_to_x, albedo_to_y, mc_fit, format_panel_tag
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

# Colors for the 5 lines (order: t91, ret, msk, ret_dt, msk_dt)
LINE_COLORS = [plt.cm.tab10(0), plt.cm.tab10(0.75), plt.cm.tab10(1), plt.cm.tab10(0.75), plt.cm.tab10(1)]
LINE_STYLES = ['-', '--', '--', '-', '-']
LINE_LABELS = [
    r'$k_{\mathrm{T91}}$=',
    r'$k_{\mathrm{ret,1030}}$=',
    r'$k_{\mathrm{msk,1030}}$=',
    r'$k_{\mathrm{ret,day}}$=',
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


def logit_fit_to_albedo(cot, k, b):
    x = cot_to_x(np.asarray(cot, dtype=float))
    y = k * x + b
    return np.exp(y) / (1 + np.exp(y))


# ============================================================
# Solar geometry (from cal_daytime_mean_sza.py)
# ============================================================

def declination(day_of_year):
    return math.radians(23.45) * math.sin(2 * math.pi * (284 + day_of_year) / 365.0)


def hourly_sza(lat_deg, doy, hour):
    """Calculate SZA for a given latitude, day of year, and hour (0-23)."""
    phi = math.radians(lat_deg)
    delta = declination(doy)
    h_angle = math.radians((hour - 12) * 15)
    cos_sza = (math.sin(phi) * math.sin(delta) +
               math.cos(phi) * math.cos(delta) * math.cos(h_angle))
    return math.degrees(math.acos(np.clip(cos_sza, -1, 1)))


def get_daytime_sza(lat_deg, doy, max_sza=70):
    """Return array of hourly SZA values < max_sza for the given lat and doy."""
    sza_list = []
    for hour in range(24):
        s = hourly_sza(lat_deg, doy, hour)
        if s < max_sza:
            sza_list.append(s)
    return np.array(sza_list)


# ============================================================
# Daytime-adjusted fit computation
# ============================================================

def compute_daytime_fit_data(df):
    """
    Compute daytime-adjusted ret and msk fit lines.
    
    For each ocean-season group, for each pixel:
      1. Compute albedo_cp_1030 using df's sza
      2. For each daytime hour (SZA < 70°), compute albedo_cp_hr
      3. ratio_cp = albedo_cp_hr / albedo_cp_1030
      4. albedo_ret_hr = ret_albedo * ratio_cp
      5. Collect all (ret_cot_cer, albedo_ret_hr) points for fitting
    
    Same for msk using cot_mod08 and albedo.
    
    Saves results to npz file for reuse.
    """
    print('Computing daytime-adjusted fit data (this may take a while)...')
    
    ret_cot_list = []
    ret_alb_list = []
    msk_cot_list = []
    msk_alb_list = []

    for ocean in oceans:
        for season in season_dict.keys():
            mask = (df['ocean'] == ocean) & (df['season'] == season)
            sub = df[mask].copy()
            if len(sub) == 0:
                continue

            # Deduplicate by (lat, time) for hourly SZA computation
            lat_time = sub[['lat', 'time']].drop_duplicates()
            lat_time['doy'] = pd.to_datetime(lat_time['time']).dt.dayofyear

            # Build lookup: (lat, time) -> daytime SZA array
            sza_cache = {}
            for _, row in lat_time.iterrows():
                key = (row['lat'], row['time'])
                sza_cache[key] = get_daytime_sza(row['lat'], row['doy'])

            print(f'  Processing {ocean}_{season}: {len(sub)} points')

            for idx, pixel in sub.iterrows():
                key = (pixel['lat'], pixel['time'])
                hr_szas = sza_cache.get(key, np.array([]))
                if len(hr_szas) == 0:
                    continue

                ret_cot = pixel['ret_cot_cer']
                cot_msk = pixel['cot_mod08']
                ret_alb = pixel['ret_albedo']
                msk_alb = pixel['albedo']
                sza_1030 = pixel['sza']

                # albedo_cp_1030 for ret and msk
                alb_cp_1030_ret = cot_to_albedo(
                    np.array([ret_cot]), 'sbdart', sza=np.array([sza_1030]),
                    table_folder='cp', ocean=ocean, season=season
                )[0]
                alb_cp_1030_msk = cot_to_albedo(
                    np.array([cot_msk]), 'sbdart', sza=np.array([sza_1030]),
                    table_folder='cp', ocean=ocean, season=season
                )[0]

                if not np.isfinite(alb_cp_1030_ret) or not np.isfinite(alb_cp_1030_msk):
                    continue

                # For each daytime hour
                for sza_hr in hr_szas:
                    alb_cp_hr_ret = cot_to_albedo(
                        np.array([ret_cot]), 'sbdart', sza=np.array([sza_hr]),
                        table_folder='cp', ocean=ocean, season=season
                    )[0]
                    alb_cp_hr_msk = cot_to_albedo(
                        np.array([cot_msk]), 'sbdart', sza=np.array([sza_hr]),
                        table_folder='cp', ocean=ocean, season=season
                    )[0]

                    if not np.isfinite(alb_cp_hr_ret) or not np.isfinite(alb_cp_hr_msk):
                        continue

                    ratio_ret = alb_cp_hr_ret / alb_cp_1030_ret
                    ratio_msk = alb_cp_hr_msk / alb_cp_1030_msk

                    alb_ret_hr = ret_alb * ratio_ret
                    alb_msk_hr = msk_alb * ratio_msk

                    ret_cot_list.append(ret_cot)
                    ret_alb_list.append(alb_ret_hr)
                    msk_cot_list.append(cot_msk)
                    msk_alb_list.append(alb_msk_hr)

    print(f'  Generated {len(ret_cot_list)} ret points and {len(msk_cot_list)} msk points')

    # Fit
    print('  Fitting ret daytime...')
    k_ret_dt, b_ret_dt, _, _ = mc_fit(
        np.array(ret_cot_list), np.array(ret_alb_list),
        cot_std=0.10, albedo_std=0.13, n_mc=300, bootstrap=True
    )

    print('  Fitting msk daytime...')
    k_msk_dt, b_msk_dt, _, _ = mc_fit(
        np.array(msk_cot_list), np.array(msk_alb_list),
        cot_std=0.10, albedo_std=0.20, n_mc=300, bootstrap=True
    )

    # Generate fit lines
    alb_ret_dt_fit = logit_fit_to_albedo(cot_range, k_ret_dt, b_ret_dt)
    alb_msk_dt_fit = logit_fit_to_albedo(cot_range, k_msk_dt, b_msk_dt)

    # Save
    np.savez(FIT_DATA_PATH,
             cot_range=cot_range,
             alb_ret_dt_fit=alb_ret_dt_fit,
             alb_msk_dt_fit=alb_msk_dt_fit,
             k_ret_dt=k_ret_dt, b_ret_dt=b_ret_dt,
             k_msk_dt=k_msk_dt, b_msk_dt=b_msk_dt)
    print(f'  Saved fit data to {FIT_DATA_PATH}')

    return alb_ret_dt_fit, alb_msk_dt_fit, k_ret_dt, k_msk_dt


# ============================================================
# Panel (a): pcolor of mean lookup table
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


def draw_pcolor(ax, sza_grid, cot_grid, albedo_mean, count):
    """Draw pcolor plot on given axes."""
    pcm = ax.pcolor(
        cot_grid, sza_grid, albedo_mean,
        shading='auto',
        cmap='viridis',
    )
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 70)
    ax.set_xlabel('COT', fontsize=13)
    ax.set_ylabel('SZA (deg)', fontsize=13)
    return pcm


# ============================================================
# Panel (b): 5 fit lines (t91, ret, msk, ret_dt, msk_dt)
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
        alb_ret_dt_fit, alb_msk_dt_fit, k_ret_dt, k_msk_dt = compute_daytime_fit_data(df)
    else:
        print(f'Loading saved fit data from {FIT_DATA_PATH}')
        data = np.load(FIT_DATA_PATH)
        alb_ret_dt_fit = data['alb_ret_dt_fit']
        alb_msk_dt_fit = data['alb_msk_dt_fit']
        k_ret_dt = float(data['k_ret_dt'])
        k_msk_dt = float(data['k_msk_dt'])

    # --- Plot all 5 lines ---
    fit_curves = [alb_t91_fit, alb_ret_fit, alb_msk_fit, alb_ret_dt_fit, alb_msk_dt_fit]
    k_values = [k_t91, k_ret, k_msk, k_ret_dt, k_msk_dt]

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
    # ---- Panel (a): pcolor ----
    print('Computing mean lookup table...')
    sza_grid, cot_grid, albedo_mean, count = compute_mean_lookup_table()
    if count == 0:
        print('No lookup tables found!')
        return
    print(f'Averaged {count} ocean-season lookup tables.')

    # ---- Create figure ----
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(wspace=0.35)

    # Panel (a): pcolor
    pcm = draw_pcolor(ax_a, sza_grid, cot_grid, albedo_mean, count)
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
