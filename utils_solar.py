"""
Utility functions for solar geometry and SZA-based daytime adjustments.

Provides:
- Solar geometry: declination, hourly_sza, get_daytime_sza
- Daily-mean SWdown: E_ext, sunset_hour_angle, H0_daily_mean, calc_monthly_swdown
- Grid cell area: calc_grid_cell_area
- Daytime-adjusted fit computation: compute_daytime_fit_data
- Per ocean-season CSV export: compute_and_save_per_ocean_season
"""

import os
import math
from datetime import date, timedelta
import numpy as np
import pandas as pd

from utils_fitting import oceans, season_dict, cot_range, cot_to_albedo, cot_to_x, mc_fit

BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIT_DATA_PATH = f'{BASE_PATH}/processed_data/fig4_panel_b_fit_data.npz'
SENSITIVITY_DAY_CSV = f'{BASE_PATH}/processed_data/sensitivity_albedo_vs_cot_day.csv'

# Physical constants
S0 = 1361.0  # Solar constant (W/m2)
R_EARTH = 6371000  # Earth radius (meters)
M2_TO_KM2 = 1e6


# ============================================================
# Solar geometry (from cal_daytime_mean_sza.py)
# ============================================================

def declination(day_of_year):
    """Solar declination in radians."""
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
# Daily-mean SWdown (from fig4_albedo_sensitivity_and_irf_erf.py)
# ============================================================

def E_ext(n):
    """Extraterrestrial solar irradiance (W/m2) from day of year."""
    return S0 * (1 + 0.033 * math.cos(2 * math.pi * n / 365.0))


def sunset_hour_angle(phi, delta):
    """Sunset hour angle (radians) from latitude and declination."""
    x = -math.tan(phi) * math.tan(delta)
    return 0.0 if x >= 1 else math.pi if x <= -1 else math.acos(x)


def H0_daily_mean(phi, n):
    """Daily-mean TOA solar radiation (W/m2)."""
    delta, E = declination(n), E_ext(n)
    omega_s = sunset_hour_angle(phi, delta)
    return (E / math.pi) * (math.cos(phi)*math.cos(delta)*math.sin(omega_s) + omega_s*math.sin(phi)*math.sin(delta))


def calc_monthly_swdown(lat, year=2020, month=None):
    """Monthly mean daily-mean SWdown for a given latitude."""
    if pd.isna(lat) or pd.isna(month) or month is None:
        return np.nan
    try:
        month = int(month)
        lat = float(lat)
        year = int(year)
    except (ValueError, TypeError):
        return np.nan
    if not (1 <= month <= 12):
        return np.nan
    start = date(year, month, 1)
    end = date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    phi = math.radians(lat)
    vals = [H0_daily_mean(phi, d.timetuple().tm_yday) for d in pd.date_range(start, end)]
    return float(np.mean(vals)) if vals else np.nan


def calc_grid_cell_area(lat, lon_res=1.0, lat_res=1.0):
    """Grid cell area (km2) from latitude and resolution."""
    lat1, lat2 = math.radians(lat - lat_res/2), math.radians(lat + lat_res/2)
    dlon = math.radians(lon_res)
    area_m2 = dlon * (math.sin(lat2) - math.sin(lat1)) * (R_EARTH ** 2)
    return area_m2 / M2_TO_KM2


def cot_k_b_to_albedo(cot, k, b):
    """Corrected albedo: Ac = b * cot^k / (1 + b * cot^k)"""
    return b * cot ** k / (1 + b * cot ** k)


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
    
    Also computes per-ocean-season fits and saves to CSV.
    
    Returns
    -------
    alb_ret_day_fit, alb_msk_day_fit, k_ret_day, k_msk_day
    """
    print('Computing daytime-adjusted fit data (this may take a while)...')
    
    ret_cot_list = []
    ret_alb_list = []
    msk_cot_list = []
    msk_alb_list = []

    # Per ocean-season results
    per_os_records = []

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

            os_ret_cot = []
            os_ret_alb = []
            os_msk_cot = []
            os_msk_alb = []

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

                    # Global lists
                    ret_cot_list.append(ret_cot)
                    ret_alb_list.append(alb_ret_hr)
                    msk_cot_list.append(cot_msk)
                    msk_alb_list.append(alb_msk_hr)

                    # Per ocean-season lists
                    os_ret_cot.append(ret_cot)
                    os_ret_alb.append(alb_ret_hr)
                    os_msk_cot.append(cot_msk)
                    os_msk_alb.append(alb_msk_hr)

            # Fit per ocean-season
            if len(os_ret_cot) >= 5:
                k_ret_os, b_ret_os, k_ret_unc, b_ret_unc = mc_fit(
                    np.array(os_ret_cot), np.array(os_ret_alb),
                    cot_std=0.10, albedo_std=0.13, n_mc=300, bootstrap=True
                )
            else:
                k_ret_os, b_ret_os, k_ret_unc, b_ret_unc = np.nan, np.nan, np.nan, np.nan

            if len(os_msk_cot) >= 5:
                k_msk_os, b_msk_os, k_msk_unc, b_msk_unc = mc_fit(
                    np.array(os_msk_cot), np.array(os_msk_alb),
                    cot_std=0.10, albedo_std=0.20, n_mc=300, bootstrap=True
                )
            else:
                k_msk_os, b_msk_os, k_msk_unc, b_msk_unc = np.nan, np.nan, np.nan, np.nan

            per_os_records.append({
                'Ocean': ocean, 'Season': season,
                'k_ret_day': k_ret_os, 'b_ret_day': b_ret_os,
                'k_ret_day_unc': k_ret_unc, 'b_ret_day_unc': b_ret_unc,
                'k_msk_day': k_msk_os, 'b_msk_day': b_msk_os,
                'k_msk_day_unc': k_msk_unc, 'b_msk_day_unc': b_msk_unc,
            })

    print(f'  Generated {len(ret_cot_list)} ret points and {len(msk_cot_list)} msk points')

    # Fit global (all ocean-season combined)
    print('  Fitting ret daytime (global)...')
    k_ret_day, b_ret_day, _, _ = mc_fit(
        np.array(ret_cot_list), np.array(ret_alb_list),
        cot_std=0.10, albedo_std=0.13, n_mc=300, bootstrap=True
    )

    print('  Fitting msk daytime (global)...')
    k_msk_day, b_msk_day, _, _ = mc_fit(
        np.array(msk_cot_list), np.array(msk_alb_list),
        cot_std=0.10, albedo_std=0.20, n_mc=300, bootstrap=True
    )

    # Generate fit lines
    alb_ret_day_fit = cot_k_b_to_albedo(cot_range, k_ret_day, b_ret_day)
    alb_msk_day_fit = cot_k_b_to_albedo(cot_range, k_msk_day, b_msk_day)

    # Save global fit data
    np.savez(FIT_DATA_PATH,
             cot_range=cot_range,
             alb_ret_day_fit=alb_ret_day_fit,
             alb_msk_day_fit=alb_msk_day_fit,
             k_ret_day=k_ret_day, b_ret_day=b_ret_day,
             k_msk_day=k_msk_day, b_msk_day=b_msk_day)
    print(f'  Saved global fit data to {FIT_DATA_PATH}')

    # Save per ocean-season CSV
    os_df = pd.DataFrame(per_os_records)
    os_df = os_df.sort_values(['Ocean', 'Season']).reset_index(drop=True)
    os_df.to_csv(SENSITIVITY_DAY_CSV, index=False)
    print(f'  Saved per ocean-season fits to {SENSITIVITY_DAY_CSV}')

    return alb_ret_day_fit, alb_msk_day_fit, k_ret_day, k_msk_day
