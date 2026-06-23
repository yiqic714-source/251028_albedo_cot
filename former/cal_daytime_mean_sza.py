# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from datetime import timedelta
from netCDF4 import Dataset
import warnings
warnings.filterwarnings("ignore")

# Configuration
LAT_STEP = 1.0
LAT_MIN = -60.0
LAT_MAX = 60.0
LAT_START = LAT_MIN + 0.5
LAT_END = LAT_MAX - 0.5
LON_STEP = 1.0
YEAR = 2020
SEASON_MAP = {1: 'DJF', 2: 'DJF', 12: 'DJF',
              3: 'MAM', 4: 'MAM', 5: 'MAM',
              6: 'JJA', 7: 'JJA', 8: 'JJA',
              9: 'SON', 10: 'SON', 11: 'SON'}
SEASONS = ['DJF', 'MAM', 'JJA', 'SON']

S0 = 1361.0
HOUR_CONV = 12.0 / math.pi

LSMASK_PATH = "/data/chenyiqi/251007_tropic/landsea.nc"
OUTPUT_OCEAN_CSV = "ocean_season_sza_weighted.csv"

# Ocean rectangles
OCEANS = {
    'NPO': [
        [-170, 20, -100, 60],
        [-180, 20, -170, 60],
        [105, 20, 180, 60]
    ],
    'NAO': [
        [-100, 55, 45, 60],
        [-100, 40, 27, 55],
        [-100, 30, 45, 40],
        [-100, 20, 30, 30]
    ],
    'TPO': [
        [-170, 16, -100, 20],
        [-170, 13, -89, 16],
        [-170, 9, -84, 13],
        [-170, -20, -70, 9],
        [100, 0, 180, 20],
        [130, -20, 180, 0],
        [-180, -20, -170, 20]
    ],
    'TAO': [
        [-100, 16, -15, 20],
        [-84, 9, -13, 16],
        [-60, -20, 15, 9]
    ],
    'TIO': [
        [30, 0, 100, 30],
        [30, -20, 130, 0]
    ],
    'SPO': [
        [-170, -60, -70, -20],
        [130, -60, 180, -20],
        [-180, -60, -170, -20]
    ],
    'SAO': [
        [-70, -60, 20, -20]
    ],
    'SIO': [
        [20, -60, 130, -20]
    ]
}

# Solar geometry utilities
def declination(day_of_year):
    return math.radians(23.45) * math.sin(2 * math.pi * (284 + day_of_year) / 365.0)

def sunset_hour_angle(phi, delta):
    x = -math.tan(phi) * math.tan(delta)
    if x >= 1.0:
        return 0.0
    if x <= -1.0:
        return math.pi
    return math.acos(x)

def H0_daily_mean(phi, doy):
    delta = declination(doy)
    E = S0 * (1 + 0.033 * math.cos(2 * math.pi * doy / 365.0))
    omega_s = sunset_hour_angle(phi, delta)
    return (E / math.pi) * (math.cos(phi) * math.cos(delta) * math.sin(omega_s) +
                            omega_s * math.sin(phi) * math.sin(delta))

def calc_daily_sunshine_hours(phi, doy):
    delta = declination(doy)
    omega_s = sunset_hour_angle(phi, delta)
    return 2.0 * omega_s * HOUR_CONV

# Compute seasonal means (swdown, sunshine) for a latitude
def calc_seasonal_swdown_and_sunshine(lat):
    phi = math.radians(lat)
    season_data = {s: {'swdown': [], 'sunshine': []} for s in SEASONS}
    for day in pd.date_range(start=f'{YEAR}-01-01', end=f'{YEAR}-12-31'):
        season = SEASON_MAP[day.month]
        doy = day.timetuple().tm_yday
        swdown = H0_daily_mean(phi, doy)
        sunshine = calc_daily_sunshine_hours(phi, doy)
        season_data[season]['swdown'].append(swdown)
        season_data[season]['sunshine'].append(sunshine)
    season_mean = {}
    for s in SEASONS:
        season_mean[s] = (np.nanmean(season_data[s]['swdown']),
                          np.nanmean(season_data[s]['sunshine']))
    return season_mean

# Read land-sea mask and prepare lon array 0..360 sorted
def read_landsea_mask():
    with Dataset(LSMASK_PATH, 'r') as ds:
        lon = ds.variables['lon'][:].astype(np.float32)
        lat = ds.variables['lat'][:].astype(np.float32)
        lsmask = ds.variables['LSMASK'][:].astype(np.int32)
    lon360 = np.where(lon < 0.0, lon + 360.0, lon)
    order = np.argsort(lon360)
    lon_sorted = lon360[order]
    lsmask_sorted = lsmask[:, order]
    return lon_sorted, lat, lsmask_sorted

def lon180_to_360_scalar(lon):
    return lon + 360.0 if lon < 0 else lon

# Count ocean pixels at a given latitude for given ocean rectangles
def get_ocean_pixel_count(lat, ocean_ranges, lon_array, lat_array, lsmask):
    lat_idx_arr = np.where(np.isclose(lat_array, lat, atol=1e-6))[0]
    if len(lat_idx_arr) == 0:
        return 0
    lat_idx = lat_idx_arr[0]
    sea_mask_row = (lsmask[lat_idx, :] == 0)
    total = 0
    for lon_w, lat_s, lon_e, lat_n in ocean_ranges:
        if not (lat_s - 1e-9 < lat <= lat_n + 1e-9):
            continue
        lw = lon180_to_360_scalar(lon_w)
        le = lon180_to_360_scalar(lon_e)
        if lw <= le:
            lon_idx = np.where((lon_array >= lw - 1e-9) & (lon_array <= le + 1e-9))[0]
        else:
            lon_idx = np.where((lon_array >= lw - 1e-9) | (lon_array <= le + 1e-9))[0]
        if len(lon_idx) == 0:
            continue
        total += np.sum(sea_mask_row[lon_idx])
    return int(total)

# Count total longitude grid cells within ocean rectangles (no land/sea check)
def calc_lon_count_in_ocean(lat, ocean_ranges, lon_array):
    total = 0
    for lon_w, lat_s, lon_e, lat_n in ocean_ranges:
        if not (lat_s - 1e-9 < lat <= lat_n + 1e-9):
            continue
        lw = lon180_to_360_scalar(lon_w)
        le = lon180_to_360_scalar(lon_e)
        if lw <= le:
            lon_idx = np.where((lon_array >= lw - 1e-9) & (lon_array <= le + 1e-9))[0]
        else:
            lon_idx = np.where((lon_array >= lw - 1e-9) | (lon_array <= le + 1e-9))[0]
        total += len(lon_idx)
    return int(total)

# For each ocean and season, compute weighted mean(swdown) and mean(sunshine) using ocean pixel counts,
# then compute ratio = (swdown_mean * 24) / (sunshine_mean * S0) and arccos -> degrees.
def calc_ocean_season_weighted_angle(lat_season_df, oceans_dict, lon_array, lat_array, lsmask):
    results = []
    for ocean_name, ranges in oceans_dict.items():
        for season in SEASONS:
            df_season = lat_season_df[lat_season_df['season'] == season].copy()
            if df_season.empty:
                results.append({'ocean': ocean_name, 'season': season,
                                'weighted_angle_deg': np.nan,
                                'total_ocean_pixels': 0, 'total_pixels': 0})
                continue
            df_season['ocean_pixels'] = df_season['lat'].apply(
                lambda x: get_ocean_pixel_count(x, ranges, lon_array, lat_array, lsmask)
            )
            df_season['total_pixels'] = df_season['lat'].apply(
                lambda x: calc_lon_count_in_ocean(x, ranges, lon_array)
            )
            valid = df_season[df_season['ocean_pixels'] > 0].copy()
            if valid.empty:
                results.append({'ocean': ocean_name, 'season': season,
                                'weighted_angle_deg': np.nan,
                                'total_ocean_pixels': 0,
                                'total_pixels': int(df_season['total_pixels'].sum())})
                continue
            total_ocean_pix = int(valid['ocean_pixels'].sum())
            # weighted averages
            swdown_weighted = (valid['swdown'] * valid['ocean_pixels']).sum() / total_ocean_pix
            sunshine_weighted = (valid['sunshine'] * valid['ocean_pixels']).sum() / total_ocean_pix
            if np.isnan(swdown_weighted) or np.isnan(sunshine_weighted) or sunshine_weighted < 1e-12:
                weighted_angle_deg = np.nan
            else:
                ratio = (swdown_weighted * 24.0) / (sunshine_weighted * S0)
                ratio_clipped = float(np.clip(ratio, -1.0, 1.0))
                weighted_angle_deg = round(math.degrees(math.acos(ratio_clipped)), 4)
            total_pix = int(valid['total_pixels'].sum())
            results.append({'ocean': ocean_name, 'season': season,
                            'weighted_angle_deg': weighted_angle_deg,
                            'total_ocean_pixels': total_ocean_pix,
                            'total_pixels': total_pix})
    res_df = pd.DataFrame(results)
    res_df['season'] = pd.Categorical(res_df['season'], categories=SEASONS, ordered=True)
    return res_df.sort_values(['ocean', 'season']).reset_index(drop=True)

def main():
    lats = np.round(np.arange(LAT_START, LAT_END + LAT_STEP, LAT_STEP), 1)
    records = []
    for lat in lats:
        season_mean = calc_seasonal_swdown_and_sunshine(lat)
        for season in SEASONS:
            swdown, sunshine = season_mean[season]
            if np.isnan(swdown) or np.isnan(sunshine) or sunshine < 1e-12:
                angle_deg = np.nan
            else:
                ratio = (swdown * 24.0) / (sunshine * S0)
                ratio_clipped = float(np.clip(ratio, -1.0, 1.0))
                angle_deg = round(math.degrees(math.acos(ratio_clipped)), 4)
            records.append({'lat': lat, 'season': season,
                            'swdown': swdown, 'sunshine': sunshine, 'angle_deg': angle_deg})
    lat_season_df = pd.DataFrame(records)
    lat_season_df['season'] = pd.Categorical(lat_season_df['season'], categories=SEASONS, ordered=True)
    lat_season_df = lat_season_df.sort_values(['lat', 'season']).reset_index(drop=True)

    lon_array, lat_array, lsmask = read_landsea_mask()
    ocean_df = calc_ocean_season_weighted_angle(lat_season_df, OCEANS, lon_array, lat_array, lsmask)
    ocean_df.to_csv(OUTPUT_OCEAN_CSV, index=False)
    print(ocean_df)
    return lat_season_df, ocean_df

if __name__ == "__main__":
    lat_df, ocean_df = main()