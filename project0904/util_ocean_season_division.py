"""
util_ocean_season_division.py

Utility module for splitting data by ocean region and season.

Provides:
  - oceans_def, oceans, season_dict: definitions
  - normalize_lon(lon)
  - is_in_ocean(lat, lon, bounds_list)
  - get_season_from_month(month)
  - split_by_ocean_season(df, output_dir, time_col='time')
      Splits df by ocean region and season, saves {ocean}_{season}.csv files.
"""

import os
import pandas as pd
import numpy as np

# ============================================================
# Ocean definitions
# ============================================================

oceans_def = {
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

oceans = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']

season_dict = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}


# ============================================================
# Helper functions
# ============================================================

def normalize_lon(lon):
    """Normalize longitude to [-180, 180)."""
    return ((lon + 180) % 360) - 180


def is_in_ocean(lat, lon, bounds_list):
    """
    Determine if coordinates are within the provided list of bounds.
    bounds_list: list of [west, south, east, north] boxes
    lon: will be normalized before comparison
    """
    lon_n = normalize_lon(lon)
    for bound in bounds_list:
        min_lon, min_lat, max_lon, max_lat = bound
        min_lon_n = normalize_lon(min_lon)
        max_lon_n = normalize_lon(max_lon)

        # latitude check
        if not (lat >= min_lat and lat <= max_lat):
            continue

        # longitude check with possible wrap
        if min_lon_n <= max_lon_n:
            if lon_n >= min_lon_n and lon_n <= max_lon_n:
                return True
        else:
            # wrapped: lon in [min_lon_n, 180] or [-180, max_lon_n]
            if lon_n >= min_lon_n or lon_n <= max_lon_n:
                return True
    return False


def get_season_from_month(month):
    """Get season name from month number."""
    for season_name, months in season_dict.items():
        if month in months:
            return season_name
    return None


# ============================================================
# Main splitting function
# ============================================================

def split_by_ocean_season(df, output_dir, time_col='time'):
    """
    Split a DataFrame by ocean region and season, and save to {ocean}_{season}.csv files.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Must contain 'lat', 'lon' columns, and a time column.
    output_dir : str
        Directory to save the output CSV files.
    time_col : str
        Name of the time column (default 'time').

    Returns
    -------
    dict
        Summary of saved files: {(ocean, season): n_rows}
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add month and season columns (temporary)
    df = df.copy()
    df['month'] = pd.to_datetime(df[time_col]).dt.month
    df['season'] = df['month'].apply(get_season_from_month)

    summary = {}

    for ocean in oceans:
        bounds = oceans_def[ocean]
        # Filter by ocean region
        ocean_mask = df.apply(
            lambda row: is_in_ocean(row['lat'], row['lon'], bounds), axis=1
        )
        df_ocean = df[ocean_mask]

        if df_ocean.empty:
            print(f"  {ocean}: no data")
            continue

        for season_name in season_dict.keys():
            df_season = df_ocean[df_ocean['season'] == season_name]
            if df_season.empty:
                continue

            # Drop helper columns before saving
            df_season = df_season.drop(columns=['month', 'season'], errors='ignore')

            output_csv = os.path.join(output_dir, f'{ocean}_{season_name}.csv')
            df_season.to_csv(output_csv, index=False)
            summary[(ocean, season_name)] = len(df_season)
            print(f"  Saved {ocean}_{season_name}.csv: {len(df_season)} rows")

    return summary
