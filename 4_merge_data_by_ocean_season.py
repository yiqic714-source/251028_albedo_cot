"""
4_merge_data_by_ocean_season.py

Merge three data sources by ocean region and season:
1. L3_product/{ocean}_{season}.csv          (remove 'cot_ceres')
2. uniform_fov_product/rsl_2020{mm}_{hemiph}.csv  (remove specified variables)
3. processed_data/cmip6_AodDiff_nat1850to1860_aer2010to2020_HadGEM3.csv  (monthly log_aod_diff)

Output: processed_data/merged_data/{ocean}_{season}.csv
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path for util import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util_ocean_season_division import oceans, season_dict

# ============================================================
# Configuration
# ============================================================
BASE_DIR = '/data/chenyiqi/251028_albedo_cot'
L3_DIR = os.path.join(BASE_DIR, 'L3_product')
UNIFORM_DIR = os.path.join(BASE_DIR, 'uniform_fov_product')
CMIP_FILE = os.path.join(BASE_DIR, 'processed_data',
                         'cmip6_AodDiff_nat1850to1860_aer2010to2020_HadGEM3.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_data', 'merged_data')

# Columns to remove from uniform_fov_product data
DROP_COLS_UNIFORM = [
    'ret_fov_fra', 'ret_albedo_uncert', 'ret_uncorrected_albedo',
    'ret_fra', 'unr_fra', 'clr_fra', 'solar_zenith', 'sensor_zenith'
]

# Column to remove from L3_product data
DROP_COLS_L3 = ['cot_ceres']


def load_cmip_data():
    """Load CMIP6 AOD difference data (monthly)."""
    cmip_df = pd.read_csv(CMIP_FILE)
    cmip_df.rename(columns={'longitude': 'lon', 'latitude': 'lat'}, inplace=True)
    return cmip_df


def load_uniform_data(year=2020):
    """Load all uniform_fov_product CSV files for the given year.
    
    Returns a single DataFrame with all months and both hemispheres.
    """
    all_dfs = []
    for mm in range(1, 13):
        for hemiph in ['east', 'west']:
            fname = f'rsl_{year}{mm:02d}_{hemiph}.csv'
            fpath = os.path.join(UNIFORM_DIR, fname)
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath)
            df['time'] = pd.to_datetime(df['time'], format='mixed')
            # Normalize to date only (for matching with L3 date-only time)
            df['time'] = df['time'].dt.normalize()
            # Drop unwanted columns (keep only those that exist)
            cols_to_drop = [c for c in DROP_COLS_UNIFORM if c in df.columns]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            all_dfs.append(df)
    
    if not all_dfs:
        raise FileNotFoundError(f"No uniform_fov_product files found in {UNIFORM_DIR}")
    
    return pd.concat(all_dfs, ignore_index=True)


def load_l3_data(ocean, season):
    """Load L3_product data for a given ocean and season."""
    fpath = os.path.join(L3_DIR, f'{ocean}_{season}.csv')
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    # Normalize to date only (for matching with uniform_fov date)
    df['time'] = df['time'].dt.normalize()
    # Drop unwanted columns
    cols_to_drop = [c for c in DROP_COLS_L3 if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df


def merge_data_for_ocean_season(ocean, season, uniform_df, cmip_df):
    """Merge L3, uniform_fov, and CMIP6 data for one ocean+season.
    
    Matching logic:
    - L3 <-> uniform_fov: match on (time, lat, lon)
    - L3 <-> CMIP6: match on (month, lat, lon) — CMIP6 is monthly, so same
      log_aod_diff applies to all days in that month.
    """
    # Load L3 data
    l3_df = load_l3_data(ocean, season)
    if l3_df is None or l3_df.empty:
        print(f"  {ocean}_{season}: no L3 data")
        return None
    
    # Merge L3 with uniform_fov on (time, lat, lon)
    merged = pd.merge(
        l3_df, uniform_df,
        on=['time', 'lat', 'lon'],
        how='left'
    )
    
    # Merge with CMIP6 on (month, lat, lon)
    # Extract month from time
    merged['month'] = merged['time'].dt.month
    
    # Round lat/lon to match CMIP6 grid (0.5-degree resolution)
    # CMIP6 data has lat/lon at 0.5-degree centers: -179.5, -178.5, ...
    # L3 data has integer lat/lon: 20.5, 21.5, ...
    # They should already match if both use 0.5-degree grid centers.
    # But to be safe, round to nearest 0.5
    merged['lat_round'] = np.round(merged['lat'] * 2) / 2
    merged['lon_round'] = np.round(merged['lon'] * 2) / 2
    
    cmip_merged = pd.merge(
        merged, cmip_df,
        left_on=['month', 'lat_round', 'lon_round'],
        right_on=['month', 'lat', 'lon'],
        how='left'
    )
    
    # Drop helper columns
    cmip_merged.drop(columns=['month', 'lat_round', 'lon_round'], inplace=True, errors='ignore')
    
    # Drop duplicate 'lat' and 'lon' columns from CMIP6 merge if they exist
    cmip_merged.drop(columns=['lat_y', 'lon_y'], inplace=True, errors='ignore')
    # Rename lat_x, lon_x back to lat, lon if needed
    if 'lat_x' in cmip_merged.columns:
        cmip_merged.rename(columns={'lat_x': 'lat', 'lon_x': 'lon'}, inplace=True)
    
    # Remove duplicate rows (can arise from overlapping east/west uniform_fov data)
    cmip_merged.drop_duplicates(inplace=True)
    
    return cmip_merged


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading CMIP6 data...")
    cmip_df = load_cmip_data()
    print(f"  CMIP6 data: {len(cmip_df)} rows")
    
    print("Loading uniform_fov_product data...")
    uniform_df = load_uniform_data(year=2020)
    print(f"  Uniform FOV data: {len(uniform_df)} rows")
    print(f"  Uniform FOV columns: {list(uniform_df.columns)}")
    
    total_saved = 0
    
    for ocean in oceans:
        for season in season_dict.keys():
            print(f"\nProcessing {ocean}_{season}...")
            result = merge_data_for_ocean_season(ocean, season, uniform_df, cmip_df)
            
            if result is None or result.empty:
                print(f"  {ocean}_{season}: no data after merge")
                continue
            
            # Save to CSV
            output_csv = os.path.join(OUTPUT_DIR, f'{ocean}_{season}.csv')
            result.to_csv(output_csv, index=False)
            n_rows = len(result)
            total_saved += n_rows
            print(f"  Saved {ocean}_{season}.csv: {n_rows} rows, columns: {list(result.columns)}")
    
    print(f"\nDone! Total rows saved: {total_saved}")


if __name__ == "__main__":
    main()
