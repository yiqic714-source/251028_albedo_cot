"""
2_get_L3_by_ocean_season.py

Merge of 2_get_daily_data.py and fig4_generate_lncf_lnnd_lnaod_npz.py.
Processes only 2020 data, computes L3 variables, splits by ocean and season,
and saves to: L3_product/{ocean}_{season}.csv

Variables saved (keeping raw variables instead of ln-transformed ones):
  From 2_get_daily_data.py:
    lat, lon, time
    cf_ret_tot_mod08, cf_mod08, cttmin, cf_ret_liq_mod08,
    cot_mod08, cer_mod08, cotstd_mod08, sza, aod_mod08
    cf_ceres, cf_liq_ceres, clr_fra, sw_clr, sw_all, solar_incoming

  From fig4_generate_lncf_lnnd_lnaod_npz.py:
    nd (computed from cot, cer)
    cwp (Cloud_Water_Path_Liquid_Mean)

Usage:
    python 2_get_L3_by_ocean_season.py
"""

import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc
import os
from datetime import datetime, timedelta, date
from pyhdf.SD import SD, SDC
import glob

from util_ocean_season_division import split_by_ocean_season

# ============================================================
# Constants
# ============================================================

YEAR = 2020

# Paths
LSMASK_PATH = "/data/chenyiqi/251007_tropic/landsea.nc"
CERES_DIR = '/home/chenyiqi/251028_albedo_cot/CERES_L3SSF_2020'
CERES_FILE = 'CERES_SSF1deg-Day_Terra-MODIS_Ed4.1_Subset_20200101-20201231.nc'
MOD08_DIR = '/data/MODIS/MxD08_D3'
OUTPUT_DIR = '/home/chenyiqi/251028_albedo_cot/L3_product'

# MOD08 variable names
VAR_COT = 'Cloud_Optical_Thickness_Liquid_Mean'
VAR_CER = 'Cloud_Effective_Radius_Liquid_Mean'
VAR_AOD = 'Aerosol_Optical_Depth_Land_Ocean_Mean'
VAR_CF_RET = 'Cloud_Retrieval_Fraction_Combined'
VAR_CWP = 'Cloud_Water_Path_Liquid_Mean'
VAR_CF = 'Cloud_Fraction_Day_Mean'
VAR_CF_RET_LIQ = 'Cloud_Retrieval_Fraction_Liquid'
VAR_COTSTD = 'Cloud_Optical_Thickness_Liquid_Standard_Deviation'
VAR_SZA = 'Solar_Zenith_Mean'
VAR_CTTMIN = 'Cloud_Top_Temperature_Day_Minimum'


# ============================================================
# Helper functions
# ============================================================

def read_and_mask_mod_variable(hdf, var_name):
    """Read HDF SDS and apply fill/offset/scale corrections."""
    sds = hdf.select(var_name)
    data = sds[:].astype(float)
    attrs = sds.attributes()
    fill_value = attrs.get('_FillValue', None)
    scale_factor = attrs.get('scale_factor')
    offset = attrs.get('add_offset')
    if fill_value is not None:
        data[data == fill_value] = np.nan
    if offset is not None:
        data = data - offset
    if scale_factor is not None:
        data = data * scale_factor
    return data


def compute_nd(cot, cer):
    """
    Compute cloud droplet number concentration.
    """
    return 1.37e-5 * np.power(cot, 0.5) * np.power(cer * 1e-6, -2.5) * 1e-6


def find_mod08_file_for_date(target_date):
    """Find the MOD08_D3 HDF file for a given date."""
    yyyyddd = target_date.strftime('%Y%j')
    pattern = os.path.join(MOD08_DIR, f'MOD08_D3.A{yyyyddd}.061.*.hdf')
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[0]


# ============================================================
# Main processing
# ============================================================

def main():
    year = YEAR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    time_min = np.datetime64(f'{year}-01-01')
    time_max = np.datetime64(f'{year}-12-31')
    lat_min, lat_max = -60, 60

    # ========== Land-Sea mask ==========
    print("Loading land-sea mask...")
    with nc.Dataset(LSMASK_PATH, 'r') as ds:
        lon_ls = ds.variables['lon'][:]
        lon_ls[lon_ls > 180] -= 360
        lat_ls = ds.variables['lat'][:]
        lsmask = ds.variables['LSMASK'][:]
    lat_mask_ls = (lat_ls >= lat_min) & (lat_ls <= lat_max)
    lsmask = lsmask[lat_mask_ls, :]
    lat_ls = lat_ls[lat_mask_ls]

    lon_grid_ls, lat_grid_ls = np.meshgrid(lon_ls, lat_ls)
    lon_flat_ocean = lon_grid_ls[lsmask == 0].flatten()
    lat_flat_ocean = lat_grid_ls[lsmask == 0].flatten()
    dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
    lon_repeated = np.repeat(lon_flat_ocean, len(dates))
    lat_repeated = np.repeat(lat_flat_ocean, len(dates))
    time_repeated = np.tile(dates.date, len(lon_flat_ocean))

    df_ls = pd.DataFrame({'lat': lat_repeated, 'lon': lon_repeated, 'time': time_repeated})
    print(f"Land-sea mask: {len(lon_flat_ocean)} ocean grid cells, {len(dates)} days")

    # ========== CERES daily data ==========
    print("Loading CERES daily data...")
    ceres_path = os.path.join(CERES_DIR, CERES_FILE)
    ds = xr.open_dataset(ceres_path)

    lon_cer = ds['lon'].values.copy()
    lon_cer[lon_cer > 180] -= 360
    lat_cer = ds['lat'].values
    latmask_cer = (lat_cer <= lat_max) & (lat_cer >= lat_min)
    lat_cer = lat_cer[latmask_cer]
    time_cer = ds['time'].values
    time_mask_cer = (time_cer >= time_min) & (time_cer <= time_max)

    toa_sw_clr = ds['toa_sw_clr_daily'].sel(time=time_mask_cer, lat=latmask_cer).values
    toa_sw_all = ds['toa_sw_all_daily'].sel(time=time_mask_cer, lat=latmask_cer).values
    toa_solar = ds['toa_solar_all_daily'].sel(time=time_mask_cer, lat=latmask_cer).values
    cld_fra = ds['cldarea_total_day_daily'].sel(time=time_mask_cer, lat=latmask_cer).values / 100
    cld_fra_liq = ds['cldarea_liq_total_day_daily'].sel(time=time_mask_cer, lat=latmask_cer).values / 100
    clr_fra = (ds['toa_sw_num_obs_clr_daily'] / ds['toa_sw_num_obs_all_daily']).sel(time=time_mask_cer, lat=latmask_cer).values
    time_cer = time_cer[time_mask_cer].astype('datetime64[D]')

    ds.close()

    lon_grid_cer, lat_grid_cer = np.meshgrid(lon_cer, lat_cer)
    lon_flat_cer = np.repeat(lon_grid_cer[np.newaxis, :, :], cld_fra_liq.shape[0], axis=0).flatten()
    lat_flat_cer = np.repeat(lat_grid_cer[np.newaxis, :, :], cld_fra_liq.shape[0], axis=0).flatten()
    time_flat_cer = np.repeat(time_cer[:, np.newaxis, np.newaxis],
                              cld_fra_liq.shape[1] * cld_fra_liq.shape[2], axis=1).flatten()

    df_cer = pd.DataFrame({
        'lat': lat_flat_cer, 'lon': lon_flat_cer, 'time': pd.to_datetime(time_flat_cer).date,
        'cf_ceres': cld_fra.flatten(), 'cf_liq_ceres': cld_fra_liq.flatten(), 'clr_fra': clr_fra.flatten(),
        'sw_clr': toa_sw_clr.flatten(), 'sw_all': toa_sw_all.flatten(),
        'solar_incoming': toa_solar.flatten()
    })
    print(f"CERES data loaded: {len(df_cer)} rows")

    # ========== MOD08 daily processing ==========
    print("Processing MOD08_D3 files...")
    all_mod_dfs = []

    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    current_date = start_date
    total_days = (end_date - start_date).days + 1
    day_count = 0
    missing_count = 0

    while current_date <= end_date:
        day_count += 1
        if day_count % 50 == 0:
            print(f"  Progress: day {day_count}/{total_days}")

        file_path = find_mod08_file_for_date(current_date)
        if file_path is None:
            missing_count += 1
            current_date += timedelta(days=1)
            continue

        try:
            hdf = SD(file_path, SDC.READ)

            # Read 1D coordinates
            lon_mod = hdf.select('XDim')[:].astype(float)
            lat_mod = hdf.select('YDim')[:].astype(float)

            # Read 2D variables
            cot_data = read_and_mask_mod_variable(hdf, VAR_COT)
            cer_data = read_and_mask_mod_variable(hdf, VAR_CER)
            aod_data = read_and_mask_mod_variable(hdf, VAR_AOD)
            cf_ret_data = read_and_mask_mod_variable(hdf, VAR_CF_RET)
            cwp_data = read_and_mask_mod_variable(hdf, VAR_CWP)
            cf = read_and_mask_mod_variable(hdf, VAR_CF)
            cf_ret_liq = read_and_mask_mod_variable(hdf, VAR_CF_RET_LIQ)
            cotstd_liq = read_and_mask_mod_variable(hdf, VAR_COTSTD)
            sza = read_and_mask_mod_variable(hdf, VAR_SZA)
            cttmin = read_and_mask_mod_variable(hdf, VAR_CTTMIN)

            hdf.end()

            # Compute nd
            nd = compute_nd(cot_data, cer_data)

            # Get lon/lat grid
            lon_grid_mod, lat_grid_mod = np.meshgrid(lon_mod, lat_mod)

            # Build DataFrame for this day
            df_day = pd.DataFrame({
                'lat': lat_grid_mod.flatten(),
                'lon': lon_grid_mod.flatten(),
                'time': current_date,
                'cf_ret_tot_mod08': cf_ret_data.flatten(),
                'cf_mod08': cf.flatten(),
                'cttmin': cttmin.flatten(),
                'cf_ret_liq_mod08': cf_ret_liq.flatten(),
                'cot_mod08': cot_data.flatten(),
                'cer_mod08': cer_data.flatten(),
                'cotstd_mod08': cotstd_liq.flatten(),
                'sza': sza.flatten(),
                'aod_mod08': aod_data.flatten(),
                'nd': nd.flatten(),
                'cwp': cwp_data.flatten(),
            })

            all_mod_dfs.append(df_day)

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")

        current_date += timedelta(days=1)

    print(f"MOD08 processing done: {len(all_mod_dfs)} days with valid data, "
          f"{missing_count} missing files out of {total_days} days.")

    if not all_mod_dfs:
        print("No valid MOD08 data found. Exiting.")
        return

    df_mod = pd.concat(all_mod_dfs, ignore_index=True)
    print(f"Total MOD08 rows: {len(df_mod)}")

    # ========== Merge with land-sea mask and CERES ==========
    print("Merging data...")
    merged_df = pd.merge(df_ls, df_mod, on=['time', 'lon', 'lat'], how='left')
    merged_df = pd.merge(merged_df, df_cer, on=['time', 'lon', 'lat'], how='left')
    print(f"Merged total rows: {len(merged_df)}")

    # ========== Split by ocean and season ==========
    print("Splitting by ocean and season...")
    split_by_ocean_season(merged_df, OUTPUT_DIR, time_col='time')
    print("All done!")


if __name__ == "__main__":
    main()
