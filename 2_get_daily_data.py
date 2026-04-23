import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc
import os
from datetime import datetime, timedelta
from pyhdf.SD import SD, SDC
import glob
import re
import sys

def read_and_mask_mod_variable(hdf, var_name):
    """read HDF variable """
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

def process_single_mod08_file(file_path, time_min, time_max, lat_min, lat_max):
    """read and process single MOD08 file"""
    file_name = os.path.basename(file_path)
    date_str = file_name.split('.A')[1].split('.')[0]
    file_date = datetime.strptime(date_str, '%Y%j').date()
    file_date_np = np.datetime64(file_date)
    if not (time_min <= file_date_np <= time_max):
        return None

    hdf = SD(file_path, SDC.READ)
    lon = hdf.select('XDim')[:]
    lat = hdf.select('YDim')[:]
    lat_mask = (lat <= lat_max) & (lat >= lat_min)
    lat = lat[lat_mask]

    cf = read_and_mask_mod_variable(hdf, 'Cloud_Fraction_Day_Mean')[lat_mask, :]
    cf_ret_liq = read_and_mask_mod_variable(hdf, 'Cloud_Retrieval_Fraction_Liquid')[lat_mask, :]
    cf_ret_tot = read_and_mask_mod_variable(hdf, 'Cloud_Retrieval_Fraction_Combined')[lat_mask, :]
    cot_liq = read_and_mask_mod_variable(hdf, 'Cloud_Optical_Thickness_Liquid_Mean')[lat_mask, :]
    cer_liq = read_and_mask_mod_variable(hdf, 'Cloud_Effective_Radius_Liquid_Mean')[lat_mask, :]
    cotstd_liq = read_and_mask_mod_variable(hdf, 'Cloud_Optical_Thickness_Liquid_Standard_Deviation')[lat_mask, :]
    sza = read_and_mask_mod_variable(hdf, 'Solar_Zenith_Mean')[lat_mask, :]
    cttmin = read_and_mask_mod_variable(hdf, 'Cloud_Top_Temperature_Day_Minimum')[lat_mask, :]
    hdf.end()

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    df = pd.DataFrame({
        'lat': lat_grid.flatten(),
        'lon': lon_grid.flatten(),
        'time': file_date,
        'cf_ret_tot_mod08': cf_ret_tot.flatten(),
        'cf_mod08': cf.flatten(),
        'cttmin': cttmin.flatten(),
        'cf_ret_liq_mod08': cf_ret_liq.flatten(),
        'cot_mod08': cot_liq.flatten(),
        'cer_mod08': cer_liq.flatten(),
        'cotstd_mod08': cotstd_liq.flatten(),
        'sza': sza.flatten(),
    })
    return df


# ======================== main ========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <year>")
        sys.exit(1)

    year = sys.argv[1]
    output_csv = f'/home/chenyiqi/251028_albedo_cot/SSFproduct/{year}.csv'

    time_min = np.datetime64(f'{year}-01-01')
    time_max = np.datetime64(f'{year}-12-31')
    lat_min, lat_max = -60, 60
    lsmask_path = "/data/chenyiqi/251007_tropic/landsea.nc"

    # ========== Land-Sea mask ==========
    with nc.Dataset(lsmask_path, 'r') as ds:
        lon = ds.variables['lon'][:]
        lon[lon > 180] -= 360
        lat = ds.variables['lat'][:]
        lsmask = ds.variables['LSMASK'][:]
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lsmask = lsmask[lat_mask, :]
    lat = lat[lat_mask]

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_flat = lon_grid[lsmask == 0].flatten()
    lat_flat = lat_grid[lsmask == 0].flatten()
    dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
    lon_repeated = np.repeat(lon_flat, len(dates))
    lat_repeated = np.repeat(lat_flat, len(dates))
    time_repeated = np.tile(dates.date, len(lon_flat))

    df_ls = pd.DataFrame({'lat': lat_repeated, 'lon': lon_repeated, 'time': time_repeated})

    # ========== CERES ==========
    cer_files = '/home/chenyiqi/251028_albedo_cot/CERES_L3SSF_2020/CERES_SSF1deg-Day_Terra-MODIS_Ed4.1_Subset_20200101-20201231.nc'
    ds = xr.open_mfdataset(cer_files, combine='nested', concat_dim='time')
    
    lon = ds['lon'].values
    lon[lon > 180] -= 360
    lat = ds['lat'].values
    # print(np.min(lat))
    latmask = (lat <= lat_max) & (lat >= lat_min)
    lat = lat[latmask]
    time_cer = ds['time'].values
    time_mask = (time_cer >= time_min) & (time_cer <= time_max)
    toa_sw_clr = ds['toa_sw_clr_daily'].sel(time=time_mask, lat=latmask).values
    toa_sw_all = ds['toa_sw_all_daily'].sel(time=time_mask, lat=latmask).values
    toa_solar  = ds['toa_solar_all_daily'].sel(time=time_mask, lat=latmask).values
    cld_fra = ds['cldarea_total_day_daily'].sel(time=time_mask, lat=latmask).values / 100
    cld_fra_liq = ds['cldarea_liq_total_day_daily'].sel(time=time_mask, lat=latmask).values / 100
    clr_fra = (ds['toa_sw_num_obs_clr_daily'] / ds['toa_sw_num_obs_all_daily']).sel(time=time_mask, lat=latmask).values
    time_cer = time_cer[time_mask].astype('datetime64[D]')

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_flat = np.repeat(lon_grid[np.newaxis, :, :], cld_fra_liq.shape[0], axis=0).flatten()
    lat_flat = np.repeat(lat_grid[np.newaxis, :, :], cld_fra_liq.shape[0], axis=0).flatten()
    time_flat = np.repeat(time_cer[:, np.newaxis, np.newaxis], cld_fra_liq.shape[1] * cld_fra_liq.shape[2], axis=1).flatten()

    df_cer = pd.DataFrame({
        'lat': lat_flat, 'lon': lon_flat, 'time': pd.to_datetime(time_flat).date,
        'cf_ceres': cld_fra.flatten(), 'cf_liq_ceres': cld_fra_liq.flatten(), 'clr_fra':clr_fra.flatten(),
        'sw_clr': toa_sw_clr.flatten(), 'sw_all': toa_sw_all.flatten(),
        'solar_incoming': toa_solar.flatten()
    })

    # ========== MOD08 ==========
    mod_files = glob.glob(f"/data/MODIS/MxD08_D3/MOD08_D3.A{year}*.hdf")
    all_dfs = [process_single_mod08_file(f, time_min, time_max, lat_min, lat_max) for f in mod_files]
    df_mod = pd.concat([df for df in all_dfs if df is not None], ignore_index=True)

    
    # ========== merge ==========
    merged_df = pd.merge(df_ls, df_mod, on=['time', 'lon', 'lat'], how='left')
    merged_df = pd.merge(merged_df, df_cer, on=['time', 'lon', 'lat'], how='left')

    # ========== save ==========
    merged_df.to_csv(output_csv, index=False)
    print(f"saved to: {output_csv}")
