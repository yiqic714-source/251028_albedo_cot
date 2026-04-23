# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 16:30:29 2025
Modified by: Copilot (for Chen111777)
Purpose: compute per 1deg x 1deg grid cloud fractions from MODIS granules:
 - retrieval_cf: fraction of pixels with cld_retrieval_mask == 1
 - cloudmask_cf: fraction of pixels with cld_mask == 1
Outputs CSV with columns: time, lat, lon, retrieval_cf, cloudmask_cf
"""
import sys
import os
import time
from datetime import datetime
import pickle
import numpy as np
from pyhdf.SD import SD, SDC
import xarray as xr
import netCDF4 as nc
import pandas as pd
import uniform_fov_tools as uft


def get_granule_cf(latlon_mod, cld_retrieval_mask, cld_mask, latlon_land, obs_window):
    """
    Compute 1deg x 1deg grid centers within the overlap of latlon_mod and obs_window,
    exclude grid centers that are within ~1 degree (Manhattan distance) of any land point in latlon_land,
    and for each valid 1deg grid compute:
      - retrieval fraction: fraction of mod pixels inside the 1deg box with cld_retrieval_mask == 1
      - cloudmask fraction: fraction of mod pixels inside the 1deg box with cld_mask == 1

    Inputs:
      latlon_mod: (N,2) array of [lat, lon] of MODIS 1km pixels
      cld_retrieval_mask: (N,) array of 0/1 flags
      cld_mask: (N,) array of 0/1 flags
      latlon_land: (M,2) array of land [lat, lon] points
      obs_window: [[lat_min, lat_max],[lon_min, lon_max]]

    Returns:
      list of [lat_center, lon_center, retrieval_frac, cloudmask_frac]
    """
    # Determine numeric bounds clipped by obs_window
    lat_min0 = max(np.min(latlon_mod[:, 0]), obs_window[0][0])
    lat_max0 = min(np.max(latlon_mod[:, 0]), obs_window[0][1])
    lon_min0 = max(np.min(latlon_mod[:, 1]), obs_window[1][0])
    lon_max0 = min(np.max(latlon_mod[:, 1]), obs_window[1][1])

    lat_min = int(np.ceil(lat_min0))
    lat_max = int(np.floor(lat_max0))
    lon_min = int(np.ceil(lon_min0))
    lon_max = int(np.floor(lon_max0))

    if lat_max < lat_min or lon_max < lon_min:
        return []

    # create grid centers (1x1 degree)
    lat_centers = np.arange(lat_min, lat_max) + 0.5
    lon_centers = np.arange(lon_min, lon_max) + 0.5
    lat_grid, lon_grid = np.meshgrid(lat_centers, lon_centers, indexing='ij')
    grid_points = np.stack((lat_grid.ravel(), lon_grid.ravel()), axis=1)

    # Land exclusion: compute Manhattan distance to land points
    if latlon_land is not None and len(latlon_land) > 0:
        # shape: (num_grid, num_land)
        land_dist = np.abs(grid_points[:, 0:1] - latlon_land[:, 0][None, :]) + \
                    np.abs(grid_points[:, 1:2] - latlon_land[:, 1][None, :])
        min_land_dist = np.min(land_dist, axis=1)
        valid_grid_mask = min_land_dist >= 1.001
    else:
        valid_grid_mask = np.ones(len(grid_points), dtype=bool)

    rsl_lst = []
    # For each valid center, compute fractions
    for (lat_c, lon_c), valid in zip(grid_points, valid_grid_mask):
        if not valid:
            continue
        sub_lat_min = lat_c - 0.5
        sub_lat_max = lat_c + 0.5
        sub_lon_min = lon_c - 0.5
        sub_lon_max = lon_c + 0.5

        in_box = (
            (latlon_mod[:, 0] >= sub_lat_min) & (latlon_mod[:, 0] <= sub_lat_max) &
            (latlon_mod[:, 1] >= sub_lon_min) & (latlon_mod[:, 1] <= sub_lon_max)
        )
        n_pixels = np.count_nonzero(in_box)
        if n_pixels <= 1:
            # skip boxes with 0 or 1 pixel (not enough statistics)
            continue

        retrieval_frac = float(np.sum(cld_retrieval_mask[in_box] == 1) / n_pixels)
        cloudmask_frac = float(np.sum(cld_mask[in_box] == 1) / n_pixels)

        rsl_lst.append([lat_c, lon_c, retrieval_frac, cloudmask_frac])

    return rsl_lst


if __name__ == "__main__":
    start = time.perf_counter()

    if len(sys.argv) < 4:
        print("Usage: python granule_cf_processor.py <year> <month> <hemisphere(east|west)>")
        sys.exit(1)

    year = sys.argv[1]
    month = int(sys.argv[2])
    hemisph = sys.argv[3].lower()

    if hemisph == "east":
        obs_window = [[-23., 23.], [0., 180.]]
        fname_ending = "lon_0_180"
    elif hemisph == "west":
        obs_window = [[-23., 23.], [-180., 0.]]
        fname_ending = "lon_m180_0"
    else:
        raise ValueError("hemisphere must be 'east' or 'west'")

    # ---------------------- get (lat, lon) of the land ---------------------------
    land_file = '/data/chenyiqi/251007_tropic/landsea.nc'
    ds = nc.Dataset(land_file, 'r')
    lsmask = ds.variables['LSMASK'][:].ravel()
    lat_ls = ds.variables['lat'][:]
    lon_ls = ds.variables['lon'][:]  # range: 0~360
    lon_ls[lon_ls > 180] -= 360
    lon_grid, lat_grid = np.meshgrid(lon_ls, lat_ls)
    lat_ls_flat = lat_grid.ravel()
    lon_ls_flat = lon_grid.ravel()
    idx_land = (lsmask == 1) & (lat_ls_flat > obs_window[0][0]) & (lat_ls_flat < obs_window[0][1]) & \
               (lon_ls_flat > obs_window[1][0]) & (lon_ls_flat < obs_window[1][1])
    latlon_land = np.stack((lat_ls_flat[idx_land], lon_ls_flat[idx_land]), axis=1)

    # ------------------ read modis filename list -----------------------
    # mod_pkl = f"/data/chenyiqi/251028_albedo_cot/mod06/MOD06_files_{year}{month:02d}S_{fname_ending}.pkl"
    mod_pkl = f"/home/chenyiqi/251007_tropic/mod06/MOD06_files_{year}_{month:02d}_{month:02d}_{fname_ending}.pkl"
    with open(mod_pkl, "rb") as f:
        mod_file_lst = pickle.load(f)['terra']

    # output csv
    csv_fname = f"/data/chenyiqi/251028_albedo_cot/cf_product/cf_{year}{month:02d}T_{fname_ending}.csv"
    csv_header = ["time", "lat", "lon", "retrieval_cf", "cloudmask_cf"]

    num_mod_files = len(mod_file_lst)
    for i_mod, mod_filename in enumerate(mod_file_lst):
        try:
            parts = mod_filename.split('.')
            timestamp_str = parts[1][1:] + parts[2]
            time_mod = datetime.strptime(timestamp_str, "%Y%j%H%M")
        except Exception:
            # if filename format unexpected, skip
            print(f"Skipping file with unexpected name format: {mod_filename}")
            continue

        # read MODIS HDF
        hdf = SD(mod_filename, SDC.READ)

        # read variables (1km and 5km as needed)
        lat_5km = hdf.select('Latitude')[:]
        lat_5km[lat_5km == -999] = np.nan
        lon_5km = hdf.select('Longitude')[:]
        lon_5km[lon_5km == -999] = np.nan

        sensor_zenith_5km = uft.read_and_mask_mod_variable(hdf, 'Sensor_Zenith')
        solar_zenith_5km = uft.read_and_mask_mod_variable(hdf, 'Solar_Zenith')

        cot_1km = uft.read_and_mask_mod_variable(hdf, 'Cloud_Optical_Thickness')
        ctt_1km = uft.read_and_mask_mod_variable(hdf, 'cloud_top_temperature_1km')

        qa1km = hdf.select('Quality_Assurance_1km').get()
        byte2 = qa1km[:, :, 2]
        Cloud_Retrieval_Phase_Flag = (byte2 >> 0) & 0b111
        Primary_Cloud_Retrieval_Outcome_Flag = (byte2 >> 3) & 0b1

        cm1km = hdf.select('Cloud_Mask_1km').get()
        byte0 = cm1km[:, :, 0]
        Cloudiness_Flag = (byte0 >> 1) & 0b11
        Sunglint_Flag = (byte0 >> 4) & 0b1  # 1 means no sunglint (keep), 0 means sunglint present

        hdf.end()

        # Upscale/interpolate 5km -> 1km grids (returns arrays matching cot_1km shape)
        lat_mod, lon_mod, solar_zenith_mod, sensor_zenith_mod = \
            uft.upscale_and_interpolate(lat_5km, lon_5km, solar_zenith_5km, sensor_zenith_5km, cot_1km.shape, obs_window)

        # generate cloud masks (1km)
        cld_retrieval_mask = np.where((ctt_1km > 273.15 - 5) & (Cloud_Retrieval_Phase_Flag == 2) &
                                      (Primary_Cloud_Retrieval_Outcome_Flag == 1), 1, 0)
        cld_mask = np.where((ctt_1km > 273.15 - 5) & (Cloud_Retrieval_Phase_Flag == 2) &
                            (Cloudiness_Flag <= 1), 1, 0)

        # define bounding box for selection (intersection with obs_window)
        lon_min = max(obs_window[1][0], np.nanmin(lon_mod))
        lon_max = min(obs_window[1][1], np.nanmax(lon_mod))
        lat_min = max(obs_window[0][0], np.nanmin(lat_mod))
        lat_max = min(obs_window[0][1], np.nanmax(lat_mod))

        # select valid pixels: zeniths and within obs window and non-sunglint
        valid_mask = (lat_mod > lat_min) & (lat_mod < lat_max) & \
                     (lon_mod > lon_min) & (lon_mod < lon_max)

        if not np.any(valid_mask):
            print(f"File {i_mod+1}/{num_mod_files}, no valid pixels in obs window")
            continue

        lat_sel = lat_mod[valid_mask].flatten()
        lon_sel = lon_mod[valid_mask].flatten()
        retrieval_sel = cld_retrieval_mask[valid_mask].flatten()
        cloudmask_sel = cld_mask[valid_mask].flatten()

        latlon_mod_flat = np.stack((lat_sel, lon_sel), axis=1)

        # compute grid-level cloud fractions
        rsl_lst = get_granule_cf(latlon_mod_flat, retrieval_sel, cloudmask_sel, latlon_land, obs_window)

        if len(rsl_lst) == 0:
            print(f"File {i_mod+1}/{num_mod_files}, 0 grids recorded")
            continue

        # prepend time and save to CSV
        rsl_with_time = [[time_mod, *r] for r in rsl_lst]
        df = pd.DataFrame(rsl_with_time, columns=csv_header)
        file_exists = os.path.exists(csv_fname)
        df.to_csv(csv_fname, mode='a' if file_exists else 'w', header=not file_exists, index=False)

        print(f"File {i_mod+1}/{num_mod_files}, {len(rsl_lst)} grid(s) recorded")

    end = time.perf_counter()
    print(f"Run time: {end - start:.2f} s. Output file: {csv_fname}")