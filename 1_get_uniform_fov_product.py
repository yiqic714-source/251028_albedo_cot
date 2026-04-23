# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 16:30:29 2025

@author: yiqi
"""
import numpy as np
from datetime import datetime, timedelta
from pyhdf.SD import SD, SDC
import pickle
import xarray as xr
import pandas as pd
import os
import time
import netCDF4 as nc
import sys
import uniform_fov_tools as uft
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def process_single_grid(
    primary_type, primary_flag, nan_flag,
    secondary_albedo, secondary_uncertainty,
    in_94power, beta, delta, albedo_all,
    primary_fra_thre, uncertainty_thre,
    cot_mod=None, cot_cer=None, cotstd_cer=None,
    unr_flag=None
):
    """
    Compute low-uncertainty (pure) FOV albedo based on pixel-level properties.

    Returns:
        Depending on `primary_type`, returns tuples containing:
        (mean_albedo, mean_primary_fraction,
         mean_uncertainty, mean_uncorrected_albedo, ...)
    """

    # Internal helper function for standardized output
    def _finalize_output(m):
        if primary_type == "ret":
            return (m['albedo'], m['prim_fra'], m['uncertainty'], m['uncorrected'],
                    m['cot_mod'], m['cotstd_mod'], m['cot_cer'], m['cotstd_cer'])
        elif primary_type == "cld":
            return (m['albedo'], m['prim_fra'], m['uncertainty'], m['uncorrected'],
                    m['cot_mod'], m['cotstd_mod'], m['cot_cer'], m['cotstd_cer'], m['unr_over_cld'])
        else:
            return (m['albedo'], m['prim_fra'], m['uncertainty'], m['uncorrected'])

    # ------------- set weight of each angle bin within an FOV---------------------
    wij = np.array([
        [.0000, .0018, .0091, .0116, .0074, .0038, .0020, .0010],
        [.0019, .0016, .0248, .0310, .0248, .0142, .0073, .0038],
        [.0055, .0191, .0304, .0362, .0334, .0213, .0111, .0058],
        [.0055, .0191, .0304, .0362, .0334, .0213, .0111, .0058]
    ])# CERES ATBD subsystem 4.4 fig, 9
    wij_flipped = np.flipud(wij)
    wij = np.vstack((wij, wij_flipped))
    wij = wij / np.sum(wij.flatten())
    beta_bins = np.arange(1.32, -1.32+0.01, -0.33)
    delta_bins = np.arange(-1.32, 1.32-0.01, 0.33)

    # Initialize output dictionary with NaNs
    means = dict(
        albedo=np.nan, prim_fra=np.nan, uncertainty=np.nan,
        uncorrected=np.nan, cot_mod=np.nan, cotstd_mod=np.nan,
        cot_cer=np.nan, cotstd_cer=np.nan, unr_over_cld=np.nan
    )

    # ---------------------------------------------------------------------
    # Step 1. Identify possibly "pure" FOVs (dominant primary fraction, no NaN)
    # ---------------------------------------------------------------------
    pixel_num = np.sum(in_94power, axis=1).astype(float)
    fra_coarse = np.divide(
        np.sum(primary_flag * in_94power, axis=1, where=in_94power > 0),
        pixel_num,
        out=np.zeros_like(pixel_num, dtype=float),
        where=pixel_num > 0
    )
    with_nan = np.any(np.isnan(nan_flag) & in_94power, axis=1)
    possible_mask = (fra_coarse > primary_fra_thre) & ~with_nan
    if not np.any(possible_mask):
        return _finalize_output(means)

    # Extract data for potentially pure FOVs
    beta, delta = beta[possible_mask], delta[possible_mask]
    albedo_pure = albedo_all[possible_mask].copy()
    possible_num = len(albedo_pure)

    # Initialize working arrays
    uncertainty = np.zeros(possible_num)
    primary_fra = np.ones(possible_num)
    cot = np.full([possible_num, len(beta_bins)*len(delta_bins)], np.nan)
    cot_variance = np.full([possible_num, len(beta_bins)*len(delta_bins)], np.nan)
    unr_over_cld = np.zeros(possible_num)

    valid_count = 0

    # ---------------------------------------------------------------------
    # Step 2. Iterate over each potentially pure FOV
    # ---------------------------------------------------------------------
    for k in range(possible_num):
        if np.isnan(albedo_pure[k]):
            uncertainty[k] = 999
            continue

        discard_flag = False
        l = 0
        for ii, beta_edge in enumerate(beta_bins):
            mask_beta = (beta[k] < beta_edge) & (beta[k] >= beta_edge - 0.33)
            for jj, delta_edge in enumerate(delta_bins):
                mask_angle = mask_beta & (delta[k] > delta_edge) & (delta[k] <= delta_edge + 0.33)
                pixel_count = np.sum(mask_angle)
                if pixel_count == 0:
                    discard_flag = True
                    break

                wij_ = wij[ii, jj]
                sec_unc = np.sum(secondary_uncertainty * mask_angle) / pixel_count
                sec_alb = np.sum(secondary_albedo * mask_angle) / pixel_count
                sec_flag = secondary_albedo > 0

                # Weighted accumulation
                uncertainty[k] += sec_unc * wij_
                albedo_pure[k] -= sec_alb * wij_
                primary_fra[k] -= np.sum(sec_flag * mask_angle) / pixel_count * wij_

                if primary_type in ["cld", "ret"]:
                    if np.any(~np.isnan(cot_mod[mask_angle])):
                        #print(ii, jj, l)
                        cot[k, l] = np.nanmean(cot_mod[mask_angle])
                        cot_variance[k, l] = np.nanvar(cot_mod[mask_angle])
                    l += 1

                if primary_type == "cld":
                    unr_over_cld[k] += np.sum(unr_flag * mask_angle) / pixel_count * wij_

            # Discard incomplete edge FOVs or high-uncertainty cases
            if uncertainty[k] / albedo_pure[k] > uncertainty_thre * 1.5 or discard_flag:
                uncertainty[k] = 999
                break

        # Early stopping: for "clr", stop once two valid FOVs found
        if primary_type == "clr" and uncertainty[k] / (albedo_pure[k]/primary_fra[k]) < uncertainty_thre:
            valid_count += 1
            if valid_count >= 2:
                break

    # ---------------------------------------------------------------------
    # Step 3. Compute mean statistics for valid FOVs
    # ---------------------------------------------------------------------
    albedo_pure /= primary_fra
    valid_idx = (uncertainty / albedo_pure) < uncertainty_thre
    if np.any(valid_idx):
        means['uncertainty'] = np.mean(uncertainty[valid_idx])
        means['albedo'] = np.mean(albedo_pure[valid_idx])
        means['uncorrected'] = np.mean(albedo_all[possible_mask][valid_idx])
        means['prim_fra'] = np.mean(primary_fra[valid_idx])

        if primary_type in ["cld", "ret"]:
            means['cot_mod'] = np.nanmean(np.nanmean(cot, axis=1)[valid_idx])
            means['cotstd_mod'] = np.nanmean(np.sqrt(
                np.nanmean(cot_variance, axis=1)[valid_idx] +
                np.nanvar(cot, axis=1)[valid_idx]
                ))
            means['cot_cer'] = np.nanmean(cot_cer[possible_mask][valid_idx])
            means['cotstd_cer'] = np.nanmean(cotstd_cer[possible_mask][valid_idx])

        if primary_type == "cld":
            means['unr_over_cld'] = np.mean(unr_over_cld[valid_idx])

    # ---------------------------------------------------------------------
    # Step 4. Return formatted output
    # ---------------------------------------------------------------------
    return _finalize_output(means)


def get_uniform_fov_product(latlon_cer, latlon_mod, latlon_subsat, latlon_land, \
                    type_data, cot_mod, cot_cer, cotstd_cer, solar_zenith_cer, sensor_zenith_cer,\
                        albedo_all):
    """
    Divide region into 1 degree * 1degree and call process_single_grid function in each subregion
    
    Parameters:
        latlon_cer: [latitude, longitude] for CERES FOV centroid points
        latlon_mod: [latitude, longitude] for MODIS pixels
        latlon_subsat: [latitude, longitude] for CERES sub-satellite points
        type_data: 2 for retrievable, 1 for unretrievable, 0 for clear-sky, and nan for non-liquid or sunglint
        cot_mod: MODIS pixel-level cloud optical thickness
        solar_zenith_mod: MODIS pixel-level solar zenith
        albedo_all: CERES FOV-level all-sky albedo
    
    Returns:
        Three lists for retrievable, unretrievable, and clear cases, respectively
        Each element is [lat, lon, 1-degree mean albedo, 1-degree mean fraction, 1-degree mean uncertainty]
    """
    # Precompute region bounds
    lat_min0 = max(np.min(latlon_mod[:, 0]), obs_window[0][0])
    lat_max0 = min(np.max(latlon_mod[:, 0]), obs_window[0][1])
    lat_min = np.ceil(lat_min0)
    lat_max = np.floor(lat_max0)
    lon_min0 = max(np.min(latlon_mod[:, 1]), obs_window[1][0])
    lon_max0 = min(np.max(latlon_mod[:, 1]), obs_window[1][1])
    lon_min = np.ceil(lon_min0)
    lon_max = np.floor(lon_max0)
    # Partition parameters
    step = 1
    overlap_lat = 0.3
    # Calculate num of partitions
    num_lat_parts = int(lat_max - lat_min)
    num_lon_parts = int(lon_max - lon_min)

    if num_lat_parts <= 0 or num_lon_parts <= 0:
        return []

    # Prepare grid centers (vectorized)
    lat_centers = lat_min + np.arange(num_lat_parts) + 0.5 * step
    lon_centers = lon_min + np.arange(num_lon_parts) + 0.5 * step
    lat_grid, lon_grid = np.meshgrid(lat_centers, lon_centers, indexing='ij')
    lat_grid_flat = lat_grid.ravel()
    lon_grid_flat = lon_grid.ravel()
    grid_points = np.stack([lat_grid_flat, lon_grid_flat], axis=1)

    # Mask out grid points near land (vectorized)
    # Calculate Manhattan distance between all grid points and all land points
    if len(latlon_land) > 0:
        land_dist = np.abs(lat_grid_flat[:, None] - latlon_land[:, 0]) + np.abs(lon_grid_flat[:, None] - latlon_land[:, 1])
        min_land_dist = np.min(land_dist, axis=1)
        valid_grid_mask = min_land_dist >= 1.001
    else:
        valid_grid_mask = np.ones(lat_grid_flat.shape)
    valid_indices = np.where(valid_grid_mask)[0]
    rsl_lst = []

    # Precompute for MODIS and CERES
    for idx in valid_indices:
        lat_1deg, lon_1deg = grid_points[idx]
        # Compute sub-region bounds
        sub_lat_min = lat_1deg - 0.5 * step
        sub_lat_max = lat_1deg + 0.5 * step
        sub_lon_min = lon_1deg - 0.5 * step
        sub_lon_max = lon_1deg + 0.5 * step

        overlap_lon = 0.375 / np.cos(np.radians(lat_1deg))
        sub_lat_min_mod = max(sub_lat_min - overlap_lat, lat_min)
        sub_lat_max_mod = min(sub_lat_max + overlap_lat, lat_max)
        sub_lon_min_mod = max(sub_lon_min - overlap_lon, lon_min)
        sub_lon_max_mod = min(sub_lon_max + overlap_lon, lon_max)

        # Vectorized masks
        cer_mask = (
            (latlon_cer[:, 0] >= sub_lat_min) & (latlon_cer[:, 0] <= sub_lat_max) &
            (latlon_cer[:, 1] >= sub_lon_min) & (latlon_cer[:, 1] <= sub_lon_max)
        )
        sub_albedo_all = albedo_all[cer_mask]
        if len(sub_albedo_all) <= 1: # in case only one fov in 1*1 degree
            continue

        mod_mask = (
            (latlon_mod[:, 0] >= sub_lat_min_mod) & (latlon_mod[:, 0] <= sub_lat_max_mod) &
            (latlon_mod[:, 1] >= sub_lon_min_mod) & (latlon_mod[:, 1] <= sub_lon_max_mod)
        )
        sub_cld_type = type_data[mod_mask]
        len_mod = len(sub_cld_type)
        if len_mod <= 1:
            continue
        nan_flag = np.isnan(sub_cld_type)
        nan_fraction = np.mean(nan_flag)
        if nan_fraction > 0.01:
            continue
        
        # 1-D cloud type flag
        ret_flag = (sub_cld_type==2)
        unr_flag = (sub_cld_type==1)
        clr_flag = (sub_cld_type==0)
        cld_flag = ret_flag | unr_flag
        ret_fra = np.sum(ret_flag) / len_mod
        cld_fra = np.sum(unr_flag) / len_mod
        clr_fra = np.sum(clr_flag) / len_mod

        sub_latlon_cer = latlon_cer[cer_mask]
        sub_latlon_subsat = latlon_subsat[cer_mask]
        sub_cot_cer = cot_cer[cer_mask]
        sub_cotstd_cer = cotstd_cer[cer_mask]
        sub_solar_zenith_cer = solar_zenith_cer[cer_mask]
        sub_sensor_zenith_cer = sensor_zenith_cer[cer_mask]
        sub_latlon_mod = latlon_mod[mod_mask]
        sub_cot_mod = cot_mod[mod_mask]
        solar_zenith_grid = np.mean(sub_solar_zenith_cer)
        sensor_zenith_grid = np.mean(sub_sensor_zenith_cer)
            
        # calculate along- and cross-track angles
        delta, beta = \
            uft.calc_delta_beta(sub_latlon_cer, sub_latlon_mod, sub_latlon_subsat)
        
        in_94power = (abs(beta) < 1.32) & (abs(delta) <= 1.32)
        uncertainty_thre = 0.03
        
        # Get albedo from "pure" FOVs
        clr_albedo_guess = 0.08
        clr_uncertainty = 0.03
        cld_albedo_guess = 0.3
        cld_uncertainty = 0.15
        
        # 2. Process "pure" unretrievable FOVs
        fov_ret = process_single_grid(
            primary_type="ret",
            primary_flag=ret_flag,
            nan_flag = nan_flag,
            cot_mod = sub_cot_mod,
            cot_cer = sub_cot_cer,
            cotstd_cer = sub_cotstd_cer,
            
            secondary_uncertainty = unr_flag * cld_uncertainty \
                + clr_flag * clr_uncertainty,
            secondary_albedo = unr_flag * cld_albedo_guess \
                + clr_flag * clr_albedo_guess ,
            
            in_94power=in_94power,
            beta=beta,
            delta=delta,
            albedo_all=sub_albedo_all,
            primary_fra_thre=0.65, 
            uncertainty_thre=uncertainty_thre
        )
        new_line = [*fov_ret, solar_zenith_grid, sensor_zenith_grid, ret_fra, cld_fra, clr_fra]
        if np.any(~np.isnan(new_line)):
            rsl_lst.append([lat_1deg, lon_1deg, *new_line])
            
    return rsl_lst
                


if __name__ == "__main__":
    start = time.perf_counter()
        
    if len(sys.argv) < 3:
        print("Usage: python script.py <year> <month> <hemisphere>")
        sys.exit(1)

    year = sys.argv[1]
    month = int(sys.argv[2])
    hemisph = sys.argv[3]

    if hemisph == "east":
        obs_window = [[-60., -23.],[0., 180.]]
        fname_ending = "lon_0_180"
    elif hemisph == "west":
        obs_window = [[-60., -23.],[-180., 0.]]
        fname_ending = "lon_m180_0"
    else:
        print('Only support hemisphere to be east or west')

    # ---------------------- get (lat, lon) of the land ---------------------------
    file_path = '/data/chenyiqi/251007_tropic/landsea.nc'
    ds = nc.Dataset(file_path, 'r')
    lsmask = ds.variables['LSMASK'][:].ravel()
    lat_ls = ds.variables['lat'][:]
    lon_ls = ds.variables['lon'][:] # range: 0~360
    lon_ls[lon_ls>180] -= 360
    lon_ls, lat_ls = np.meshgrid(lon_ls, lat_ls)
    lat_ls = lat_ls.ravel()
    lon_ls = lon_ls.ravel()
    idx = (lsmask==1) & (lat_ls>obs_window[0][0]) & (lat_ls<obs_window[0][1]) & \
        (lon_ls>obs_window[1][0]) & (lon_ls<obs_window[1][1])
    latlon_land = np.stack((lat_ls[idx], lon_ls[idx]), axis = 1)

    # ------------------ read modis and ceres filename lists-----------------------
    # mod_pkl = f"/data/chenyiqi/251007_tropic/mod06/MOD06_files_{year}_{month:02d}_{month:02d}_{fname_ending}.pkl"
    mod_pkl = f"/data/chenyiqi/251028_albedo_cot/mod06/MOD06_files_{year}{month:02d}S_{fname_ending}.pkl"
        
    with open(mod_pkl, "rb") as f:
        mod_file_lst = pickle.load(f)['terra']

    # cer_pkl = f"/data/chenyiqi/251007_tropic/ceres_ssf_L2/CERES_{year}_files_and_times.pkl"
    cer_pkl = f"/data/chenyiqi/251028_albedo_cot/CERES_L2SSF_2020/CERES_{year}_files_and_times.pkl"
    with open(cer_pkl, 'rb') as f:
        ssf_files_times_lst = pickle.load(f)['ssf_files']

    # determine output csv filename
    output_fname = mod_pkl.split("MOD06_files_")[-1].replace(".pkl", "")
    csv_fname = f"/data/chenyiqi/251028_albedo_cot/uniform_fov_product/rsl_{year}{month:02d}S_{fname_ending}.csv"

    # header for csv
    csv_header = ["time", "lat", "lon", 
                    "ret_albedo", "ret_fov_fra", "ret_albedo_uncert", "ret_uncorrected_albedo", 
                    "ret_cot_mod", "ret_cotstd_mod", "ret_cot_cer", "ret_cotstd_cer", 
                    "solar_zenith", "sensor_zenith", "ret_fra", "unr_fra", "clr_fra"]

    num_mod_files = len(mod_file_lst)
    for i_mod, mod_filename in enumerate(mod_file_lst):
        
        # --------------------- find corresponding ceres file----------------------
        parts = mod_filename.split('.')
        timestamp_str = parts[1][1:] + parts[2]
        time_mod = datetime.strptime(timestamp_str, "%Y%j%H%M")
        
        cer_found_flag = 0
        for item in ssf_files_times_lst:
            if item['start_time'] <= time_mod <= item['end_time']:
                cere_ssf_filename = item['filename']
                cer_found_flag = 1
                break
        if cer_found_flag == 0:
            print(time_mod, "has no CERES-Terra data")
            continue
        
        # ------------------get and process modis variables------------------------
        print(mod_filename)
        hdf = SD(mod_filename, SDC.READ)
        
        # read 5-km resolution variables
        lat_mod = hdf.select('Latitude')[:]
        lat_mod[lat_mod==-999] = np.nan
        lon_mod = hdf.select('Longitude')[:]
        lon_mod[lon_mod==-999] = np.nan
        # print(np.min(lon_mod), np.max(lon_mod))
        sensor_zenith_mod = uft.read_and_mask_mod_variable(hdf, 'Sensor_Zenith')
        solar_zenith_mod = uft.read_and_mask_mod_variable(hdf, 'Solar_Zenith')
        # read 1-km resolution variables
        cot_mod = uft.read_and_mask_mod_variable(hdf, 'Cloud_Optical_Thickness')
        ctt_mod = uft.read_and_mask_mod_variable(hdf, 'cloud_top_temperature_1km')
        qa1km = hdf.select('Quality_Assurance_1km').get()
        byte2 = qa1km[:, :, 2]  # Get byte 2 for all pixels
        Cloud_Retrieval_Phase_Flag = (byte2 >> 0) & 0b111  # Get bits 0-2
        Primary_Cloud_Retrieval_Outcome_Flag = (byte2 >> 3) & 0b1  # Get bit 3
        cm1km = hdf.select('Cloud_Mask_1km').get()
        byte0 = cm1km[:, :, 0]  # Get byte 0 for all pixels
        Cloudiness_Flag = (byte0 >> 1) & 0b11  # Get bits 1-2
        Sunglint_Flag = (byte0 >> 4) & 0b1 # Get bit 0, 0 is yes
        hdf.end()

        # Interpolate 5km resolution to 1km resolution
        lat_mod, lon_mod, solar_zenith_mod, sensor_zenith_mod = \
            uft.upscale_and_interpolate(lat_mod, lon_mod, solar_zenith_mod, sensor_zenith_mod, cot_mod.shape, obs_window)
        
        # generate cld_type
        cld_retrieval_mask = np.where((Cloud_Retrieval_Phase_Flag == 2)\
                                    & (Primary_Cloud_Retrieval_Outcome_Flag == 1), 1, 0) # liquid phase, retrievable
        # cld_retrieval_mask = np.where((Cloud_Retrieval_Phase_Flag == 2)\
        #                               & (cot_mod > 0), 1, 0) # liquid phase, retrievable, equal to the above
        cld_mask = np.where((Cloud_Retrieval_Phase_Flag == 2)\
                                    & (Cloudiness_Flag <= 1), 1, 0) # liquid phase, cloudy
        # cld_type is 2 for retrievable, 1 for unretrievable, 0 for clear-sky, and nan for no-liquid
        nan_mask = ((Cloud_Retrieval_Phase_Flag != 2) & (Primary_Cloud_Retrieval_Outcome_Flag == 1)) | (ctt_mod < 270)
        cld_type = cld_retrieval_mask.astype(int) + cld_mask.astype(int)
        cld_type = np.where(nan_mask, np.nan, cld_type)
        # plt.figure()
        # plt.scatter(lon_mod, lat_mod, s=1, c=cld_type)
        # plt.colorbar()
        
        lon_min = max(obs_window[1][0], np.min(lon_mod))
        lon_max = min(obs_window[1][1], np.max(lon_mod))
        lat_min = max(obs_window[0][0], np.min(lat_mod))
        lat_max = min(obs_window[0][1], np.max(lat_mod))
        time_low = time_mod - timedelta(minutes=5)
        time_high = time_mod + timedelta(minutes=5)
        
        indices = np.where((sensor_zenith_mod < 55) & (solar_zenith_mod < 55)
                            & (lat_mod > lat_min) & (lat_mod < lat_max)
                            & (lon_mod > lon_min) & (lon_mod < lon_max)
                            & (Sunglint_Flag == 1))
        if not np.any(indices):
            print(f"File {i_mod} in {num_mod_files}, 0 grid recorded")
            continue
        
        cot_mod = cot_mod[indices].flatten()
        solar_zenith_mod = solar_zenith_mod[indices].flatten()
        cld_type = cld_type[indices].flatten()
        lat_mod = lat_mod[indices].flatten()
        lon_mod = lon_mod[indices].flatten()
        latlon_mod = np.stack((lat_mod, lon_mod), axis = 1)
        
        # ------------------- get and process cere variables ----------------------
        ds_ssf = xr.open_dataset(cere_ssf_filename)
        # print(ds_ssf.variables)
        
        time_ssf = uft.julian_to_datetime(ds_ssf['Time_of_observation'].values)
        lon_ssf = ds_ssf['Longitude_of_CERES_FOV_at_surface'].values
        lon_ssf[lon_ssf > 180] -= 360
        lat_ssf = 90 - ds_ssf['Colatitude_of_CERES_FOV_at_surface'].values
        # print(np.max(lat_ssf))
        latlon_ssf = np.stack((lat_ssf, lon_ssf), axis = 1)
        lon_subsat = ds_ssf['Longitude_of_subsatellite_point_at_surface_at_observation'].values
        lon_subsat[lon_subsat > 180] -= 360
        lat_subsat = 90 - ds_ssf['Colatitude_of_subsatellite_point_at_surface_at_observation'].values
        latlon_subsat = np.stack((lat_subsat, lon_subsat), axis = 1)
        
        sw_incoming = ds_ssf['TOA_Incoming_Solar_Radiation'].values# W m-2
        sw_toa_all = ds_ssf['CERES_SW_TOA_flux___upwards'].values # W m-2
        solar_zenith_cer = ds_ssf['CERES_solar_zenith_at_surface'].values
        sensor_zenith_cer = ds_ssf['CERES_viewing_zenith_at_surface'].values# range: [ 0. 90.]
        cot_cer = ds_ssf['Mean_visible_optical_depth_for_cloud_layer'].values
        cotstd_cer = ds_ssf['Stddev_of_visible_optical_depth_for_cloud_layer'].values
        time_cond = np.array([t >= time_low for t in time_ssf]) & np.array([t <= time_high for t in time_ssf])
        indices = time_cond & (latlon_ssf[:, 0] > lat_min) & (latlon_ssf[:, 0] < lat_max)\
                            & (latlon_ssf[:, 1] > lon_min) & (latlon_ssf[:, 1] < lon_max)\
                            & (sensor_zenith_cer < 55) & (solar_zenith_cer < 55)
        if not np.any(indices):
            print(f"File {i_mod}/{num_mod_files}, month {month:02d}, 0 grid recorded")
            continue
        
        latlon_cer = latlon_ssf[indices]
        latlon_subsat = latlon_subsat[indices]
        solar_zenith_cer = solar_zenith_cer[indices]
        sensor_zenith_cer = sensor_zenith_cer[indices]
        cot_cer = cot_cer[indices]
        cotstd_cer = cotstd_cer[indices]
        albedo_all = sw_toa_all[indices] / sw_incoming[indices]
        
        # ------------ match cere-modis, get albedo for 3 cases -------------------
        rsl_lst = \
            get_uniform_fov_product(latlon_cer, latlon_mod, latlon_subsat, latlon_land,
                                cld_type, cot_mod, cot_cer, cotstd_cer, solar_zenith_cer, sensor_zenith_cer, albedo_all) 
        
        len_rsl = len(rsl_lst)
        if len_rsl > 0: rsl_lst = [[time_mod, *rsl_lst[i]] for i in range(len_rsl)] 
        print(f"File {i_mod} in month {month:02d}, {len_rsl} grid(s) recorded")
        # ----------------------------- save data ---------------------------------
        file_exists = os.path.exists(csv_fname)
        df = pd.DataFrame(rsl_lst, columns=csv_header)
        df.to_csv(
            csv_fname,
            mode='a' if file_exists else 'w',
            header=not file_exists,
            index=False
        )

    end = time.perf_counter()
    print(f"Run time：{end - start:.6f} s. Output file: {csv_fname}")