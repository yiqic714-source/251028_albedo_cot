# -*- coding: utf-8 -*-
"""
Create a CERES-footprint-level MODIS/CERES matched product.

Each output row represents one CERES footprint. The saved lat/lon are the
centers of the 1° x 1° grid containing that CERES footprint.

For each accepted footprint, the MODIS COT distribution is saved as 76
CERES-response-weighted frequency columns for COT intervals 0-1, 1-2, ...,
75-76. For one CERES response bin with weight w, if n MODIS pixels fall in a
COT interval, that response bin contributes w * n to the interval frequency.

@author: yiqi
"""

import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timedelta

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from pyhdf.SD import SD, SDC

import utils_uniform_fov as uft

warnings.filterwarnings("ignore", category=RuntimeWarning)

CODE_VERSION = "2026-09-04-per-footprint-v4"
PRIMARY_FRA_THRE = 1.0


# CERES FOV response weights: CERES ATBD subsystem 4.4, Fig. 9.
_WIJ_TOP = np.array(
    [
        [0.0000, 0.0018, 0.0091, 0.0116, 0.0074, 0.0038, 0.0020, 0.0010],
        [0.0019, 0.0016, 0.0248, 0.0310, 0.0248, 0.0142, 0.0073, 0.0038],
        [0.0055, 0.0191, 0.0304, 0.0362, 0.0334, 0.0213, 0.0111, 0.0058],
        [0.0055, 0.0191, 0.0304, 0.0362, 0.0334, 0.0213, 0.0111, 0.0058],
    ],
    dtype=float,
)
FOV_WEIGHTS = np.vstack((_WIJ_TOP, np.flipud(_WIJ_TOP)))
FOV_WEIGHTS /= np.sum(FOV_WEIGHTS)

BETA_BINS = np.arange(1.32, -1.32 + 0.01, -0.33)
DELTA_BINS = np.arange(-1.32, 1.32 - 0.01, 0.33)

# 76 COT intervals: [0,1), [1,2), ..., [74,75), [75,76].
# np.histogram includes the rightmost edge in the final interval.
COT_EDGES = np.arange(0.0, 77.0, 1.0)
COT_FREQ_COLUMNS = [f"cot_freq_{i}_{i + 1}" for i in range(76)]


def process_single_footprint(
    ret_flag,
    in_94power,
    beta,
    delta,
    albedo_all,
    cot_mod,
):
    """Process one CERES footprint.

    A footprint is retained only when every MODIS pixel inside the CERES
    94%-power footprint is a retrievable warm liquid-cloud pixel, i.e.
    ret_fraction == 1.0.

    Returns
    -------
    tuple or None
        (ret_albedo, ret_fov_fra, cot_freq[0], ..., cot_freq[75])
        if the footprint passes all filters; otherwise None.
    """

    ret_flag = np.asarray(ret_flag, dtype=bool).ravel()
    in_94power = np.asarray(in_94power, dtype=bool).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()
    cot_mod = np.asarray(cot_mod, dtype=float).ravel()

    n_pix = in_94power.size
    for name, arr in {
        "ret_flag": ret_flag,
        "beta": beta,
        "delta": delta,
        "cot_mod": cot_mod,
    }.items():
        if arr.size != n_pix:
            raise ValueError(
                "Single-footprint pixel dimension mismatch: "
                f"in_94power has {n_pix} pixels but {name} has {arr.size}."
            )

    # Require a complete pure retrievable-cloud footprint.
    pixel_num = int(np.sum(in_94power))
    if pixel_num == 0:
        return None

    ret_fraction = np.sum(ret_flag & in_94power) / pixel_num
    # PRIMARY_FRA_THRE = 1.0 means exactly 100% retrievable pixels are required.
    if ret_fraction < PRIMARY_FRA_THRE:
        return None

    ret_albedo = float(np.asarray(albedo_all).squeeze())
    if not np.isfinite(ret_albedo):
        return None

    cot_frequency = np.zeros(76, dtype=float)

    # Every one of the 8 x 8 CERES response bins must contain MODIS pixels,
    # preserving the original complete-footprint requirement.
    for ii, beta_edge in enumerate(BETA_BINS):
        mask_beta = (beta < beta_edge) & (beta >= beta_edge - 0.33)

        for jj, delta_edge in enumerate(DELTA_BINS):
            mask_angle = (
                mask_beta
                & (delta > delta_edge)
                & (delta <= delta_edge + 0.33)
            )

            if not np.any(mask_angle):
                return None

            valid_cot = cot_mod[mask_angle]
            valid_cot = valid_cot[np.isfinite(valid_cot)]

            if valid_cot.size == 0:
                continue

            counts, _ = np.histogram(valid_cot, bins=COT_EDGES)
            cot_frequency += FOV_WEIGHTS[ii, jj] * counts

    return (
        ret_albedo,
        ret_fraction,
        *cot_frequency.tolist(),
    )


def get_uniform_fov_product(
    latlon_cer,
    latlon_mod,
    latlon_subsat,
    latlon_land,
    ret_flag,
    cot_mod,
    solar_zenith_cer,
    sensor_zenith_cer,
    albedo_all,
):
    """Match MODIS pixels to individual CERES footprints.

    Each returned row corresponds to one CERES footprint. The first two
    values are the center latitude/longitude of the 1° grid containing that
    CERES footprint center.
    """

    lat_min = np.ceil(max(np.min(latlon_mod[:, 0]), obs_window[0][0]))
    lat_max = np.floor(min(np.max(latlon_mod[:, 0]), obs_window[0][1]))
    lon_min = np.ceil(max(np.min(latlon_mod[:, 1]), obs_window[1][0]))
    lon_max = np.floor(min(np.max(latlon_mod[:, 1]), obs_window[1][1]))

    step = 1.0
    overlap_lat = 0.3
    num_lat_parts = int(lat_max - lat_min)
    num_lon_parts = int(lon_max - lon_min)

    if num_lat_parts <= 0 or num_lon_parts <= 0:
        return []

    lat_centers = lat_min + np.arange(num_lat_parts) + 0.5 * step
    lon_centers = lon_min + np.arange(num_lon_parts) + 0.5 * step
    lat_grid, lon_grid = np.meshgrid(lat_centers, lon_centers, indexing="ij")
    lat_grid_flat = lat_grid.ravel()
    lon_grid_flat = lon_grid.ravel()
    grid_points = np.stack((lat_grid_flat, lon_grid_flat), axis=1)

    # Keep only ocean grids sufficiently far from land, as in the original code.
    if len(latlon_land) > 0:
        land_dist = (
            np.abs(lat_grid_flat[:, None] - latlon_land[:, 0])
            + np.abs(lon_grid_flat[:, None] - latlon_land[:, 1])
        )
        valid_grid_mask = np.min(land_dist, axis=1) >= 1.001
    else:
        valid_grid_mask = np.ones(lat_grid_flat.shape, dtype=bool)

    results = []

    for idx in np.where(valid_grid_mask)[0]:
        lat_1deg, lon_1deg = grid_points[idx]

        sub_lat_min = lat_1deg - 0.5 * step
        sub_lat_max = lat_1deg + 0.5 * step
        sub_lon_min = lon_1deg - 0.5 * step
        sub_lon_max = lon_1deg + 0.5 * step

        overlap_lon = 0.375 / np.cos(np.radians(lat_1deg))
        sub_lat_min_mod = max(sub_lat_min - overlap_lat, lat_min)
        sub_lat_max_mod = min(sub_lat_max + overlap_lat, lat_max)
        sub_lon_min_mod = max(sub_lon_min - overlap_lon, lon_min)
        sub_lon_max_mod = min(sub_lon_max + overlap_lon, lon_max)

        # Half-open grid bounds ensure one CERES footprint center belongs to
        # only one 1° grid even if it lies exactly on an integer boundary.
        cer_mask = (
            (latlon_cer[:, 0] >= sub_lat_min)
            & (latlon_cer[:, 0] < sub_lat_max)
            & (latlon_cer[:, 1] >= sub_lon_min)
            & (latlon_cer[:, 1] < sub_lon_max)
        )
        cer_indices = np.where(cer_mask)[0]
        if cer_indices.size == 0:
            continue

        mod_mask = (
            (latlon_mod[:, 0] >= sub_lat_min_mod)
            & (latlon_mod[:, 0] <= sub_lat_max_mod)
            & (latlon_mod[:, 1] >= sub_lon_min_mod)
            & (latlon_mod[:, 1] <= sub_lon_max_mod)
        )
        if np.sum(mod_mask) <= 1:
            continue

        sub_latlon_mod = latlon_mod[mod_mask]
        sub_ret_flag = ret_flag[mod_mask]
        sub_cot_mod = cot_mod[mod_mask]

        for cer_idx in cer_indices:
            delta, beta = uft.calc_delta_beta(
                latlon_cer[cer_idx : cer_idx + 1],
                sub_latlon_mod,
                latlon_subsat[cer_idx : cer_idx + 1],
            )

            delta = np.asarray(delta).ravel()
            beta = np.asarray(beta).ravel()
            in_94power = (np.abs(beta) < 1.32) & (np.abs(delta) <= 1.32)

            fov_ret = process_single_footprint(
                ret_flag=sub_ret_flag,
                in_94power=in_94power,
                beta=beta,
                delta=delta,
                albedo_all=albedo_all[cer_idx],
                cot_mod=sub_cot_mod,
            )

            if fov_ret is None:
                continue

            results.append(
                [
                    lat_1deg,
                    lon_1deg,
                    *fov_ret,
                    solar_zenith_cer[cer_idx],
                    sensor_zenith_cer[cer_idx],
                ]
            )

    return results


if __name__ == "__main__":
    start = time.perf_counter()
    print(f"Code version: {CODE_VERSION}")

    if len(sys.argv) < 4:
        print("Usage: python script.py <year> <month> <hemisphere>")
        sys.exit(1)

    year = sys.argv[1]
    month = int(sys.argv[2])
    hemisph = sys.argv[3]

    if hemisph == "east":
        obs_window = [[-60.0, 60.0], [0.0, 180.0]]
    elif hemisph == "west":
        obs_window = [[-60.0, 60.0], [-180.0, 0.0]]
    else:
        raise ValueError("Only support hemisphere to be 'east' or 'west'.")

    # ---------------------- land coordinates ----------------------
    landsea_file = "/data/chenyiqi/251007_tropic/landsea.nc"
    with nc.Dataset(landsea_file, "r") as ds:
        lsmask = ds.variables["LSMASK"][:].ravel()
        lat_ls = ds.variables["lat"][:]
        lon_ls = ds.variables["lon"][:].copy()

    lon_ls[lon_ls > 180] -= 360
    lon_mesh, lat_mesh = np.meshgrid(lon_ls, lat_ls)
    lat_flat = lat_mesh.ravel()
    lon_flat = lon_mesh.ravel()
    land_mask = (
        (lsmask == 1)
        & (lat_flat > obs_window[0][0])
        & (lat_flat < obs_window[0][1])
        & (lon_flat > obs_window[1][0])
        & (lon_flat < obs_window[1][1])
    )
    latlon_land = np.stack((lat_flat[land_mask], lon_flat[land_mask]), axis=1)

    # ------------------ MODIS/CERES file lists ------------------
    mod_pkl = (
        f"/data/chenyiqi/251028_albedo_cot/mod06/"
        f"MOD06_files_{year}{month:02d}_{hemisph}.pkl"
    )
    with open(mod_pkl, "rb") as f:
        mod_file_lst = pickle.load(f)["terra"]

    cer_pkl = (
        f"/data/chenyiqi/251028_albedo_cot/CERES_L2SSF_2020/"
        f"CERES_{year}_files_and_times.pkl"
    )
    with open(cer_pkl, "rb") as f:
        ssf_files_times_lst = pickle.load(f)["ssf_files"]

    csv_fname = (
        f"/data/chenyiqi/251028_albedo_cot/project0904/uniform_fov_product/"
        f"rsl_fov_{year}{month:02d}_{hemisph}.csv"
    )

    csv_header = [
        "time",
        "lat",
        "lon",
        "ret_albedo",
        "ret_fov_fra",
        *COT_FREQ_COLUMNS,
        "solar_zenith",
        "sensor_zenith",
    ]

    num_mod_files = len(mod_file_lst)

    for i_mod, mod_filename in enumerate(mod_file_lst):
        # ---------------- find corresponding CERES file ----------------
        parts = mod_filename.split(".")
        timestamp_str = parts[1][1:] + parts[2]
        time_mod = datetime.strptime(timestamp_str, "%Y%j%H%M")

        ceres_filename = None
        for item in ssf_files_times_lst:
            if item["start_time"] <= time_mod <= item["end_time"]:
                ceres_filename = item["filename"]
                break

        if ceres_filename is None:
            print(time_mod, "has no CERES-Terra data")
            continue

        # ---------------- read/process MODIS ----------------
        print(mod_filename)
        hdf = SD(mod_filename, SDC.READ)

        lat_mod = hdf.select("Latitude")[:]
        lat_mod[lat_mod == -999] = np.nan
        lon_mod = hdf.select("Longitude")[:]
        lon_mod[lon_mod == -999] = np.nan

        sensor_zenith_mod = uft.read_and_mask_mod_variable(hdf, "Sensor_Zenith")
        solar_zenith_mod = uft.read_and_mask_mod_variable(hdf, "Solar_Zenith")
        cot_mod = uft.read_and_mask_mod_variable(hdf, "Cloud_Optical_Thickness")
        ctt_mod = uft.read_and_mask_mod_variable(hdf, "cloud_top_temperature_1km")

        qa1km = hdf.select("Quality_Assurance_1km").get()
        byte2 = qa1km[:, :, 2]
        cloud_phase_flag = (byte2 >> 0) & 0b111
        retrieval_outcome_flag = (byte2 >> 3) & 0b1

        cm1km = hdf.select("Cloud_Mask_1km").get()
        byte0 = cm1km[:, :, 0]
        cloudiness_flag = (byte0 >> 1) & 0b11
        sunglint_flag = (byte0 >> 4) & 0b1
        hdf.end()

        lat_mod, lon_mod, solar_zenith_mod, sensor_zenith_mod = (
            uft.upscale_and_interpolate(
                lat_mod,
                lon_mod,
                solar_zenith_mod,
                sensor_zenith_mod,
                cot_mod.shape,
                obs_window,
            )
        )

        # True only for retrievable, cloudy, warm liquid-cloud pixels.
        ret_flag = (
            (cloud_phase_flag == 2)
            & (retrieval_outcome_flag == 1)
            & (cloudiness_flag <= 1)
            & (ctt_mod >= 270)
        )

        lon_min = max(obs_window[1][0], np.nanmin(lon_mod))
        lon_max = min(obs_window[1][1], np.nanmax(lon_mod))
        lat_min = max(obs_window[0][0], np.nanmin(lat_mod))
        lat_max = min(obs_window[0][1], np.nanmax(lat_mod))

        time_low = time_mod - timedelta(minutes=5)
        time_high = time_mod + timedelta(minutes=5)

        valid_mod = (
            (sensor_zenith_mod < 55)
            & (solar_zenith_mod < 55)
            & (lat_mod > lat_min)
            & (lat_mod < lat_max)
            & (lon_mod > lon_min)
            & (lon_mod < lon_max)
            & (sunglint_flag == 1)
        )

        if not np.any(valid_mod):
            print(f"File {i_mod} in {num_mod_files}, 0 footprint recorded")
            continue

        # These arrays must use the same MODIS-pixel mask to remain aligned.
        cot_mod = cot_mod[valid_mod].ravel()
        ret_flag = ret_flag[valid_mod].ravel()
        lat_mod = lat_mod[valid_mod].ravel()
        lon_mod = lon_mod[valid_mod].ravel()
        latlon_mod = np.stack((lat_mod, lon_mod), axis=1)

        # ---------------- read/process CERES ----------------
        with xr.open_dataset(ceres_filename) as ds_ssf:
            time_ssf = uft.julian_to_datetime(ds_ssf["Time_of_observation"].values)

            lon_ssf = ds_ssf["Longitude_of_CERES_FOV_at_surface"].values.copy()
            lon_ssf[lon_ssf > 180] -= 360
            lat_ssf = 90 - ds_ssf["Colatitude_of_CERES_FOV_at_surface"].values
            latlon_ssf = np.stack((lat_ssf, lon_ssf), axis=1)

            lon_subsat = ds_ssf[
                "Longitude_of_subsatellite_point_at_surface_at_observation"
            ].values.copy()
            lon_subsat[lon_subsat > 180] -= 360
            lat_subsat = 90 - ds_ssf[
                "Colatitude_of_subsatellite_point_at_surface_at_observation"
            ].values
            latlon_subsat = np.stack((lat_subsat, lon_subsat), axis=1)

            sw_incoming = ds_ssf["TOA_Incoming_Solar_Radiation"].values
            sw_toa_all = ds_ssf["CERES_SW_TOA_flux___upwards"].values
            solar_zenith_cer = ds_ssf["CERES_solar_zenith_at_surface"].values
            sensor_zenith_cer = ds_ssf["CERES_viewing_zenith_at_surface"].values

        time_cond = np.array(
            [(time_low <= t <= time_high) for t in time_ssf],
            dtype=bool,
        )
        valid_cer = (
            time_cond
            & (latlon_ssf[:, 0] > lat_min)
            & (latlon_ssf[:, 0] < lat_max)
            & (latlon_ssf[:, 1] > lon_min)
            & (latlon_ssf[:, 1] < lon_max)
            & (sensor_zenith_cer < 55)
            & (solar_zenith_cer < 55)
        )

        if not np.any(valid_cer):
            print(
                f"File {i_mod}/{num_mod_files}, month {month:02d}, "
                "0 footprint recorded"
            )
            continue

        latlon_cer = latlon_ssf[valid_cer]
        latlon_subsat = latlon_subsat[valid_cer]
        solar_zenith_cer = solar_zenith_cer[valid_cer]
        sensor_zenith_cer = sensor_zenith_cer[valid_cer]
        albedo_all = sw_toa_all[valid_cer] / sw_incoming[valid_cer]

        # ---------------- MODIS-CERES matching ----------------
        rsl_lst = get_uniform_fov_product(
            latlon_cer,
            latlon_mod,
            latlon_subsat,
            latlon_land,
            ret_flag,
            cot_mod,
            solar_zenith_cer,
            sensor_zenith_cer,
            albedo_all,
        )

        if rsl_lst:
            rsl_lst = [[time_mod, *row] for row in rsl_lst]

        print(
            f"File {i_mod} in month {month:02d}, "
            f"{len(rsl_lst)} footprint(s) recorded"
        )

        # ---------------- save ----------------
        file_exists = os.path.exists(csv_fname)
        df = pd.DataFrame(rsl_lst, columns=csv_header)
        df.to_csv(
            csv_fname,
            mode="a" if file_exists else "w",
            header=not file_exists,
            index=False,
        )

    end = time.perf_counter()
    print(f"Run time: {end - start:.6f} s. Output file: {csv_fname}")
