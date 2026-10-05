# -*- coding: utf-8 -*-
"""Find two RFOV examples and plot their granule, grid, and CERES weights."""

import argparse
import os
import pickle
from datetime import datetime, timedelta

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import xarray as xr
from matplotlib.cm import ScalarMappable
from pyhdf.SD import SD, SDC

import utils_retrievable_fov as urf
from utils_fitting import format_panel_tag


DATA_DIR = "/data/chenyiqi/251028_albedo_cot"
PROJECT_DIR = "/home/chenyiqi/251028_albedo_cot/project0904"
LANDSEA_FILE = "/data/chenyiqi/251007_tropic/landsea.nc"
MAX_FIGURES = 4
PRIMARY_FRA_THRE = 1.0

_WIJ_TOP = np.array([
    [0.0000, 0.0018, 0.0091, 0.0116, 0.0074, 0.0038, 0.0020, 0.0010],
    [0.0019, 0.0016, 0.0248, 0.0310, 0.0248, 0.0142, 0.0073, 0.0038],
    [0.0055, 0.0191, 0.0304, 0.0362, 0.0334, 0.0213, 0.0111, 0.0058],
    [0.0055, 0.0191, 0.0304, 0.0362, 0.0334, 0.0213, 0.0111, 0.0058],
], dtype=float)
FOV_WEIGHTS = np.vstack((_WIJ_TOP, np.flipud(_WIJ_TOP)))
FOV_WEIGHTS /= np.sum(FOV_WEIGHTS)
BETA_BINS = np.arange(1.32, -1.32 + 0.01, -0.33)
DELTA_BINS = np.arange(-1.32, 1.32 - 0.01, 0.33)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find and plot the first two retrievable CERES FOVs."
    )
    parser.add_argument("year", nargs="?", default="2020")
    parser.add_argument("month", nargs="?", type=int, default=6)
    parser.add_argument(
        "hemisphere", nargs="?", choices=("east", "west"), default="west"
    )
    return parser.parse_args()


def get_obs_window(hemisphere):
    if hemisphere == "east":
        return [[-60.0, 60.0], [0.0, 180.0]]
    return [[-60.0, 60.0], [-180.0, 0.0]]


def read_land_coordinates(obs_window):
    with nc.Dataset(LANDSEA_FILE, "r") as ds:
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
    return np.stack((lat_flat[land_mask], lon_flat[land_mask]), axis=1)


def footprint_weights(ret_flag, in_94power, beta, delta, albedo, cot_cer):
    """Return per-pixel response-bin weights when a footprint is an RFOV."""
    ret_flag = np.asarray(ret_flag, dtype=bool).ravel()
    in_94power = np.asarray(in_94power, dtype=bool).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()

    if not np.any(in_94power):
        return None
    if np.mean(ret_flag[in_94power]) < PRIMARY_FRA_THRE:
        return None
    if not np.isfinite(albedo):
        return None

    finite_cot = np.asarray(cot_cer, dtype=float).ravel()
    finite_cot = finite_cot[np.isfinite(finite_cot)]
    if finite_cot.size == 0:
        return None

    weights = np.full(in_94power.shape, np.nan, dtype=float)
    for ii, beta_edge in enumerate(BETA_BINS):
        mask_beta = (beta < beta_edge) & (beta >= beta_edge - 0.33)
        for jj, delta_edge in enumerate(DELTA_BINS):
            mask_angle = (
                mask_beta & (delta > delta_edge) & (delta <= delta_edge + 0.33)
            )
            if not np.any(mask_angle):
                return None
            weights[mask_angle] = FOV_WEIGHTS[ii, jj]

    return weights, float(finite_cot[0])


def find_rfov_cases(
    latlon_cer, latlon_mod, latlon_subsat, latlon_land, ret_flag,
    albedo_all, cot_cer, obs_window, max_cases,
):
    """Return RFOVs using the selection in 1_RFOV_product_per_footprint.py."""
    lat_min = np.ceil(max(np.nanmin(latlon_mod[:, 0]), obs_window[0][0]))
    lat_max = np.floor(min(np.nanmax(latlon_mod[:, 0]), obs_window[0][1]))
    lon_min = np.ceil(max(np.nanmin(latlon_mod[:, 1]), obs_window[1][0]))
    lon_max = np.floor(min(np.nanmax(latlon_mod[:, 1]), obs_window[1][1]))
    num_lat_parts = int(lat_max - lat_min)
    num_lon_parts = int(lon_max - lon_min)
    if num_lat_parts <= 0 or num_lon_parts <= 0:
        return []

    lat_centers = lat_min + np.arange(num_lat_parts) + 0.5
    lon_centers = lon_min + np.arange(num_lon_parts) + 0.5
    lat_grid, lon_grid = np.meshgrid(lat_centers, lon_centers, indexing="ij")
    grid_points = np.stack((lat_grid.ravel(), lon_grid.ravel()), axis=1)

    if len(latlon_land) > 0:
        land_dist = (
            np.abs(grid_points[:, 0, None] - latlon_land[:, 0])
            + np.abs(grid_points[:, 1, None] - latlon_land[:, 1])
        )
        valid_grid = np.min(land_dist, axis=1) >= 1.001
    else:
        valid_grid = np.ones(len(grid_points), dtype=bool)

    cases = []
    for grid_idx in np.where(valid_grid)[0]:
        grid_lat, grid_lon = grid_points[grid_idx]
        grid_window = [
            [grid_lat - 0.5, grid_lat + 0.5],
            [grid_lon - 0.5, grid_lon + 0.5],
        ]
        cer_mask = (
            (latlon_cer[:, 0] >= grid_window[0][0])
            & (latlon_cer[:, 0] < grid_window[0][1])
            & (latlon_cer[:, 1] >= grid_window[1][0])
            & (latlon_cer[:, 1] < grid_window[1][1])
        )
        cer_indices = np.where(cer_mask)[0]
        if cer_indices.size == 0:
            continue

        overlap_lat = 0.3
        overlap_lon = 0.375 / np.cos(np.radians(grid_lat))
        mod_mask = (
            (latlon_mod[:, 0] >= max(grid_window[0][0] - overlap_lat, lat_min))
            & (latlon_mod[:, 0] <= min(grid_window[0][1] + overlap_lat, lat_max))
            & (latlon_mod[:, 1] >= max(grid_window[1][0] - overlap_lon, lon_min))
            & (latlon_mod[:, 1] <= min(grid_window[1][1] + overlap_lon, lon_max))
        )
        if np.sum(mod_mask) <= 1:
            continue

        sub_latlon_mod = latlon_mod[mod_mask]
        sub_ret_flag = ret_flag[mod_mask]
        for cer_idx in cer_indices:
            delta, beta = urf.calc_delta_beta(
                latlon_cer[cer_idx:cer_idx + 1],
                sub_latlon_mod,
                latlon_subsat[cer_idx:cer_idx + 1],
            )
            delta = np.asarray(delta).ravel()
            beta = np.asarray(beta).ravel()
            in_94power = (np.abs(beta) < 1.32) & (np.abs(delta) <= 1.32)
            result = footprint_weights(
                sub_ret_flag, in_94power, beta, delta,
                albedo_all[cer_idx], cot_cer[cer_idx],
            )
            if result is None:
                continue

            weights, cot_value = result
            cases.append({
                "grid_window": grid_window,
                "center": latlon_cer[cer_idx],
                "albedo": float(albedo_all[cer_idx]),
                "cot": cot_value,
                "weight_latlon": sub_latlon_mod,
                "weights": weights,
            })
            if len(cases) >= max_cases:
                return cases
    return cases


def longitude_label(value, decimals=0):
    if value < 0:
        return f"{abs(value):.{decimals}f}°W"
    if value > 0:
        return f"{value:.{decimals}f}°E"
    return "0°"


def latitude_label(value, decimals=0):
    if value < 0:
        return f"{abs(value):.{decimals}f}°S"
    if value > 0:
        return f"{value:.{decimals}f}°N"
    return "0°"


def format_geo_axes(ax, grid_window=None, decimals=0):
    if grid_window is not None:
        ax.set_xlim(grid_window[1])
        ax.set_ylim(grid_window[0])
        ax.set_xticks(np.linspace(grid_window[1][0], grid_window[1][1], 4))
        ax.set_yticks(np.linspace(grid_window[0][0], grid_window[0][1], 5))
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([longitude_label(x, decimals) for x in xticks], fontsize=7.5)
    ax.set_yticklabels([latitude_label(y, decimals) for y in yticks], fontsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.5)


def plot_case(case, granule, time_mod, figure_number, icon_style="science"):
    grid_window = case["grid_window"]
    lat_min, lat_max = grid_window[0]
    lon_min, lon_max = grid_window[1]
    regional_valid = (
        (granule["lat_valid"] > lat_min) & (granule["lat_valid"] < lat_max)
        & (granule["lon_valid"] > lon_min) & (granule["lon_valid"] < lon_max)
    )
    regional_invalid = (
        granule["invalid_mask"]
        & (granule["lat_all"] > lat_min) & (granule["lat_all"] < lat_max)
        & (granule["lon_all"] > lon_min) & (granule["lon_all"] < lon_max)
    )

    cmap_cloud = mcolors.ListedColormap(["grey", "red", "orange", "thistle"])
    norm_cloud = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    cmap_cloud_bar = mcolors.ListedColormap(
        ["lightgray", "grey", "red", "orange", "thistle"]
    )
    norm_cloud_bar = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], 5)

    fig = plt.figure(figsize=(11.8, 4.0))
    gs = fig.add_gridspec(
        1, 3, wspace=0.30, left=0.06, right=0.97, bottom=0.23, top=0.88
    )
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]

    ax_a.scatter(
        granule["lon_all"][granule["invalid_mask"]],
        granule["lat_all"][granule["invalid_mask"]],
        s=0.04, c="lightgray", edgecolors="none",
    )
    ax_a.scatter(
        granule["lon_valid"], granule["lat_valid"],
        c=granule["type_valid"], s=0.04, cmap=cmap_cloud,
        norm=norm_cloud, edgecolors="none",
    )
    ax_a.add_patch(patches.Rectangle(
        (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
        linewidth=1.5, edgecolor="blue", facecolor="none",
    ))
    format_geo_axes(ax_a)

    ax_b.scatter(
        granule["lon_all"][regional_invalid], granule["lat_all"][regional_invalid],
        s=8, c="lightgray", edgecolors="none",
    )
    ax_b.scatter(
        granule["lon_valid"][regional_valid], granule["lat_valid"][regional_valid],
        c=granule["type_valid"][regional_valid], s=8,
        cmap=cmap_cloud, norm=norm_cloud, edgecolors="none",
    )
    center_lat, center_lon = case["center"]
    ax_b.scatter(center_lon, center_lat, c="black", s=18, zorder=5)
    format_geo_axes(ax_b, grid_window, decimals=1)

    ax_c.scatter(
        case["weight_latlon"][:, 1], case["weight_latlon"][:, 0],
        s=8, c="lightgray", edgecolors="none",
    )
    weight_mask = np.isfinite(case["weights"])
    weight_scatter = ax_c.scatter(
        case["weight_latlon"][weight_mask, 1],
        case["weight_latlon"][weight_mask, 0],
        c=case["weights"][weight_mask], s=13, cmap="viridis",
        vmin=0.0, vmax=np.max(FOV_WEIGHTS), edgecolors="none", zorder=3,
    )
    ax_c.scatter(center_lon, center_lat, c="black", s=18, zorder=5)
    format_geo_axes(ax_c, grid_window, decimals=1)

    for panel_index, ax in enumerate((ax_a, ax_b, ax_c)):
        ax.text(
            -0.01, 1.01, format_panel_tag(panel_index, icon_style),
            transform=ax.transAxes, fontsize=14, va="bottom", ha="left",
        )

    pos_a, pos_b, pos_c = [ax.get_position() for ax in (ax_a, ax_b, ax_c)]
    cloud_cax = fig.add_axes(
        [pos_a.x0, pos_a.y0 - 0.115, pos_b.x1 - pos_a.x0, 0.030]
    )
    cloud_mappable = ScalarMappable(cmap=cmap_cloud_bar, norm=norm_cloud_bar)
    cloud_mappable.set_array([])
    cloud_cbar = fig.colorbar(
        cloud_mappable, cax=cloud_cax, orientation="horizontal",
        ticks=[0, 1, 2, 3, 4], spacing="proportional",
    )
    cloud_cbar.ax.set_xticklabels([
        "Sunglint or\nZenith Angles > 55°", "Clear Sky",
        "Unretrieved\nLiquid Cloud", "Retrieved\nLiquid Cloud", "Ice Cloud",
    ], fontsize=9)
    cloud_cbar.ax.tick_params(axis="x", pad=4)

    weight_cax = fig.add_axes([pos_c.x0, pos_c.y0 - 0.115, pos_c.width, 0.030])
    weight_cbar = fig.colorbar(weight_scatter, cax=weight_cax, orientation="horizontal")
    weight_cbar.set_label("CERES Response Weight", fontsize=9)
    weight_cbar.ax.tick_params(labelsize=8)

    output_dir = os.path.join(PROJECT_DIR, "figs")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time_mod.strftime("%Y%j_%H%M")
    output_path = os.path.join(
        output_dir, f"figsupp_method_RFOV_{figure_number:02d}_{timestamp}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    obs_window = get_obs_window(args.hemisphere)
    latlon_land = read_land_coordinates(obs_window)
    mod_pkl = os.path.join(
        DATA_DIR, "mod06",
        f"MOD06_files_{args.year}{args.month:02d}_{args.hemisphere}.pkl",
    )
    cer_pkl = os.path.join(
        DATA_DIR, "CERES_L2SSF_2020", f"CERES_{args.year}_files_and_times.pkl",
    )
    with open(mod_pkl, "rb") as stream:
        mod_file_list = pickle.load(stream)["terra"]
    with open(cer_pkl, "rb") as stream:
        ssf_files_times = pickle.load(stream)["ssf_files"]

    figure_count = 0
    for file_index, mod_filename in enumerate(mod_file_list):
        basename_parts = os.path.basename(mod_filename).split(".")
        timestamp = basename_parts[1][1:] + basename_parts[2]
        time_mod = datetime.strptime(timestamp, "%Y%j%H%M")
        ceres_filename = next((
            item["filename"] for item in ssf_files_times
            if item["start_time"] <= time_mod <= item["end_time"]
        ), None)
        if ceres_filename is None:
            continue

        print(f"[{file_index + 1}/{len(mod_file_list)}] {mod_filename}")
        hdf = SD(mod_filename, SDC.READ)
        lat_coarse = hdf.select("Latitude")[:].astype(float)
        lon_coarse = hdf.select("Longitude")[:].astype(float)
        lat_coarse[lat_coarse == -999] = np.nan
        lon_coarse[lon_coarse == -999] = np.nan
        sensor_zenith = urf.read_and_mask_mod_variable(hdf, "Sensor_Zenith")
        solar_zenith = urf.read_and_mask_mod_variable(hdf, "Solar_Zenith")
        cot_mod = urf.read_and_mask_mod_variable(hdf, "Cloud_Optical_Thickness")
        ctt_mod = urf.read_and_mask_mod_variable(hdf, "cloud_top_temperature_1km")
        qa1km = hdf.select("Quality_Assurance_1km").get()
        byte2 = qa1km[:, :, 2]
        cloud_phase = (byte2 >> 0) & 0b111
        retrieval_outcome = (byte2 >> 3) & 0b1
        cm1km = hdf.select("Cloud_Mask_1km").get()
        byte0 = cm1km[:, :, 0]
        cloudiness = (byte0 >> 1) & 0b11
        sunglint = (byte0 >> 4) & 0b1
        hdf.end()

        lat_mod, lon_mod, solar_zenith, sensor_zenith = urf.upscale_and_interpolate(
            lat_coarse, lon_coarse, solar_zenith, sensor_zenith,
            cot_mod.shape, obs_window,
        )
        ret_flag = (
            (cloud_phase == 2) & (retrieval_outcome == 1)
            & (cloudiness <= 1) & (ctt_mod >= 270)
        )
        retrieved_liquid = (cloud_phase == 2) & (retrieval_outcome == 1)
        cloudy_liquid = (cloud_phase == 2) & (cloudiness <= 1)
        ice_or_cold = (
            ((cloud_phase != 2) & (retrieval_outcome == 1)) | (ctt_mod < 270)
        )
        cloud_type = retrieved_liquid.astype(int) + cloudy_liquid.astype(int)
        cloud_type = np.where(ice_or_cold, 3, cloud_type)

        lon_min = max(obs_window[1][0], np.nanmin(lon_mod))
        lon_max = min(obs_window[1][1], np.nanmax(lon_mod))
        lat_min = max(obs_window[0][0], np.nanmin(lat_mod))
        lat_max = min(obs_window[0][1], np.nanmax(lat_mod))
        valid_mod = (
            (sensor_zenith < 55) & (solar_zenith < 55)
            & (lat_mod > lat_min) & (lat_mod < lat_max)
            & (lon_mod > lon_min) & (lon_mod < lon_max)
            & (sunglint == 1)
        )
        if not np.any(valid_mod):
            continue

        granule = {
            "lat_all": lat_mod, "lon_all": lon_mod, "invalid_mask": ~valid_mod,
            "lat_valid": lat_mod[valid_mod].ravel(),
            "lon_valid": lon_mod[valid_mod].ravel(),
            "type_valid": cloud_type[valid_mod].ravel(),
        }
        ret_flag_valid = ret_flag[valid_mod].ravel()
        latlon_mod = np.stack(
            (granule["lat_valid"], granule["lon_valid"]), axis=1
        )

        time_low = time_mod - timedelta(minutes=5)
        time_high = time_mod + timedelta(minutes=5)
        with xr.open_dataset(ceres_filename) as ds_ssf:
            time_ssf = urf.julian_to_datetime(ds_ssf["Time_of_observation"].values)
            lon_cer = ds_ssf["Longitude_of_CERES_FOV_at_surface"].values.copy()
            lon_cer[lon_cer > 180] -= 360
            lat_cer = 90 - ds_ssf["Colatitude_of_CERES_FOV_at_surface"].values
            latlon_cer_all = np.stack((lat_cer, lon_cer), axis=1)
            lon_subsat = ds_ssf[
                "Longitude_of_subsatellite_point_at_surface_at_observation"
            ].values.copy()
            lon_subsat[lon_subsat > 180] -= 360
            lat_subsat = 90 - ds_ssf[
                "Colatitude_of_subsatellite_point_at_surface_at_observation"
            ].values
            latlon_subsat_all = np.stack((lat_subsat, lon_subsat), axis=1)
            sw_incoming = ds_ssf["TOA_Incoming_Solar_Radiation"].values
            sw_up = ds_ssf["CERES_SW_TOA_flux___upwards"].values
            cot_cer_all = ds_ssf[
                "Mean_visible_optical_depth_for_cloud_layer"
            ].values
            solar_zenith_cer = ds_ssf["CERES_solar_zenith_at_surface"].values
            sensor_zenith_cer = ds_ssf["CERES_viewing_zenith_at_surface"].values

        time_cond = np.array(
            [time_low <= value <= time_high for value in time_ssf], dtype=bool
        )
        valid_cer = (
            time_cond
            & (latlon_cer_all[:, 0] > lat_min) & (latlon_cer_all[:, 0] < lat_max)
            & (latlon_cer_all[:, 1] > lon_min) & (latlon_cer_all[:, 1] < lon_max)
            & (sensor_zenith_cer < 55) & (solar_zenith_cer < 55)
            & np.isfinite(sw_incoming) & (sw_incoming > 0) & np.isfinite(sw_up)
        )
        if not np.any(valid_cer):
            continue

        latlon_cer = latlon_cer_all[valid_cer]
        latlon_subsat = latlon_subsat_all[valid_cer]
        albedo_all = sw_up[valid_cer] / sw_incoming[valid_cer]
        cot_cer = cot_cer_all[valid_cer]
        cases = find_rfov_cases(
            latlon_cer, latlon_mod, latlon_subsat, latlon_land,
            ret_flag_valid, albedo_all, cot_cer, obs_window,
            MAX_FIGURES - figure_count,
        )
        for case in cases:
            figure_count += 1
            plot_case(case, granule, time_mod, figure_count)
            if figure_count >= MAX_FIGURES:
                print("Generated two RFOV figures; stopping.")
                return

    print(f"Search completed; generated {figure_count} RFOV figure(s).")


if __name__ == "__main__":
    main()
