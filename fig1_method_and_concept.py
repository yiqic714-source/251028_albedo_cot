# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime, timedelta
from pyhdf.SD import SD, SDC
from scipy.interpolate import RegularGridInterpolator
from scipy import stats
import pickle
import xarray as xr
import os
import utils_geom as gu
import utils_uniform_fov as uft
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from utils_fitting import format_panel_tag

# Set weight for each angle bin within CERES FOV
wij = np.array([
    [.0000, .0018, .0091, .0116, .0074, .0038, .0020, .0010],
    [.0019, .0016, .0248, .0310, .0248, .0142, .0073, .0038],
    [.0055, .0191, .0304, .0362, .0334, .0213, .0111, .0058],
    [.0055, .0191, .0304, .0362, .0334, .0213, .0111, .0058]
])
wij_flipped = np.flipud(wij)
wij = np.vstack((wij, wij_flipped))
wij = wij / np.sum(wij.flatten())
beta_bins = np.arange(1.32, -1.32 + 0.01, -0.33)
delta_bins = np.arange(-1.32, 1.32 - 0.01, 0.33)


# Calculate delta and beta angles between CERES and MODIS
def calc_delta_beta(latlon_cer, latlon_mod, latlon_subsat,
                    sensor_altitude=705., earth_radius=6367.):
    eq_sat = gu.get_equatorial_vectors(
        latitude=latlon_subsat[..., 0],
        longitude=latlon_subsat[..., 1],
    )
    eq_cer = gu.get_equatorial_vectors(
        latitude=latlon_cer[..., 0],
        longitude=latlon_cer[..., 1],
    )
    eq_mod = gu.get_equatorial_vectors(
        latitude=latlon_mod[..., 0],
        longitude=latlon_mod[..., 1],
    )

    CX, CY, CZ = np.split(np.expand_dims(gu.get_view_vectors(
        sensor_equatorial_vectors=eq_sat,
        pixel_equatorial_vectors=eq_cer,
        sensor_altitude=sensor_altitude,
        earth_radius=earth_radius,
    ), axis=(1, 2)), 3, axis=-1)

    MX, MY, MZ = np.split(gu.get_view_vectors(
        sensor_equatorial_vectors=np.expand_dims(eq_sat, axis=(1, 2)),
        pixel_equatorial_vectors=eq_mod,
        sensor_altitude=sensor_altitude,
        earth_radius=earth_radius,
    ), 3, axis=-1)

    delta = np.rad2deg(np.arcsin(np.sum(MY * CZ, axis=-1)))
    delta = np.squeeze(delta)

    tmp = np.cross(CZ, MY)
    tmp /= np.linalg.norm(tmp, axis=-1, keepdims=True)
    beta = np.rad2deg(np.arcsin(-1. * np.sum(tmp * CY, axis=-1)))
    beta = np.squeeze(beta)

    return delta, beta


# Calculate pixel weight for single cloud type
def get_pixel_weight_1type(
    primary_flag,
    nan_flag,
    in_94power,
    beta,
    delta,
    albedo_all,
    secondary_albedo,
    secondary_uncertainty,
    primary_fra_thre=0.9,
    uncertainty_thre=0.05
):
    pixel_weight = np.zeros(primary_flag.shape)
    possible_mask = np.zeros(albedo_all.shape[0], dtype=bool)

    pixel_num_in_94power = np.sum(in_94power, axis=1)
    fra_coarse = np.zeros_like(pixel_num_in_94power, dtype=float)
    valid = pixel_num_in_94power > 0
    fra_coarse[valid] = np.sum(primary_flag * in_94power[valid, :], axis=1) / pixel_num_in_94power[valid]

    has_nan = np.any(np.isnan(nan_flag) & in_94power, axis=1)
    possible_mask = (fra_coarse > primary_fra_thre) & ~has_nan

    if not np.any(possible_mask):
        return pixel_weight, possible_mask

    beta_pure = beta[possible_mask, :]
    delta_pure = delta[possible_mask, :]
    albedo_pure = albedo_all[possible_mask]

    uncertainty = np.zeros_like(albedo_pure)
    primary_fra = np.ones_like(albedo_pure)

    for k in range(len(albedo_pure)):
        if np.isnan(albedo_pure[k]):
            uncertainty[k] = np.inf
            continue

        discard_flag = False
        for ii, beta_edge in enumerate(beta_bins):
            mask_beta = (beta_pure[k, :] < beta_edge) & (beta_pure[k, :] >= beta_edge - 0.33)
            for jj, delta_edge in enumerate(delta_bins):
                mask_angle = mask_beta & (delta_pure[k, :] > delta_edge) & (delta_pure[k, :] <= delta_edge + 0.33)
                pixel_num = np.sum(mask_angle)
                if pixel_num == 0:
                    discard_flag = True
                    continue

                weight_ij = wij[ii, jj]
                uncertainty[k] += np.mean(secondary_uncertainty[mask_angle]) * weight_ij
                albedo_pure[k] -= np.mean(secondary_albedo[mask_angle]) * weight_ij
                primary_fra[k] -= np.mean(primary_flag[mask_angle]) * weight_ij

            if (uncertainty[k] / albedo_pure[k]) > uncertainty_thre * 1.5 or discard_flag:
                uncertainty[k] = np.inf
                break

    valid_mask = (uncertainty / albedo_pure) < uncertainty_thre
    valid_mask = valid_mask & (albedo_pure != 0) & ~np.isnan(albedo_pure)
    beta_valid = beta_pure[valid_mask, :]
    delta_valid = delta_pure[valid_mask, :]
    possible_mask[possible_mask] = valid_mask

    for k in range(len(beta_valid)):
        for ii, beta_edge in enumerate(beta_bins):
            mask_beta = (beta_valid[k, :] < beta_edge) & (beta_valid[k, :] >= beta_edge - 0.33)
            for jj, delta_edge in enumerate(delta_bins):
                mask_angle = mask_beta & (delta_pure[k, :] > delta_edge) & (delta_pure[k, :] <= delta_edge + 0.33)
                if np.any(mask_angle):
                    pixel_weight[mask_angle & primary_flag] += wij[ii, jj]

    return pixel_weight, possible_mask


# Convert Julian day to datetime
def julian_to_datetime(julian_days):
    base_julian = 2440587.5
    base_datetime = datetime(1970, 1, 1, 0, 0, 0)
    days_since_epoch = julian_days - base_julian
    return [base_datetime + timedelta(days=days) for days in days_since_epoch]


# Calculate pixel weight for MODIS within CERES FOV
def get_pixel_weight(
    cld_type, latlon_cer, latlon_mod, latlon_subsat,
    albedo_all, cot_cer,
    cld_albedo_guess=0.3, cld_uncertainty=0.15,
    clr_albedo_guess=0.08, clr_uncertainty=0.03,
    primary_fra_thre=0.65
):
    nan_flag = np.isnan(cld_type)
    nan_fraction = np.mean(nan_flag)
    if nan_fraction > 0.1:
        print("Warning: nan_fra > 0.1, skipping this FOV.")
        pixel_weight_ret = np.full(cld_type.shape, np.nan)
        valid_cer_ret_mask = np.zeros(albedo_all.shape[0], dtype=bool)
        return pixel_weight_ret, valid_cer_ret_mask

    ret_flag = (cld_type == 2)
    unr_flag = (cld_type == 1)
    clr_flag = (cld_type == 0)

    delta, beta = calc_delta_beta(latlon_cer, latlon_mod, latlon_subsat)
    in_94power = (np.abs(beta) < 1.32) & (np.abs(delta) <= 1.32)
    uncertainty_thre = 0.03

    pixel_weight_ret, valid_cer_ret_mask = get_pixel_weight_1type(
        primary_flag=ret_flag,
        nan_flag=nan_flag,
        secondary_albedo=unr_flag * cld_albedo_guess + clr_flag * clr_albedo_guess,
        secondary_uncertainty=unr_flag * cld_uncertainty + clr_flag * clr_uncertainty,
        in_94power=in_94power,
        beta=beta,
        delta=delta,
        albedo_all=albedo_all,
        primary_fra_thre=primary_fra_thre,
        uncertainty_thre=uncertainty_thre
    )

    return pixel_weight_ret, latlon_cer[valid_cer_ret_mask], albedo_all[valid_cer_ret_mask], cot_cer[valid_cer_ret_mask]


# Upscale and interpolate MODIS data to 1km resolution
def upscale_and_interpolate(lat, lon, solar_zenith, sensor_zenith, target_shape):
    Ny, Nx = lat.shape
    Ny_new = target_shape[0]
    Nx_new = target_shape[1]

    y = np.arange(Ny)
    x = np.arange(Nx)

    y_new = np.linspace(0, Ny - 1, Ny_new)
    x_new = np.linspace(0, Nx - 1, Nx_new)

    mesh_y, mesh_x = np.meshgrid(y_new, x_new, indexing='ij')
    points = np.stack([mesh_y.ravel(), mesh_x.ravel()], axis=-1)

    interp_lat = RegularGridInterpolator((y, x), lat)
    interp_lon = RegularGridInterpolator((y, x), lon)
    interp_solar = RegularGridInterpolator((y, x), solar_zenith)
    interp_sensor = RegularGridInterpolator((y, x), sensor_zenith)

    lat_new = interp_lat(points).reshape(Ny_new, Nx_new)
    lon_new = interp_lon(points).reshape(Ny_new, Nx_new)
    solar_zenith_new = interp_solar(points).reshape(Ny_new, Nx_new)
    sensor_zenith_new = interp_sensor(points).reshape(Ny_new, Nx_new)

    return lat_new, lon_new, solar_zenith_new, sensor_zenith_new


# Main process
if __name__ == "__main__":
    # Choose panel tag style here: 'nature' -> (a)(b)(c), 'science' -> A B C.
    icon_style = 'nature'
    if icon_style not in ('nature', 'science'):
        raise ValueError("icon_style must be 'nature' or 'science'.")

    # File paths
    mod_filename = "/home/chenyiqi/251028_albedo_cot/mod06/202006N/MOD06_L2.A2020172.2000.061.2020173075052.pscs_000502425053.hdf"
    cer_pkl = "/home/chenyiqi/251028_albedo_cot/CERES_L2SSF_2020/CERES_2020_files_and_times.pkl"

    # Load CERES file list
    with open(cer_pkl, 'rb') as f:
        ssf_files_times_lst = pickle.load(f)['ssf_files']

    # Match corresponding CERES file
    parts = mod_filename.split('.')
    timestamp_str = parts[1][1:] + parts[2]
    time_mod = datetime.strptime(timestamp_str, "%Y%j%H%M")
    cere_ssf_filename = None
    for item in ssf_files_times_lst:
        if item['start_time'] <= time_mod <= item['end_time']:
            cere_ssf_filename = item['filename']
            break

    # Read and process MODIS data
    hdf = SD(mod_filename, SDC.READ)
    lat_mod = hdf.select('Latitude')[:]
    lat_mod[lat_mod == -999] = np.nan
    lon_mod = hdf.select('Longitude')[:]
    lon_mod[lon_mod == -999] = np.nan
    sensor_zenith_mod = uft.read_and_mask_mod_variable(hdf, 'Sensor_Zenith')
    solar_zenith_mod = uft.read_and_mask_mod_variable(hdf, 'Solar_Zenith')
    lon_min = np.nanmin(lon_mod)
    lon_max = np.nanmax(lon_mod)
    if (lon_max - lon_min) > 180:
        lon_mod[lon_mod < 0] = lon_mod[lon_mod < 0] + 360

    # Read 1km MODIS data
    qa1km = hdf.select('Quality_Assurance_1km').get()
    ctt_mod = uft.read_and_mask_mod_variable(hdf, 'cloud_top_temperature_1km')
    byte2 = qa1km[:, :, 2]
    Cloud_Retrieval_Phase_Flag = (byte2 >> 0) & 0b111
    Primary_Cloud_Retrieval_Outcome_Flag = (byte2 >> 3) & 0b1
    cm1km = hdf.select('Cloud_Mask_1km').get()
    byte0 = cm1km[:, :, 0]
    Cloudiness_Flag = (byte0 >> 1) & 0b11
    Sunglint_Flag = (byte0 >> 4) & 0b1
    hdf.end()

    # Upscale MODIS data
    lat_mod, lon_mod, solar_zenith_mod, sensor_zenith_mod = upscale_and_interpolate(
        lat_mod, lon_mod, solar_zenith_mod, sensor_zenith_mod, cm1km.shape
    )

    # Classify cloud types
    cld_retrieval_mask = np.where(
        (Cloud_Retrieval_Phase_Flag == 2) &
        (Primary_Cloud_Retrieval_Outcome_Flag == 1), 1, 0
    )
    cld_mask = np.where(
        (Cloud_Retrieval_Phase_Flag == 2) &
        (Cloudiness_Flag <= 1), 1, 0
    )
    nan_mask = (
        ((Cloud_Retrieval_Phase_Flag != 2) &
         (Primary_Cloud_Retrieval_Outcome_Flag == 1)) |
        (ctt_mod < 273.15 - 5)
    )
    cld_type = cld_retrieval_mask.astype(int) + cld_mask.astype(int)
    cld_type = np.where(nan_mask, 3, cld_type)

    # Save full original data (unfiltered)
    lat_mod_full_original = lat_mod.copy()
    lon_mod_full_original = lon_mod.copy()
    cld_type_full_original = cld_type.copy()
    solar_zenith_mod_original = solar_zenith_mod.copy()
    sensor_zenith_mod_original = sensor_zenith_mod.copy()
    Sunglint_Flag_original = Sunglint_Flag.copy()

    # Define filter condition masks
    valid_mask = (sensor_zenith_mod < 55) & (solar_zenith_mod < 55) & (Sunglint_Flag == 1)
    invalid_mask = ~valid_mask

    # Filter valid data
    indices_valid = np.where(valid_mask)
    cld_type_valid = cld_type[indices_valid].flatten()
    lat_mod_valid = lat_mod[indices_valid].flatten()
    lon_mod_valid = lon_mod[indices_valid].flatten()

    # Backup filtered valid global data
    lat_mod_full = lat_mod_valid.copy()
    lon_mod_full = lon_mod_valid.copy()
    cld_type_full = cld_type_valid.copy()

    # Extract invalid data
    indices_invalid_full = np.where(invalid_mask)
    lat_mod_full_invalid = lat_mod_full_original[indices_invalid_full].flatten()
    lon_mod_full_invalid = lon_mod_full_original[indices_invalid_full].flatten()

    # Define regional grid window
    grid_window = [[54., 55.], [-136., -135.]]
    lat_min, lat_max = grid_window[0]
    lon_min, lon_max = grid_window[1]

    # Regional valid data mask
    regional_valid_mask = (
        (lat_mod_valid > lat_min - 0.5) & (lat_mod_valid < lat_max + 0.5) &
        (lon_mod_valid > lon_min - 0.5) & (lon_mod_valid < lon_max + 0.5)
    )
    indices_regional_valid = np.where(regional_valid_mask)
    cld_type = cld_type_valid[indices_regional_valid].flatten()
    lat_mod = lat_mod_valid[indices_regional_valid].flatten()
    lon_mod = lon_mod_valid[indices_regional_valid].flatten()
    latlon_mod = np.stack((lat_mod, lon_mod), axis=1)

    # Regional invalid data mask
    regional_invalid_mask = (
        invalid_mask &
        (lat_mod_full_original > lat_min - 0.5) &
        (lat_mod_full_original < lat_max + 0.5) &
        (lon_mod_full_original > lon_min - 0.5) &
        (lon_mod_full_original < lon_max + 0.5)
    )
    indices_regional_invalid = np.where(regional_invalid_mask)
    lat_mod_regional_invalid = lat_mod_full_original[indices_regional_invalid].flatten()
    lon_mod_regional_invalid = lon_mod_full_original[indices_regional_invalid].flatten()

    # Read CERES data
    ds_ssf = xr.open_dataset(cere_ssf_filename)
    time_ssf = julian_to_datetime(ds_ssf['Time_of_observation'].values)
    lon_ssf = ds_ssf['Longitude_of_CERES_FOV_at_surface'].values
    lon_ssf[lon_ssf > 180] -= 360
    lat_ssf = 90 - ds_ssf['Colatitude_of_CERES_FOV_at_surface'].values
    latlon_cer = np.stack((lat_ssf, lon_ssf), axis=1)
    sw_incoming = ds_ssf['TOA_Incoming_Solar_Radiation'].values
    sw_toa_all = ds_ssf['CERES_SW_TOA_flux___upwards'].values
    cot_cer = ds_ssf['Mean_visible_optical_depth_for_cloud_layer'].values
    lon_subsat = ds_ssf['Longitude_of_subsatellite_point_at_surface_at_observation'].values
    lon_subsat[lon_subsat > 180] -= 360
    lat_subsat = 90 - ds_ssf['Colatitude_of_subsatellite_point_at_surface_at_observation'].values
    latlon_subsat = np.stack((lat_subsat, lon_subsat), axis=1)

    # Time and spatial condition filter
    time_low = time_mod - timedelta(minutes=5)
    time_high = time_mod + timedelta(minutes=5)
    time_cond = np.array([t >= time_low for t in time_ssf]) & np.array([t <= time_high for t in time_ssf])
    spatial_cond = (
        (latlon_cer[:, 0] > lat_min) & (latlon_cer[:, 0] < lat_max) &
        (latlon_cer[:, 1] > lon_min) & (latlon_cer[:, 1] < lon_max)
    )
    indices = time_cond & spatial_cond

    if not np.any(indices):
        print(f"File has 0 valid grids for MODIS {mod_filename}")
        print("ceres file spatial range:", np.min(lat_ssf), np.max(lat_ssf), np.min(lon_ssf), np.max(lon_ssf))

    latlon_cer = latlon_cer[indices]
    latlon_subsat = latlon_subsat[indices]
    albedo_all = sw_toa_all[indices] / sw_incoming[indices]
    cot_cer = cot_cer[indices]

    # Get pixel weight
    weight_ret, center_latlon, center_albedo, center_cot = get_pixel_weight(
        cld_type, latlon_cer, latlon_mod, latlon_subsat, albedo_all, cot_cer
    )

    # =========================
    # Create figure with new layout (2x2 grid)
    # First two panels are blank with custom titles
    # Panels 3 and 4 are Full Granule and Processed Grid
    # Shared colorbar below the second row, spanning both columns
    # =========================
    fig = plt.figure(figsize=(8, 7))

    gs = fig.add_gridspec(
        2, 2,
        hspace=0.2, wspace=0.25,
        bottom=0.12, top=0.95
    )

    ax1 = fig.add_subplot(gs[0, 0])  # (a) Blank: domains of variables in Eq. (3)
    ax2 = fig.add_subplot(gs[0, 1])  # (b) Blank: data used to build Eq. (3)
    ax3 = fig.add_subplot(gs[1, 0])  # (c) Full Granule
    ax4 = fig.add_subplot(gs[1, 1])  # (d) Processed Grid

    from matplotlib.cm import ScalarMappable

    cmap_cld = mcolors.ListedColormap(['grey', 'red', 'orange', 'thistle'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap_cld.N)

    cmap_cld_cbar = mcolors.ListedColormap(['lightgray', 'grey', 'red', 'orange', 'thistle'])
    bounds_cbar = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm_cbar = mcolors.BoundaryNorm(bounds_cbar, cmap_cld_cbar.N)
    sm_cld_cbar = ScalarMappable(cmap=cmap_cld_cbar, norm=norm_cbar)
    sm_cld_cbar.set_array([])

    # -------------------------
    # (a) Blank: domains of variables in Eq. (3)
    # -------------------------
    ax1.text(
        -0.01, 1.01,
        format_panel_tag(0, icon_style),
        transform=ax1.transAxes,
        fontsize=13.5,
        va='bottom',
        ha='left'
    )
    ax1.set_title('Domains of variables', fontsize=11, loc='center', pad=5.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(True)

    # -------------------------
    # (b) Blank: data used to build Eq. (3)
    # -------------------------
    ax2.text(
        -0.01, 1.01,
        format_panel_tag(1, icon_style),
        transform=ax2.transAxes,
        fontsize=13.5,
        va='bottom',
        ha='left'
    )
    ax2.set_title('Data to build relationships', fontsize=11, loc='center', pad=5.5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(True)

    # -------------------------
    # (c) Full Granule
    # -------------------------
    ax3.scatter(lon_mod_full_invalid, lat_mod_full_invalid, s=0.04, c='lightgray',
                marker='o', edgecolors='None')
    sc1 = ax3.scatter(lon_mod_full, lat_mod_full, s=0.04, c=cld_type_full,
                      marker='o', cmap=cmap_cld, norm=norm, edgecolors='None')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.text(
        -0.01, 1.01,
        format_panel_tag(2, icon_style),
        transform=ax3.transAxes,
        fontsize=14,
        va='bottom',
        ha='left'
    )
    ax3.set_title('Full granule', fontsize=11, loc='center', pad=5.5)

    rect = patches.Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        linewidth=1.5,
        edgecolor='blue',
        facecolor='none'
    )
    ax3.add_patch(rect)

    # Customize longitude ticks for full granule
    lon_ticks_full = ax3.get_xticks()
    lon_tick_labels_full = []
    for lon in lon_ticks_full:
        if not np.isnan(lon):
            if lon < 0:
                lon_tick_labels_full.append(f"{int(abs(lon))}°W")
            elif lon > 0:
                lon_tick_labels_full.append(f"{int(lon)}°E")
            else:
                lon_tick_labels_full.append("0°")
        else:
            lon_tick_labels_full.append("")
    ax3.set_xticklabels(lon_tick_labels_full, fontsize=9)

    # Customize latitude ticks for full granule
    lat_ticks_full = ax3.get_yticks()
    lat_tick_labels_full = []
    for lat in lat_ticks_full:
        if not np.isnan(lat):
            if lat < 0:
                lat_tick_labels_full.append(f"{int(abs(lat))}°S")
            elif lat > 0:
                lat_tick_labels_full.append(f"{int(lat)}°N")
            else:
                lat_tick_labels_full.append("0°")
        else:
            lat_tick_labels_full.append("")
    ax3.set_yticklabels(lat_tick_labels_full, fontsize=9)

    # -------------------------
    # (d) Processed Grid with black FOSRs overlay
    # -------------------------
    ax4.scatter(lon_mod_regional_invalid, lat_mod_regional_invalid, s=8, c='lightgray',
                marker='o', edgecolors='None')
    sc2 = ax4.scatter(lon_mod, lat_mod, c=cld_type, s=8, marker='o',
                      cmap=cmap_cld, norm=norm, edgecolors='None')

    # Add black scatter points for CERES FOV centers
    if center_latlon is not None and len(center_latlon) > 0:
        ax4.scatter(center_latlon[:, 1], center_latlon[:, 0],
                    c='black', s=18, marker='o', zorder=5)

    ax4.set_xlim(grid_window[1])
    ax4.set_ylim(grid_window[0])
    ax4.text(
        -0.01, 1.01,
        format_panel_tag(3, icon_style),
        transform=ax4.transAxes,
        fontsize=14,
        va='bottom',
        ha='left'
    )
    ax4.set_title('Processed grid', fontsize=11, loc='center', pad=5.5)

    lon_start = grid_window[1][0]
    lon_end = grid_window[1][1]
    lon_ticks_reg = np.arange(lon_start, lon_end + 0.01, 0.3)
    ax4.set_xticks(lon_ticks_reg)

    lon_tick_labels_reg = []
    for lon in lon_ticks_reg:
        if not np.isnan(lon):
            if lon < 0:
                lon_tick_labels_reg.append(f"{abs(lon):.1f}°W")
            elif lon > 0:
                lon_tick_labels_reg.append(f"{lon:.1f}°E")
            else:
                lon_tick_labels_reg.append("0°")
        else:
            lon_tick_labels_reg.append("")
    ax4.set_xticklabels(lon_tick_labels_reg, fontsize=9)

    lat_ticks_reg = ax4.get_yticks()
    lat_tick_labels_reg = []
    for lat in lat_ticks_reg:
        if not np.isnan(lat):
            if lat < 0:
                lat_tick_labels_reg.append(f"{abs(lat):.1f}°S")
            elif lat > 0:
                lat_tick_labels_reg.append(f"{lat:.1f}°N")
            else:
                lat_tick_labels_reg.append("0°")
        else:
            lat_tick_labels_reg.append("")
    ax4.set_yticklabels(lat_tick_labels_reg, fontsize=9)

    # -------------------------
    # Shared colorbar below the second row, spanning both columns
    # -------------------------
    pos3 = ax3.get_position()
    pos4 = ax4.get_position()
    cbar_ax1 = fig.add_axes([pos3.x0, pos3.y0 - 0.075, pos4.x1 - pos3.x0, 0.025])
    cbar1 = plt.colorbar(
        sm_cld_cbar,
        cax=cbar_ax1,
        orientation='horizontal',
        ticks=[0, 1, 2, 3, 4],
        spacing='proportional'
    )
    cbar1.ax.set_xticklabels([
        'Sunglint or\nZenith Angles > 55°',
        'Clear Sky',
        'Unretrieved\nLiquid Cloud',
        'Retrieved\nLiquid Cloud',
        'Ice Cloud'
    ], fontsize=9.5, rotation=0, ha='center')
    cbar1.ax.tick_params(axis='x', pad=4)

    # Adjust layout
    plt.tight_layout()

    # Ensure figs directory exists
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/fig1_illustration.png', dpi=300, bbox_inches='tight')
    plt.show()