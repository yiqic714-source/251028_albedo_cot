# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime, timedelta
from pyhdf.SD import SD, SDC
from scipy.interpolate import RegularGridInterpolator
from scipy import stats
import pickle
import xarray as xr
import os
import geom_utils as gu
import uniform_fov_tools as uft
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import pandas as pd
import Ac_cot_fitting_utils as acfu

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


def format_panel_tag(panel_idx, icon_style):
    if icon_style == 'science':
        letter = chr(ord('A') + panel_idx)
        return rf'$\mathbf{{{letter}}}$'

    letter = chr(ord('a') + panel_idx)
    return rf'$\mathbf{{({letter})}}$'


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


# Plot global fitting figure
def plot_global_ax(ax):
    _, global_processed_data = acfu.preprocess_ocean_data(
		min_cot_mod08=2.5,
		min_ret_cot_cer=2.5)

    # background density
    acfu.plot_density_overlay(
        global_processed_data['x1_ret'], global_processed_data['y1_list_ret'][0],
        global_processed_data['x1_msk'], global_processed_data['y1_msk'], ax
    )

    keys = ['ret', 'cp', 'dcp', 'msk', 'LH74']
    seasons = list(acfu.season_dict.keys())

    global_result_row = {'Ocean': 'Global'}

    # Fit directly on global samples to keep consistent with sensitivity_test_satellite.py.
    fig_tmp, ax_tmp = plt.subplots(figsize=(4, 3))
    try:
        line_handles_tmp = []
        line_labels_tmp = []

        ret_result = acfu.plot_weighted_fit_line(
            global_processed_data['ret_cot'],
            global_processed_data['ret_albedo_list'][0],
            global_processed_data['sza'],
            global_processed_data['season'],
            global_processed_data['x2'],
            'blue',
            'ret',
            line_handles_tmp,
            line_labels_tmp,
            ax_tmp,
            linestyle='--',
            cot_std=0.1,
            albedo_std=0.13,
        )

        cp_result = acfu.plot_weighted_fit_line(
            global_processed_data['ret_cot'],
            global_processed_data['ret_albedo_list'][1],
            global_processed_data['sza'],
            global_processed_data['season'],
            global_processed_data['x2'],
            'orange',
            'cp',
            line_handles_tmp,
            line_labels_tmp,
            ax_tmp,
            linestyle='-',
            cot_std=0.0,
            albedo_std=0.03,
        )

        dcp_result = acfu.plot_weighted_fit_line(
            global_processed_data['ret_cot'],
            global_processed_data['ret_albedo_list'][2],
            global_processed_data['sza'],
            global_processed_data['season'],
            global_processed_data['x2'],
            'red',
            'dcp',
            line_handles_tmp,
            line_labels_tmp,
            ax_tmp,
            linestyle='--',
            cot_std=0.0,
            albedo_std=0.03,
        )

        msk_result = acfu.plot_weighted_fit_line(
            global_processed_data['msk_cot'],
            global_processed_data['msk_albedo'],
            global_processed_data['sza'],
            global_processed_data['season'],
            global_processed_data['x2'],
            'magenta',
            'msk',
            line_handles_tmp,
            line_labels_tmp,
            ax_tmp,
            linestyle='-',
            cot_std=0.1,
            albedo_std=0.20,
        )
    finally:
        plt.close(fig_tmp)

    # LH74 line follows existing fig2 logic (deterministic line in transformed space).
    k_lh74, b_lh74, _, _, _ = stats.linregress(global_processed_data['x2'], global_processed_data['y22'])

    fit_results = {
        'ret': ret_result[:8],
        'cp': cp_result[:8],
        'dcp': dcp_result[:8],
        'msk': msk_result[:8],
        'LH74': (k_lh74, b_lh74, 0.0, 0.0, {}, {}, {}, {}),
    }

    for key in keys:
        (
            global_k, global_b, global_k_unc, global_b_unc,
            season_k, season_b, season_k_unc, season_b_unc
        ) = fit_results[key]

        global_result_row[f'Ann_Slope_{key}'] = global_k
        global_result_row[f'Ann_Intercept_{key}'] = global_b
        global_result_row[f'Ann_SlopeUnc_{key}'] = global_k_unc
        global_result_row[f'Ann_InterceptUnc_{key}'] = global_b_unc

        for s_name in seasons:
            global_result_row[f'{s_name}_Slope_{key}'] = season_k.get(s_name, np.nan)
            global_result_row[f'{s_name}_Intercept_{key}'] = season_b.get(s_name, np.nan)
            global_result_row[f'{s_name}_SlopeUnc_{key}'] = season_k_unc.get(s_name, np.nan)
            global_result_row[f'{s_name}_InterceptUnc_{key}'] = season_b_unc.get(s_name, np.nan)

    # plot lines using mean results
    x2 = global_processed_data['x2']
    line_handles = []
    line_labels = []

    plot_specs = [
        ('LH74', 'black', '-'),
        ('dcp', 'red', '--'),
        ('cp', 'orange', '-'),
        ('ret', 'blue', '--'),
        ('msk', 'magenta', '-')
    ]

    for key, color, linestyle in plot_specs:
        k = global_result_row[f'Ann_Slope_{key}']
        b = global_result_row[f'Ann_Intercept_{key}']
        y_line = k * x2 + b

        sign = '+' if b >= 0 else ''
        eq = f'y={k:.2f}x{sign}{b:.1f}'
        line = ax.plot(x2, y_line, color=color, linestyle=linestyle, lw=1.5, label=f'{key}: {eq}')
        line_handles.append(line[0])
        line_labels.append(f'{key}: {eq}')

    ax.legend(handles=line_handles, labels=line_labels, fontsize=9, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel(r'ln(COT)', fontsize=14)
    ax.set_ylabel(r'$\ln\left[A_{\mathrm{c}}/(1-A_{\mathrm{c}})\right]$', fontsize=14)
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(-2.0, 1.5)

    # save global mean results
    output_csv_path = '/home/chenyiqi/251028_albedo_cot/processed_data/k_lnb_global_by_seasons.csv'
    output_df = pd.DataFrame([global_result_row])
    output_df.to_csv(output_csv_path, index=False)
    print(f"Global slope and intercept results saved to: {output_csv_path}")

    return global_result_row


# Main process
if __name__ == "__main__":
    # Choose panel tag style here: 'nature' -> (a)(b)(c), 'science' -> A B C.
    icon_style = 'science'
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
    cld_retrieval_mask = np.where((Cloud_Retrieval_Phase_Flag == 2) & (Primary_Cloud_Retrieval_Outcome_Flag == 1), 1, 0)
    cld_mask = np.where((Cloud_Retrieval_Phase_Flag == 2) & (Cloudiness_Flag <= 1), 1, 0)
    nan_mask = ((Cloud_Retrieval_Phase_Flag != 2) & (Primary_Cloud_Retrieval_Outcome_Flag == 1)) | (ctt_mod < 273.15 - 5)
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
        (lat_mod_full_original > lat_min - 0.5) & (lat_mod_full_original < lat_max + 0.5) &
        (lon_mod_full_original > lon_min - 0.5) & (lon_mod_full_original < lon_max + 0.5)
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
    # Create figure with new layout
    # (a) Data Domains
    # (b) Full Granule
    # (c) Processed Grid
    # (d) Identified FOSRs
    # (e) Slope Fittings
    # =========================
    fig = plt.figure(figsize=(15, 8.5))

    gs = fig.add_gridspec(
        2, 3,
        hspace=0.5, wspace=0.3,
        height_ratios=[1, 0.9],
        bottom=0.08, top=0.95
    )

    ax1 = fig.add_subplot(gs[0, 0])  # (a) Data Domains
    ax2 = fig.add_subplot(gs[0, 1])  # (b) Full Granule
    ax3 = fig.add_subplot(gs[0, 2])  # (c) Processed Grid
    ax4 = fig.add_subplot(gs[1, 0])  # (d) Identified FOSRs
    ax5 = fig.add_subplot(gs[1, 1])  # (e) Slope Fittings

    # Hide unused panel
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

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
    # (a) Data Domains
    # -------------------------
    ax1.text(-0.01, 1.01, f'{format_panel_tag(0, icon_style)}   Data Domains',
             transform=ax1.transAxes, fontsize=15, va='bottom', ha='left')
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(True)

    # -------------------------
    # (b) Full Granule
    # -------------------------
    ax2.scatter(lon_mod_full_invalid, lat_mod_full_invalid, s=0.04, c='lightgray',
                marker='o', edgecolors='None')
    sc1 = ax2.scatter(lon_mod_full, lat_mod_full, s=0.04, c=cld_type_full,
                      marker='o', cmap=cmap_cld, norm=norm, edgecolors='None')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.text(-0.01, 1.01, f'{format_panel_tag(1, icon_style)}   Full Granule',
             transform=ax2.transAxes, fontsize=15, va='bottom', ha='left')

    rect = patches.Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        linewidth=1.5,
        edgecolor='blue',
        facecolor='none'
    )
    ax2.add_patch(rect)

    # Customize longitude ticks for full granule
    lon_ticks_full = ax2.get_xticks()
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
    ax2.set_xticklabels(lon_tick_labels_full, fontsize=9)

    # Customize latitude ticks for full granule
    lat_ticks_full = ax2.get_yticks()
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
    ax2.set_yticklabels(lat_tick_labels_full, fontsize=9)

    # -------------------------
    # (c) Processed Grid
    # -------------------------
    ax3.scatter(lon_mod_regional_invalid, lat_mod_regional_invalid, s=8, c='lightgray',
                marker='o', edgecolors='None')
    sc2 = ax3.scatter(lon_mod, lat_mod, c=cld_type, s=8, marker='o',
                      cmap=cmap_cld, norm=norm, edgecolors='None')
    ax3.set_xlim(grid_window[1])
    ax3.set_ylim(grid_window[0])
    ax3.text(-0.01, 1.01, f'{format_panel_tag(2, icon_style)}   Processed Grid',
             transform=ax3.transAxes, fontsize=15, va='bottom', ha='left')

    lon_start = grid_window[1][0]
    lon_end = grid_window[1][1]
    lon_ticks_reg = np.arange(lon_start, lon_end + 0.01, 0.3)
    ax3.set_xticks(lon_ticks_reg)

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
    ax3.set_xticklabels(lon_tick_labels_reg, fontsize=9)

    lat_ticks_reg = ax3.get_yticks()
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
    ax3.set_yticklabels(lat_tick_labels_reg, fontsize=9)

    # -------------------------
    # (d) Identified FOSRs
    # -------------------------
    colors_white2blue = [
        (1.0, 1.0, 1.0, 1.0),
        (0.1216, 0.4039, 0.6745)
    ]
    cmap_white2blue = mcolors.LinearSegmentedColormap.from_list(
        'white2blue', colors_white2blue, N=256
    )

    sc3 = ax4.scatter(lon_mod, lat_mod, c=weight_ret, s=8,
                      cmap=cmap_white2blue, marker='o', edgecolors='None')

    if center_latlon is not None and len(center_latlon) > 0:
        ax4.scatter(center_latlon[:, 1], center_latlon[:, 0],
                    c='orange', s=14, marker='o')

    ax4.set_xlim(grid_window[1])
    ax4.set_ylim(grid_window[0])
    ax4.text(-0.01, 1.01, f'{format_panel_tag(3, icon_style)}   Identified FOSRs',
             transform=ax4.transAxes, fontsize=15, va='bottom', ha='left')

    # Annotate orange points
    if center_latlon is not None and len(center_latlon) > 0:
        for idx in range(len(center_latlon)):
            lon_pt = center_latlon[idx, 1]
            lat_pt = center_latlon[idx, 0]
            cot_val = center_cot[idx]
            albedo_val = center_albedo[idx]

            try:
                cot_val = float(np.asarray(cot_val).ravel()[0])
            except Exception:
                cot_val = np.nan
            try:
                albedo_val = float(np.asarray(albedo_val).ravel()[0])
            except Exception:
                albedo_val = np.nan

            if np.isnan(cot_val) or np.isnan(albedo_val):
                continue

            label_text = "({:.1f}, {:.2f})".format(cot_val, albedo_val)

            ax4.text(
                x=lon_pt - 0.06,
                y=lat_pt + 0.02,
                s=label_text,
                fontsize=11.5,
                ha='center',
                va='bottom',
                color='black'
            )

    ax4.text(0.01, 0.68, r'(COT$_{\mathrm{ret}}, A_{\mathrm{c,ret}})$:',
             transform=ax4.transAxes, fontsize=11.5, va='top', ha='left')

    ax4.set_xticks(lon_ticks_reg)
    lon_tick_labels_ret = []
    for lon in lon_ticks_reg:
        if not np.isnan(lon):
            if lon < 0:
                lon_tick_labels_ret.append(f"{abs(lon):.1f}°W")
            elif lon > 0:
                lon_tick_labels_ret.append(f"{lon:.1f}°E")
            else:
                lon_tick_labels_ret.append("0°")
        else:
            lon_tick_labels_ret.append("")
    ax4.set_xticklabels(lon_tick_labels_ret, fontsize=9)

    lat_ticks_ret = ax4.get_yticks()
    lat_tick_labels_ret = []
    for lat in lat_ticks_ret:
        if not np.isnan(lat):
            if lat < 0:
                lat_tick_labels_ret.append(f"{abs(lat):.1f}°S")
            elif lat > 0:
                lat_tick_labels_ret.append(f"{lat:.1f}°N")
            else:
                lat_tick_labels_ret.append("0°")
        else:
            lat_tick_labels_ret.append("")
    ax4.set_yticklabels(lat_tick_labels_ret, fontsize=9)

    # -------------------------
    # (e) Slope Fittings
    # -------------------------
    ax5.text(-0.01, 1.01, f'{format_panel_tag(4, icon_style)}   Slope Fittings',
             transform=ax5.transAxes, fontsize=15, va='bottom', ha='left')
    plot_global_ax(ax5)

    # -------------------------
    # Colorbars
    # -------------------------
    # colorbar for panels (b) and (c)
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()
    cbar_ax1 = fig.add_axes([pos2.x0, pos2.y0 - 0.06, pos3.x1 - pos2.x0, 0.02])
    cbar1 = plt.colorbar(
        sm_cld_cbar,
        cax=cbar_ax1,
        orientation='horizontal',
        ticks=[0, 1, 2, 3, 4],
        spacing='proportional'
    )
    cbar1.ax.set_xticklabels([
        'Sunglint/\nLarge Zenith Angles',
        'Clear Sky',
        'Unretrieved\nLiquid Cloud',
        'Retrieved\nLiquid Cloud',
        'Ice Cloud'
    ], fontsize=10.5)
    cbar1.ax.tick_params(axis='x', pad=8)

    # colorbar for panel (d)
    pos4 = ax4.get_position()
    cbar_ax2 = fig.add_axes([pos4.x0, pos4.y0 - 0.06, pos4.width, 0.02])
    cbar2 = plt.colorbar(
        sc3, cax=cbar_ax2, orientation='horizontal', label='Cumulative Weight'
    )
    cbar2.ax.get_xaxis().get_label().set_fontsize(10.5)

    # Adjust layout
    plt.tight_layout()

    # Ensure figs directory exists
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/illustration_5panels.png', dpi=300, bbox_inches='tight')
    plt.show()