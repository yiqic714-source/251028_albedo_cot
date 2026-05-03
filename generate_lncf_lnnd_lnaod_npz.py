"""
Generate NPZ file with ln(nd), ln(aod), ln(cf_ret), ln(cwp), ln(cf_msk),
lon, lat data for a given year.

Can be used both as a standalone script and as an importable module.

Usage (standalone):
    python generate_cf_lnnd_lnaod_npz.py yyyy

Example:
    python generate_cf_lnnd_lnaod_npz.py 2020

This will create: processed_data/cf_lnnd_lnaod_2020.npz
"""

import os
import sys
import glob
import datetime as dt
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC

# ============================================================
# Constants
# ============================================================

# Ocean definitions
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

MOD08_DIR = '/data/MODIS/MxD08_D3'
MOD_PREFIX = 'MOD08'  # Terra

# Variables to read from MOD08_D3
VAR_COT = 'Cloud_Optical_Thickness_Liquid_Mean'
VAR_CER = 'Cloud_Effective_Radius_Liquid_Mean'
VAR_AOD = 'Aerosol_Optical_Depth_Land_Ocean_Mean'
VAR_CF_RET = 'Cloud_Retrieval_Fraction_Combined'
VAR_CWP = 'Cloud_Water_Path_Liquid_Mean'

OUTPUT_DIR = '/home/chenyiqi/251028_albedo_cot/processed_data'


# ============================================================
# Helper functions
# ============================================================

def normalize_lon(lon):
    """Normalize longitude to [-180, 180)."""
    return ((lon + 180) % 360) - 180


def is_in_ocean(lat, lon, bounds_list):
    """Check if (lat, lon) falls within any of the bounding boxes."""
    lon_n = normalize_lon(lon)
    for bound in bounds_list:
        min_lon, min_lat, max_lon, max_lat = bound
        min_lon_n = normalize_lon(min_lon)
        max_lon_n = normalize_lon(max_lon)

        if not (lat >= min_lat and lat <= max_lat):
            continue

        if min_lon_n <= max_lon_n:
            if lon_n >= min_lon_n and lon_n <= max_lon_n:
                return True
        else:
            if lon_n >= min_lon_n or lon_n <= max_lon_n:
                return True
    return False


def read_and_mask_mod_variable(hdf, var_name):
    """Read HDF SDS and apply fill/offset/scale corrections."""
    sds = hdf.select(var_name)
    data = sds[:].astype(float)
    attrs = sds.attributes()
    fill_value = attrs.get('_FillValue', None)
    scale_factor = attrs.get('scale_factor')
    offset = attrs.get('add_offset')
    if fill_value is not None:
        data[data == fill_value] = float('nan')
    if offset is not None:
        data = data - offset
    if scale_factor is not None:
        data = data * scale_factor
    return data


def find_mod08_file_for_date(target_date):
    """Find the MOD08_D3 HDF file for a given date."""
    yyyyddd = target_date.strftime('%Y%j')
    pattern = os.path.join(MOD08_DIR, f'{MOD_PREFIX}_D3.A{yyyyddd}.061.*.hdf')
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[0]


def get_season_from_month(month):
    """Get season name from month number."""
    for season_name, months in season_dict.items():
        if month in months:
            return season_name
    return None


def compute_nd(cot, cer):
    """
    Compute cloud droplet number concentration.
    """
    return 1.37e-5 * np.power(cot, 0.5) * np.power(cer * 1e-6, -2.5) * 1e-6


def init_ocean_masks(lat_1d, lon_1d):
    """
    Pre-compute ocean masks using the 1D lat/lon grids from MOD08.
    lat_1d: 1D array of shape (180,), lon_1d: 1D array of shape (360,).
    Returns dict of {ocean_name: bool_mask_2d} with shape (180, 360).
    """
    print("Pre-computing ocean masks from XDim/YDim...")
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    ocean_masks = {}
    for ocean in oceans:
        bounds = oceans_def[ocean]
        mask = np.zeros((180, 360), dtype=bool)
        for i in range(180):
            for j in range(360):
                if is_in_ocean(lat_2d[i, j], lon_2d[i, j], bounds):
                    mask[i, j] = True
        ocean_masks[ocean] = mask
        print(f"  {ocean}: {np.sum(mask)} grid cells")
    return ocean_masks


# ============================================================
# Plotting
# ============================================================

def plot_pdf(ocean_data, year):
    """
    Plot overall probability density distributions of saved variables
    (all oceans and seasons combined into a single histogram per variable).
    """
    fig_dir = os.path.join(os.path.dirname(OUTPUT_DIR), 'figs')
    os.makedirs(fig_dir, exist_ok=True)

    variables = [
        ('ln_nd', 'ln(nd)'),
        ('ln_aod', 'ln(AOD)'),
        ('ln_cf_ret', 'ln(cf_ret)'),
        ('ln_cwp', 'ln(CWP)'),
        ('ln_cf_msk', 'ln(cf_msk)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    fig.suptitle(f'Overall PDF of Saved Variables ({year})',
                 fontsize=16, y=0.98)

    for ax, (var_key, var_label) in zip(axes.flat, variables):
        all_data = []
        for ocean in oceans:
            for season in season_dict:
                # ln_cf_msk is derived from cf_msk (not stored directly)
                src_key = 'cf_msk' if var_key == 'ln_cf_msk' else var_key
                values = ocean_data[ocean][season][src_key]
                for v in values:
                    all_data.append(v)

        if len(all_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(var_label)
            continue

        data = np.concatenate(all_data)
        data = data[np.isfinite(data)]

        # For ln_cf_msk, log-transform cf_msk (only where cf_msk > 0)
        if var_key == 'ln_cf_msk':
            data = data[data > 0]
            if len(data) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(var_label)
                continue
            data = np.log(data)

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(var_label)
            continue

        ax.hist(data, bins=80, density=True, alpha=0.7,
                color='steelblue', edgecolor='none')
        ax.set_title(var_label)
        ax.set_xlabel(var_label)
        ax.set_ylabel('PDF')

    # Hide the unused 6th subplot
    axes.flat[-1].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(fig_dir, f'pdf_overall_{year}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"PDF plot saved: pdf_overall_{year}.png")


# ============================================================
# CERES cf_liq_ceres loading (daily, matched by time)
# ============================================================

def load_ceres_cf_daily(year):
    """
    Load CERES L3 SSF daily data for the given year.
    Returns (ceres_lon_1d, ceres_lat_1d, ceres_time_1d, cf_liq_3d)
    where cf_liq_3d has shape (ntime, nlat, nlon).
    """
    ceres_path = f'/home/chenyiqi/251028_albedo_cot/CERES_L3SSF_{year}/CERES_SSF1deg-Day_Terra-MODIS_Ed4.1_Subset_{year}0101-{year}1231.nc'
    if not os.path.isfile(ceres_path):
        print(f"  Warning: CERES file not found: {ceres_path}")
        return None, None, None, None

    print(f"  Loading CERES daily data from {ceres_path}...")
    ds = xr.open_dataset(ceres_path)

    lon = ds['lon'].values.copy()
    lon[lon > 180] -= 360
    lat = ds['lat'].values
    time = ds['time'].values.astype('datetime64[D]')

    # cf_liq_ceres: liquid cloud fraction
    cf_liq = ds['cldarea_liq_total_day_daily'].values / 100.0

    ds.close()
    print(f"  CERES data shape: time={len(time)}, lat={len(lat)}, lon={len(lon)}")
    return lon, lat, time, cf_liq


# ============================================================
# Core function (importable)
# ============================================================

def generate_npz_for_year(year):
    """
    Read MOD08_D3 files for the given year, compute ln(nd), ln(aod), ln(cf_ret),
    ln(cwp), ln(cf_msk), lon, lat, and save to npz.

    CERES cf_liq_ceres is matched with MODIS data by time (daily), lat, and lon.

    The npz contains flat arrays with keys:
        'ln_nd', 'ln_aod', 'ln_cf_ret', 'ln_cwp', 'ln_cf_msk', 'lon', 'lat',
        'ocean_idx', 'season_idx'

    Returns True on success, False on failure.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load CERES daily cf_liq data
    ceres_lon, ceres_lat, ceres_time, ceres_cf_liq = load_ceres_cf_daily(year)

    # Ocean masks will be initialized from the first available HDF file's XDim/YDim
    ocean_masks = None
    lon_1d = None
    lat_1d = None

    # Collect data for each ocean x season
    ocean_data = {ocean: {s: {'ln_nd': [], 'ln_aod': [],
                              'ln_cf_ret': [], 'ln_cwp': [],
                              'cf_msk': [],
                              'lon': [], 'lat': []}
                          for s in season_dict.keys()}
                  for ocean in oceans}

    # Iterate over the given year
    start_date = dt.date(year, 1, 1)
    end_date = dt.date(year, 12, 31)
    current_date = start_date

    total_days = (end_date - start_date).days + 1
    day_count = 0
    processed_count = 0
    missing_count = 0

    print(f"Processing MOD08_D3 files for year {year}...")

    while current_date <= end_date:
        day_count += 1
        if day_count % 50 == 0:
            print(f"  Progress: day {day_count}/{total_days} "
                  f"(processed {processed_count}, missing {missing_count})")

        file_path = find_mod08_file_for_date(current_date)
        if file_path is None:
            missing_count += 1
            current_date += dt.timedelta(days=1)
            continue

        month = current_date.month
        season = get_season_from_month(month)
        if season is None:
            current_date += dt.timedelta(days=1)
            continue

        try:
            hdf = SD(file_path, SDC.READ)

            # Read 2D variables (180 x 360)
            cot_data = read_and_mask_mod_variable(hdf, VAR_COT)
            cer_data = read_and_mask_mod_variable(hdf, VAR_CER)
            aod_data = read_and_mask_mod_variable(hdf, VAR_AOD)
            cf_ret_data = read_and_mask_mod_variable(hdf, VAR_CF_RET)
            cwp_data = read_and_mask_mod_variable(hdf, VAR_CWP)

            # Read 1D coordinate variables
            # XDim: longitude (360,), YDim: latitude (180,)
            lon_1d_file = hdf.select('XDim')[:].astype(float)
            lat_1d_file = hdf.select('YDim')[:].astype(float)

            hdf.end()

            # Initialize ocean masks from the first file's XDim/YDim
            if ocean_masks is None:
                lon_1d = lon_1d_file
                lat_1d = lat_1d_file
                ocean_masks = init_ocean_masks(lat_1d, lon_1d)

            # --- Validity mask (all variables finite and positive where needed) ---
            valid_mask = (
                np.isfinite(cot_data) & (cot_data > 0) &
                np.isfinite(cer_data) & (cer_data > 0) &
                np.isfinite(aod_data) & (aod_data > 0) &
                np.isfinite(cf_ret_data) & (cf_ret_data > 0) &
                np.isfinite(cwp_data) & (cwp_data > 0)
            )

            if not np.any(valid_mask):
                current_date += dt.timedelta(days=1)
                continue

            # --- Compute nd for valid pixels ---
            nd = compute_nd(cot_data[valid_mask], cer_data[valid_mask])
            aod = aod_data[valid_mask]
            cf_ret = cf_ret_data[valid_mask]
            cwp = cwp_data[valid_mask]

            # Additional filter: nd > 0 and cf_ret > 0
            valid_nd = (nd > 0) & (cf_ret > 0)
            nd = nd[valid_nd]
            aod = aod[valid_nd]
            cf_ret = cf_ret[valid_nd]
            cwp = cwp[valid_nd]

            if len(nd) == 0:
                current_date += dt.timedelta(days=1)
                continue

            # Log-transform for nd, aod, cwp, cf_ret
            ln_nd = np.log(nd)
            ln_aod = np.log(aod)
            ln_cwp = np.log(cwp)
            ln_cf_ret = np.log(cf_ret)

            # Get grid indices of valid pixels
            valid_indices = np.where(valid_mask)
            valid_indices = (valid_indices[0][valid_nd], valid_indices[1][valid_nd])

            # Get lon/lat for each valid pixel from the 1D coordinate arrays
            pixel_lon = lon_1d[valid_indices[1]]
            pixel_lat = lat_1d[valid_indices[0]]

            # --- Match CERES cf_liq by time (daily), lat, lon ---
            cf_msk = np.full(len(pixel_lon), np.nan)
            if ceres_lon is not None:
                date64 = np.datetime64(current_date)
                time_idx = np.where(ceres_time == date64)[0]
                if len(time_idx) > 0:
                    ti = time_idx[0]
                    # For each valid pixel, find nearest CERES grid cell
                    for i in range(len(pixel_lon)):
                        lat_idx = np.argmin(np.abs(ceres_lat - pixel_lat[i]))
                        lon_idx = np.argmin(np.abs(ceres_lon - pixel_lon[i]))
                        cf_msk[i] = ceres_cf_liq[ti, lat_idx, lon_idx]

            # --- Assign to oceans ---
            for ocean in oceans:
                ocean_mask_2d = ocean_masks[ocean]
                in_ocean = ocean_mask_2d[valid_indices]
                if not np.any(in_ocean):
                    continue
                ocean_data[ocean][season]['ln_nd'].append(ln_nd[in_ocean])
                ocean_data[ocean][season]['ln_aod'].append(ln_aod[in_ocean])
                ocean_data[ocean][season]['ln_cf_ret'].append(ln_cf_ret[in_ocean])
                ocean_data[ocean][season]['ln_cwp'].append(ln_cwp[in_ocean])
                ocean_data[ocean][season]['cf_msk'].append(cf_msk[in_ocean])
                ocean_data[ocean][season]['lon'].append(pixel_lon[in_ocean])
                ocean_data[ocean][season]['lat'].append(pixel_lat[in_ocean])

            processed_count += 1

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")

        current_date += dt.timedelta(days=1)

    print(f"\nYear {year} done: processed {processed_count} files, "
          f"missing {missing_count} files out of {total_days} days.")

    # --- Concatenate all data and save as flat arrays ---
    npz_path = os.path.join(OUTPUT_DIR, f'cf_lnnd_lnaod_{year}.npz')
    print(f"Saving to: {npz_path}")

    save_dict = {}
    for key in ['ln_nd', 'ln_aod', 'ln_cf_ret', 'ln_cwp', 'lon', 'lat']:
        arrays = [v for ocean in oceans for s in season_dict for v in ocean_data[ocean][s][key]]
        save_dict[key] = np.concatenate(arrays) if arrays else np.array([])

    # ln_cf_msk: log-transform cf_msk (only where cf_msk > 0)
    cf_msk_arrays = [v for ocean in oceans for s in season_dict for v in ocean_data[ocean][s]['cf_msk']]
    if cf_msk_arrays:
        cf_msk_all = np.concatenate(cf_msk_arrays)
        valid_cf = np.isfinite(cf_msk_all) & (cf_msk_all > 0)
        save_dict['ln_cf_msk'] = np.full_like(cf_msk_all, np.nan)
        save_dict['ln_cf_msk'][valid_cf] = np.log(cf_msk_all[valid_cf])
    else:
        save_dict['ln_cf_msk'] = np.array([])

    # Ocean and season indices for grouping
    season_names = list(season_dict.keys())
    ocean_idx, season_idx = [], []
    for oi, ocean in enumerate(oceans):
        for si, s in enumerate(season_names):
            n = sum(len(v) for v in ocean_data[ocean][s]['ln_nd'])
            ocean_idx.extend([oi] * n)
            season_idx.extend([si] * n)
    save_dict['ocean_idx'] = np.array(ocean_idx, dtype=int)
    save_dict['season_idx'] = np.array(season_idx, dtype=int)

    np.savez_compressed(npz_path, **save_dict)
    print("Done.")

    # Plot probability density distributions
    plot_pdf(ocean_data, year)

    return True


# ============================================================
# CLI entry point
# ============================================================

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_cf_lnnd_lnaod_npz.py yyyy")
        sys.exit(1)

    year = int(sys.argv[1])
    generate_npz_for_year(year)


if __name__ == "__main__":
    main()
