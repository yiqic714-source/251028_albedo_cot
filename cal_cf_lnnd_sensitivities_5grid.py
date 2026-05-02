"""
Read per-year NPZ files for years 2001-2020, and for each ocean x season:
  - Divide the ocean into 5°x5° lat/lon grid cells
  - Within each grid cell, apply density filter (>= 1e-2) and linear fit
  - Average slopes across grid cells to get ocean-mean slope
  - Plot a map showing the spatial distribution of slopes

Two relationships:
  1. ln(nd) vs ln(aod)
  2. cf_ret vs ln(nd)

If any year's NPZ file is missing, it will be automatically generated.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

from generate_lncf_lnnd_lnaod_npz import generate_npz_for_year, oceans_def, is_in_ocean

# ============================================================
# Constants
# ============================================================

oceans = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']

season_dict = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}

OUTPUT_DIR = '/home/chenyiqi/251028_albedo_cot/processed_data'
FIG_DIR = '/home/chenyiqi/251028_albedo_cot/figs'

YEAR_START = 2014
YEAR_END = 2020

BINS = 80
GRID_SIZE = 5
MIN_POINTS_PER_CELL = 10
DENSITY_PERCENTILE = 40  # remove lowest 20% density bins


# ============================================================
# Data loading
# ============================================================

def load_or_generate_year(year):
    path = os.path.join(OUTPUT_DIR, f'lncf_lnnd_lnaod_{year}.npz')
    if not os.path.isfile(path):
        print(f'Generating NPZ for {year}...')
        generate_npz_for_year(year)
    if not os.path.isfile(path):
        print(f'Failed to load {year}')
        return None
    return np.load(path, allow_pickle=True)['ocean_data'].item()


def merge_years(yearly_data):
    merged = {
        ocean: {
            season: {'ln_nd': [], 'ln_aod': [], 'ln_cf_ret': [],
                     'lon': [], 'lat': []}
            for season in season_dict
        }
        for ocean in oceans
    }
    for data in yearly_data:
        if data is None:
            continue
        for ocean in oceans:
            for season in season_dict:
                for key in ['ln_nd', 'ln_aod', 'ln_cf_ret', 'lon', 'lat']:
                    merged[ocean][season][key].extend(data[ocean][season][key])
    return merged


def get_array(ocean_data, ocean, season, key):
    values = ocean_data[ocean][season][key]
    if len(values) == 0:
        return np.array([])
    return np.concatenate(values)


# ============================================================
# Grid and fitting
# ============================================================

def density_filter(x, y, bins=BINS, percentile=DENSITY_PERCENTILE):
    """Filter data to keep only points in the top (100 - percentile)% density bins."""
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)

    # Compute threshold at the given percentile of non-zero density values
    non_zero = hist[hist > 0]
    if len(non_zero) == 0:
        return x, y  # no data to filter
    threshold = np.percentile(non_zero, percentile)

    dense_mask = hist >= threshold

    x_idx = np.digitize(x, x_edges) - 1
    y_idx = np.digitize(y, y_edges) - 1

    valid = (
        (x_idx >= 0) & (x_idx < len(x_edges) - 1) &
        (y_idx >= 0) & (y_idx < len(y_edges) - 1)
    )

    keep = np.zeros_like(valid, dtype=bool)
    keep[valid] = dense_mask[x_idx[valid], y_idx[valid]]

    return x[keep], y[keep]


def fit_slope_in_grid_cell(lon, lat, x, y):
    """
    Bin points into 5°x5° grid cells and compute the linear slope in each cell
    after applying density filter.

    Returns
    -------
    slopes : 2D array (n_lat x n_lon) with slope values (NaN where no data)
    lon_centers, lat_centers : 1D arrays of bin centers
    """
    # Define bin edges
    lon_edges = np.arange(-180, 180 + GRID_SIZE, GRID_SIZE)
    lat_edges = np.arange(-90, 90 + GRID_SIZE, GRID_SIZE)

    lon_idx = np.digitize(lon, lon_edges) - 1
    lat_idx = np.digitize(lat, lat_edges) - 1
    lon_idx = np.clip(lon_idx, 0, len(lon_edges) - 2)
    lat_idx = np.clip(lat_idx, 0, len(lat_edges) - 2)

    lon_centers = lon_edges[:-1] + GRID_SIZE / 2
    lat_centers = lat_edges[:-1] + GRID_SIZE / 2

    n_lat = len(lat_centers)
    n_lon = len(lon_centers)

    slopes = np.full((n_lat, n_lon), np.nan)

    for i in range(n_lat):
        for j in range(n_lon):
            mask = (lat_idx == i) & (lon_idx == j)
            count = np.sum(mask)
            if count < MIN_POINTS_PER_CELL:
                continue

            x_cell = x[mask]
            y_cell = y[mask]

            # Apply density filter within this cell
            x_f, y_f = density_filter(x_cell, y_cell)

            if len(x_f) < MIN_POINTS_PER_CELL:
                continue
            if np.unique(x_f).size < 2:
                continue

            try:
                slope, _, _, _, _ = stats.linregress(x_f, y_f)
                slopes[i, j] = slope
            except Exception:
                continue

    return slopes, lon_centers, lat_centers


def ocean_mean_slope(slopes, ocean_name):
    """Average slopes over grid cells within the given ocean."""
    bounds_list = oceans_def[ocean_name]

    n_lat, n_lon = slopes.shape
    lat_centers = np.linspace(-90 + GRID_SIZE / 2, 90 - GRID_SIZE / 2, n_lat)
    lon_centers = np.linspace(-180 + GRID_SIZE / 2, 180 - GRID_SIZE / 2, n_lon)

    valid = []
    for i in range(n_lat):
        for j in range(n_lon):
            if not np.isfinite(slopes[i, j]):
                continue
            if is_in_ocean(lat_centers[i], lon_centers[j], bounds_list):
                valid.append(slopes[i, j])

    if len(valid) == 0:
        return np.nan, np.nan, 0

    return np.mean(valid), np.std(valid), len(valid)


# ============================================================
# Plotting
# ============================================================

def plot_slope_map(ax, slopes, lon_centers, lat_centers, ocean, season):
    """Plot a map of slopes on the given axes."""
    masked = np.ma.masked_invalid(slopes)

    # Symmetric color range
    valid = slopes[np.isfinite(slopes)]
    if len(valid) > 0:
        abs_max = max(abs(np.nanpercentile(valid, 5)), abs(np.nanpercentile(valid, 95)))
    else:
        abs_max = 1

    im = ax.pcolormesh(
        lon_centers - GRID_SIZE / 2,
        lat_centers - GRID_SIZE / 2,
        masked,
        cmap='RdBu_r',
        vmin=-abs_max,
        vmax=abs_max
    )

    ax.set_title(f'{ocean} {season}')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    return im


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load data
    yearly_data = [
        load_or_generate_year(year)
        for year in range(YEAR_START, YEAR_END + 1)
    ]
    ocean_data = merge_years(yearly_data)

    # ============================================================
    # Define analyses
    # ============================================================
    analyses = [
        {
            'x_key': 'ln_aod',
            'y_key': 'ln_nd',
            'x_label': 'ln(AOD)',
            'y_label': 'ln(nd)',
            'title': 'Slope of ln(nd) vs ln(AOD) in 5°x5° grid cells (2020)',
            'fig_name': 'lnnd_vs_lnaod_grid_slopes.png',
            'convert_y': None,
        },
        {
            'x_key': 'ln_nd',
            'y_key': 'ln_cf_ret',
            'x_label': 'ln(nd)',
            'y_label': 'cf_ret',
            'title': 'Slope of cf_ret vs ln(nd) in 5°x5° grid cells (2020)',
            'fig_name': 'cf_ret_vs_lnnd_grid_slopes.png',
            'convert_y': np.exp,
        },
    ]

    for analysis in analyses:
        print(f"\n{'='*60}")
        print(f"Processing: {analysis['fig_name']}")
        print(f"{'='*60}")

        fig, axes = plt.subplots(4, 8, figsize=(32, 16))
        fig.suptitle(analysis['title'], fontsize=20, y=0.98)

        last_im = None

        for ri, season in enumerate(['MAM', 'JJA', 'SON', 'DJF']):
            for ci, ocean in enumerate(oceans):
                ax = axes[ri, ci]

                lon = get_array(ocean_data, ocean, season, 'lon')
                lat = get_array(ocean_data, ocean, season, 'lat')
                x = get_array(ocean_data, ocean, season, analysis['x_key'])
                y = get_array(ocean_data, ocean, season, analysis['y_key'])

                if analysis['convert_y'] is not None and len(y) > 0:
                    y = analysis['convert_y'](y)

                good = np.isfinite(x) & np.isfinite(y) & np.isfinite(lon) & np.isfinite(lat)
                lon = lon[good]
                lat = lat[good]
                x = x[good]
                y = y[good]

                if len(x) < MIN_POINTS_PER_CELL:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax.transAxes)
                    ax.set_title(f'{ocean} {season}')
                    continue

                # Compute slopes in 5°x5° grid cells
                slopes, lon_centers, lat_centers = fit_slope_in_grid_cell(lon, lat, x, y)

                # Plot slope map
                im = plot_slope_map(ax, slopes, lon_centers, lat_centers, ocean, season)
                last_im = im

                # Compute and annotate ocean-mean slope
                mean_slope, std_slope, n_cells = ocean_mean_slope(slopes, ocean)
                if np.isfinite(mean_slope):
                    ax.text(
                        0.05, 0.05,
                        f'mean k={mean_slope:.3f}\nσ={std_slope:.3f}\nn={n_cells}',
                        transform=ax.transAxes, fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                    )

                if ci == 0:
                    ax.set_ylabel('Latitude')
                if ri == 3:
                    ax.set_xlabel('Longitude')

        # Colorbar
        if last_im is not None:
            fig.subplots_adjust(right=0.92, hspace=0.3, wspace=0.3)
            cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
            cbar = fig.colorbar(last_im, cax=cbar_ax)
            cbar.set_label(
                f'Slope ({analysis["y_label"]} vs {analysis["x_label"]})',
                fontsize=14
            )

        fig.savefig(
            os.path.join(FIG_DIR, analysis['fig_name']),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig)
        print(f"Figure saved: {analysis['fig_name']}")

        # Print ocean-mean summary
        print(f"\nOcean-mean slopes ({analysis['y_label']} vs {analysis['x_label']}):")
        print(f"{'Ocean':<6} {'MAM':>8} {'JJA':>8} {'SON':>8} {'DJF':>8}")
        for ocean in oceans:
            row = f"{ocean:<6}"
            for season in ['MAM', 'JJA', 'SON', 'DJF']:
                lon = get_array(ocean_data, ocean, season, 'lon')
                lat = get_array(ocean_data, ocean, season, 'lat')
                x = get_array(ocean_data, ocean, season, analysis['x_key'])
                y = get_array(ocean_data, ocean, season, analysis['y_key'])
                if analysis['convert_y'] is not None and len(y) > 0:
                    y = analysis['convert_y'](y)
                good = np.isfinite(x) & np.isfinite(y) & np.isfinite(lon) & np.isfinite(lat)
                if good.sum() < MIN_POINTS_PER_CELL:
                    row += f" {'NaN':>8}"
                else:
                    slopes, _, _ = fit_slope_in_grid_cell(
                        lon[good], lat[good], x[good], y[good]
                    )
                    mean_s, _, _ = ocean_mean_slope(slopes, ocean)
                    row += f" {mean_s:>8.3f}" if np.isfinite(mean_s) else f" {'NaN':>8}"
            print(row)


if __name__ == '__main__':
    main()
