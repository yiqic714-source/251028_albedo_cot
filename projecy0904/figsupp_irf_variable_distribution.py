# -*- coding: utf-8 -*-
"""
Plot grid-level time means of selected variables.

Variables:
    1. swdown
    2. log_aod_diff
    3. cf_liq_ceres
    4. cot_mod08

The script loads all merged ocean-season CSV files, does not apply the
cloud/retrieval mask, computes time means at each lat-lon grid cell,
and draws four global maps.
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import format_panel_tag, oceans, season_dict
from utils_solar import calc_monthly_swdown


# ============================================================
# Paths
# ============================================================

# This script now resides in
# /home/chenyiqi/251028_albedo_cot/project0904.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MERGED_DATA_DIR = PROJECT_ROOT / 'processed_data' / 'merged_data'
FIG_DIR = SCRIPT_DIR / 'figs'
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = FIG_DIR / 'figsupp_grid_mean_variables.png'


# ============================================================
# Plot settings
# ============================================================

MAP_EXTENT = [-180, 180, -60, 60]

VAR_INFO = [
    {
        'name': 'swdown',
        'title': r'SW$_\mathrm{down}$',
        'cbar_label': r'SW$_\mathrm{down}$ (W m$^{-2}$)',
        'cmap': 'YlOrRd',
        'robust': True,
    },
    {
        'name': 'log_aod_diff',
        'title': r'$\Delta \ln\mathrm{AOD}$',
        'cbar_label': r'$\Delta \ln\mathrm{AOD}$',
        'cmap': 'YlGnBu',
        'robust': True,
    },
    {
        'name': 'cf_liq_ceres',
        'title': r'CF$_\mathrm{msk}$',
        'cbar_label': r'CF$_\mathrm{msk}$',
        'cmap': 'Blues',
        'robust': False,
        'vmin': 0,
        'vmax': 1,
    },
    {
        'name': 'cot_mod08',
        'title': r'COT',
        'cbar_label': r'COT',
        'cmap': 'viridis',
        'robust': True,
    },
]


# ============================================================
# Data loading and processing
# ============================================================

def load_global_data():
    """Load merged data without applying any cloud/retrieval mask."""
    frames = []

    for ocean in oceans:
        for season_name in season_dict:
            file_path = MERGED_DATA_DIR / f'{ocean}_{season_name}.csv'
            if not file_path.exists():
                print(f'Skip missing file: {file_path}')
                continue

            data = pd.read_csv(file_path)
            data['ocean'] = ocean
            data['season'] = season_name
            frames.append(data)

    if not frames:
        raise FileNotFoundError(
            f'No merged ocean-season CSV files were found in {MERGED_DATA_DIR}.')

    return pd.concat(frames, ignore_index=True)


def add_swdown(data):
    """Compute monthly incoming shortwave flux by latitude and month."""
    if 'time' not in data.columns:
        raise KeyError("Column 'time' is required to compute swdown.")

    data = data.copy()
    data['month'] = pd.to_datetime(data['time']).dt.month

    unique_lat_month = data[['lat', 'month']].drop_duplicates().copy()
    unique_lat_month['swdown'] = unique_lat_month.apply(
        lambda row: calc_monthly_swdown(row['lat'], month=row['month']),
        axis=1,
    )

    return data.merge(unique_lat_month, on=['lat', 'month'], how='left')


def compute_grid_means(data):
    """Compute the time mean of each plotted variable by lat-lon cell."""
    required_columns = ['lat', 'lon'] + [item['name'] for item in VAR_INFO]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise KeyError(f'Missing required columns: {missing}')

    aggregations = {item['name']: 'mean' for item in VAR_INFO}
    return data.groupby(['lat', 'lon'], as_index=False).agg(aggregations)


# ============================================================
# Plotting
# ============================================================

def robust_limits(values, pmin=2, pmax=98):
    """Return percentile-based color limits for finite values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None, None

    vmin = np.nanpercentile(values, pmin)
    vmax = np.nanpercentile(values, pmax)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)

    if np.isclose(vmin, vmax):
        return None, None

    return vmin, vmax


def draw_global_map(ax, grid_mean, var_info, panel_index):
    """Draw one global map."""
    variable = var_info['name']
    data = grid_mean[['lat', 'lon', variable]].dropna().copy()

    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(
        cfeature.LAND,
        facecolor='white',
        edgecolor='black',
        linewidth=0.35,
        zorder=3,
    )
    ax.coastlines(linewidth=0.45, color='black', zorder=4)

    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.35,
        color='gray',
        linestyle='--',
        alpha=0.30,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {'size': 8.5}
    gridlines.ylabel_style = {'size': 8.5}

    if var_info.get('robust', True):
        vmin, vmax = robust_limits(data[variable].values)
    else:
        vmin = var_info.get('vmin')
        vmax = var_info.get('vmax')

    latitudes = np.sort(data['lat'].unique())
    longitudes = np.sort(data['lon'].unique())
    values = (
        data.pivot_table(
            index='lat', columns='lon', values=variable, aggfunc='mean')
        .reindex(index=latitudes, columns=longitudes)
        .to_numpy(dtype=float)
    )
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)

    if np.count_nonzero(np.isfinite(values)) >= 4:
        mappable = ax.pcolormesh(
            longitude_grid,
            latitude_grid,
            values,
            cmap=var_info['cmap'],
            vmin=vmin,
            vmax=vmax,
            shading='auto',
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
    else:
        mappable = ax.scatter(
            data['lon'],
            data['lat'],
            c=data[variable],
            s=5,
            cmap=var_info['cmap'],
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

    ax.set_title(var_info['title'], fontsize=12, pad=6)
    ax.text(
        -0.01, 1.01, format_panel_tag(panel_index, 'science'),
        transform=ax.transAxes,
        fontsize=15, va='bottom', ha='left',
    )

    return mappable


def plot_four_maps(grid_mean):
    """Plot the four selected global maps in a 2-by-2 layout."""
    fig = plt.figure(figsize=(13.5, 5.8))
    grid = fig.add_gridspec(
        2, 2,
        left=0.055,
        right=0.975,
        bottom=0.075,
        top=0.94,
        hspace=0.34,
        wspace=0.08,
    )

    axes = [
        fig.add_subplot(grid[0, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[1, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid[1, 1], projection=ccrs.PlateCarree()),
    ]

    for panel_index, (ax, var_info) in enumerate(zip(axes, VAR_INFO)):
        mappable = draw_global_map(ax, grid_mean, var_info, panel_index)
        if mappable is None:
            continue
        colorbar = fig.colorbar(
            mappable,
            ax=ax,
            orientation='vertical',
            shrink=0.82,
            pad=0.025,
        )
        colorbar.set_label(var_info['cbar_label'], fontsize=9.5)
        colorbar.ax.tick_params(labelsize=8.5)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {OUT_PNG}')


# ============================================================
# Main
# ============================================================

def main():
    print(f'Loading merged data from {MERGED_DATA_DIR}...')
    data = load_global_data()

    print('Computing swdown...')
    data = add_swdown(data)

    print('Computing grid-cell time means...')
    grid_mean = compute_grid_means(data)

    print('Plotting four global maps...')
    plot_four_maps(grid_mean)


if __name__ == '__main__':
    main()
