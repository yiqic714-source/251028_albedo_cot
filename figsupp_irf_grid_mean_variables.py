# -*- coding: utf-8 -*-
"""
Plot grid-level time means of selected variables.

Variables:
    1. swdown
    2. log_aod_diff
    3. cf_liq_ceres
    4. cf_ret_liq_mod08
    5. cot_mod08

The script loads all merged ocean-season CSV files, does not apply the
cloud/retrieval mask, computes time means at each lat-lon grid cell,
and draws five global maps.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from utils_fitting import oceans, season_dict, format_panel_tag
from utils_solar import calc_monthly_swdown


# ============================================================
# Paths
# ============================================================

BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIG_DIR = f'{BASE_PATH}/figs'
os.makedirs(FIG_DIR, exist_ok=True)

OUT_PNG = os.path.join(FIG_DIR, 'figsupp_grid_mean_variables.png')


# ============================================================
# Plot settings
# ============================================================

MAP_EXTENT = [-180, 180, -60, 60]

VAR_INFO = [
    {
        'name': 'swdown',
        'title': r'SW$_{\downarrow}$',
        'cbar_label': r'SW$_{\downarrow}$ (W m$^{-2}$)',
        'cmap': 'YlOrRd',
        'robust': True,
    },
    {
        'name': 'log_aod_diff',
        'title': r'$\Delta \ln(\mathrm{AOD})$',
        'cbar_label': r'$\Delta \ln(\mathrm{AOD})$',
        'cmap': 'YlGnBu',
        'robust': True,
    },
    {
        'name': 'cf_liq_ceres',
        'title': r'Liquid CF, mask domain',
        'cbar_label': r'Liquid CF, mask domain',
        'cmap': 'Blues',
        'robust': False,
        'vmin': 0,
        'vmax': 1,
    },
    {
        'name': 'cf_ret_liq_mod08',
        'title': r'Liquid CF, retrieval domain',
        'cbar_label': r'Liquid CF, retrieval domain',
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
    dfs = []

    for ocean in oceans:
        for season_name in season_dict:
            file_path = f'{BASE_PATH}/processed_data/merged_data/{ocean}_{season_name}.csv'
            if not os.path.exists(file_path):
                print(f'Skip missing file: {file_path}')
                continue

            df = pd.read_csv(file_path)
            df['ocean'] = ocean
            df['season'] = season_name
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError('No merged ocean-season CSV files were found.')

    return pd.concat(dfs, ignore_index=True)


def add_swdown(df):
    """Compute monthly clear-sky incoming shortwave proxy swdown by lat and month."""
    if 'time' not in df.columns:
        raise KeyError("Column 'time' is required to compute swdown.")

    df = df.copy()
    df['month'] = pd.to_datetime(df['time']).dt.month

    unique_lat_month = df[['lat', 'month']].drop_duplicates()
    unique_lat_month['swdown'] = unique_lat_month.apply(
        lambda r: calc_monthly_swdown(r['lat'], month=r['month']),
        axis=1
    )

    df = df.merge(unique_lat_month, on=['lat', 'month'], how='left')
    return df


def compute_grid_means(df):
    """Compute time mean of selected variables at each lat-lon grid cell."""
    required_cols = ['lat', 'lon'] + [v['name'] for v in VAR_INFO]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f'Missing required columns: {missing}')

    agg_cols = {v['name']: 'mean' for v in VAR_INFO}
    grid_mean = df.groupby(['lat', 'lon'], as_index=False).agg(agg_cols)

    return grid_mean


# ============================================================
# Plotting
# ============================================================

def robust_limits(values, pmin=2, pmax=98):
    """Robust color limits using percentiles."""
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
        vmin = None
        vmax = None

    return vmin, vmax


def draw_global_map(ax, grid_mean, var_info, panel_idx):
    """Draw one global map for one variable."""
    var = var_info['name']
    df = grid_mean[['lat', 'lon', var]].dropna().copy()

    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='black', linewidth=0.35, zorder=3)
    ax.coastlines(linewidth=0.45, color='black', zorder=4)

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, color='gray', linestyle='--', alpha=0.30)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8.5}
    gl.ylabel_style = {'size': 8.5}

    if df.empty:
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        ax.set_title(var_info['title'], fontsize=12)
        ax.text(-0.01, 1.01, format_panel_tag(panel_idx, 'nature'),
                transform=ax.transAxes, fontsize=15, va='bottom', ha='left')
        return None

    if var_info.get('robust', True):
        vmin, vmax = robust_limits(df[var].values)
    else:
        vmin = var_info.get('vmin', None)
        vmax = var_info.get('vmax', None)

    # Prefer pcolormesh when the data are on a regular lat-lon grid.
    lat_vals = np.sort(df['lat'].unique())
    lon_vals = np.sort(df['lon'].unique())
    z = (
        df.pivot_table(index='lat', columns='lon', values=var, aggfunc='mean')
          .reindex(index=lat_vals, columns=lon_vals)
    )

    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    zvals = z.values.astype(float)

    if np.sum(np.isfinite(zvals)) >= 4:
        pcm = ax.pcolormesh(
            lon2d, lat2d, zvals,
            cmap=var_info['cmap'],
            vmin=vmin, vmax=vmax,
            shading='auto',
            transform=ccrs.PlateCarree(),
            zorder=2
        )
    else:
        pcm = ax.scatter(
            df['lon'], df['lat'],
            c=df[var], s=5,
            cmap=var_info['cmap'],
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(),
            zorder=2
        )

    ax.set_title(var_info['title'], fontsize=12, pad=6)
    ax.text(-0.01, 1.01, format_panel_tag(panel_idx, 'nature'),
            transform=ax.transAxes, fontsize=15, va='bottom', ha='left')

    return pcm


def plot_five_maps(grid_mean):
    """Plot five global maps in one figure."""
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(
        3, 2,
        left=0.055, right=0.975,
        bottom=0.055, top=0.955,
        hspace=0.34, wspace=0.16
    )

    axes = [
        fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree()),
    ]

    for i, (ax, var_info) in enumerate(zip(axes, VAR_INFO)):
        mappable = draw_global_map(ax, grid_mean, var_info, i)
        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=ax, orientation='vertical', shrink=0.82, pad=0.025)
            cbar.set_label(var_info['cbar_label'], fontsize=9.5)
            cbar.ax.tick_params(labelsize=8.5)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {OUT_PNG}')


# ============================================================
# Main
# ============================================================

def main():
    print('Loading merged data...')
    df = load_global_data()

    print('Computing swdown...')
    df = add_swdown(df)

    print('Computing grid-cell time means...')
    grid_mean = compute_grid_means(df)
    print('Plotting five global maps...')
    plot_five_maps(grid_mean)


if __name__ == '__main__':
    main()
