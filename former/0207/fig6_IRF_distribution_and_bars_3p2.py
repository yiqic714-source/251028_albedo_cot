import os
from pathlib import Path
import math
import warnings
import pickle
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes

# Suppress non-critical visualization warnings
warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=UserWarning, module='pyproj.network')

# -------------------------- Configuration --------------------------
OUTPUT_DIR = Path('/home/chenyiqi/251028_albedo_cot/processed_data/')
INTERMEDIATE_DATA_PATH = OUTPUT_DIR / "irf_plot2_data.pkl"

# Physical constants
R_EARTH = 6371000  # Earth radius (meters)
M2_TO_KM2 = 1e6    # Conversion factor: m² to km²

size_paras = {
    'figure_size': (18, 10),
    'xtick': 12,
    'ylabel': 12.5,
    'title': 15,
    'legend': 11,
}

# Column-level colorbar ranges (updated values)
COLUMN_COLORBAR_RANGES = {
    0: {'vmin': -0.5, 'vmax': 4},   # Second column (purple maps)
    1: {'vmin': -0.5, 'vmax': 2}  # Third column (green maps)
}

# -------------------------- Reusable Helper Functions --------------------------
def calc_grid_cell_area(lat, lon_res=1.0, lat_res=1.0):
    """Calculate grid cell area (km²) based on latitude and resolution."""
    lat1, lat2 = math.radians(lat - lat_res/2), math.radians(lat + lat_res/2)
    dlon = math.radians(lon_res)
    area_m2 = dlon * (math.sin(lat2) - math.sin(lat1)) * (R_EARTH ** 2)
    return area_m2 / M2_TO_KM2

def plot_single_subplot(ax, lon_grid, lat_grid, data_grid, area_grid, title, cmap, vmin=None, vmax=None):
    """
    Plot spatial data on cartopy GeoAxes with column-level colorbar range.
    Forces extend triangles at both ends of colorbar.
    """
    # Print data min/max for current subplot
    data_min = np.nanmin(data_grid)
    data_max = np.nanmax(data_grid)
    print(f"Subplot: {title} | Min: {data_min:.3f} | Max: {data_max:.3f}")

    # Apply geographic features only to cartopy GeoAxes
    if isinstance(ax, GeoAxes):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='k', alpha=0.7)
        ax.add_feature(cfeature.LAND, color='#f5f5f5', alpha=0.6)
        ax.add_feature(cfeature.OCEAN, color='#eaf6fa', alpha=0.3)
        ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                          linestyle='--', alpha=0.6, color='gray')
        gl.top_labels = gl.right_labels = False

    # Plot data with column-level color range (force both extend triangles)
    pc = ax.pcolormesh(lon_grid, lat_grid, data_grid, 
                   cmap=cmap,
                   vmin=vmin, vmax=vmax, 
                   transform=ccrs.PlateCarree(),
                   edgecolors='none',
                   linewidth=0)
    ax.set_title(title, fontsize=size_paras['title'], pad=5, loc='left')

    # Add global weighted mean annotation
    valid_mask = ~np.isnan(data_grid)
    if np.any(valid_mask):
        mean_val = np.average(data_grid[valid_mask], weights=area_grid[valid_mask])
        ax.text(1.0, 1.02, f'Mean: {mean_val:.3f}', 
                transform=ax.transAxes, ha='right', va='bottom', fontsize=13.5)
    
    return pc

def plot_maps_and_barrows(combined_df, ocean_area_order, fig_save_path=None):
    """
    Generate 3x3 plot layout: 
    - 6 spatial IRF maps (cartopy GeoAxes) with shared colorbars at column bottom
    - 3 ocean mean bar charts (regular matplotlib Axes)
    """
    # Add grid cell area column if missing
    if 'grid_area_km2' not in combined_df.columns:
        combined_df['grid_area_km2'] = combined_df['lat'].apply(calc_grid_cell_area)
    
    # -------------------------- 1. Prepare Grid Data --------------------------
    # Aggregate data by lat/lon grid cells
    agg_cols = [
        'IRF_retr_orig', 'IRF_mask_orig', 'IRF_retr_corr', 'IRF_mask_corr',
        'IRF_mean_orig', 'IRF_mean_corr', 'grid_area_km2'
    ]
    agg_cols = [col for col in agg_cols if col in combined_df.columns]
    
    df_grid = combined_df.groupby(['lat', 'lon']).agg({col: 'mean' for col in agg_cols}).reset_index()

    # Create grid matrices for plotting
    lats, lons = np.sort(df_grid['lat'].unique()), np.sort(df_grid['lon'].unique())
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    def make_grid(col):
        """Convert DataFrame column to 2D grid matrix."""
        pivot_grid = df_grid.pivot(index='lat', columns='lon', values=col)
        return pivot_grid.reindex(index=lats, columns=lons).values

    # -------------------------- 2. Calculate Weighted Ocean Mean Values --------------------------
    def weighted_avg(series, weights):
        """Calculate weighted average (ignore NaN values)."""
        mask = series.notna()
        return np.nan if not mask.any() else float(np.average(series[mask], weights=weights[mask]))

    # Compute area-weighted IRF means for each ocean basin
    ocean_means = {}
    for ocean, g in combined_df.groupby('ocean'):
        ocean_means[ocean] = {
            'IRF_mask_orig': weighted_avg(g['IRF_mask_orig'], g['grid_area_km2']),
            'IRF_mask_corr': weighted_avg(g['IRF_mask_corr'], g['grid_area_km2']),
            'IRF_retr_orig': weighted_avg(g['IRF_retr_orig'], g['grid_area_km2']),
            'IRF_retr_corr': weighted_avg(g['IRF_retr_corr'], g['grid_area_km2']),
            'IRF_mean_orig': weighted_avg(g['IRF_mean_orig'], g['grid_area_km2']) if 'IRF_mean_orig' in g.columns else np.nan,
            'IRF_mean_corr': weighted_avg(g['IRF_mean_corr'], g['grid_area_km2']) if 'IRF_mean_corr' in g.columns else np.nan
        }
    ocean_means_df = pd.DataFrame.from_dict(ocean_means, orient='index').reindex(
        [o for o in ocean_area_order if o in ocean_means]
    )

    # -------------------------- 3. Create Plot Layout --------------------------
    fig = plt.figure(figsize=size_paras['figure_size'])
    proj = ccrs.PlateCarree()
    
    # Define subplot positions: 6 GeoAxes for maps, 3 regular Axes for bar charts
    geo_axes_pos = [2,3,5,6,8,9]  # Spatial map positions (b,c,e,f,h,i)
    bar_axes_pos = [1,4,7]         # Bar chart positions (a,d,g)

    axes = []
    # Create cartopy GeoAxes for spatial maps
    for pos in geo_axes_pos:
        ax = fig.add_subplot(3, 3, pos, projection=proj)
        axes.append(ax)
    # Create regular matplotlib Axes for bar charts
    for pos in bar_axes_pos:
        ax = fig.add_subplot(3, 3, pos)
        axes.append(ax)

    # Adjust subplot spacing to make room for horizontal colorbars
    # plt.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.95, 
    #                     wspace=0.3, hspace=0.4)

    # -------------------------- 6. Plot Ocean Mean Bar Charts --------------------------
    x = np.arange(len(ocean_means_df))
    bar_width = 0.37
    size_ratio = [0.77, 0.62]
    
    # Bar chart 1: Method 1 (mask)
    ax = axes[6]
    ax.bar(x - bar_width/2, ocean_means_df['IRF_mask_orig'], bar_width, label='Uncorrected', color='tab:purple', alpha=0.8)
    ax.bar(x + bar_width/2, ocean_means_df['IRF_mask_corr'], bar_width, label='Corrected', color='tab:green', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ocean_means_df.index, fontsize=size_paras['xtick'], ha='center')
    ax.set_ylabel('IRF (W m$^{-2}$)', fontsize=size_paras['ylabel'])
    ax.set_title(r'$\mathbf{(a)}$ Method 1', fontsize=size_paras['title'], loc='left')
    ax.legend(fontsize=size_paras['legend'])
    ax.grid(axis='y', alpha=0.3)
    box = ax.get_position()
    ax.set_position([
        box.x0 + 0.25 * box.width,
        box.y0 + 0.17 * box.height,
        box.width * size_ratio[0],
        box.height * size_ratio[1]
    ])

    # Bar chart 2: Method 2 (retrieval)
    ax = axes[7]
    ax.bar(x - bar_width/2, ocean_means_df['IRF_retr_orig'], bar_width, label='Uncorrected', color='tab:purple', alpha=0.8)
    ax.bar(x + bar_width/2, ocean_means_df['IRF_retr_corr'], bar_width, label='Corrected', color='tab:green', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ocean_means_df.index, fontsize=size_paras['xtick'], ha='center')
    ax.set_ylabel('IRF (W/m²)', fontsize=size_paras['ylabel'])
    ax.set_title(r'$\mathbf{(d)}$ Method 2', fontsize=size_paras['title'], loc='left')
    ax.legend(fontsize=size_paras['legend'])
    ax.grid(axis='y', alpha=0.3)
    box = ax.get_position()
    ax.set_position([
        box.x0 + 0.25 * box.width,
        box.y0 + 0.42 * box.height,
        box.width * size_ratio[0],
        box.height * size_ratio[1]
    ])

    # Bar chart 3: Method 3 (mean)
    ax = axes[8]
    ax.bar(x - bar_width/2, ocean_means_df['IRF_mean_orig'], bar_width, label='Uncorrected', color='tab:purple', alpha=0.8)
    ax.bar(x + bar_width/2, ocean_means_df['IRF_mean_corr'], bar_width, label='Corrected', color='tab:green', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ocean_means_df.index, fontsize=size_paras['xtick'], ha='center')
    ax.set_ylabel('IRF (W/m²)', fontsize=size_paras['ylabel'])
    ax.set_title(r'$\mathbf{(g)}$ Method 3', fontsize=size_paras['title'], loc='left')
    ax.legend(fontsize=size_paras['legend'])
    ax.grid(axis='y', alpha=0.3)
    box = ax.get_position()
    ax.set_position([
        box.x0 + 0.25 * box.width,
        box.y0 + 0.67 * box.height,
        box.width * size_ratio[0],
        box.height * size_ratio[1]
    ])

    # -------------------------- 4. Plot Spatial IRF Maps --------------------------
    area_grid = make_grid('grid_area_km2')
    
    # IRF map configurations (no per-plot color ranges - use column-level config)
    # Format: (data_grid, ax_idx, title, cmap, column_group)
    irf_configs = [
        (make_grid('IRF_mask_orig'), 0, r'$\mathbf{(b)}$ Uncorrected', plt.cm.plasma, 0),
        (make_grid('IRF_mask_corr'), 1, r'$\mathbf{(c)}$ Corrected', plt.cm.viridis, 1),
        (make_grid('IRF_retr_orig'), 2, r'$\mathbf{(e)}$ Uncorrected', plt.cm.plasma, 0),
        (make_grid('IRF_retr_corr'), 3, r'$\mathbf{(f)}$ Corrected', plt.cm.viridis, 1),
        (make_grid('IRF_mean_orig'), 4, r'$\mathbf{(h)}$ Uncorrected', plt.cm.plasma, 0),
        (make_grid('IRF_mean_corr'), 5, r'$\mathbf{(i)}$ Corrected', plt.cm.viridis, 1),
    ]

    # Store plot objects for shared colorbars
    plot_objects = {0: [], 1: []}  # 0 = second column, 1 = third column

    # Plot each spatial map with column-level color ranges
    for grid, ax_idx, title, cmap, col_group in irf_configs:
        ax = axes[ax_idx]
        # Get column-level color range
        vmin = COLUMN_COLORBAR_RANGES[col_group]['vmin']
        vmax = COLUMN_COLORBAR_RANGES[col_group]['vmax']
        
        pc = plot_single_subplot(ax, lon_grid, lat_grid, grid, area_grid, title, cmap, vmin, vmax)
        plot_objects[col_group].append(pc)

    # -------------------------- 5. Add Shared Horizontal Colorbars (at column bottom) --------------------------
    # Get the bottom axes for each column to position colorbars

    # Shared colorbar for second column (purple maps) - below bottom subplot
    cbar_col2 = fig.colorbar(
        plot_objects[0][0],
        ax=[axes[0], axes[2], axes[4]],  # All axes in second column (b, e, h)
        orientation='horizontal',        # Horizontal colorbar
        location='bottom',               # Position at bottom of column
        fraction=0.15,                   # Size relative to subplot
        shrink=0.9,                      # Scale down slightly
        pad=0.05,                         # Padding from subplot
        extend='both',                   # Force both extend triangles
        label='IRF (W m$^{-2}$)'
    )
    cbar_col2.set_label('IRF (W m$^{-2}$)', fontsize=size_paras['ylabel'])
    cbar_col2.ax.tick_params(labelsize=10)  # Adjust tick label size

    # Shared colorbar for third column (green maps) - below bottom subplot
    cbar_col3 = fig.colorbar(
        plot_objects[1][0],
        ax=[axes[1], axes[3], axes[5]],  # All axes in third column (c, f, i)
        orientation='horizontal',        # Horizontal colorbar
        location='bottom',               # Position at bottom of column
        fraction=0.15,                   # Size relative to subplot
        shrink=0.9,                      # Scale down slightly
        pad=0.05,                         # Padding from subplot
        extend='both',                   # Force both extend triangles
        label='IRF (W m$^{-2}$)'
    )
    cbar_col3.set_label('IRF (W m$^{-2}$)', fontsize=size_paras['ylabel'])
    cbar_col3.ax.tick_params(labelsize=10)  # Adjust tick label size

    # -------------------------- 7. Save Figure --------------------------
    if fig_save_path:
        plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_save_path}")
    plt.close(fig)

# -------------------------- Main Execution --------------------------
if __name__ == "__main__":
    # Load preprocessed intermediate data
    if not INTERMEDIATE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Intermediate data file not found at {INTERMEDIATE_DATA_PATH}. "
            "Please run data_processing_and_plot1.py first."
        )
    
    with open(INTERMEDIATE_DATA_PATH, 'rb') as f:
        plot2_data = pickle.load(f)
    
    combined_df = plot2_data["combined_df"]
    ocean_area = plot2_data["ocean_area"]

    # Validate required data columns exist
    required_cols = ['IRF_mean_orig', 'IRF_mean_corr']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        raise ValueError(f"Combined DataFrame missing required columns: {missing_cols}")

    # Create output directory and generate plot
    figs_dir = Path("/data/chenyiqi/251028_albedo_cot/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    plot_maps_and_barrows(
        combined_df, 
        list(ocean_area.keys()), 
        fig_save_path=str(figs_dir / "IRF_distribution_and_bars.png")
    )