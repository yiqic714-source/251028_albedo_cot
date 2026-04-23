from pathlib import Path
import math
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes

warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=UserWarning, module='pyproj.network')

OUTPUT_DIR = Path('/home/chenyiqi/251028_albedo_cot/processed_data/')
INTERMEDIATE_DATA_PATH = OUTPUT_DIR / "IRF_distribution_data.pkl"

R_EARTH = 6371000
M2_TO_KM2 = 1e6

size_paras = {
    'figure_size': (18, 5.2),
    'xtick': 12,
    'ylabel': 12.5,
    'title': 15,
    'legend': 11,
}

COLORBAR_CONFIG = {
    'vmin': -2.5,
    'vmax': 2.5,
    'vcenter': 0.0
}

DIVERGING_CMAP = plt.cm.seismic
DIVERGING_NORM = mcolors.TwoSlopeNorm(
    vmin=COLORBAR_CONFIG['vmin'],
    vcenter=COLORBAR_CONFIG['vcenter'],
    vmax=COLORBAR_CONFIG['vmax']
)

def calc_grid_cell_area(lat, lon_res=1.0, lat_res=1.0):
    """Calculate grid cell area (km2) based on latitude and resolution."""
    lat1, lat2 = math.radians(lat - lat_res/2), math.radians(lat + lat_res/2)
    dlon = math.radians(lon_res)
    area_m2 = dlon * (math.sin(lat2) - math.sin(lat1)) * (R_EARTH ** 2)
    return area_m2 / M2_TO_KM2

def plot_single_subplot(ax, lon_grid, lat_grid, data_grid, area_grid, title, cmap, norm):
    """
    Plot spatial data on cartopy GeoAxes with proper 0-centered normalization.
    Forces extend triangles at both ends of colorbar.
    """
    data_min = np.nanmin(data_grid)
    data_max = np.nanmax(data_grid)
    print(f"Subplot: {title} | Min: {data_min:.3f} | Max: {data_max:.3f}")

    if isinstance(ax, GeoAxes):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='k', alpha=0.7)
        ax.add_feature(cfeature.LAND, color='#f5f5f5', alpha=0.6)
        ax.add_feature(cfeature.OCEAN, color='#eaf6fa', alpha=0.3)
        ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                          linestyle='--', alpha=0.6, color='gray')
        gl.top_labels = gl.right_labels = False

    pc = ax.pcolormesh(lon_grid, lat_grid, data_grid, 
                   cmap=cmap,
                   norm=norm,
                   transform=ccrs.PlateCarree(),
                   edgecolors='none',
                   linewidth=0)
    ax.set_title(title, fontsize=size_paras['title'], pad=5, loc='left')

    valid_mask = ~np.isnan(data_grid)
    if np.any(valid_mask):
        mean_val = np.average(data_grid[valid_mask], weights=area_grid[valid_mask])
        ax.text(1.0, 1.02, f'Mean: {mean_val:.3f}', 
                transform=ax.transAxes, ha='right', va='bottom', fontsize=13.5)
    
    return pc

def plot_maps_and_barrows(combined_df, ocean_area_order, fig_save_path=None):
    """
    Generate 2x3 plot layout with 4 spatial IRF maps and 2 ocean mean bar charts.
    """
    if 'grid_area_km2' not in combined_df.columns:
        combined_df['grid_area_km2'] = combined_df['lat'].apply(calc_grid_cell_area)
    
    agg_cols = [
        'IRF_ret_orig', 'IRF_msk_orig', 'IRF_ret_corr1', 'IRF_msk_corr',
        'grid_area_km2'
    ]
    agg_cols = [col for col in agg_cols if col in combined_df.columns]
    
    df_grid = combined_df.groupby(['lat', 'lon']).agg({col: 'mean' for col in agg_cols}).reset_index()

    lats, lons = np.sort(df_grid['lat'].unique()), np.sort(df_grid['lon'].unique())
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    def make_grid(col):
        """Convert DataFrame column to 2D grid matrix."""
        pivot_grid = df_grid.pivot(index='lat', columns='lon', values=col)
        return pivot_grid.reindex(index=lats, columns=lons).values

    def weighted_avg(series, weights):
        """Calculate weighted average (ignore NaN values)."""
        mask = series.notna()
        return np.nan if not mask.any() else float(np.average(series[mask], weights=weights[mask]))

    ocean_ratios = {}
    for ocean, g in combined_df.groupby('ocean'):
        mask_orig = weighted_avg(g['IRF_msk_orig'], g['grid_area_km2'])
        mask_corr = weighted_avg(g['IRF_msk_corr'], g['grid_area_km2'])
        retr_orig = weighted_avg(g['IRF_ret_orig'], g['grid_area_km2'])
        retr_corr = weighted_avg(g['IRF_ret_corr1'], g['grid_area_km2'])
        
        ocean_ratios[ocean] = {
            'IRF_mask_ratio': mask_corr / mask_orig if (not np.isnan(mask_orig) and mask_orig != 0) else np.nan,
            'IRF_retr_ratio': retr_corr / retr_orig if (not np.isnan(retr_orig) and retr_orig != 0) else np.nan
        }
    ocean_ratios_df = pd.DataFrame.from_dict(ocean_ratios, orient='index').reindex(
        [o for o in ocean_area_order if o in ocean_ratios]
    )

    fig = plt.figure(figsize=size_paras['figure_size'])
    proj = ccrs.PlateCarree()
    
    geo_axes_pos = [2, 3, 5, 6]
    bar_axes_pos = [1, 4]

    axes = []
    for pos in geo_axes_pos:
        ax = fig.add_subplot(2, 3, pos, projection=proj)
        axes.append(ax)
    for pos in bar_axes_pos:
        ax = fig.add_subplot(2, 3, pos)
        axes.append(ax)

    x = np.arange(len(ocean_ratios_df))
    bar_width = 0.7
    size_ratio = [0.78, 0.75]
    
    ax = axes[4]
    ax.bar(x, ocean_ratios_df['IRF_mask_ratio'], bar_width, color='gray', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ocean_ratios_df.index, fontsize=size_paras['xtick'], ha='center')
    ax.set_ylabel('Corrected IRF Ratio', fontsize=size_paras['ylabel'])
    ax.set_title(r'$\mathbf{(a)}$ Method 1', fontsize=size_paras['title'], loc='left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 0.625])
    box = ax.get_position()
    ax.set_position([
        box.x0 + 0.23 * box.width,
        box.y0 + 0.11 * box.height,
        box.width * size_ratio[0],
        box.height * size_ratio[1]
    ])

    ax = axes[5]
    ax.bar(x, ocean_ratios_df['IRF_retr_ratio'], bar_width, color='gray', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ocean_ratios_df.index, fontsize=size_paras['xtick'], ha='center')
    ax.set_ylabel('Corrected IRF Ratio', fontsize=size_paras['ylabel'])
    ax.set_title(r'$\mathbf{(d)}$ Method 2', fontsize=size_paras['title'], loc='left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 0.625])
    box = ax.get_position()
    ax.set_position([
        box.x0 + 0.23 * box.width,
        box.y0 + 0.11 * box.height,
        box.width * size_ratio[0],
        box.height * size_ratio[1]
    ])

    area_grid = make_grid('grid_area_km2')
    
    irf_configs = [
        (make_grid('IRF_msk_orig'), 0, r'$\mathbf{(b)}$ Uncorrected', DIVERGING_CMAP, DIVERGING_NORM),
        (make_grid('IRF_msk_corr'), 1, r'$\mathbf{(c)}$ Corrected', DIVERGING_CMAP, DIVERGING_NORM),
        (make_grid('IRF_ret_orig'), 2, r'$\mathbf{(e)}$ Uncorrected', DIVERGING_CMAP, DIVERGING_NORM),
        (make_grid('IRF_ret_corr1'), 3, r'$\mathbf{(f)}$ Corrected', DIVERGING_CMAP, DIVERGING_NORM),
    ]

    all_plot_objects = []

    for grid, ax_idx, title, cmap, norm in irf_configs:
        ax = axes[ax_idx]
        pc = plot_single_subplot(ax, lon_grid, lat_grid, grid, area_grid, title, cmap, norm)
        all_plot_objects.append(pc)

    cbar_ax = fig.add_axes([0.44, 0.04, 0.4, 0.025])
    
    cbar_unified = fig.colorbar(
        all_plot_objects[0],
        cax=cbar_ax,
        orientation='horizontal',
        extend='max',
        label='IRF (W m$^{-2}$)'
    )
    cbar_unified.set_label('IRF (W m$^{-2}$)', fontsize=size_paras['ylabel'])
    cbar_unified.ax.tick_params(labelsize=10)

    if fig_save_path:
        plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_save_path}")
    plt.close(fig)

if __name__ == "__main__":
    if not INTERMEDIATE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Intermediate data file not found at {INTERMEDIATE_DATA_PATH}. "
            "Please run data_processing_and_plot1.py first."
        )
    
    with open(INTERMEDIATE_DATA_PATH, 'rb') as f:
        plot2_data = pickle.load(f)
    
    combined_df = plot2_data["combined_df"]
    ocean_area = plot2_data["ocean_area"]

    figs_dir = Path("/data/chenyiqi/251028_albedo_cot/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    plot_maps_and_barrows(
        combined_df, 
        list(ocean_area.keys()), 
        fig_save_path=str(figs_dir / "IRF_distribution_and_ratio_bars.png")
    )