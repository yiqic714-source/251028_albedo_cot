import os
from pathlib import Path
from datetime import date, timedelta
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import pickle  # For saving intermediate data

# Suppress non-fatal visualization warnings
warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=UserWarning, module='pyproj.network')

# -------------------------- Configuration --------------------------
INPUT_DIR = Path('/home/chenyiqi/251028_albedo_cot/processed_data/IRF_oceanic_data/')
OUTPUT_DIR = Path('/home/chenyiqi/251028_albedo_cot/processed_data/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Intermediate data save path
INTERMEDIATE_DATA_PATH = OUTPUT_DIR / "irf_plot2_data.pkl"

# Coefficient file paths (consolidated dict for brevity)
SLOPE_FILES = {
    'k1': OUTPUT_DIR / 'table_slope1.csv',
    'k2': OUTPUT_DIR / 'table_slope2.csv',
    'lnb2': OUTPUT_DIR / 'table_intercept2.csv',
    'bello': OUTPUT_DIR / 'Bellouin2013.csv'
}

# Season mapping and order
SEASON_MAP = {1: 'DJF', 2: 'DJF', 12: 'DJF',
              3: 'MAM', 4: 'MAM', 5: 'MAM',
              6: 'JJA', 7: 'JJA', 8: 'JJA',
              9: 'SON', 10: 'SON', 11: 'SON'}
SEASONS = ['DJF', 'MAM', 'JJA', 'SON']

# Physical constants
S0 = 1361.0  # Solar constant (W/m2)
R_EARTH = 6371000  # Earth radius (meters)
M2_TO_KM2 = 1e6  # Conversion factor: m² to km²
HOUR_CONV = 12 / math.pi  # 日落时角转日照时长系数

# -------------------------- Solar Geometry Helpers --------------------------
def declination(n):
    """Calculate solar declination angle (radians) from day of year."""
    return math.radians(23.45) * math.sin(2 * math.pi * (284 + n) / 365.0)

def E_ext(n):
    """Calculate extraterrestrial solar irradiance (W/m2) from day of year."""
    return S0 * (1 + 0.033 * math.cos(2 * math.pi * n / 365.0))

def sunset_hour_angle(phi, delta):
    """Calculate sunset hour angle (radians) from latitude and declination."""
    x = -math.tan(phi) * math.tan(delta)
    return 0.0 if x >= 1 else math.pi if x <= -1 else math.acos(x)

def H0_daily_mean(phi, n):
    """Calculate daily-mean top-of-atmosphere solar radiation (W/m2)."""
    delta, E = declination(n), E_ext(n)
    omega_s = sunset_hour_angle(phi, delta)
    return (E / math.pi) * (math.cos(phi)*math.cos(delta)*math.sin(omega_s) + omega_s*math.sin(phi)*math.sin(delta))

def calc_daily_sunshine_hours(phi, n):
    """Calculate daily sunshine hours (hours/day) from latitude and day of year."""
    delta = declination(n)
    omega_s = sunset_hour_angle(phi, delta)
    return 2 * omega_s * HOUR_CONV

def calc_monthly_swdown(lat, year=2020, month=None):
    """Calculate monthly mean daily-mean SWdown (W/m2) for a given latitude."""
    if pd.isna(lat) or pd.isna(month) or month is None:
        return np.nan
    try:
        month = int(month)
        lat = float(lat)
        year = int(year)
    except (ValueError, TypeError):
        return np.nan
    if not (1 <= month <= 12):
        return np.nan
    start = date(year, month, 1)
    end = date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    phi = math.radians(lat)
    vals = [H0_daily_mean(phi, d.timetuple().tm_yday) for d in pd.date_range(start, end)]
    return float(np.mean(vals)) if vals else np.nan

def calc_monthly_sunshine_hours(lat, year=2020, month=None):
    """Calculate monthly mean daily sunshine hours (hours/day) for a given latitude."""
    if pd.isna(lat) or pd.isna(month) or month is None:
        return np.nan
    try:
        month = int(month)
        lat = float(lat)
        year = int(year)
    except (ValueError, TypeError):
        return np.nan
    if not (1 <= month <= 12):
        return np.nan
    start = date(year, month, 1)
    end = date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    phi = math.radians(lat)
    sunshine_hours_daily = [calc_daily_sunshine_hours(phi, d.timetuple().tm_yday) for d in pd.date_range(start, end)]
    return float(np.mean(sunshine_hours_daily)) if sunshine_hours_daily else np.nan

def calc_grid_cell_area(lat, lon_res=1.0, lat_res=1.0):
    """Calculate grid cell area (km²) from latitude and resolution."""
    lat1, lat2 = math.radians(lat - lat_res/2), math.radians(lat + lat_res/2)
    dlon = math.radians(lon_res)
    area_m2 = dlon * (math.sin(lat2) - math.sin(lat1)) * (R_EARTH ** 2)
    return area_m2 / M2_TO_KM2

def spatial_interpolate_grid(df_grid, value_cols, n_neighbors=15, power=2):
    """Inverse Distance Weighting (IDW) interpolation for spatial data."""
    df_grid = df_grid.copy()
    from scipy.spatial import cKDTree
    for col in value_cols:
        valid_data = df_grid[~df_grid[col].isna()]
        if len(valid_data) < 1:
            df_grid[col] = df_grid[col].fillna(df_grid[col].mean())
            continue
        valid_coords = valid_data[['lon', 'lat']].values
        valid_vals = valid_data[col].values
        all_coords = df_grid[['lon', 'lat']].values
        tree = cKDTree(valid_coords)
        interpolated_vals = []
        for coord in all_coords:
            distances, indices = tree.query(coord, k=min(n_neighbors, len(valid_data)))
            distances[distances == 0] = 1e-10
            weights = 1 / (distances ** power)
            weighted_val = np.sum(weights * valid_vals[indices]) / np.sum(weights)
            interpolated_vals.append(weighted_val)
        df_grid[col] = interpolated_vals
    return df_grid

# -------------------------- IO / Coefficient Table Helpers --------------------------
def melt_slope_table(df, value_name):
    """Melt coefficient tables to long format (ocean, season, value)."""
    df = df.rename(columns={'Ocean': 'ocean'})
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    melted_df = pd.melt(df, id_vars=['ocean'], value_vars=SEASONS,
                        var_name='season', value_name=value_name)
    for col in ['ocean', 'season']:
        melted_df[col] = melted_df[col].str.strip()
    return melted_df

# -------------------------- Plot Helpers (Global Distributions) --------------------------
def plot_single_subplot(ax, lon_grid, lat_grid, data_grid, area_grid, title, cmap):
    """Plot single geographic subplot with coastlines, land, and mean annotation."""
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='k', alpha=0.7)
    ax.add_feature(cfeature.LAND, color='#f5f5f5', alpha=0.6)
    ax.add_feature(cfeature.OCEAN, color='#eaf6fa', alpha=0.3)
    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                      linestyle='--', alpha=0.6, color='gray')
    gl.top_labels = gl.right_labels = False
    pc = ax.pcolormesh(lon_grid, lat_grid, data_grid, 
                   cmap=cmap,
                   transform=ccrs.PlateCarree(),
                   edgecolors='none',
                   linewidth=0)
    ax.set_title(title, fontsize=16, pad=5, loc='left')
    valid_mask = ~np.isnan(data_grid)
    if np.any(valid_mask):
        mean_val = np.average(data_grid[valid_mask], weights=area_grid[valid_mask])
        ax.text(1.0, 1.02, f'Mean: {mean_val:.2f}', 
                transform=ax.transAxes, ha='right', va='bottom', fontsize=13.5)
    return pc

def plot_global_distributions(df, fig_save_path=None):
    """Generate 3x2 global distribution plot for core variables."""
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    if 'grid_area_km2' not in df.columns:
        df['grid_area_km2'] = df['lat'].apply(calc_grid_cell_area)
    if 'log_aod_diff' not in df.columns or df['log_aod_diff'].isna().all():
        aod_cols = [c for c in df.columns if 'aod' in c.lower()]
        if len(aod_cols) >= 2:
            ret_col = [c for c in aod_cols if any(k in c.lower() for k in ('retriev', 'sat', 'obs'))][0]
            mod_col = [c for c in aod_cols if any(k in c.lower() for k in ('model', 'cmip', 'ref'))][0]
            a1 = pd.to_numeric(df[ret_col], errors='coerce').replace({0: np.nan})
            a2 = pd.to_numeric(df[mod_col], errors='coerce').replace({0: np.nan})
            with np.errstate(invalid='ignore', divide='ignore'):
                df['log_aod_diff'] = np.log10(a1) - np.log10(a2)
    if ('swdown' not in df.columns or df['swdown'].isna().all()) and 'month' in df.columns:
        unique_lat_month = df[['lat', 'month']].drop_duplicates()
        unique_lat_month['swdown'] = unique_lat_month.apply(
            lambda r: calc_monthly_swdown(r['lat'], month=r['month']), axis=1
        )
        df = df.merge(unique_lat_month, on=['lat', 'month'], how='left', suffixes=('', '_computed'))
        if 'swdown_computed' in df.columns:
            df['swdown'] = df['swdown'].fillna(df['swdown_computed']).drop(columns=['swdown_computed'])
    agg_cols = ['cf_ret_liq_mod08', 'cf_liq_ceres', 'Ac', 'cot_mod08', 'log_aod_diff', 'swdown', 'grid_area_km2']
    df_grid = df.groupby(['lat', 'lon']).agg({col: 'mean' for col in agg_cols}).reset_index()
    lats, lons = np.sort(df_grid['lat'].unique()), np.sort(df_grid['lon'].unique())
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    def make_grid(col):
        pivot_grid = df_grid.pivot(index='lat', columns='lon', values=col)
        return pivot_grid.reindex(index=lats, columns=lons).values
    grid_configs = [
        (make_grid('swdown'), plt.cm.Blues, r'$\mathbf{(a)}$ SW$_{\mathrm{down}}$'),
        (make_grid('log_aod_diff'), plt.cm.Blues, r'$\mathbf{(b)}$ $\ln\text{AOD}_{\mathrm{PD}} - \ln\text{AOD}_{\mathrm{PI}}$'),
        (make_grid('cf_liq_ceres'), plt.cm.Blues, r'$\mathbf{(c)}$ CF$_{\mathrm{msk}}$'),
        (make_grid('Ac') * (1 - make_grid('Ac')), plt.cm.Blues, r'$\mathbf{(d)}$ $A_{\mathrm{c,msk}} × (1-A_{\mathrm{c,msk}})$'),
        (make_grid('cf_ret_liq_mod08'), plt.cm.Blues, r'$\mathbf{(e)}$ CF$_{\mathrm{ret}}$'),
        (np.log(make_grid('cot_mod08')), plt.cm.Blues, r'$\mathbf{(f)}$ $\ln$COT'),
    ]
    area_grid = make_grid('grid_area_km2')
    fig, axes = plt.subplots(3, 2, figsize=(13, 7.5), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.17, hspace=0.25)
    for idx, (grid, cmap, title) in enumerate(grid_configs):
        ax = axes.flat[idx]
        pc = plot_single_subplot(ax, lon_grid, lat_grid, grid, area_grid, title, cmap)
        cbar = fig.colorbar(pc, ax=ax, orientation='vertical', fraction=0.05, shrink=0.89)
        if idx==0:
            cbar.set_label('W m$^{-2}$', fontsize=12)
    if fig_save_path:
        plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fig_save_path}")
    plt.close(fig)

# -------------------------- Main Execution --------------------------
if __name__ == "__main__":
    # Validate all coefficient files exist
    if not all(f.exists() for f in SLOPE_FILES.values()):
        raise FileNotFoundError("One or more coefficient files are missing.")

    # Load and melt all coefficient tables
    coeff_dfs = {key: melt_slope_table(pd.read_csv(filepath), key) for key, filepath in SLOPE_FILES.items()}

    # Load ocean data files
    ocean_files = [p for p in INPUT_DIR.iterdir() if p.suffix == '.csv' and p.stem]
    if not ocean_files:
        raise FileNotFoundError(f"No valid CSV files found in {INPUT_DIR}")

    # Precompute SWdown + 月平均日照时长 for all unique lat-month combinations
    combined_lat_month = pd.concat([pd.read_csv(p)[['lat', 'month']] for p in ocean_files if 'month' in pd.read_csv(p).columns])
    combined_lat_month = combined_lat_month.drop_duplicates().reset_index(drop=True)
    # Clean month column
    combined_lat_month = combined_lat_month.dropna(subset=['lat', 'month'])
    combined_lat_month['month'] = combined_lat_month['month'].astype(float).astype(int)
    combined_lat_month = combined_lat_month[(combined_lat_month['month'] >= 1) & (combined_lat_month['month'] <= 12)]
    # 同步计算SWdown和日照时长
    combined_lat_month['swdown'] = combined_lat_month.apply(
        lambda r: calc_monthly_swdown(r['lat'], month=r['month']), axis=1
    )
    combined_lat_month['sunshine_hours_mean'] = combined_lat_month.apply(
        lambda r: calc_monthly_sunshine_hours(r['lat'], month=r['month']), axis=1
    )

    # Process each ocean file
    all_dfs, ocean_area = [], {}
    for p in ocean_files:
        ocean_name = p.stem
        df = pd.read_csv(p)

        # Add metadata and season
        df['ocean'] = ocean_name
        df['season'] = df['month'].map(SEASON_MAP) if 'month' in df.columns else np.nan
        df.loc[(df['Ac'] > 1) | (df['Ac'] < 0), 'Ac'] = np.nan

        # Merge precomputed SWdown + 日照时长
        if 'swdown' not in df.columns or df['swdown'].isna().all() or 'sunshine_hours_mean' not in df.columns:
            df = df.merge(combined_lat_month, on=['lat', 'month'], how='left', suffixes=('', '_computed'))
            # 填充SWdown
            if 'swdown_computed' in df.columns:
                df['swdown'] = df['swdown'].fillna(df['swdown_computed']).drop(columns=['swdown_computed'])
            # 填充日照时长
            if 'sunshine_hours_mean_computed' in df.columns:
                df['sunshine_hours_mean'] = df['sunshine_hours_mean'].fillna(df['sunshine_hours_mean_computed']).drop(columns=['sunshine_hours_mean_computed'])

        # Calculate grid area and merge coefficients
        df['grid_area_km2'] = df['lat'].apply(calc_grid_cell_area)
        for coeff_df in coeff_dfs.values():
            df = df.merge(coeff_df, on=['ocean', 'season'], how='left')

        # Calculate corrected and original Ac
        with np.errstate(invalid='ignore', divide='ignore'):
            num = np.exp(df['lnb2']) * df['cot_mod08'] ** df['k2']
            df['Ac_retr_corr'] = num / (1 + num)
            df['Ac_retr_orig'] = 0.13 * df['cot_mod08'] / (1 + 0.13 * df['cot_mod08'])

        # Track total ocean area
        ocean_area[ocean_name] = df.drop_duplicates(subset=['lat', 'lon'])['grid_area_km2'].sum()
        all_dfs.append(df)

    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Calculate IRF
    Ac_ret_gm = np.sum(combined_df['Ac_retr_corr'] * combined_df['grid_area_km2']) / np.sum(combined_df['grid_area_km2'])
    print(f'Global mean cloud retrieval albedo (corrected): {Ac_ret_gm:.2f}')
    Ac_msk_gm = np.sum(combined_df['Ac'] * combined_df['grid_area_km2']) / np.sum(combined_df['grid_area_km2'].where(~np.isnan(combined_df['Ac'])))
    print(f'Global mean cloud mask albedo: {Ac_msk_gm:.2f}')
    # 全球平均日照时长
    global_mean_sunshine = np.sum(combined_df['sunshine_hours_mean'] * combined_df['grid_area_km2']) / np.sum(combined_df['grid_area_km2'].where(~np.isnan(combined_df['sunshine_hours_mean'])))
    print(f'Global mean monthly sunshine hours (hours/day): {global_mean_sunshine:.2f}')
    print(np.std(combined_df['sunshine_hours_mean'].dropna()))

    irf_base = combined_df['bello'] / 3 * combined_df['swdown'] * combined_df['log_aod_diff']
    combined_df['IRF_retr_orig'] = irf_base * combined_df['cf_ret_liq_mod08'] * combined_df['Ac_retr_orig'] * (1 - combined_df['Ac_retr_orig'])
    combined_df['IRF_mean_orig'] = irf_base * combined_df['cf_ret_liq_mod08'] * Ac_ret_gm * (1 - Ac_ret_gm)
    combined_df['IRF_mask_orig'] = irf_base * combined_df['cf_liq_ceres'] * combined_df['Ac'] * (1 - combined_df['Ac'])
    combined_df['IRF_retr_corr'] = irf_base * combined_df['cf_ret_liq_mod08'] * combined_df['Ac_retr_corr'] * (1 - combined_df['Ac_retr_corr']) * combined_df['k2']
    combined_df['IRF_mean_corr'] = irf_base * combined_df['cf_ret_liq_mod08'] * Ac_ret_gm * (1 - Ac_ret_gm) * combined_df['k2']
    combined_df['IRF_mask_corr'] = irf_base * combined_df['cf_liq_ceres'] * combined_df['Ac'] * (1 - combined_df['Ac']) * combined_df['k1']
    
    # 1. Plot and save IRF_data_used.png
    figs_dir = Path("/data/chenyiqi/251028_albedo_cot/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    plot_global_distributions(combined_df, fig_save_path=str(figs_dir / "IRF_data_used.png"))

    # 2. Save intermediate data
    plot2_data = {
        "combined_df": combined_df,
        "ocean_area": ocean_area
    }
    with open(INTERMEDIATE_DATA_PATH, 'wb') as f:
        pickle.dump(plot2_data, f)
    print(f"\nIntermediate data (with sunshine hours) saved to: {INTERMEDIATE_DATA_PATH}")

    # ===================== 核心：按 ocean + season 分组计算并打印目标角度 =====================
    # 过滤无效行，避免除0、arccos越界报错
    df_calc = combined_df[
        (combined_df['swdown'].notna()) &
        (combined_df['sunshine_hours_mean'].notna()) &
        (combined_df['sunshine_hours_mean'] > 1e-3) &  # 避免日照时长接近0
        (combined_df['ocean'].notna()) &
        (combined_df['season'].notna())
    ].copy()

    # 计算目标角度（严格按公式，做合法性校验）
    with np.errstate(invalid='ignore', divide='ignore'):
        # 先计算分式部分
        df_calc['calc_ratio'] = (df_calc['swdown'] * 24) / (df_calc['sunshine_hours_mean'] * S0)
        # 强制限制在arccos定义域[-1,1]，防止数值溢出
        df_calc['calc_ratio_clipped'] = df_calc['calc_ratio'].clip(-1.0, 1.0)
        # 计算弧度角度
        df_calc['target_angle_deg'] = np.degrees(np.arccos(df_calc['calc_ratio_clipped']))

    # 按海洋、季节分组求平均，统计有效样本数
    grouped_result = df_calc.groupby(['ocean', 'season'])['target_angle_deg'].agg(
        平均角度='mean',
        有效样本数='count'
    ).reset_index()
    # 按海洋、季节排序，结果更整洁
    grouped_result = grouped_result.sort_values(['ocean', 'season']).reset_index(drop=True)

    # 格式化打印结果（一定会输出）
    print("\n" + "="*90)
    print("各海洋-季节平均角度：arccos( swdown * 24 / sunshine_hours_mean / S0 )")
    print("="*90)
    print(f"{'海洋名':<12} {'季节':<6} {'平均角度':<15} {'有效样本数':<8}")
    print("-"*90)
    for _, row in grouped_result.iterrows():
        print(f"{row['ocean']:<12} {row['season']:<6} {row['平均角度']:<15.4f} {row['有效样本数']:<8d}")
    print("="*90 + "\n")