import os
import math
from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_fitting import oceans, format_panel_tag

# =========================
# Paths and basic settings
# =========================
BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
COEF_CSV = os.path.join(BASE_PATH, 'processed_data', 'sensitivity_albedo_vs_cot.csv')
FIG_SAVE_PATH = os.path.join(BASE_PATH, 'figs', 'fig4_plt_corr_coef.png')
os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)

OCEANS = oceans  # 8 ocean regions
SEASONS = ['MAM', 'JJA', 'SON', 'DJF']

HEATMAP_CMAP = plt.cm.GnBu
B_CMAP = plt.cm.pink_r
M_CMAP = plt.cm.YlOrRd

SIZE_PARAMS = {
    'small_tick': 8,
    'title': 14,
    'cbar_tick': 10,
    'cbar_label': 12,
    'legend': 10,
    'text_label': 22,
}

# Physical constants for SWdown calculation
S0 = 1361.0  # Solar constant (W/m2)
R_EARTH = 6371000  # Earth radius (meters)
M2_TO_KM2 = 1e6

# =========================
# Solar geometry helpers (from fig4_s3_IRF_data_used.py)
# =========================

def declination(n):
    """Solar declination angle (radians) from day of year."""
    return math.radians(23.45) * math.sin(2 * math.pi * (284 + n) / 365.0)

def E_ext(n):
    """Extraterrestrial solar irradiance (W/m2) from day of year."""
    return S0 * (1 + 0.033 * math.cos(2 * math.pi * n / 365.0))

def sunset_hour_angle(phi, delta):
    """Sunset hour angle (radians) from latitude and declination."""
    x = -math.tan(phi) * math.tan(delta)
    return 0.0 if x >= 1 else math.pi if x <= -1 else math.acos(x)

def H0_daily_mean(phi, n):
    """Daily-mean TOA solar radiation (W/m2)."""
    delta, E = declination(n), E_ext(n)
    omega_s = sunset_hour_angle(phi, delta)
    return (E / math.pi) * (math.cos(phi)*math.cos(delta)*math.sin(omega_s) + omega_s*math.sin(phi)*math.sin(delta))

def calc_monthly_swdown(lat, year=2020, month=None):
    """Monthly mean daily-mean SWdown for a given latitude."""
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

def calc_grid_cell_area(lat, lon_res=1.0, lat_res=1.0):
    """Grid cell area (km2) from latitude and resolution."""
    lat1, lat2 = math.radians(lat - lat_res/2), math.radians(lat + lat_res/2)
    dlon = math.radians(lon_res)
    area_m2 = dlon * (math.sin(lat2) - math.sin(lat1)) * (R_EARTH ** 2)
    return area_m2 / M2_TO_KM2

# =========================
# Data loading
# =========================

def load_coef_data():
    """Load the merged coefficient CSV."""
    df = pd.read_csv(COEF_CSV)
    return df


def get_data_matrix(df, method, var_type, value_name):
    """
    Extract a (8 oceans x 4 seasons) matrix for a given variable.
    """
    mdf = df[(df['Method'] == method)].copy()
    matrix = np.full((len(OCEANS), len(SEASONS)), np.nan)
    for i, ocean in enumerate(OCEANS):
        for j, season in enumerate(SEASONS):
            mask = (mdf['Ocean'] == ocean) & (mdf['Season'] == season)
            if mask.any():
                matrix[i, j] = mdf.loc[mask, value_name].values[0]
    return matrix


def plot_single_heatmap(ax, data, x_labels, y_labels, cmap, vmin=None, vmax=None, text_label=''):
    """
    Plot a single heatmap on the given axes.
    """
    masked_data = np.ma.masked_invalid(data)
    im = ax.imshow(masked_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=SIZE_PARAMS['small_tick'])
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=SIZE_PARAMS['small_tick'])
    
    # Remove tick marks (keep labels)
    ax.tick_params(length=0)
    
    # Add text label centered on the heatmap
    if text_label:
        n_y, n_x = data.shape
        ax.text(n_x / 2 - 0.5, n_y / 2 - 0.5, text_label,
                fontsize=SIZE_PARAMS['text_label'], fontweight='bold',
                color='k', ha='center', va='center', fontname='DejaVu Sans')
    
    return im


# =========================
# IRF calculation
# =========================

def load_merged_data():
    """Load all merged ocean-season CSV files."""
    dfs = []
    for ocean in OCEANS:
        for season in SEASONS:
            fpath = os.path.join(BASE_PATH, 'processed_data', 'merged_data', f'{ocean}_{season}.csv')
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath)
            df['ocean'] = ocean
            df['season'] = season
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_lnnd_o_lnaod():
    """Load the sensitivity_lnnd_vs_lnaod.csv coefficients."""
    fpath = os.path.join(BASE_PATH, 'processed_data', 'sensitivity_lnnd_vs_lnaod.csv')
    df = pd.read_csv(fpath)
    df.columns = [c.strip() for c in df.columns]
    df['Ocean'] = df['Ocean'].str.strip()
    df['Season'] = df['Season'].str.strip()
    return df


def calc_irf_ratios():
    """
    Calculate IRF ratios for each ocean:
      IRF_corrected / IRF_original
    
    Workflow:
      1. Load merged data (daily grid-cell data)
      2. Compute Ac_msk_origin from radiation fluxes:
         Ac_msk = (sw_all - sw_clr * (1 - cf_ceres)) / cf_ceres / solar_incoming
      3. Aggregate to seasonal means at each lat/lon grid point
      4. Compute IRF using seasonal means + coefficients
    
    Returns
    -------
    dict with keys:
      'ret_1030_ratio', 'ret_daytime_ratio', 'msk_1030_ratio', 'msk_daytime_ratio'
      'ret_orig', 'msk_orig'
    Each value is a dict {ocean: (mean, std)}
    """
    # Load coefficients
    coef_df = load_coef_data()
    
    # Load merged data
    merged_df = load_merged_data()
    
    # Load lnnd_o_lnaod
    lnnd_df = load_lnnd_o_lnaod()
    
    # ---- Step 1: Compute Ac_msk_origin from radiation fluxes ----
    # Ac_msk = (sw_all - sw_clr * (1 - cf_ceres)) / cf_ceres / solar_incoming
    with np.errstate(invalid='ignore', divide='ignore'):
        merged_df['Ac_msk_origin'] = (
            (merged_df['sw_all'] - merged_df['sw_clr'] * (1 - merged_df['cf_ceres']))
            / merged_df['cf_ceres']
            / merged_df['solar_incoming']
        )
    
    # Apply filter for Ac_msk
    msk_filter = (
        # (merged_df['cf_liq_ceres'] / merged_df['cf_ceres'] > 0.99)
        # & 
        (merged_df['Ac_msk_origin'] >= 0)
        & (merged_df['Ac_msk_origin'] <= 1)
    )
    merged_df.loc[~msk_filter, 'Ac_msk_origin'] = np.nan
    
    # ---- Step 2: Compute SWdown for each grid cell ----
    merged_df['month'] = pd.to_datetime(merged_df['time']).dt.month
    unique_lat_month = merged_df[['lat', 'month']].drop_duplicates()
    unique_lat_month['swdown'] = unique_lat_month.apply(
        lambda r: calc_monthly_swdown(r['lat'], month=r['month']), axis=1
    )
    merged_df = merged_df.merge(unique_lat_month, on=['lat', 'month'], how='left')
    
    # Compute grid area
    merged_df['grid_area_km2'] = merged_df['lat'].apply(calc_grid_cell_area)
    
    # ---- Step 3: Aggregate to seasonal means at each lat/lon ----
    # Variables to aggregate: swdown, log_aod_diff, Ac_msk_origin, cf_liq_ceres (CF_msk),
    #   cf_ret_liq_mod08 (CF_ret), cot_mod08, grid_area_km2
    agg_cols = {
        'swdown': 'mean',
        'log_aod_diff': 'mean',
        'Ac_msk_origin': 'mean',
        'cf_liq_ceres': 'mean',       # CF_msk
        'cf_ret_liq_mod08': 'mean',   # CF_ret
        'cot_mod08': 'mean',
        'grid_area_km2': 'first',     # area is same for same lat/lon
    }
    seasonal_grid = merged_df.groupby(['ocean', 'season', 'lat', 'lon']).agg(agg_cols).reset_index()
    
    # ---- Step 4: Merge coefficients ----
    # Merge lnnd_o_lnaod
    lnnd_lookup = lnnd_df.set_index(['Ocean', 'Season'])['Slope'].to_dict()
    seasonal_grid['lnnd_o_lnaod'] = seasonal_grid.apply(
        lambda r: lnnd_lookup.get((r['ocean'], r['season']), np.nan), axis=1
    )
    
    # Build coefficient lookup
    coef_lookup = {}
    for _, row in coef_df.iterrows():
        key = (row['Method'], row['Ocean'], row['Season'])
        coef_lookup[(key, 'k_1030')] = row['Slope_1030']
        coef_lookup[(key, 'k_day')] = row['Slope_Daytime']
        coef_lookup[(key, 'lnb_1030')] = row['Intercept_1030']
        coef_lookup[(key, 'lnb_day')] = row['Intercept_Daytime']
        coef_lookup[(key, 'k_1030_unc')] = row['Slope_1030_Unc']
        coef_lookup[(key, 'k_day_unc')] = row['Slope_Daytime_Unc']
        coef_lookup[(key, 'lnb_1030_unc')] = row['Intercept_1030_Unc']
        coef_lookup[(key, 'lnb_day_unc')] = row['Intercept_Daytime_Unc']
        coef_lookup[(key, 'm_day')] = row['Albedo_Ratio_Daytime_o_1030']
        coef_lookup[(key, 'm_day_unc')] = row['Albedo_Ratio_Unc']
    
    def get_coef(method, ocean, season, var):
        return coef_lookup.get(((method, ocean, season), var), np.nan)
    
    # ---- Step 5: Compute IRF for each ocean-season ----
    # IRF_base = (lnnd_o_lnaod / 3) * swdown * log_aod_diff
    irf_base = (seasonal_grid['lnnd_o_lnaod'] / 3.0) * seasonal_grid['swdown'] * seasonal_grid['log_aod_diff']
    
    # Original IRF
    # ret: Ac_ret_orig = 0.13 * cot / (1 + 0.13 * cot)
    cot = seasonal_grid['cot_mod08'].values
    Ac_ret_orig = 0.13 * cot / (1 + 0.13 * cot)
    cf_ret = seasonal_grid['cf_ret_liq_mod08'].values
    irf_ret_orig = irf_base.values * cf_ret * Ac_ret_orig * (1 - Ac_ret_orig)
    
    # msk: Ac_msk from radiation flux calculation
    Ac_msk = seasonal_grid['Ac_msk_origin'].values
    cf_msk = seasonal_grid['cf_liq_ceres'].values
    irf_msk_orig = irf_base.values * cf_msk * Ac_msk * (1 - Ac_msk)
    
    results = {
        'ret_1030_ratio': {}, 'ret_daytime_ratio': {},
        'msk_1030_ratio': {}, 'msk_daytime_ratio': {},
        'ret_orig': {}, 'msk_orig': {},
        'Ac_corr_ret_1030': {}, 'Ac_corr_ret_day': {},
        'Ac_corr_msk_1030': {}, 'Ac_corr_msk_day': {},
    }
    
    for ocean in OCEANS:
        for season in SEASONS:
            mask = (seasonal_grid['ocean'] == ocean) & (seasonal_grid['season'] == season)
            if not mask.any():
                continue
            
            sub = seasonal_grid[mask]
            area = sub['grid_area_km2'].values
            total_area = np.nansum(area)
            if total_area <= 0:
                continue
            
            # Area-weighted mean original IRF
            irf_ret_o = np.nansum(irf_ret_orig[mask.values] * area) / total_area
            irf_msk_o = np.nansum(irf_msk_orig[mask.values] * area) / total_area
            
            # Get coefficients for this ocean-season
            k_ret_1030 = get_coef('ret', ocean, season, 'k_1030')
            k_ret_day = get_coef('ret', ocean, season, 'k_day')
            lnb_ret_1030 = get_coef('ret', ocean, season, 'lnb_1030')
            lnb_ret_day = get_coef('ret', ocean, season, 'lnb_day')
            b_ret_1030 = np.exp(lnb_ret_1030) if not np.isnan(lnb_ret_1030) else np.nan
            b_ret_day = np.exp(lnb_ret_day) if not np.isnan(lnb_ret_day) else np.nan
            
            k_ret_1030_unc = get_coef('ret', ocean, season, 'k_1030_unc')
            k_ret_day_unc = get_coef('ret', ocean, season, 'k_day_unc')
            lnb_ret_1030_unc = get_coef('ret', ocean, season, 'lnb_1030_unc')
            lnb_ret_day_unc = get_coef('ret', ocean, season, 'lnb_day_unc')
            b_ret_1030_unc = b_ret_1030 * lnb_ret_1030_unc if not np.isnan(b_ret_1030) else np.nan
            b_ret_day_unc = b_ret_day * lnb_ret_day_unc if not np.isnan(b_ret_day) else np.nan
            
            k_msk_1030 = get_coef('msk', ocean, season, 'k_1030')
            k_msk_day = get_coef('msk', ocean, season, 'k_day')
            m_msk_1030 = 1.0  # m=1 for 10:30
            m_msk_day = get_coef('msk', ocean, season, 'm_day')
            
            k_msk_1030_unc = get_coef('msk', ocean, season, 'k_1030_unc')
            k_msk_day_unc = get_coef('msk', ocean, season, 'k_day_unc')
            m_msk_day_unc = get_coef('msk', ocean, season, 'm_day_unc')
            m_msk_1030_unc = 0.0
            
            # Per-grid-cell IRF ratios
            cot_sub = sub['cot_mod08'].values
            
            # ret 1030
            if not np.isnan(k_ret_1030) and not np.isnan(b_ret_1030):
                Ac_corr_1030 = b_ret_1030 * cot_sub ** k_ret_1030 / (1 + b_ret_1030 * cot_sub ** k_ret_1030)
                ratio_1030 = k_ret_1030 * Ac_corr_1030 * (1 - Ac_corr_1030) / (Ac_ret_orig[mask.values] * (1 - Ac_ret_orig[mask.values]))
                irf_ret_1030 = np.nansum(irf_ret_orig[mask.values] * ratio_1030 * area) / total_area
                ratio_mean = irf_ret_1030 / irf_ret_o if irf_ret_o != 0 else np.nan
            else:
                ratio_mean = np.nan
            
            # ret Daytime
            if not np.isnan(k_ret_day) and not np.isnan(b_ret_day):
                Ac_corr_day = b_ret_day * cot_sub ** k_ret_day / (1 + b_ret_day * cot_sub ** k_ret_day)
                ratio_day = k_ret_day * Ac_corr_day * (1 - Ac_corr_day) / (Ac_ret_orig[mask.values] * (1 - Ac_ret_orig[mask.values]))
                irf_ret_day = np.nansum(irf_ret_orig[mask.values] * ratio_day * area) / total_area
                ratio_mean_day = irf_ret_day / irf_ret_o if irf_ret_o != 0 else np.nan
            else:
                ratio_mean_day = np.nan
            
            # msk 1030
            if not np.isnan(k_msk_1030):
                Ac_corr_msk_1030 = m_msk_1030 * Ac_msk[mask.values]
                ratio_msk_1030 = k_msk_1030 * Ac_corr_msk_1030 * (1 - Ac_corr_msk_1030) / (Ac_msk[mask.values] * (1 - Ac_msk[mask.values]))
                irf_msk_1030 = np.nansum(irf_msk_orig[mask.values] * ratio_msk_1030 * area) / total_area
                ratio_mean_msk_1030 = irf_msk_1030 / irf_msk_o if irf_msk_o != 0 else np.nan
            else:
                ratio_mean_msk_1030 = np.nan
            
            # msk Daytime
            if not np.isnan(k_msk_day) and not np.isnan(m_msk_day):
                Ac_corr_msk_day = m_msk_day * Ac_msk[mask.values]
                ratio_msk_day = k_msk_day * Ac_corr_msk_day * (1 - Ac_corr_msk_day) / (Ac_msk[mask.values] * (1 - Ac_msk[mask.values]))
                irf_msk_day = np.nansum(irf_msk_orig[mask.values] * ratio_msk_day * area) / total_area
                ratio_mean_msk_day = irf_msk_day / irf_msk_o if irf_msk_o != 0 else np.nan
            else:
                ratio_mean_msk_day = np.nan
            
            # Store results
            results['ret_1030_ratio'][(ocean, season)] = ratio_mean
            results['ret_daytime_ratio'][(ocean, season)] = ratio_mean_day
            results['msk_1030_ratio'][(ocean, season)] = ratio_mean_msk_1030
            results['msk_daytime_ratio'][(ocean, season)] = ratio_mean_msk_day
            results['ret_orig'][(ocean, season)] = irf_ret_o
            results['msk_orig'][(ocean, season)] = irf_msk_o
            
            # Store area-weighted mean Ac_corr values
            if not np.isnan(k_ret_1030) and not np.isnan(b_ret_1030):
                ac_ret_1030_mean = np.nansum(Ac_corr_1030 * area) / total_area
            else:
                ac_ret_1030_mean = np.nan
            if not np.isnan(k_ret_day) and not np.isnan(b_ret_day):
                ac_ret_day_mean = np.nansum(Ac_corr_day * area) / total_area
            else:
                ac_ret_day_mean = np.nan
            if not np.isnan(k_msk_1030):
                ac_msk_1030_mean = np.nansum(Ac_corr_msk_1030 * area) / total_area
            else:
                ac_msk_1030_mean = np.nan
            if not np.isnan(k_msk_day) and not np.isnan(m_msk_day):
                ac_msk_day_mean = np.nansum(Ac_corr_msk_day * area) / total_area
            else:
                ac_msk_day_mean = np.nan
            
            results['Ac_corr_ret_1030'][(ocean, season)] = ac_ret_1030_mean
            results['Ac_corr_ret_day'][(ocean, season)] = ac_ret_day_mean
            results['Ac_corr_msk_1030'][(ocean, season)] = ac_msk_1030_mean
            results['Ac_corr_msk_day'][(ocean, season)] = ac_msk_day_mean
    
    # Aggregate to ocean level (average across seasons)
    ocean_results = {}
    for key in ['ret_1030_ratio', 'ret_daytime_ratio', 'msk_1030_ratio', 'msk_daytime_ratio', 'ret_orig', 'msk_orig']:
        ocean_results[key] = {}
        for ocean in OCEANS:
            vals = [results[key].get((ocean, s), np.nan) for s in SEASONS]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                ocean_results[key][ocean] = (np.mean(vals), np.std(vals) if len(vals) > 1 else 0.0)
            else:
                ocean_results[key][ocean] = (np.nan, np.nan)
    
    # Print mean Ac_corr values
    print("\n=== Mean Ac_corr values (area-weighted, global) ===")
    for ac_key, ac_label in [('Ac_corr_ret_1030', 'Ac_corr_ret_1030'),
                              ('Ac_corr_ret_day', 'Ac_corr_ret_daytime'),
                              ('Ac_corr_msk_1030', 'Ac_corr_msk_1030'),
                              ('Ac_corr_msk_day', 'Ac_corr_msk_daytime')]:
        all_vals = []
        for ocean in OCEANS:
            for season in SEASONS:
                v = results.get(ac_key, {}).get((ocean, season), np.nan)
                if not np.isnan(v):
                    all_vals.append(v)
        if all_vals:
            print(f"  {ac_label}: mean = {np.mean(all_vals):.4f}, std = {np.std(all_vals):.4f}")
        else:
            print(f"  {ac_label}: no data")
    
    return ocean_results


# =========================
# Global distribution plotting (3x2 map)
# =========================

def plot_global_distributions(df, fig_save_path, icon_style='nature'):
    """Generate 3x2 global distribution plot for core variables."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Aggregate to annual mean at each lat/lon grid point
    agg_cols = {
        'swdown': 'mean',
        'log_aod_diff': 'mean',
        'Ac_msk_origin': 'mean',
        'cf_liq_ceres': 'mean',
        'cf_ret_liq_mod08': 'mean',
        'cot_mod08': 'mean',
        'grid_area_km2': 'first',
    }
    df_grid = df.groupby(['lat', 'lon']).agg(agg_cols).reset_index()
    
    # Create gridded matrices
    lats = np.sort(df_grid['lat'].unique())
    lons = np.sort(df_grid['lon'].unique())
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    def make_grid(col):
        pivot_grid = df_grid.pivot(index='lat', columns='lon', values=col)
        return pivot_grid.reindex(index=lats, columns=lons).values
    
    area_grid = make_grid('grid_area_km2')
    
    # Define plot configurations
    grid_configs = [
        (make_grid('swdown'), plt.cm.Blues,
         f'{format_panel_tag(0, icon_style)} SW$_{{\mathrm{{down}}}}$', 'W m$^{-2}$'),
        (make_grid('log_aod_diff'), plt.cm.Blues,
         f'{format_panel_tag(1, icon_style)} $\\ln\\text{{AOD}}_{{\mathrm{{PD}}}} - \\ln\\text{{AOD}}_{{\mathrm{{PI}}}}$', ''),
        (make_grid('Ac_msk_origin'), plt.cm.Blues,
         f'{format_panel_tag(2, icon_style)} $A_{{\mathrm{{c,msk}}}}$', ''),
        (make_grid('cf_liq_ceres'), plt.cm.Blues,
         f'{format_panel_tag(3, icon_style)} CF$_{{\mathrm{{msk}}}}$', ''),
        (make_grid('cf_ret_liq_mod08'), plt.cm.Blues,
         f'{format_panel_tag(4, icon_style)} CF$_{{\mathrm{{ret}}}}$', ''),
        (np.log(make_grid('cot_mod08')), plt.cm.Blues,
         f'{format_panel_tag(5, icon_style)} $\\ln$COT', ''),
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(13, 7.5),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        wspace=0.17, hspace=0.25)
    
    for idx, (grid, cmap, title, cbar_label) in enumerate(grid_configs):
        ax = axes.flat[idx]
        
        # Add geographic features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='k', alpha=0.7)
        ax.add_feature(cfeature.LAND, color='#f5f5f5', alpha=0.6)
        ax.add_feature(cfeature.OCEAN, color='#eaf6fa', alpha=0.3)
        ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())
        
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                          linestyle='--', alpha=0.6, color='gray')
        gl.top_labels = gl.right_labels = False
        
        pc = ax.pcolormesh(lon_grid, lat_grid, grid,
                           cmap=cmap, transform=ccrs.PlateCarree(),
                           edgecolors='none', linewidth=0)
        ax.set_title(title, fontsize=16, pad=5, loc='left')
        
        # Add global weighted mean annotation
        valid_mask = ~np.isnan(grid)
        if np.any(valid_mask):
            mean_val = np.average(grid[valid_mask], weights=area_grid[valid_mask])
            ax.text(1.0, 1.02, f'Mean: {mean_val:.2f}',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=13.5)
        
        cbar = fig.colorbar(pc, ax=ax, orientation='vertical', fraction=0.05, shrink=0.89)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=12)
    
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    print(f"Global distribution figure saved to: {fig_save_path}")
    plt.close(fig)


# =========================
# Main plotting
# =========================

def main():
    df = load_coef_data()
    
    # Extract data matrices
    # ret 1030: slope = k, intercept = lnb -> b = exp(lnb)
    ret_k_1030 = get_data_matrix(df, 'ret', '1030', 'Slope_1030')
    ret_lnb_1030 = get_data_matrix(df, 'ret', '1030', 'Intercept_1030')
    ret_b_1030 = np.exp(ret_lnb_1030)  # convert lnb to b
    
    # msk 1030
    msk_k_1030 = get_data_matrix(df, 'msk', '1030', 'Slope_1030')
    msk_m_1030 = np.full_like(msk_k_1030, 1.0)  # m=1 for all
    
    # ret Daytime
    ret_k_day = get_data_matrix(df, 'ret', 'Daytime', 'Slope_Daytime')
    ret_lnb_day = get_data_matrix(df, 'ret', 'Daytime', 'Intercept_Daytime')
    ret_b_day = np.exp(ret_lnb_day)  # convert lnb to b
    
    # msk Daytime
    msk_k_day = get_data_matrix(df, 'msk', 'Daytime', 'Slope_Daytime')
    msk_m_day = get_data_matrix(df, 'msk', 'Daytime', 'Albedo_Ratio_Daytime_o_1030')
    
    # Determine global vmin/vmax for shared colorbars
    # k values (ret and msk share similar range)
    all_k = np.concatenate([ret_k_1030.ravel(), msk_k_1030.ravel(),
                            ret_k_day.ravel(), msk_k_day.ravel()])
    all_k = all_k[np.isfinite(all_k)]
    k_vmin = np.floor(np.min(all_k) * 10) / 10 if len(all_k) > 0 else 0
    k_vmax = np.ceil(np.max(all_k) * 10) / 10 if len(all_k) > 0 else 1
    
    # b values (ret only, now in linear space)
    all_b = np.concatenate([ret_b_1030.ravel(), ret_b_day.ravel()])
    all_b = all_b[np.isfinite(all_b)]
    b_vmin = np.floor(np.min(all_b) * 10) / 10 if len(all_b) > 0 else 0
    b_vmax = np.ceil(np.max(all_b) * 10) / 10 if len(all_b) > 0 else 1
    
    # m values (msk only)
    all_m = np.concatenate([msk_m_1030.ravel(), msk_m_day.ravel()])
    all_m = all_m[np.isfinite(all_m)]
    m_vmin = np.floor(np.min(all_m) * 10) / 10 if len(all_m) > 0 else 0.8
    m_vmax = np.ceil(np.max(all_m) * 10) / 10 if len(all_m) > 0 else 1.4
    
    # =========================
    # Calculate IRF ratios
    # =========================
    print("Calculating IRF ratios...")
    irf_results = calc_irf_ratios()
    
    # =========================
    # Plot global distributions
    # =========================
    print("Plotting global distributions...")
    merged_df = load_merged_data()
    # Compute Ac_msk_origin (same as in calc_irf_ratios)
    with np.errstate(invalid='ignore', divide='ignore'):
        merged_df['Ac_msk_origin'] = (
            (merged_df['sw_all'] - merged_df['sw_clr'] * (1 - merged_df['cf_ceres']))
            / merged_df['cf_ceres']
            / merged_df['solar_incoming']
        )
    msk_filter = (
        (merged_df['cf_liq_ceres'] / merged_df['cf_ceres'] > 0.90)
        & (merged_df['ret_albedo'] >= 0)
        & (merged_df['ret_albedo'] <= 1)
    )
    merged_df.loc[~msk_filter, 'Ac_msk_origin'] = np.nan
    # Compute SWdown
    merged_df['month'] = pd.to_datetime(merged_df['time']).dt.month
    unique_lat_month = merged_df[['lat', 'month']].drop_duplicates()
    unique_lat_month['swdown'] = unique_lat_month.apply(
        lambda r: calc_monthly_swdown(r['lat'], month=r['month']), axis=1
    )
    merged_df = merged_df.merge(unique_lat_month, on=['lat', 'month'], how='left')
    merged_df['grid_area_km2'] = merged_df['lat'].apply(calc_grid_cell_area)
    
    GLOBAL_DIST_PATH = os.path.join(BASE_PATH, 'figs', 'fig4_global_distributions.png')
    plot_global_distributions(merged_df, GLOBAL_DIST_PATH, icon_style='nature')
    
    # =========================
    # Create figure layout
    # =========================
    # 2x3 grid: top row = (a)(b), middle row = (c)(d), bottom row = (e) spanning full width
    # Each of (a)-(d) has 2 heatmaps (k + b/m)
    # (e) is a bar+scatter plot
    
    fig = plt.figure(figsize=(12, 9), dpi=100)
    
    # Layout parameters for heatmap rows
    left_margin = 0.06
    right_margin = 0.02
    top_margin = 0.03
    bottom_margin = 0.06
    h_space = 0.10
    w_space = 0.14
    inner_w_space = 0.0
    
    # Space for colorbars below bottom heatmap row
    cbar_height = 0.02
    cbar_gap_from_heatmap = 0.04
    
    # Heatmap rows occupy top portion
    heatmap_total_height = 0.55  # fraction of figure height for heatmaps
    bar_height = 1 - top_margin - bottom_margin - heatmap_total_height - 0.02  # remaining for bar plot
    
    # Heatmap group dimensions
    group_width = (1 - left_margin - right_margin - w_space) / 2
    group_height = (heatmap_total_height - h_space - cbar_height - cbar_gap_from_heatmap) / 2
    
    inner_width = (group_width - inner_w_space) / 2
    
    def get_group_rect(row, col):
        """Get the bounding box for a subplot group (row, col)."""
        left = left_margin + col * (group_width + w_space)
        if row == 0:
            bottom = bottom_margin + bar_height + 0.02 + cbar_height + cbar_gap_from_heatmap + group_height + h_space
        else:
            bottom = bottom_margin + bar_height + 0.02 + cbar_height + cbar_gap_from_heatmap
        return left, bottom, group_width, group_height
    
    def get_inner_rect(group_left, group_bottom, is_left):
        if is_left:
            return (group_left, group_bottom, inner_width, group_height)
        else:
            return (group_left + inner_width + inner_w_space, group_bottom, inner_width, group_height)
    
    def get_cbar_rect(group_left, is_left):
        if is_left:
            cbar_left = group_left + 0.005
        else:
            cbar_left = group_left + inner_width + inner_w_space + 0.005
        return (cbar_left, bottom_margin + bar_height + 0.02, inner_width - 0.01, cbar_height)
    
    # =========================
    # Plot subplots (a)-(d): heatmaps
    # =========================
    
    # Subplot (a): ret 1030
    gl_a, gb_a, gw_a, gh_a = get_group_rect(0, 0)
    
    ax_a_k = fig.add_axes(get_inner_rect(gl_a, gb_a, True))
    im_a_k = plot_single_heatmap(ax_a_k, ret_k_1030, SEASONS, OCEANS,
                                  HEATMAP_CMAP, vmin=k_vmin, vmax=k_vmax,
                                  text_label='$k_{\\mathrm{ret}}$')
    ax_a_k.set_title(
        f'{format_panel_tag(0, "nature")} Retrieval-Domain Coef., 10:30',
        fontsize=SIZE_PARAMS['title'], pad=10, loc='left', x=-0.40
    )
    
    ax_a_b = fig.add_axes(get_inner_rect(gl_a, gb_a, False))
    im_a_b = plot_single_heatmap(ax_a_b, ret_b_1030, SEASONS, OCEANS,
                                 B_CMAP, vmin=b_vmin, vmax=b_vmax,
                                 text_label='$l$')
    ax_a_b.set_yticklabels([])
    
    # Subplot (b): msk 1030
    gl_b, gb_b, gw_b, gh_b = get_group_rect(0, 1)
    
    ax_b_k = fig.add_axes(get_inner_rect(gl_b, gb_b, True))
    im_b_k = plot_single_heatmap(ax_b_k, msk_k_1030, SEASONS, OCEANS,
                                 HEATMAP_CMAP, vmin=k_vmin, vmax=k_vmax,
                                 text_label='$k_{\\mathrm{msk}}$')
    ax_b_k.set_title(
        f'{format_panel_tag(1, "nature")} Mask-Domain Coef., 10:30',
        fontsize=SIZE_PARAMS['title'], pad=10, loc='left', x=-0.40
    )
    
    ax_b_m = fig.add_axes(get_inner_rect(gl_b, gb_b, False))
    im_b_m = plot_single_heatmap(ax_b_m, msk_m_1030, SEASONS, OCEANS,
                                 M_CMAP, vmin=m_vmin, vmax=m_vmax,
                                 text_label='$m$')
    ax_b_m.set_yticklabels([])
    
    # Subplot (c): ret Daytime
    gl_c, gb_c, gw_c, gh_c = get_group_rect(1, 0)
    
    ax_c_k = fig.add_axes(get_inner_rect(gl_c, gb_c, True))
    im_c_k = plot_single_heatmap(ax_c_k, ret_k_day, SEASONS, OCEANS,
                                 HEATMAP_CMAP, vmin=k_vmin, vmax=k_vmax,
                                 text_label='$k_{\\mathrm{ret}}$')
    ax_c_k.set_title(
        f'{format_panel_tag(2, "nature")} Retrieval-Domain Coef., Daytime',
        fontsize=SIZE_PARAMS['title'], pad=10, loc='left', x=-0.40
    )
    
    ax_c_b = fig.add_axes(get_inner_rect(gl_c, gb_c, False))
    im_c_b = plot_single_heatmap(ax_c_b, ret_b_day, SEASONS, OCEANS,
                                 B_CMAP, vmin=b_vmin, vmax=b_vmax,
                                 text_label='$l$')
    ax_c_b.set_yticklabels([])
    
    # Subplot (d): msk Daytime
    gl_d, gb_d, gw_d, gh_d = get_group_rect(1, 1)
    
    ax_d_k = fig.add_axes(get_inner_rect(gl_d, gb_d, True))
    im_d_k = plot_single_heatmap(ax_d_k, msk_k_day, SEASONS, OCEANS,
                                 HEATMAP_CMAP, vmin=k_vmin, vmax=k_vmax,
                                 text_label='$k_{\\mathrm{msk}}$')
    ax_d_k.set_title(
        f'{format_panel_tag(3, "nature")} Mask-Domain Coef., Daytime',
        fontsize=SIZE_PARAMS['title'], pad=10, loc='left', x=-0.40
    )
    
    ax_d_m = fig.add_axes(get_inner_rect(gl_d, gb_d, False))
    im_d_m = plot_single_heatmap(ax_d_m, msk_m_day, SEASONS, OCEANS,
                                 M_CMAP, vmin=m_vmin, vmax=m_vmax,
                                 text_label='$m$')
    ax_d_m.set_yticklabels([])
    
    # =========================
    # Colorbars below heatmaps
    # =========================
    cbar_specs = [
        (im_a_k, gl_c, True, '$k$'),
        (im_a_b, gl_c, False, '$l$'),
        (im_b_k, gl_d, True, '$k$'),
        (im_b_m, gl_d, False, '$m$'),
    ]
    
    for im, group_left, is_left, label in cbar_specs:
        cax = fig.add_axes(get_cbar_rect(group_left, is_left))
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label(label, fontsize=SIZE_PARAMS['cbar_label'])
        cbar.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])
    
    # =========================
    # Subplot (e): IRF ratio bar chart
    # =========================
    ax_e = fig.add_axes([left_margin, bottom_margin, 1 - left_margin - right_margin, bar_height])
    
    # Prepare data
    ocean_names = OCEANS
    x = np.arange(len(ocean_names))
    width = 0.18
    
    # Ratios (left y-axis)
    ratio_keys = ['ret_1030_ratio', 'ret_daytime_ratio', 'msk_1030_ratio', 'msk_daytime_ratio']
    ratio_labels = ['Ret 10:30', 'Ret Daytime', 'Msk 10:30', 'Msk Daytime']
    colors = ['steelblue', 'lightblue', 'coral', 'salmon']
    
    for i, (key, label, color) in enumerate(zip(ratio_keys, ratio_labels, colors)):
        means = [irf_results[key].get(o, (np.nan, np.nan))[0] for o in ocean_names]
        stds = [irf_results[key].get(o, (np.nan, np.nan))[1] for o in ocean_names]
        ax_e.bar(x + i * width - 1.5 * width, means, width, yerr=stds,
                 label=label, color=color, capsize=3, edgecolor='k', linewidth=0.5)
    
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(ocean_names, fontsize=SIZE_PARAMS['small_tick'])
    ax_e.set_ylabel('IRF Ratio (Corrected / Original)', fontsize=SIZE_PARAMS['title'] - 1, color='k')
    ax_e.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
    ax_e.tick_params(axis='y', labelsize=SIZE_PARAMS['small_tick'])
    
    # Original IRF values (right y-axis)
    ax_e2 = ax_e.twinx()
    
    ret_orig_vals = [irf_results['ret_orig'].get(o, (np.nan, np.nan))[0] for o in ocean_names]
    msk_orig_vals = [irf_results['msk_orig'].get(o, (np.nan, np.nan))[0] for o in ocean_names]
    
    ax_e2.scatter(x - 0.5 * width, ret_orig_vals, marker='o', color='darkblue', s=40,
                  label='IRF$_{\\mathrm{ret,orig}}$', zorder=5)
    ax_e2.scatter(x + 0.5 * width, msk_orig_vals, marker='s', color='darkred', s=40,
                  label='IRF$_{\\mathrm{msk,orig}}$', zorder=5)
    
    ax_e2.set_ylabel('Original IRF (W m$^{-2}$)', fontsize=SIZE_PARAMS['title'] - 1, color='k')
    ax_e2.tick_params(axis='y', labelsize=SIZE_PARAMS['small_tick'])
    
    # Title and legend
    ax_e.set_title(
        f'{format_panel_tag(4, "nature")} IRF Ratio by Ocean',
        fontsize=SIZE_PARAMS['title'], pad=5, loc='left', x=-0.05
    )
    
    # Combine legends
    lines1, labels1 = ax_e.get_legend_handles_labels()
    lines2, labels2 = ax_e2.get_legend_handles_labels()
    ax_e.legend(lines1 + lines2, labels1 + labels2, fontsize=SIZE_PARAMS['legend'] - 1,
                loc='upper left', ncol=2, framealpha=0.8)
    
    # =========================
    # Save figure
    # =========================
    plt.savefig(FIG_SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {FIG_SAVE_PATH}")
    plt.close(fig)


if __name__ == '__main__':
    main()
