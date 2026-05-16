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
COEF_CSV = os.path.join(BASE_PATH, 'processed_data', 'sensitivity_albedo_vs_cot_ratio.csv')
FIG_SAVE_PATH = os.path.join(BASE_PATH, 'figs', 'fig4_plt_corr_coef.png')
os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)

# Switch to control whether to plot global distributions (3x2 map)
PLOT_GLOBAL_DIST = False

OCEANS = oceans  # 8 ocean regions
SEASONS = ['MAM', 'JJA', 'SON', 'DJF']

HEATMAP_CMAP = plt.cm.GnBu
B_CMAP = plt.cm.pink_r

SIZE_PARAMS = {
    'small_tick': 9.5,
    'title': 16,
    'xylabel': 14,
    'cbar_tick': 9.5,
    'cbar_label': 12,
    'legend': 13,
    'text_label': 22,
}

# Physical constants for SWdown calculation
S0 = 1361.0  # Solar constant (W/m2)
R_EARTH = 6371000  # Earth radius (meters)
M2_TO_KM2 = 1e6

# =========================
# Solar geometry helpers
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


def load_cf_sensitivity(method):
    """Load sensitivity_cf{method}_vs_lnnd.csv coefficients."""
    fpath = os.path.join(BASE_PATH, 'processed_data', f'sensitivity_cf{method}_vs_lnnd.csv')
    df = pd.read_csv(fpath)
    df.columns = [c.strip() for c in df.columns]
    df['Ocean'] = df['Ocean'].str.strip()
    df['Season'] = df['Season'].str.strip()
    return df


def calc_Ac(b, k, cot):
    """Corrected albedo: Ac = b * cot^k / (1 + b * cot^k)"""
    return b * cot ** k / (1 + b * cot ** k)


def calc_irf(k, cf, Ac, irf_base):
    """IRF per grid cell: k * irf_base * cf * Ac * (1 - Ac)"""
    return irf_base * k * Ac * (1 - Ac) * cf


def calc_beta(k, cf, Ac, Aclr, dcf):
    """Beta adjustment per grid cell: k * cf * Ac * (1-Ac) / 3 / (Ac - Aclr) / dcf"""
    return k / 3.0 * Ac * (1 - Ac) * cf / ((Ac - Aclr) * dcf + k / 3.0 * Ac * (1 - Ac) * cf)


def calc_irf_and_beta_ratios():
    """
    Calculate IRF ratios for each ocean:
      IRF_corrected / IRF_original
    
    Workflow:
      1. Load merged data (daily grid-cell data)
      2. Compute SWdown for each grid cell
      3. Aggregate to seasonal means at each lat/lon grid point
      4. Compute IRF using seasonal means + coefficients
    
    Returns
    -------
    dict with keys:
      'irf_ret_1030_ratio', 'irf_ret_day_ratio', 'irf_msk_1030_ratio', 'irf_msk_day_ratio'
      'irf_ret_orig', 'irf_msk_orig'
    Each value is a dict {ocean: (mean, std)}
    """
    # Load coefficients
    coef_df = load_coef_data()
    
    # Load merged data
    merged_df = load_merged_data()
    
    # Load lnnd_o_lnaod
    lnnd_df = load_lnnd_o_lnaod()
    
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
    # Variables to aggregate: swdown, log_aod_diff, cf_liq_ceres (CF_msk),
    #   cf_ret_liq_mod08 (CF_ret), cot_mod08, grid_area_km2
    agg_cols = {
        'swdown': 'mean',
        'log_aod_diff': 'mean',
        'cf_liq_ceres': 'mean',       # CF_msk
        'cf_ret_liq_mod08': 'mean',   # CF_ret
        'cot_mod08': 'mean',
        'grid_area_km2': 'first',     # area is same for same lat/lon
    }
    seasonal_grid = merged_df.groupby(['ocean', 'season', 'lat', 'lon']).agg(agg_cols).reset_index()
    
    # ---- Compute Aclr = sw_clr / solar_incoming for each grid cell ----
    merged_df['Aclr'] = merged_df['sw_clr'] / merged_df['solar_incoming']
    # Aggregate Aclr to seasonal means at each lat/lon
    aclr_agg = merged_df.groupby(['ocean', 'season', 'lat', 'lon'])['Aclr'].mean().reset_index()
    seasonal_grid = seasonal_grid.merge(aclr_agg, on=['ocean', 'season', 'lat', 'lon'], how='left')
    
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
        coef_lookup[(key, 'k_day')] = row['Slope_Daytime']
        coef_lookup[(key, 'lnb_day')] = row['Intercept_Daytime']
        coef_lookup[(key, 'k_day_unc')] = row['Slope_Daytime_Unc']
        coef_lookup[(key, 'lnb_day_unc')] = row['Intercept_Daytime_Unc']
    
    def get_coef(method, ocean, season, var):
        return coef_lookup.get(((method, ocean, season), var), np.nan)
    
    # ---- Step 5: Compute IRF for each ocean-season ----
    # IRF_base = (lnnd_o_lnaod / 3) * swdown * log_aod_diff
    irf_base = (seasonal_grid['lnnd_o_lnaod'] / 3.0) * seasonal_grid['swdown'] * seasonal_grid['log_aod_diff']
    
    # Original albedo
    cot = seasonal_grid['cot_mod08'].values
    Ac_orig = calc_Ac(0.13, 1, cot)
    
    # Load CF sensitivity coefficients
    cf_msk_df = load_cf_sensitivity('msk')
    cf_ret_df = load_cf_sensitivity('ret')
    cf_msk_lookup = cf_msk_df.set_index(['Ocean', 'Season'])['Slope'].to_dict()
    cf_ret_lookup = cf_ret_df.set_index(['Ocean', 'Season'])['Slope'].to_dict()
    
    results = {
        'irf_ret_orig': {}, 'irf_msk_orig': {},
        'irf_ret_day': {}, 'irf_msk_day': {},
        'irf_ret_day_unc': {}, 'irf_msk_day_unc': {},
        'Ac_corr_ret_day': {},
        'Ac_corr_msk_day': {},
        'beta_ret_orig': {}, 'beta_msk_orig': {},
        'beta_ret_day': {}, 'beta_msk_day': {},
        'beta_ret_day_unc': {}, 'beta_msk_day_unc': {},
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
            
            # Get coefficients for this ocean-season
            k_ret_day = get_coef('ret', ocean, season, 'k_day')
            lnb_ret_day = get_coef('ret', ocean, season, 'lnb_day')
            b_ret_day = np.exp(lnb_ret_day) if not np.isnan(lnb_ret_day) else np.nan
            
            k_ret_day_unc = get_coef('ret', ocean, season, 'k_day_unc')
            lnb_ret_day_unc = get_coef('ret', ocean, season, 'lnb_day_unc')
            b_ret_day_unc = b_ret_day * lnb_ret_day_unc if not np.isnan(b_ret_day) else np.nan
            
            k_msk_day = get_coef('msk', ocean, season, 'k_day')
            lnb_msk_day = get_coef('msk', ocean, season, 'lnb_day')
            b_msk_day = np.exp(lnb_msk_day) if not np.isnan(lnb_msk_day) else np.nan
            
            k_msk_day_unc = get_coef('msk', ocean, season, 'k_day_unc')
            lnb_msk_day_unc = get_coef('msk', ocean, season, 'lnb_day_unc')
            b_msk_day_unc = b_msk_day * lnb_msk_day_unc if not np.isnan(b_msk_day) else np.nan
            
            # Per-grid-cell data
            cot_sub = sub['cot_mod08'].values
            cf_ret_vals = sub['cf_ret_liq_mod08'].values
            cf_msk_vals = sub['cf_liq_ceres'].values
            aclr_vals = sub['Aclr'].values
            Ac_orig_sub = Ac_orig[mask.values]
            irf_base_sub = irf_base.values[mask.values]
            
            # Original IRF (computed inside loop for consistency with beta)
            irf_ret_orig = calc_irf(1.0, cf_ret_vals, Ac_orig_sub, irf_base_sub)
            irf_msk_orig = calc_irf(1.0, cf_msk_vals, Ac_orig_sub, irf_base_sub)
            irf_ret_orig_mean = np.nansum(irf_ret_orig * area) / total_area
            irf_msk_orig_mean = np.nansum(irf_msk_orig * area) / total_area
            
            # ---- Compute IRF / CF Adjustment (beta) ----
            dcf_ret = cf_ret_lookup.get((ocean, season), np.nan)
            dcf_msk = cf_msk_lookup.get((ocean, season), np.nan)
            
            # Original beta (using original Ac)
            beta_ret_orig = calc_beta(1.0, cf_ret_vals, Ac_orig_sub, aclr_vals, dcf_ret)
            beta_msk_orig = calc_beta(1.0, cf_msk_vals, Ac_orig_sub, aclr_vals, dcf_msk)
            beta_ret_orig_mean = np.nansum(beta_ret_orig * area) / total_area
            beta_msk_orig_mean = np.nansum(beta_msk_orig * area) / total_area
            
            # Corrected IRF and beta values (per grid cell, then area-weighted)
            Ac_corr_day = calc_Ac(b_ret_day, k_ret_day, cot_sub)
            Ac_corr_msk_day = calc_Ac(b_msk_day, k_msk_day, cot_sub)
            
            irf_ret_day = calc_irf(k_ret_day, cf_ret_vals, Ac_corr_day, irf_base_sub)
            irf_msk_day = calc_irf(k_msk_day, cf_msk_vals, Ac_corr_msk_day, irf_base_sub)
            beta_ret_day = calc_beta(k_ret_day, cf_ret_vals, Ac_corr_day, aclr_vals, dcf_ret)
            beta_msk_day = calc_beta(k_msk_day, cf_msk_vals, Ac_corr_msk_day, aclr_vals, dcf_msk)
            
            # Area-weighted mean and standard deviation across grid cells
            irf_ret_day_mean = np.nansum(irf_ret_day * area) / total_area
            irf_msk_day_mean = np.nansum(irf_msk_day * area) / total_area
            beta_ret_day_mean = np.nansum(beta_ret_day * area) / total_area
            beta_msk_day_mean = np.nansum(beta_msk_day * area) / total_area
            
            # Weighted std across grid cells
            def weighted_std(vals, w):
                valid = np.isfinite(vals) & np.isfinite(w) & (w > 0)
                v, wgt = vals[valid], w[valid]
                if len(v) < 2:
                    return 0.0
                mean = np.average(v, weights=wgt)
                variance = np.average((v - mean)**2, weights=wgt)
                return np.sqrt(variance)
            
            irf_ret_day_unc = weighted_std(irf_ret_day, area)
            irf_msk_day_unc = weighted_std(irf_msk_day, area)
            beta_ret_day_unc = weighted_std(beta_ret_day, area)
            beta_msk_day_unc = weighted_std(beta_msk_day, area)
            
            # Store results (area-weighted means)
            results['irf_ret_orig'][(ocean, season)] = irf_ret_orig_mean
            results['irf_msk_orig'][(ocean, season)] = irf_msk_orig_mean
            results['irf_ret_day'][(ocean, season)] = irf_ret_day_mean
            results['irf_msk_day'][(ocean, season)] = irf_msk_day_mean
            results['irf_ret_day_unc'][(ocean, season)] = irf_ret_day_unc
            results['irf_msk_day_unc'][(ocean, season)] = irf_msk_day_unc
            
            results['beta_ret_orig'][(ocean, season)] = beta_ret_orig_mean
            results['beta_msk_orig'][(ocean, season)] = beta_msk_orig_mean
            results['beta_ret_day'][(ocean, season)] = beta_ret_day_mean
            results['beta_msk_day'][(ocean, season)] = beta_msk_day_mean
            results['beta_ret_day_unc'][(ocean, season)] = beta_ret_day_unc
            results['beta_msk_day_unc'][(ocean, season)] = beta_msk_day_unc
            
            # Store area-weighted mean Ac_corr values
            results['Ac_corr_ret_day'][(ocean, season)] = np.nansum(Ac_corr_day * area) / total_area
            results['Ac_corr_msk_day'][(ocean, season)] = np.nansum(Ac_corr_msk_day * area) / total_area
    
    # Aggregate to ocean level (average across seasons)
    ocean_results = {}
    
    # Original values: error bars = 0 (no k/b uncertainty)
    for key in ['irf_ret_orig', 'irf_msk_orig', 'beta_ret_orig', 'beta_msk_orig']:
        ocean_results[key] = {}
        for ocean in OCEANS:
            vals = [results[key].get((ocean, s), np.nan) for s in SEASONS]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                ocean_results[key][ocean] = (np.mean(vals), 0.0)
            else:
                ocean_results[key][ocean] = (np.nan, np.nan)
    
    # For corrected values, use uncertainty from k/lnb propagation (not seasonal std)
    for key in ['irf_ret_day', 'irf_msk_day', 'beta_ret_day', 'beta_msk_day']:
        unc_key = key + '_unc'
        ocean_results[key] = {}
        for ocean in OCEANS:
            vals = [results[key].get((ocean, s), np.nan) for s in SEASONS]
            unc_vals = [results[unc_key].get((ocean, s), np.nan) for s in SEASONS]
            vals = [v for v in vals if not np.isnan(v)]
            unc_vals = [u for u in unc_vals if not np.isnan(u)]
            if vals:
                # Mean across seasons, uncertainty = RMS of seasonal uncertainties
                mean_val = np.mean(vals)
                rms_unc = np.sqrt(np.nanmean([u**2 for u in unc_vals])) if unc_vals else 0.0
                ocean_results[key][ocean] = (mean_val, rms_unc)
            else:
                ocean_results[key][ocean] = (np.nan, np.nan)
    
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
    
    # Define plot configurations (5 subplots: 3 rows x 2 cols, last one empty)
    grid_configs = [
        (make_grid('swdown'), plt.cm.Blues,
         f'{format_panel_tag(0, icon_style)} SW$_{{\mathrm{{down}}}}$', 'W m$^{-2}$'),
        (make_grid('log_aod_diff'), plt.cm.Blues,
         f'{format_panel_tag(1, icon_style)} $\\ln\\text{{AOD}}_{{\mathrm{{PD}}}} - \\ln\\text{{AOD}}_{{\mathrm{{PI}}}}$', ''),
        (make_grid('cf_liq_ceres'), plt.cm.Blues,
         f'{format_panel_tag(2, icon_style)} CF$_{{\mathrm{{msk}}}}$', ''),
        (make_grid('cf_ret_liq_mod08'), plt.cm.Blues,
         f'{format_panel_tag(3, icon_style)} CF$_{{\mathrm{{ret}}}}$', ''),
        (np.log(make_grid('cot_mod08')), plt.cm.Blues,
         f'{format_panel_tag(4, icon_style)} $\\ln$COT', ''),
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 7.5),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        wspace=0.17, hspace=0.25)
    
    # Hide the last (6th) subplot since we only have 5
    axes.flat[-1].set_visible(False)
    
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
    msk_lnb_1030 = get_data_matrix(df, 'msk', '1030', 'Intercept_1030')
    msk_b_1030 = np.exp(msk_lnb_1030)  # convert lnb to b
    
    # ret day
    ret_k_day = get_data_matrix(df, 'ret', 'day', 'Slope_Daytime')
    ret_lnb_day = get_data_matrix(df, 'ret', 'day', 'Intercept_Daytime')
    ret_b_day = np.exp(ret_lnb_day)  # convert lnb to b
    
    # msk day
    msk_k_day = get_data_matrix(df, 'msk', 'day', 'Slope_Daytime')
    msk_lnb_day = get_data_matrix(df, 'msk', 'day', 'Intercept_Daytime')
    msk_b_day = np.exp(msk_lnb_day)  # convert lnb to b
    
    # Determine global vmin/vmax for shared colorbars
    # k values (ret and msk share similar range)
    all_k = np.concatenate([ret_k_1030.ravel(), msk_k_1030.ravel(),
                            ret_k_day.ravel(), msk_k_day.ravel()])
    all_k = all_k[np.isfinite(all_k)]
    k_vmin = np.floor(np.min(all_k) * 10) / 10 if len(all_k) > 0 else 0
    k_vmax = np.ceil(np.max(all_k) * 10) / 10 if len(all_k) > 0 else 1
    
    # b values (all methods, in linear space)
    all_b = np.concatenate([ret_b_1030.ravel(), ret_b_day.ravel(),
                            msk_b_1030.ravel(), msk_b_day.ravel()])
    all_b = all_b[np.isfinite(all_b)]
    b_vmin = np.floor(np.min(all_b) * 10) / 10 if len(all_b) > 0 else 0
    b_vmax = np.ceil(np.max(all_b) * 10) / 10 if len(all_b) > 0 else 1
    
    # =========================
    # Calculate IRF ratios
    # =========================
    print("Calculating IRF ratios...")
    irf_results = calc_irf_and_beta_ratios()
    
    # =========================
    # Plot global distributions (optional)
    # =========================
    if PLOT_GLOBAL_DIST:
        print("Plotting global distributions...")
        merged_df = load_merged_data()
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
    # Single subplot with 8 heatmaps in 2x4 grid:
    #   Row 0: k_ret_1030, k_ret_day, k_msk_1030, k_msk_day
    #   Row 1: b_ret_1030, b_ret_day, b_msk_1030, b_msk_day
    # Two shared colorbars (k and b) on the right
    # Bottom subplots: (b) IRF bar chart, (c) Beta bar chart
    
    fig = plt.figure(figsize=(11, 11), dpi=100)
    
    left_margin = 0.06
    right_margin = 0.02
    top_margin = 0.04
    bottom_margin = 0.06
    
    # Heatmap area
    n_rows, n_cols = 2, 4
    heatmap_total_height = 0.42
    bar_total_height = 1 - top_margin - bottom_margin - heatmap_total_height - 0.04
    bar_height = bar_total_height / 2
    
    # Colorbar settings
    cbar_height = 0.018
    cbar_gap = 0.03
    
    # Individual heatmap dimensions (reserve space for vertical colorbars on right)
    cbar_width = 0.012
    hm_w = (1 - left_margin - right_margin - cbar_width - 0.01) / n_cols
    hm_h = (heatmap_total_height - cbar_height - cbar_gap) / n_rows
    
    # Heatmap data and labels
    k_data_list = [
        (ret_k_1030, '$k_{\\mathrm{ret,1030}}$'),
        (ret_k_day,   '$k_{\\mathrm{ret,day}}$'),
        (msk_k_1030,  '$k_{\\mathrm{msk,1030}}$'),
        (msk_k_day,   '$k_{\\mathrm{msk,day}}$'),
    ]
    b_data_list = [
        (ret_b_1030, '$b_{\\mathrm{ret,1030}}$'),
        (ret_b_day,   '$b_{\\mathrm{ret,day}}$'),
        (msk_b_1030,  '$b_{\\mathrm{msk,1030}}$'),
        (msk_b_day,   '$b_{\\mathrm{msk,day}}$'),
    ]
    
    # Row 0: k heatmaps
    im_k_list = []
    for col, (data, label) in enumerate(k_data_list):
        left = left_margin + col * hm_w
        bottom = bottom_margin + bar_height + 0.03 + cbar_height + cbar_gap + hm_h
        ax = fig.add_axes([left, bottom, hm_w, hm_h])
        im = plot_single_heatmap(ax, data, SEASONS, OCEANS,
                                 HEATMAP_CMAP, vmin=k_vmin, vmax=k_vmax,
                                 text_label=label)
        im_k_list.append(im)
        if col > 0:
            ax.set_yticklabels([])
    
    # Row 1: b heatmaps
    im_b_list = []
    for col, (data, label) in enumerate(b_data_list):
        left = left_margin + col * hm_w
        bottom = bottom_margin + bar_height + 0.03 + cbar_height + cbar_gap
        ax = fig.add_axes([left, bottom, hm_w, hm_h])
        im = plot_single_heatmap(ax, data, SEASONS, OCEANS,
                                 B_CMAP, vmin=b_vmin, vmax=b_vmax,
                                 text_label=label)
        im_b_list.append(im)
        if col > 0:
            ax.set_yticklabels([])
    
    # Title above the heatmap rows (left-aligned)
    fig.text(left_margin, bottom_margin + bar_height + 0.02 + cbar_height + cbar_gap + hm_h * 2 + 0.01,
             f'{format_panel_tag(0, "nature")} Correction Coefficients',
             fontsize=SIZE_PARAMS['title'], ha='left', va='bottom')
    
    # Colorbars on the right side
    cbar_width = 0.012
    cbar_right_x = 1 - right_margin - cbar_width
    
    # k colorbar (right of row 0)
    cax_k = fig.add_axes([cbar_right_x, bottom_margin + bar_height + 0.04 + cbar_height + cbar_gap + hm_h,
                          cbar_width, hm_h-0.02])
    cbar_k = fig.colorbar(im_k_list[0], cax=cax_k, orientation='vertical')
    cbar_k.set_label('$k$', fontsize=SIZE_PARAMS['cbar_label'])
    cbar_k.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])
    
    # b colorbar (right of row 1)
    cax_b = fig.add_axes([cbar_right_x, bottom_margin + bar_height + 0.04 + cbar_height + cbar_gap,
                          cbar_width, hm_h-0.02])
    cbar_b = fig.colorbar(im_b_list[0], cax=cax_b, orientation='vertical')
    cbar_b.set_label('$b$', fontsize=SIZE_PARAMS['cbar_label'])
    cbar_b.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])
    
    # Compute global area-weighted mean for each variable
    # Use ocean areas as weights (approximate total ocean area per region)
    ocean_areas = {
        'NPO': 103.3, 'NAO': 41.5, 'TPO': 165.2, 'TAO': 60.0,
        'TIO': 73.4, 'SPO': 77.0, 'SAO': 40.3, 'SIO': 60.6,
    }  # in million km2 (approximate)
    total_ocean_area = sum(ocean_areas.values())
    
    def global_mean(results_dict, key):
        vals = [results_dict[key].get(o, (np.nan, np.nan))[0] for o in OCEANS]
        weights = [ocean_areas[o] for o in OCEANS]
        valid = [(v, w) for v, w in zip(vals, weights) if not np.isnan(v)]
        if not valid:
            return np.nan
        return np.average([v for v, _ in valid], weights=[w for _, w in valid])
    
    # =========================
    # Subplot (b): IRF bar chart (actual values, no ratios)
    # =========================
    ax_b = fig.add_axes([left_margin, bottom_margin + 0.02, 1 - left_margin - right_margin, bar_height])
    
    ocean_names = OCEANS
    x = np.arange(len(ocean_names))
    width = 0.18
    
    irf_keys = ['irf_ret_day', 'irf_ret_orig', 'irf_msk_day', 'irf_msk_orig']
    irf_labels = ['IRF$_{\\mathrm{ret,corr}}$', 'IRF$_{\\mathrm{ret,orig}}$',
                  'IRF$_{\\mathrm{msk,corr}}$', 'IRF$_{\\mathrm{msk,orig}}$']
    colors = ['blue', 'lightblue', 'firebrick', 'pink']
    
    for i, (key, label, color) in enumerate(zip(irf_keys, irf_labels, colors)):
        means = [irf_results[key].get(o, (np.nan, np.nan))[0] for o in ocean_names]
        stds = [irf_results[key].get(o, (np.nan, np.nan))[1] for o in ocean_names]
        gmean = global_mean(irf_results, key)
        ax_b.bar(x + i * width - 1.5 * width, means, width, #yerr=stds,
                 label=f'{label} ({gmean:.2f})', color=color, capsize=3)
    
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(ocean_names, fontsize=SIZE_PARAMS['small_tick'])
    ax_b.set_ylabel('IRF (W m$^{-2}$)', fontsize=SIZE_PARAMS['xylabel'] - 1, color='k')
    ax_b.tick_params(axis='y', labelsize=SIZE_PARAMS['small_tick'])
    ax_b.legend(fontsize=SIZE_PARAMS['legend'] - 1, loc='upper right', ncol=4, framealpha=0.8)
    
    ax_b.set_title(
        f'{format_panel_tag(1, "nature")} IRF',
        fontsize=SIZE_PARAMS['title'], pad=5, loc='left'
    )
    
    # =========================
    # Subplot (c): Beta bar chart (actual values, no ratios)
    # =========================
    ax_c = fig.add_axes([left_margin, bottom_margin - bar_height - 0.04, 1 - left_margin - right_margin, bar_height])
    
    beta_keys = ['beta_ret_day', 'beta_ret_orig', 'beta_msk_day', 'beta_msk_orig']
    beta_labels = ['$\\beta_{\\mathrm{ret,corr}}$', '$\\beta_{\\mathrm{ret,orig}}$',
                   '$\\beta_{\\mathrm{msk,corr}}$', '$\\beta_{\\mathrm{msk,orig}}$']
    
    for i, (key, label, color) in enumerate(zip(beta_keys, beta_labels, colors)):
        means = [irf_results[key].get(o, (np.nan, np.nan))[0] for o in ocean_names]
        stds = [irf_results[key].get(o, (np.nan, np.nan))[1] for o in ocean_names]
        gmean = global_mean(irf_results, key)
        ax_c.bar(x + i * width - 1.5 * width, means, width, #yerr=stds,
                 label=f'{label} ({gmean:.2f})', color=color, capsize=3)
    
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(ocean_names, fontsize=SIZE_PARAMS['small_tick'])
    ax_c.set_ylabel(r'$\beta$', fontsize=SIZE_PARAMS['xylabel'] - 1, color='k')
    ax_c.tick_params(axis='y', labelsize=SIZE_PARAMS['small_tick'])
    ax_c.legend(fontsize=SIZE_PARAMS['legend'] - 1, loc='upper center', ncol=4, framealpha=0.8)
    
    ax_c.set_title(
        rf'{format_panel_tag(2, "nature")} IRF / CF Adjustment ($\beta$)',
        fontsize=SIZE_PARAMS['title'], pad=5, loc='left'
    )
    
    # =========================
    # Save figure
    # =========================
    plt.savefig(FIG_SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {FIG_SAVE_PATH}")
    plt.close(fig)


if __name__ == '__main__':
    main()
