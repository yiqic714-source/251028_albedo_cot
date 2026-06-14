import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_fitting import oceans, format_panel_tag
from utils_solar import (
    calc_monthly_swdown, calc_grid_cell_area, calc_Ac
)

# =========================
# Paths and basic settings
# =========================
BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
COEF_CSV = os.path.join(BASE_PATH, 'processed_data', 'sensitivity_albedo_vs_cot_ratio.csv')
FIG_SAVE_PATH = os.path.join(BASE_PATH, 'figs', 'fig4_dac_o_dlncot_irf_and_erf.png')
os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)

# Switch to control whether to plot global distributions (3x2 map)
PLOT_GLOBAL_DIST = False

OCEANS = oceans  # 8 ocean regions
SEASONS = ['MAM', 'JJA', 'SON', 'DJF']

SIZE_PARAMS = {
    'small_tick': 10,
    'title': 16,
    'xylabel': 14,
    'legend': 12.5,
}

# =========================
# Data loading
# =========================

def load_coef_data():
    """Load the merged coefficient CSV."""
    df = pd.read_csv(COEF_CSV)
    return df


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


def calc_ac_sensitivity(k, Ac):
    """ Ac_sensitivity = k * Ac * (1-Ac) """
    return k * Ac * (1 - Ac)


def calc_irf(ac_sensitivity, cf, irf_base):
    """ IRF = IRF_base * Ac_sensitivity * CF """
    return irf_base /3 * ac_sensitivity * cf


def calc_erf(irf, irf_base, Ac, Aclr, dcf):
    """ ERF = IRF + IRF_base * (Ac - Aclr) * dcf """
    return irf + irf_base * (Ac - Aclr) * dcf


# =========================
# Bar data calculation
# =========================

def get_bar_data():
    """
    Calculate albedo sensitivity (as), IRF, and ERF for each ocean.
    
    Returns
    -------
    dict with keys:
      'as_ret_1030', 'as_ret_day', 'as_ret_orig',
      'as_msk_1030', 'as_msk_day', 'as_msk_orig',
      'irf_ret_1030', 'irf_ret_day', 'irf_ret_orig',
      'irf_msk_1030', 'irf_msk_day', 'irf_msk_orig',
      'erf_ret_1030', 'erf_ret_day', 'erf_ret_orig',
      'erf_msk_1030', 'erf_msk_day', 'erf_msk_orig'
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
    agg_cols = {
        'swdown': 'mean',
        'log_aod_diff': 'mean',
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
    
    # Build coefficient lookup (both 1030 and day)
    coef_lookup = {}
    for _, row in coef_df.iterrows():
        key = (row['Method'], row['Ocean'], row['Season'])
        coef_lookup[(key, 'k_1030')] = row['Slope_1030']
        coef_lookup[(key, 'lnb_1030')] = row['Intercept_1030']
        coef_lookup[(key, 'k_day')] = row['Slope_Daytime']
        coef_lookup[(key, 'lnb_day')] = row['Intercept_Daytime']
    
    def get_coef(method, ocean, season, var):
        return coef_lookup.get(((method, ocean, season), var), np.nan)
    
    # ---- Step 5: Compute IRF for each ocean-season ----
    # IRF_base = (lnnd_o_lnaod / 3) * swdown * log_aod_diff
    irf_base = seasonal_grid['swdown'] * seasonal_grid['lnnd_o_lnaod'] * seasonal_grid['log_aod_diff']
    
    # Original albedo
    cot = seasonal_grid['cot_mod08'].values
    Ac_orig = calc_Ac(0.13, 1, cot)
    
    # Load CF sensitivity coefficients (dcf)
    cf_msk_df = load_cf_sensitivity('msk')
    cf_ret_df = load_cf_sensitivity('ret')
    cf_msk_lookup = cf_msk_df.set_index(['Ocean', 'Season'])['Slope'].to_dict()
    cf_ret_lookup = cf_ret_df.set_index(['Ocean', 'Season'])['Slope'].to_dict()
    
    results = {
        'as_ret_1030': {}, 'as_ret_day': {}, 'as_ret_orig': {},
        'as_msk_1030': {}, 'as_msk_day': {}, 'as_msk_orig': {},
        'irf_ret_1030': {}, 'irf_ret_day': {}, 'irf_ret_orig': {},
        'irf_msk_1030': {}, 'irf_msk_day': {}, 'irf_msk_orig': {},
        'erf_ret_1030': {}, 'erf_ret_day': {}, 'erf_ret_orig': {},
        'erf_msk_1030': {}, 'erf_msk_day': {}, 'erf_msk_orig': {},
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
            
            # Get coefficients
            k_ret_1030 = get_coef('ret', ocean, season, 'k_1030')
            lnb_ret_1030 = get_coef('ret', ocean, season, 'lnb_1030')
            b_ret_1030 = np.exp(lnb_ret_1030) if not np.isnan(lnb_ret_1030) else np.nan
            
            k_ret_day = get_coef('ret', ocean, season, 'k_day')
            lnb_ret_day = get_coef('ret', ocean, season, 'lnb_day')
            b_ret_day = np.exp(lnb_ret_day) if not np.isnan(lnb_ret_day) else np.nan
            
            k_msk_1030 = get_coef('msk', ocean, season, 'k_1030')
            lnb_msk_1030 = get_coef('msk', ocean, season, 'lnb_1030')
            b_msk_1030 = np.exp(lnb_msk_1030) if not np.isnan(lnb_msk_1030) else np.nan
            
            k_msk_day = get_coef('msk', ocean, season, 'k_day')
            lnb_msk_day = get_coef('msk', ocean, season, 'lnb_day')
            b_msk_day = np.exp(lnb_msk_day) if not np.isnan(lnb_msk_day) else np.nan
            
            # Per-grid-cell data
            cot_sub = sub['cot_mod08'].values
            cf_ret_vals = sub['cf_ret_liq_mod08'].values
            cf_msk_vals = sub['cf_liq_ceres'].values
            Ac_orig_sub = Ac_orig[mask.values]
            irf_base_sub = irf_base.values[mask.values]
            
            # Compute corrected albedo for each variant
            Ac_ret_1030 = calc_Ac(b_ret_1030, k_ret_1030, cot_sub)
            Ac_ret_day = calc_Ac(b_ret_day, k_ret_day, cot_sub)
            Ac_msk_1030 = calc_Ac(b_msk_1030, k_msk_1030, cot_sub)
            Ac_msk_day = calc_Ac(b_msk_day, k_msk_day, cot_sub)
            
            # Albedo sensitivity: as = k * Ac * (1-Ac)
            as_ret_1030 = calc_ac_sensitivity(k_ret_1030, Ac_ret_1030)
            as_ret_day = calc_ac_sensitivity(k_ret_day, Ac_ret_day)
            as_ret_orig = calc_ac_sensitivity(1.0, Ac_orig_sub)
            as_msk_1030 = calc_ac_sensitivity(k_msk_1030, Ac_msk_1030)
            as_msk_day = calc_ac_sensitivity(k_msk_day, Ac_msk_day)
            as_msk_orig = calc_ac_sensitivity(1.0, Ac_orig_sub)
            
            # IRF = IRF_base * as * CF
            irf_ret_1030 = calc_irf(as_ret_1030, cf_ret_vals, irf_base_sub)
            irf_ret_day = calc_irf(as_ret_day, cf_ret_vals, irf_base_sub)
            irf_ret_orig = calc_irf(as_ret_orig, cf_ret_vals, irf_base_sub)
            irf_msk_1030 = calc_irf(as_msk_1030, cf_msk_vals, irf_base_sub)
            irf_msk_day = calc_irf(as_msk_day, cf_msk_vals, irf_base_sub)
            irf_msk_orig = calc_irf(as_msk_orig, cf_msk_vals, irf_base_sub)
            
            # CF sensitivity (dcf)
            dcf_ret = cf_ret_lookup.get((ocean, season), np.nan)
            dcf_msk = cf_msk_lookup.get((ocean, season), np.nan)
            
            # Clear-sky albedo (Aclr): Ac with k=0, b=0.13
            Aclr = calc_Ac(0.13, 0, cot_sub)
            
            # ERF = IRF + IRF_base * (Ac - Aclr) * dcf
            # Note: irf_base here is swdown * lnnd_o_lnaod * log_aod_diff (without /3)
            # The IRF already includes /3, but the (Ac-Aclr)*dcf term uses irf_base without /3
            erf_ret_1030 = calc_erf(irf_ret_1030, irf_base_sub, Ac_ret_1030, Aclr, dcf_ret)
            erf_ret_day = calc_erf(irf_ret_day, irf_base_sub, Ac_ret_day, Aclr, dcf_ret)
            erf_ret_orig = calc_erf(irf_ret_orig, irf_base_sub, Ac_orig_sub, Aclr, dcf_ret)
            erf_msk_1030 = calc_erf(irf_msk_1030, irf_base_sub, Ac_msk_1030, Aclr, dcf_msk)
            erf_msk_day = calc_erf(irf_msk_day, irf_base_sub, Ac_msk_day, Aclr, dcf_msk)
            erf_msk_orig = calc_erf(irf_msk_orig, irf_base_sub, Ac_orig_sub, Aclr, dcf_msk)
            
            # Weighted std function
            def weighted_std(vals, w):
                valid = np.isfinite(vals) & np.isfinite(w) & (w > 0)
                v, wgt = vals[valid], w[valid]
                if len(v) < 2:
                    return 0.0
                mean = np.average(v, weights=wgt)
                variance = np.average((v - mean)**2, weights=wgt)
                return np.sqrt(variance)
            
            # Store area-weighted means and stds
            for key, vals in [
                ('as_ret_1030', as_ret_1030), ('as_ret_day', as_ret_day), ('as_ret_orig', as_ret_orig),
                ('as_msk_1030', as_msk_1030), ('as_msk_day', as_msk_day), ('as_msk_orig', as_msk_orig),
                ('irf_ret_1030', irf_ret_1030), ('irf_ret_day', irf_ret_day), ('irf_ret_orig', irf_ret_orig),
                ('irf_msk_1030', irf_msk_1030), ('irf_msk_day', irf_msk_day), ('irf_msk_orig', irf_msk_orig),
                ('erf_ret_1030', erf_ret_1030), ('erf_ret_day', erf_ret_day), ('erf_ret_orig', erf_ret_orig),
                ('erf_msk_1030', erf_msk_1030), ('erf_msk_day', erf_msk_day), ('erf_msk_orig', erf_msk_orig),
            ]:
                mean_val = np.nansum(vals * area) / total_area
                std_val = weighted_std(vals, area)
                results[key][(ocean, season)] = (mean_val, std_val)
    
    # Aggregate to ocean level (average across seasons)
    ocean_results = {}
    for key in results.keys():
        ocean_results[key] = {}
        for ocean in OCEANS:
            vals = [results[key].get((ocean, s), (np.nan, np.nan)) for s in SEASONS]
            means = [v[0] for v in vals if not np.isnan(v[0])]
            stds = [v[1] for v in vals if not np.isnan(v[1])]
            if means:
                ocean_results[key][ocean] = (np.mean(means), np.sqrt(np.nanmean([s**2 for s in stds])) if stds else 0.0)
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
    
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 7.5),
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
    # =========================
    # Calculate bar data
    # =========================
    print("Calculating albedo sensitivity and IRF...")
    bar_results = get_bar_data()
    
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
        
        GLOBAL_DIST_PATH = os.path.join(BASE_PATH, 'figs', 'supp_global_distributions.png')
        plot_global_distributions(merged_df, GLOBAL_DIST_PATH, icon_style='nature')
    
    # =========================
    # Create figure layout
    # =========================
    # Three subplots stacked vertically:
    #   (a) Albedo sensitivity bar chart (5 bars per ocean)
    #   (b) IRF bar chart (6 bars per ocean)
    #   (c) ERF bar chart (6 bars per ocean)
    
    fig = plt.figure(figsize=(11, 11.5), dpi=100)
    
    left_margin = 0.08
    right_margin = 0.02
    top_margin = 0.04
    bottom_margin = 0.06
    bar_height = 0.27
    gap = 0.06
    
    ocean_names = OCEANS
    x = np.arange(len(ocean_names))
    width = 0.12
    
    # Compute global area-weighted mean for each variable
    ocean_areas = {
        'NPO': 103.3, 'NAO': 41.5, 'TPO': 165.2, 'TAO': 60.0,
        'TIO': 73.4, 'SPO': 77.0, 'SAO': 40.3, 'SIO': 60.6,
    }
    
    def global_mean(results_dict, key):
        vals = [results_dict[key].get(o, (np.nan, np.nan))[0] for o in OCEANS]
        weights = [ocean_areas[o] for o in OCEANS]
        valid = [(v, w) for v, w in zip(vals, weights) if not np.isnan(v)]
        if not valid:
            return np.nan
        return np.average([v for v, _ in valid], weights=[w for _, w in valid])
    
    # =========================
    # Subplot (a): Albedo sensitivity bar chart
    # =========================
    ax_a = fig.add_axes([left_margin, bottom_margin + 2 * (bar_height + gap),
                         1 - left_margin - right_margin, bar_height])
    
    # as_ret_orig and as_msk_orig are identical, merge into one gray bar at the rightmost position
    as_keys = ['as_ret_day', 'as_msk_day',
               'as_ret_1030', 'as_msk_1030', 'as_orig']
    # Legend: subscript content outside parentheses, shared between subplots
    as_legend_labels = ['Ret, Day', 'Msk, Day',
                        'Ret, 10:30', 'Msk, 10:30', 'Uncorr.']
    as_colors = ['steelblue', 'firebrick',
                 'lightblue', 'lightcoral', 'whitesmoke']
    
    for i, (key, label, color) in enumerate(zip(as_keys, as_legend_labels, as_colors)):
        if key == 'as_orig':
            # Use as_ret_orig (same as as_msk_orig)
            means = [bar_results['as_ret_orig'].get(o, (np.nan, np.nan))[0] for o in ocean_names]
            stds = [bar_results['as_ret_orig'].get(o, (np.nan, np.nan))[1] for o in ocean_names]
        else:
            means = [bar_results[key].get(o, (np.nan, np.nan))[0] for o in ocean_names]
            stds = [bar_results[key].get(o, (np.nan, np.nan))[1] for o in ocean_names]
        gmean = global_mean(bar_results, key if key != 'as_orig' else 'as_ret_orig')
        ax_a.bar(x + i * width - 2.0 * width, means, width, yerr=stds,
                 label=f'{label} ({gmean:.3f})', color=color, edgecolor='k', linewidth=0.5,
                 capsize=2, error_kw={'linewidth': 0.8})
    
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(ocean_names, fontsize=SIZE_PARAMS['small_tick'])
    ax_a.set_ylabel(r'$\mathrm{d}\mathit{A}_{\mathrm{c}} / \mathrm{d}\ln\mathrm{COT}$', fontsize=SIZE_PARAMS['xylabel'] - 1, color='k')
    ax_a.set_ylim(0, 0.32)
    ax_a.tick_params(axis='y', labelsize=SIZE_PARAMS['small_tick'])
    ax_a.legend(fontsize=SIZE_PARAMS['legend'] - 2, loc='upper right', ncol=5, framealpha=0.8)
    
    ax_a.set_title(
        f'{format_panel_tag(0, "nature")} ' + r'$\mathrm{d}\mathit{A}_{\mathrm{c}} / \mathrm{d}\ln\mathrm{COT}$',
        fontsize=SIZE_PARAMS['title'], pad=5, loc='left'
    )
    
    # =========================
    # Subplot (b): IRF bar chart
    # =========================
    ax_b = fig.add_axes([left_margin, bottom_margin + bar_height + gap,
                         1 - left_margin - right_margin, bar_height])
    
    # irf_ret_orig and irf_msk_orig are different, keep as separate bars (6 total)
    # Order: day (left), 1030 (middle), uncorr (right)
    irf_keys = ['irf_ret_day', 'irf_msk_day',
                'irf_ret_1030', 'irf_msk_1030',
                'irf_ret_orig', 'irf_msk_orig']
    irf_legend_labels = ['Ret, Day', 'Msk, Day',
                         'Ret, 10:30', 'Msk, 10:30',
                         'Ret, Uncorr.', 'Msk, Uncorr.']
    irf_colors = ['steelblue', 'firebrick',
                  'lightblue', 'lightcoral',
                  'lightcyan', 'mistyrose']
    
    for i, (key, label, color) in enumerate(zip(irf_keys, irf_legend_labels, irf_colors)):
        means = [bar_results[key].get(o, (np.nan, np.nan))[0] for o in ocean_names]
        stds = [bar_results[key].get(o, (np.nan, np.nan))[1] for o in ocean_names]
        gmean = global_mean(bar_results, key)
        ax_b.bar(x + i * width - 2.5 * width, means, width, yerr=stds,
                 label=f'{label} ({gmean:.2f})', color=color, edgecolor='k', linewidth=0.5,
                 capsize=2, error_kw={'linewidth': 0.8})
    
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(ocean_names, fontsize=SIZE_PARAMS['small_tick'])
    ax_b.set_ylabel('IRF (W m$^{-2}$)', fontsize=SIZE_PARAMS['xylabel'] - 1, color='k')
    # ax_b.set_ylim(-0.25, 3.5)
    ax_b.tick_params(axis='y', labelsize=SIZE_PARAMS['small_tick'])
    ax_b.legend(fontsize=SIZE_PARAMS['legend'] - 2, loc='upper right', ncol=3, framealpha=0.8)
    
    ax_b.set_title(
        f'{format_panel_tag(1, "nature")} IRF',
        fontsize=SIZE_PARAMS['title'], pad=5, loc='left'
    )
    
    # =========================
    # Subplot (c): ERF bar chart
    # =========================
    ax_c = fig.add_axes([left_margin, bottom_margin,
                         1 - left_margin - right_margin, bar_height])
    
    # erf_ret_orig and erf_msk_orig are different, keep as separate bars (6 total)
    # Order: day (left), 1030 (middle), uncorr (right)
    erf_keys = ['erf_ret_day', 'erf_msk_day',
                'erf_ret_1030', 'erf_msk_1030',
                'erf_ret_orig', 'erf_msk_orig']
    erf_legend_labels = ['Ret, Day', 'Msk, Day',
                         'Ret, 10:30', 'Msk, 10:30',
                         'Ret, Uncorr.', 'Msk, Uncorr.']
    erf_colors = ['steelblue', 'firebrick',
                  'lightblue', 'lightcoral',
                  'lightcyan', 'mistyrose']
    
    for i, (key, label, color) in enumerate(zip(erf_keys, erf_legend_labels, erf_colors)):
        means = [bar_results[key].get(o, (np.nan, np.nan))[0] for o in ocean_names]
        stds = [bar_results[key].get(o, (np.nan, np.nan))[1] for o in ocean_names]
        gmean = global_mean(bar_results, key)
        ax_c.bar(x + i * width - 2.5 * width, means, width, yerr=stds,
                 label=f'{label} ({gmean:.2f})', color=color, edgecolor='k', linewidth=0.5,
                 capsize=2, error_kw={'linewidth': 0.8})
    
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(ocean_names, fontsize=SIZE_PARAMS['small_tick'])
    ax_c.set_ylabel('ERF (W m$^{-2}$)', fontsize=SIZE_PARAMS['xylabel'] - 1, color='k')
    ax_c.tick_params(axis='y', labelsize=SIZE_PARAMS['small_tick'])
    ax_c.legend(fontsize=SIZE_PARAMS['legend'] - 2, loc='upper right', ncol=3, framealpha=0.8)
    
    ax_c.set_title(
        f'{format_panel_tag(2, "nature")} ERF',
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
