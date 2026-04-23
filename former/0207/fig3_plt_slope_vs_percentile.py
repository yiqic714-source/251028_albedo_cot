import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

np.random.seed(0) 
# Global variables - Ocean regions and seasonal month mapping
oceans = ['NPO', 'NAO','TPO',  'TAO', 'TIO', 
          'SPO', 'SAO', 'SIO']
season_dict = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}

# Color configuration for plot elements
COLORS = {
    'low': 'steelblue',
    'high': 'coral'
}

# Core utility functions
def cot_to_albedo(cot, method, sza=None, season=None, ocean=None):
    """
    Convert cloud optical thickness (COT) to albedo using specified method
    :param cot: Cloud optical thickness array
    :param method: Calculation method ('sbdart' or 'l74')
    :param sza: Solar zenith angle array
    :param season: Season label array
    :param ocean: Ocean region label
    :return: Albedo array
    """
    if method == 'sbdart':
        albedo = np.full(cot.shape, np.nan)
        for season_processed in season_dict.keys():
            # Load lookup table for specific ocean and season
            cot_sza_to_albedo_csv_path =  f'/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table/cot_sza_to_albedo_lookup_table_cp/cot_sza_to_albedo_lookup_table_{ocean}_{season_processed}.csv'
            df = pd.read_csv(cot_sza_to_albedo_csv_path, index_col=0)
            
            # Prepare grid data for interpolation
            sz_grid = np.array(df.index, dtype=float)
            tval_grid = np.array(df.columns, dtype=float)
            albedo_grid = df.values
            sz_mesh, tval_mesh = np.meshgrid(sz_grid, tval_grid, indexing='ij')
            
            # Reshape data for griddata interpolation
            points = np.column_stack([sz_mesh.ravel(), tval_mesh.ravel()])
            values = albedo_grid.ravel()
            valid_mask = ~np.isnan(values)
            points_valid = points[valid_mask]
            values_valid = values[valid_mask]
            
            if len(points_valid) == 0:
                raise ValueError("No valid albedo data in lookup table")
            
            # Apply season mask and perform interpolation
            season_mask_current = (season == season_processed)
            target_points = np.column_stack([np.atleast_1d(sza[season_mask_current]), np.atleast_1d(cot[season_mask_current])])
            interpolated = griddata(points_valid, values_valid, target_points, method='linear', fill_value=np.nan)
            albedo[season_mask_current] = interpolated
        return albedo
    
    elif method == 'l74':
        # L74 empirical formula for albedo calculation
        cot = np.asarray(cot)
        g = 0.85
        albedo = (1 - g) * cot / (1 + (1 - g) * cot)
        return albedo
    
    else:
        print("Supported methods: ['sbdart', 'l74']")
        return None

def cot_to_x(cot):
    """Convert COT to logarithmic x variable"""
    return np.log(cot)

def albedo_to_y(albedo):
    """Convert albedo to logit-transformed y variable (clipped to avoid log(0))"""
    albedo = np.clip(albedo, 1e-6, 1 - 1e-6)
    return np.log(albedo / (1 - albedo))

def ransac_linear_fit(x, y):
    """
    Robust linear regression using RANSAC to exclude outliers
    :param x: Independent variable array
    :param y: Dependent variable array
    :return: Slope (k) and intercept (b) of fitted line (NaN if insufficient data)
    """
    # Filter out NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask].reshape(-1, 1)
    y_clean = y[mask]
    
    # Return NaN if insufficient valid data points
    if len(x_clean) < 5:
        return np.nan, np.nan
    
    # Initialize RANSAC regressor
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=5,
        residual_threshold=0.5,
        max_trials=100
    )
    
    try:
        # Perform RANSAC fitting
        ransac.fit(x_clean, y_clean)
        k = ransac.estimator_.coef_[0]
        b = ransac.estimator_.intercept_
        return k, b
    except:
        # Fallback to standard linear regression if RANSAC fails
        try:
            k, b, _, _, _ = stats.linregress(x_clean.ravel(), y_clean)
            return k, b
        except:
            return np.nan, np.nan

def calc_global_slope(x1, y1, season, x2):
    """
    Calculate weighted global slope across seasons
    :param x1: Input x values (log COT)
    :param y1: Input y values (logit albedo)
    :param season: Season label array
    :param x2: Reference x array for global line calculation
    :return: Global slope, global intercept, global fitted line
    """
    slope_anns = []
    global_intercepts = []
    global_weights = []
    
    for season_processed in season_dict:
        season_mask = ~np.isnan(y1) & (season == season_processed)
        if not np.any(season_mask):
            continue
        
        x1_season = x1[season_mask]
        y1_season = y1[season_mask]
        n_points = len(x1_season)
        
        if n_points < 5:
            continue
        
        # Get slope and intercept using RANSAC robust fitting
        k, b = ransac_linear_fit(x1_season, y1_season)
        
        if np.isnan(k) or np.isnan(b):
            continue
        
        slope_anns.append(k)
        global_intercepts.append(b)
        global_weights.append(n_points)
    
    # Calculate weighted average of slopes and intercepts
    global_weights_arr = np.array(global_weights)
    if len(global_weights_arr) > 0 and np.sum(global_weights_arr) > 0:
        global_norm_weights = global_weights_arr / np.sum(global_weights_arr)
        global_slope = np.sum(np.array(slope_anns) * global_norm_weights)
        global_intercept = np.sum(np.array(global_intercepts) * global_norm_weights)
        global_line = global_slope * x2 + global_intercept
    else:
        global_slope = np.nan
        global_intercept = np.nan
        global_line = np.full_like(x2, np.nan)
    
    return global_slope, global_intercept, global_line

def split_data_by_percentile(df, col_name, n_bins):
    """
    Split data into percentile-based bins
    :param df: Input DataFrame
    :param col_name: Column name for binning
    :param n_bins: Number of target bins
    :return: Bin labels array, bin edges array
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(df[col_name].dropna(), percentiles)
    bin_edges = np.unique(bin_edges)
    
    # Adjust bin count if duplicate edges exist
    if len(bin_edges) < n_bins + 1:
        n_bins = len(bin_edges) - 1
        print(f"Adjusted to {n_bins} groups due to duplicate data in {col_name}")
    
    df['group_label'] = pd.cut(df[col_name], bins=bin_edges, labels=False, include_lowest=True)
    return df['group_label'].values, bin_edges

def process_ocean_group_by_percentile(ocean, n_bins=2):
    """
    Process ocean-specific data and calculate slope differences by percentile groups
    :param ocean: Ocean region label
    :param n_bins: Number of percentile bins
    :return: Dictionary containing slope results by cot_disp/unr_fra (移除SZA相关计算)
    """
    # Load ocean-specific data
    file_path = f"/home/chenyiqi/251028_albedo_cot/processed_data/merged_msk_and_ret_csv/{ocean}.csv"
    columns = [
        'ret_albedo', 'ret_cot_mod', 'ret_cotstd_mod', 
        'ret_cot_cer', 'ret_cotstd_cer',
        'time', 'lat', 'sw_all', 'sw_clr', 'solar_incoming', 'cf_ceres', 'cot_mod08', 'cotstd_mod08', 'sza', 
        'cf_ret_liq_mod08', 'ret_cot_cer', 'ret_albedo'
    ]
    df = pd.read_csv(file_path, usecols=columns)
    
    # Calculate albedo from radiation data
    df['albedo'] = ((df['sw_all'] - df['sw_clr'] * (1 - df['cf_ceres'])) / df['cf_ceres'] / df['solar_incoming'])
    df['month'] = pd.to_datetime(df['time'], format='mixed').dt.month
    
    # Calculate derived variables
    df['cot_disp'] = df['ret_cotstd_cer'] / df['ret_cot_cer']
    df['unr_fra'] = ((df['cf_ceres'] - df['cf_ret_liq_mod08']) / df['cf_ceres']).values
    
    # Apply quality control mask
    mask1 = (df['cf_ceres'] > 0.3) & (df['cot_mod08'] > 3) & (df['ret_cot_cer'] > 3) & \
            (df['ret_albedo'] > 0) & (df['ret_albedo'] < 1) & (df['albedo'] > 0) & (df['albedo'] < 1)
    df = df[mask1].dropna()
    
    # Assign season labels based on month
    for season_processed, season_months in season_dict.items():
        season_mask = df['month'].isin(season_months)
        df.loc[season_mask, 'season'] = season_processed
    df['season'] = df['season'].astype('object')
    
    # Reference COT range for slope calculation
    cot_range = np.exp(np.linspace(np.log(3), 4.50, 15))
    x2 = cot_to_x(cot_range)

    # 移除所有SZA相关的计算逻辑
    # Process cot_disp percentile groups
    group0_results = []
    ret_df = df.copy()
    g0_group_labels, _ = split_data_by_percentile(ret_df, 'cot_disp', n_bins)
    
    for bin_idx in range(n_bins):
        bin_mask = (g0_group_labels == bin_idx)
        if not np.any(bin_mask):
            continue
        bin_df = ret_df[bin_mask].copy()
        if len(bin_df) < 5:
            continue
        
        # Calculate albedo and slopes for ret and SBDART methods
        albedo_sbd = cot_to_albedo(bin_df['ret_cot_cer'].values, 'sbdart', 
                                  sza=bin_df['sza'].values, 
                                  season=bin_df['season'].values,
                                  ocean=ocean)
        
        x1 = cot_to_x(bin_df['ret_cot_cer'].values)
        y1_ret = albedo_to_y(bin_df['ret_albedo'].values)
        y1_sbd = albedo_to_y(albedo_sbd)
        
        slope_ret, _, _ = calc_global_slope(x1, y1_ret, bin_df['season'].values, x2)
        slope_sbd, _, _ = calc_global_slope(x1, y1_sbd, bin_df['season'].values, x2)
        
        # Store results for current bin
        bin_result = {
            'Ocean': ocean,
            'Group_Type': 'cot_disp',
            'Bin_Index': bin_idx,
            'Global_Slope_ret': slope_ret,
            'Global_Slope_SBD': slope_sbd,
            'Slope_Diff': slope_sbd - slope_ret
        }
        group0_results.append(bin_result)

    # Process unr_fra percentile groups
    group1_results = []
    msk_df = df.copy()
    g1_group_labels, unr_fra_bin_edges = split_data_by_percentile(msk_df, 'unr_fra', n_bins)
    
    ret_df['unr_fra_bin'] = pd.cut(ret_df['unr_fra'], bins=unr_fra_bin_edges, labels=False, include_lowest=True)
    n_unr_bins = len(unr_fra_bin_edges) - 1

    for bin_idx in range(n_unr_bins):
        msk_bin_mask = (g1_group_labels == bin_idx)
        if not np.any(msk_bin_mask):
            continue
        msk_bin_df = msk_df[msk_bin_mask].copy()
        if len(msk_bin_df) < 5:
            continue
        
        # Match ret data to msk bins
        ret_bin_mask = (ret_df['unr_fra_bin'] == bin_idx)
        ret_bin_df = ret_df[ret_bin_mask].copy() if np.any(ret_bin_mask) else None
        
        # Calculate msk slope
        x1_msk = cot_to_x(msk_bin_df['cot_mod08'].values)
        y1_msk = albedo_to_y(msk_bin_df['albedo'].values)
        slope_msk, _, _ = calc_global_slope(x1_msk, y1_msk, msk_bin_df['season'].values, x2)
        
        # Calculate ret slope (if valid data exists)
        slope_ret_unr = np.nan
        if ret_bin_df is not None and len(ret_bin_df) >= 5:
            x1_ret_unr = cot_to_x(ret_bin_df['ret_cot_cer'].values)
            y1_ret_unr = albedo_to_y(ret_bin_df['ret_albedo'].values)
            slope_ret_unr, _, _ = calc_global_slope(x1_ret_unr, y1_ret_unr, ret_bin_df['season'].values, x2)
        
        # Store results for current bin
        bin_result = {
            'Ocean': ocean,
            'Group_Type': 'unr_fra',
            'Bin_Index': bin_idx,
            'Global_Slope_ret': slope_ret_unr,
            'Global_Slope_msk': slope_msk,
            'Slope_Diff': slope_ret_unr - slope_msk
        }
        group1_results.append(bin_result)

    # 移除返回值中的sza_data
    return {
        'cot_disp_data': group0_results,
        'unr_fra_data': group1_results
    }

def plot_two_panels(all_results, output_fig_path):
    """
    重构绘图函数为双面板（移除SZA面板），绘制cot_disp和unr_fra的斜率差异柱状图
    :param all_results: List of processed results for each ocean region
    :param output_fig_path: Output file path for the figure
    """
    # Convert results to DataFrames for easy filtering (移除SZA相关数据组织)
    cot_disp_list = []
    unr_fra_list = []
    for res in all_results:
        cot_disp_list.extend(res['cot_disp_data'])
        unr_fra_list.extend(res['unr_fra_data'])
        
    df_cot_disp = pd.DataFrame(cot_disp_list)
    df_unr_fra = pd.DataFrame(unr_fra_list)

    # Create figure and axes - 改为1行2列的双面板
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1, ax2 = axes
    
    # Bar plot configuration
    bar_width = 0.35
    x = np.arange(len(oceans))

    # Panel (a): d_COT (原b面板，改为第一个面板)
    cot_disp_data = {}
    for ocean in oceans:
        ocean_df = df_cot_disp[df_cot_disp['Ocean'] == ocean].sort_values('Bin_Index')
        cot_disp_data[ocean] = {
            'low': ocean_df[ocean_df['Bin_Index'] == 0]['Slope_Diff'].values[0] if len(ocean_df[ocean_df['Bin_Index'] == 0]) > 0 else np.nan,
            'high': ocean_df[ocean_df['Bin_Index'] == 1]['Slope_Diff'].values[0] if len(ocean_df[ocean_df['Bin_Index'] == 1]) > 0 else np.nan
        }

    ax1.bar(x - bar_width/2, [cot_disp_data[o]['low'] for o in oceans], 
            bar_width, label=r'low $d_{\mathrm{COT}}$', color=COLORS['low'], 
            alpha=0.8, edgecolor=None)
    ax1.bar(x + bar_width/2, [cot_disp_data[o]['high'] for o in oceans], 
            bar_width, label=r'high $d_{\mathrm{COT}}$', color=COLORS['high'], 
            alpha=0.8, edgecolor=None)

    ax1.set_title(r'$\mathbf{(a)}$', fontsize=17.5, loc='left')
    ax1.set_ylabel(r'$k_{\mathrm{cp}}-k_{\mathrm{ret}}$', fontsize=17.5)
    ax1.set_ylim(0, 0.21)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(oceans, fontsize=12.5, ha='center')
    ax1.legend(fontsize=13, loc='best')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Panel (b): CUR (原c面板，改为第二个面板)
    unr_fra_data = {}
    for ocean in oceans:
        ocean_df = df_unr_fra[df_unr_fra['Ocean'] == ocean].sort_values('Bin_Index')
        unr_fra_data[ocean] = {
            'low': ocean_df[ocean_df['Bin_Index'] == 0]['Slope_Diff'].values[0] if len(ocean_df[ocean_df['Bin_Index'] == 0]) > 0 else np.nan,
            'high': ocean_df[ocean_df['Bin_Index'] == 1]['Slope_Diff'].values[0] if len(ocean_df[ocean_df['Bin_Index'] == 1]) > 0 else np.nan
        }

    ax2.bar(x - bar_width/2, [unr_fra_data[o]['low'] for o in oceans], 
            bar_width, label='low RUR', color=COLORS['low'], 
            alpha=0.8, edgecolor=None)
    ax2.bar(x + bar_width/2, [unr_fra_data[o]['high'] for o in oceans], 
            bar_width, label='high RUR', color=COLORS['high'], 
            alpha=0.8, edgecolor=None)

    ax2.set_title(r'$\mathbf{(b)}$', fontsize=17.5, loc='left')
    ax2.set_ylabel('$k_{\mathrm{ret}}-k_{\mathrm{msk}}$', fontsize=17.5)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(oceans, fontsize=12.5, ha='center')
    # ax2.legend(fontsize=13, loc='best')
    ax2.legend(fontsize=13, bbox_to_anchor=(0.18, 0.73))
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # Adjust plot spacing with direct numeric values
    plt.subplots_adjust(
        left=0.08,    # Left margin (fraction of figure width)
        right=0.98,   # Right margin (fraction of figure width)
        wspace=0.35,  # Horizontal space between subplots
        bottom=0.2,   # Bottom margin (增加底部边距，适配旋转的x轴标签)
        top=0.95      # Top margin (fraction of figure height)
    )

    # Save and close figure
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"--- Figure saved to: {output_fig_path} ---")

# Main execution
if __name__ == "__main__":
    n_bins = 2
    # 更新输出文件名，移除bins标识（或保留，根据需求）
    fig_output_path = f'/home/chenyiqi/251028_albedo_cot/figs/slope_difference_cotdisp_unrfra.png'

    # Calculate slopes for all ocean regions
    all_results = []
    for ocean in oceans:
        print(f"Processing ocean region: {ocean}")
        result_dict = process_ocean_group_by_percentile(ocean, n_bins=n_bins)
        all_results.append(result_dict)

    # Generate and save 双面板plot (替换原三面板函数)
    plot_two_panels(all_results, fig_output_path)