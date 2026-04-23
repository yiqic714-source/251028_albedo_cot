# cot_albedo_functions.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import os

# Set random seed for reproducibility (ensure identical plots every run)
np.random.seed(0)  # Fixed seed: same random sampling results for density overlay

# Configuration parameters
oceans = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']
season_dict = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}
input_dir = "/home/chenyiqi/251028_albedo_cot/processed_data/merged_msk_and_ret_csv/"
columns = [
    'ret_albedo', 'ret_cot_mod', 'ret_cotstd_mod', 
    'ret_cot_cer', 'ret_cotstd_cer',  
    'time', 'lat', 'sw_all', 'sw_clr', 'solar_incoming', 'cf_liq_ceres', 'cot_mod08', 'cotstd_mod08', 'sza'
]
cot_range = np.exp(np.linspace(np.log(2), 4.50, 15))

# Core functions
def cot_to_albedo(cot, method, sza=None, season=None, ocean_name=None):
    """
    Convert cloud optical thickness (COT) to albedo using specified method.
    """
    if method == 'sbdart_cp':
        albedo = np.full(cot.shape, np.nan)
        
        for season_processed in season_dict.keys():
            cot_sza_to_albedo_csv_path =  (
                f'/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table/'
                f'cot_sza_to_albedo_lookup_table_cp/'
                f'cot_sza_to_albedo_lookup_table_{ocean_name}_{season_processed}.csv'
            )
            try:
                df = pd.read_csv(cot_sza_to_albedo_csv_path, index_col=0)
            except FileNotFoundError:
                print(f"Warning: Lookup table not found for {ocean_name} {season_processed} (cp). Skipping.")
                continue

            sz_grid = np.array(df.index, dtype=float)
            tval_grid = np.array(df.columns, dtype=float)
            albedo_grid = df.values
            sz_mesh, tval_mesh = np.meshgrid(sz_grid, tval_grid, indexing='ij')
            points = np.column_stack([sz_mesh.ravel(), tval_mesh.ravel()])
            values = albedo_grid.ravel()
            valid_mask = ~np.isnan(values)
            points_valid = points[valid_mask]
            values_valid = values[valid_mask]
            if len(points_valid) == 0:
                continue

            target_points = np.column_stack([np.atleast_1d(sza[season==season_processed]), np.atleast_1d(cot[season==season_processed])])
            interpolated = griddata(points_valid, values_valid, target_points, method='linear', fill_value=np.nan)
            albedo[season==season_processed] = interpolated
        return albedo

    elif method == 'sbdart_dcp':
        albedo = np.full(cot.shape, np.nan)
        FIXED_SZA = 54.7
        
        for season_processed in season_dict.keys():
        
            cot_sza_to_albedo_csv_path = f'/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table/cot_sza_to_albedo_lookup_table_dcp/cot_sza_to_albedo_lookup_table_TPO_MAM.csv'
            try:
                df = pd.read_csv(cot_sza_to_albedo_csv_path, index_col=0)
            except FileNotFoundError:
                print(f"Warning: Lookup table not found for {ocean_name} {season_processed} (dcp). Skipping.")
                continue
        
            sz_grid = np.array(df.index, dtype=float)
            tval_grid = np.array(df.columns, dtype=float)
            albedo_grid = df.values
        
            sz_mesh, tval_mesh = np.meshgrid(sz_grid, tval_grid, indexing='ij')
            points = np.column_stack([sz_mesh.ravel(), tval_mesh.ravel()])
            values = albedo_grid.ravel()
        
            valid_mask = ~np.isnan(values)
            points_valid = points[valid_mask]
            values_valid = values[valid_mask]
        
            if len(points_valid) == 0:
                continue
        
            # 关键修改：不管原始 sza 是什么，这里全部使用 FIXED_SZA
            # target_points 的第一列全部填成 54.7
            target_points = np.column_stack([
                np.full(cot[season == season_processed].shape, FIXED_SZA),
                np.atleast_1d(cot[season == season_processed])
            ])
        
            interpolated = griddata(
                points_valid, 
                values_valid, 
                target_points, 
                method='linear', 
                fill_value=np.nan
            )
        
            albedo[season == season_processed] = interpolated
        
        return albedo
    
    elif method == 'l74':
        cot = np.asarray(cot)
        b=0.13
        albedo = b * cot / (1 + b * cot)
        return albedo
    
    else:
        print("supported methods: ['sbdart_cp', 'sbdart_dcp', 'l74']")
        return np.nan

def cot_to_x(cot):
    """Convert COT to logarithmic x-axis value for fitting"""
    return np.log(cot)

def albedo_to_y(albedo):
    """Convert albedo to log-odds y-axis value for fitting (clipped to avoid infinity)"""
    albedo = np.clip(albedo, 1e-6, 1 - 1e-6)
    return np.log(albedo / (1 - albedo))

def robust_fit_ransac(x, y):
    """
    Perform robust linear fitting using RANSAC to exclude outliers.
    Fall back to standard linear regression if RANSAC fails or data is insufficient.
    Return both slope (k) and intercept (b) for the linear model y = kx + b
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 5:
        try:
            k, b, _, _, _ = stats.linregress(x_clean, y_clean)
            return k, b
        except:
            return np.nan, np.nan
    
    x_2d = x_clean.reshape(-1, 1)
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=5,
        residual_threshold=0.5,
        max_trials=100
    )
    
    try:
        ransac.fit(x_2d, y_clean)
        k = ransac.estimator_.coef_[0]
        b = ransac.estimator_.intercept_
        return k, b
    except:
        try:
            k, b, _, _, _ = stats.linregress(x_clean.ravel(), y_clean)
            return k, b
        except:
            return np.nan, np.nan

def plot_weighted_fit_line(x1, y1, sza, season, x2, color, label, line_handles, line_labels, ax, linestyle=None, n_sza_groups=2):
    """
    Plot weighted linear fit line, with weights based on number of points in each SZA/season group.
    Return global slope, global intercept, seasonal slopes, seasonal intercepts, and updated plot handles/labels.
    """
    slope_anns = []
    intercept_anns = []
    global_weights = []
    season_fit_results = {s: [[], [], [], []] for s in season_dict.keys()}  # [slopes, intercepts, weights, ...]
    
    for season_processed in season_dict:
        season_mask = ~np.isnan(y1) & (season == season_processed)
        if not np.any(season_mask):
            continue
        
        sza_season_valid = sza[season_mask]
        if len(sza_season_valid) < n_sza_groups:
            continue
        
        sza_percentiles = np.percentile(sza_season_valid, np.linspace(0, 100, n_sza_groups + 1))
        sza_groups = []
        for i in range(n_sza_groups):
            sza_low = sza_percentiles[i]
            sza_high = sza_percentiles[i+1]
            if i == n_sza_groups - 1:
                sza_group_mask = (sza >= sza_low) & (sza <= sza_high)
            else:
                sza_group_mask = (sza >= sza_low) & (sza < sza_high)
            sza_groups.append(sza_group_mask)
        
        for sza_group_mask in sza_groups:
            final_mask = season_mask & sza_group_mask
            n_points = np.sum(final_mask)
            if n_points < 5:
                continue
            
            x1_bin, y1_bin = x1[final_mask], y1[final_mask]
            k, b = robust_fit_ransac(x1_bin, y1_bin)
            
            if np.isnan(k):
                continue
            
            slope_anns.append(k)
            intercept_anns.append(b)
            global_weights.append(n_points)
            season_fit_results[season_processed][0].append(k)
            season_fit_results[season_processed][1].append(b)
            season_fit_results[season_processed][2].append(n_points)
    
    global_weights_arr = np.array(global_weights)
    if len(global_weights_arr) > 0 and np.sum(global_weights_arr) > 0:
        global_norm_weights = global_weights_arr / np.sum(global_weights_arr)
        slope_ann = np.sum(np.array(slope_anns) * global_norm_weights)
        intercept_ann = np.sum(np.array(intercept_anns) * global_norm_weights)
        global_line = slope_ann * x2 + intercept_ann
    else:
        slope_ann = np.nan
        intercept_ann = np.nan
        global_line = np.full_like(x2, np.nan)
    
    sign = '+' if intercept_ann >= 0 else ''
    eq = f'y={slope_ann:.2f}x{sign}{intercept_ann:.1f}'
    line_obj = ax.plot(x2, global_line, color=color, linestyle=linestyle, 
                    lw=1.5, label=f'{label}: {eq}')
    
    line_handles.append(line_obj[0])
    line_labels.append(f'{label}: {eq}')
    
    slopes_season = {}
    intercepts_season = {}
    for s in season_dict.keys():
        s_slopes = np.array(season_fit_results[s][0])
        s_intercepts = np.array(season_fit_results[s][1])
        s_weights = np.array(season_fit_results[s][2])
        if len(s_weights) > 0 and np.sum(s_weights) > 0:
            s_norm_weights = s_weights / np.sum(s_weights)
            s_slope = np.sum(s_slopes * s_norm_weights)
            s_intercept = np.sum(s_intercepts * s_norm_weights)
        else:
            s_slope = np.nan
            s_intercept = np.nan
        slopes_season[s] = s_slope
        intercepts_season[s] = s_intercept
    
    return slope_ann, intercept_ann, slopes_season, intercepts_season, line_handles, line_labels

def plot_density_overlay(x_ret, y_ret, x_msk, y_msk, ax, sample_size=5000):
    """
    Plot density overlay for ret (blue filled contour) and msk (magenta contour) data.
    Use fixed random seed for sampling to ensure reproducible plots.
    """
    # ret: Blue filled contour (no edge line)
    mask_ret = ~(np.isnan(x_ret) | np.isnan(y_ret))
    x_u = x_ret[mask_ret]
    y_u = y_ret[mask_ret]
    
    if len(x_u) > sample_size:
        # Use fixed seed for random sampling to ensure identical results every run
        idx = np.random.choice(len(x_u), sample_size, replace=False)
        x_u = x_u[idx]
        y_u = y_u[idx]

    if len(x_u) >= 20:
        xi, yi = np.mgrid[0.5:4.25:100j, -2.0:1.5:100j]
        positions = np.vstack([xi.ravel(), yi.ravel()])
        values = np.vstack([x_u, y_u])
        kernel = gaussian_kde(values)
        zi = np.reshape(kernel(positions).T, xi.shape)
        
        ax.contourf(
            xi, yi, zi, 
            levels=5, 
            cmap='Blues', 
            alpha=0.6, 
            antialiased=False
        )

    # msk: Magenta contour line
    mask_msk = ~(np.isnan(x_msk) | np.isnan(y_msk))
    x_f = x_msk[mask_msk]
    y_f = y_msk[mask_msk]

    if len(x_f) > sample_size:
        # Use fixed seed for random sampling to ensure identical results every run
        idx = np.random.choice(len(x_f), sample_size, replace=False)
        x_f = x_f[idx]
        y_f = y_f[idx]

    if len(x_f) >= 20:
        xi, yi = np.mgrid[0.5:4.25:100j, -2.0:1.5:100j]
        positions = np.vstack([xi.ravel(), yi.ravel()])
        values = np.vstack([x_f, y_f])
        kernel = gaussian_kde(values)
        zi = np.reshape(kernel(positions).T, xi.shape)

        ax.contour(xi, yi, zi, levels=5, colors='magenta', alpha=0.6, linewidths=0.8)

def plot_axes_content(data, ax, title=None):
    """
    Populate single subplot with data, density overlay, and fit lines.
    Return slope AND intercept results for further analysis.
    """
    
    # Set axis limits
    ax.set_xlim([0.5, 3.5])
    ax.set_ylim([-2, 1.5])
    
    # Plot density overlay
    plot_density_overlay(data['x1_ret'], data['y1_list_ret'][0], data['x1_msk'], data['y1_msk'], ax)

    line_handles = []
    line_labels = []
    all_results = {}
    
    # 1. L74: Black solid line (store intercept as well)
    k, b, r_value, p_value, std_err = stats.linregress(data['x2'], data['y22'])
    sign = '+' if b >= 0 else ''
    eq = f'y={k:.2f}x{sign}{b:.1f}'
    line_obj = ax.plot(data['x2'], data['y22'], color='black', lw=1.5, label=f'LH74: {eq}')
    line_handles.append(line_obj[0])
    line_labels.append(f'LH74: {eq}')
    # Store L74 slope and intercept
    all_results['LH74'] = (k, b, {}, {})
    
    # 3. SBD_dcp
    slope_SBD_noO3, intercept_SBD_noO3, slopes_season_SBD_noO3, intercepts_season_SBD_noO3, line_handles, line_labels = plot_weighted_fit_line(
        data['x1_ret'], data['y1_list_ret'][2], data['sza'], data['season'], data['x2'],
        'red', 'dcp', line_handles, line_labels, ax, linestyle='--'
    )
    all_results['dcp'] = (slope_SBD_noO3, intercept_SBD_noO3, slopes_season_SBD_noO3, intercepts_season_SBD_noO3)
    
    # 2. SBD_cp
    slope_SBD, intercept_SBD, slopes_season_SBD, intercepts_season_SBD, line_handles, line_labels = plot_weighted_fit_line(
        data['x1_ret'], data['y1_list_ret'][1], data['sza'], data['season'], data['x2'],
        'orange', 'cp', line_handles, line_labels, ax, linestyle='-'
    )
    all_results['cp'] = (slope_SBD, intercept_SBD, slopes_season_SBD, intercepts_season_SBD)
    
    # 4. ret Obs
    slope_ret, intercept_ret, slopes_season_ret, intercepts_season_ret, line_handles, line_labels = plot_weighted_fit_line(
        data['x1_ret'], data['y1_list_ret'][0], data['sza'], data['season'], data['x2'],
        'blue', 'ret', line_handles, line_labels, ax, linestyle='--'
    )
    all_results['ret'] = (slope_ret, intercept_ret, slopes_season_ret, intercepts_season_ret)
    
    # 5. msk Obs
    slope_msk, intercept_msk, slopes_season_msk, intercepts_season_msk, line_handles, line_labels = plot_weighted_fit_line(
        data['x1_msk'], data['y1_msk'], data['sza'], data['season'], data['x2'],
        'magenta', 'msk', line_handles, line_labels, ax, linestyle='-'
    )
    all_results['msk'] = (slope_msk, intercept_msk, slopes_season_msk, intercepts_season_msk)
    
    # Legend and title
    ax.legend(handles=line_handles, labels=line_labels, fontsize=9, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)
    if title:
        ax.set_title(title, fontsize=13, loc='left')
        
    return all_results

def preprocess_ocean_data():
    """预处理所有海洋数据并返回字典"""
    albedo_l74 = cot_to_albedo(cot_range, 'l74')
    all_processed_ocean_data = {}  # Store preprocessed data for 8 oceans
    global_data_collector = {
        'x1_ret': [], 'y1_list_ret': [[], [], []],
        'x1_msk': [], 'y1_msk': [],
        'sza': [], 'season': [],
        'x2': cot_to_x(cot_range), 'y22': albedo_to_y(albedo_l74)
    }

    print("Starting one-time preprocessing for all ocean data...")
    for ocean in oceans:
        file_path = os.path.join(input_dir, f"{ocean}.csv")
        try:
            # Read raw data
            df = pd.read_csv(file_path, usecols=columns)
            
            # Data preprocessing
            df['albedo'] = ((df['sw_all'] - df['sw_clr'] * (1 - df['cf_liq_ceres'])) / df['cf_liq_ceres'] / df['solar_incoming'])
            df['month'] = pd.to_datetime(df['time'], format='mixed').dt.month
            
            # Assign seasons
            for season_name, months in season_dict.items():
                df.loc[df['month'].isin(months), 'season'] = season_name
            
            # Data filtering
            mask1 = (df['cot_mod08'] > 2) & (df['ret_cot_cer'] > 2) & \
                    (df['ret_albedo'] > 0) & (df['ret_albedo'] < 1) & (df['albedo'] > 0) & (df['albedo'] < 1)
            df_filtered = df[mask1].dropna().reset_index(drop=True)
            
            if len(df_filtered) == 0:
                print(f"{ocean} has no valid data, skipping.")
                all_processed_ocean_data[ocean] = None
                continue
            
            albedo_sbd_ret = cot_to_albedo(df_filtered['ret_cot_cer'].values, 'sbdart_cp',
                                           sza=df_filtered['sza'].values,
                                           season=df_filtered['season'].values,
                                           ocean_name=ocean)
            albedo_sbd_noO3_ret = cot_to_albedo(df_filtered['ret_cot_cer'].values, 'sbdart_dcp',
                                               sza=df_filtered['sza'].values,
                                               season=df_filtered['season'].values,
                                               ocean_name=ocean)
            
            # Convert to x/y values for plotting
            x1_ret = cot_to_x(df_filtered['ret_cot_cer'].values)
            y1_ret_obs = albedo_to_y(df_filtered['ret_albedo'].values)
            y1_ret_SBD = albedo_to_y(albedo_sbd_ret)
            y1_ret_SBD_noO3 = albedo_to_y(albedo_sbd_noO3_ret)
            x1_msk = cot_to_x(df_filtered['cot_mod08'].values)
            y1_msk = albedo_to_y(df_filtered['albedo'].values)
            
            # Store preprocessed data
            ocean_processed_data = {
                'x1_ret': x1_ret,
                'y1_list_ret': [y1_ret_obs, y1_ret_SBD, y1_ret_SBD_noO3],
                'x1_msk': x1_msk,
                'y1_msk': y1_msk,
                'x2': global_data_collector['x2'],
                'y22': global_data_collector['y22'],
                'sza': df_filtered['sza'].values,
                'season': df_filtered['season'].values,
                'data_count': len(df_filtered)
            }
            all_processed_ocean_data[ocean] = ocean_processed_data
            
            # Collect global data
            valid_global_mask = ~(np.isnan(y1_ret_obs) & np.isnan(y1_ret_SBD) & np.isnan(y1_ret_SBD_noO3))
            global_data_collector['x1_ret'].extend(x1_ret[valid_global_mask])
            global_data_collector['y1_list_ret'][0].extend(y1_ret_obs[valid_global_mask])
            global_data_collector['y1_list_ret'][1].extend(y1_ret_SBD[valid_global_mask])
            global_data_collector['y1_list_ret'][2].extend(y1_ret_SBD_noO3[valid_global_mask])
            global_data_collector['x1_msk'].extend(x1_msk[valid_global_mask])
            global_data_collector['y1_msk'].extend(y1_msk[valid_global_mask])
            global_data_collector['sza'].extend(df_filtered['sza'].values[valid_global_mask])
            global_data_collector['season'].extend(df_filtered['season'].values[valid_global_mask])
            
            print(f"{ocean} preprocessing completed, valid data count: {len(df_filtered)}")
        
        except Exception as e:
            print(f"Error processing {ocean}: {e}")
            import traceback
            traceback.print_exc()
            all_processed_ocean_data[ocean] = None
    
    # 构建全球数据
    global_processed_data = {
        'x1_ret': np.array(global_data_collector['x1_ret']),
        'y1_list_ret': [np.array(lst) for lst in global_data_collector['y1_list_ret']],
        'x1_msk': np.array(global_data_collector['x1_msk']),
        'y1_msk': np.array(global_data_collector['y1_msk']),
        'x2': global_data_collector['x2'],
        'y22': global_data_collector['y22'],
        'sza': np.array(global_data_collector['sza']),
        'season': np.array(global_data_collector['season']),
        'data_count': len(global_data_collector['x1_ret'])
    }
    print(f"Global data construction completed, integrated valid data count: {global_processed_data['data_count']}")
    
    return all_processed_ocean_data, global_processed_data