import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy import stats
import os
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

np.random.seed(0)
BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIG_SAVE_PATH = f'{BASE_PATH}/figs/bias_explain_4panels.png'
os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)

season_dict = {'MAM': [3,4,5], 'JJA': [6,7,8], 'SON': [9,10,11], 'DJF': [12,1,2]}
oceans = ['NPO', 'NAO','TPO',  'TAO', 'TIO', 'SPO', 'SAO', 'SIO']
sza_list = [55, 15]
cot_range = np.exp(np.linspace(np.log(3), 3, 15))

METHODS = ['sbdart', 'quadrature', 'eddington']
COLORS_ALBEDO = ["#50A2E9", 'orange', 'purple']
SZA_LINESTYLE = {55: '-', 15: '--'}



def cot_to_x(cot):
    return np.log(cot)

def albedo_to_y(albedo):
    albedo = np.clip(albedo, 1e-6, 1 - 1e-6)
    return np.log(albedo / (1 - albedo))

def cot_to_albedo(cot, method, sza=None, season=None, ocean=None, lookup_type='dcp', use_tpo_mam_only=False, table_folder='dcp'):
    """
    Calculate cloud albedo from cloud optical thickness.
    
    Parameters:
    - use_tpo_mam_only: if True, only use TPO_MAM lookup table regardless of ocean and season parameters
    - lookup_type: 'dcp', 'gasdcp_surcp', or 'surdcp_gascp'
    - table_folder: folder name containing lookup tables ('dcp', 'cp', 'gasdcp_surcp', 'surdcp_gascp')
    """
    if method == 'sbdart':
        albedo = np.full(cot.shape, np.nan)
        if use_tpo_mam_only or (season is not None and ocean is not None):
            if use_tpo_mam_only:
                seasons_to_use = ['MAM']
                ocean_to_use = 'TPO'
            else:
                seasons_to_use = season_dict.keys() if season is not None else ['MAM']
                ocean_to_use = ocean
            
            for season_p in seasons_to_use:
                file_path = f'{BASE_PATH}/build_sbdart_lookup_table/cot_sza_to_albedo_lookup_table_{table_folder}/cot_sza_to_albedo_lookup_table_{ocean_to_use}_{season_p}.csv'
                
                if not os.path.exists(file_path):
                    continue
                try:
                    df = pd.read_csv(file_path, index_col=0)
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
                    continue
                    
                sz_grid = np.array(df.index, float)
                tval_grid = np.array(df.columns, float)
                albedo_grid = df.values
                sz_mesh, tval_mesh = np.meshgrid(sz_grid, tval_grid, indexing='ij')
                points = np.column_stack([sz_mesh.ravel(), tval_mesh.ravel()])
                values = albedo_grid.ravel()
                valid_mask = ~np.isnan(values)
                if not np.any(valid_mask):
                    continue
                
                if use_tpo_mam_only:
                    cot_arr = np.atleast_1d(cot)
                    sza_arr = np.full_like(cot_arr, sza, dtype=float)
                    target_points = np.column_stack([sza_arr, cot_arr])
                else:
                    season_mask = (season == season_p)
                    target_points = np.column_stack([sza[season_mask], cot[season_mask]])
                
                albedo_interp = griddata(points[valid_mask], values[valid_mask], target_points, method='linear', fill_value=np.nan)
                
                if use_tpo_mam_only:
                    albedo = albedo_interp
                else:
                    albedo[season_mask] = albedo_interp
        else:
            file_path = f'{BASE_PATH}/build_sbdart_lookup_table/cot_sza_to_albedo_lookup_table_{table_folder}/cot_sza_to_albedo_lookup_table_TPO_MAM.csv'
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                return albedo
            try:
                df = pd.read_csv(file_path, index_col=0)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return albedo
                
            sz_grid = np.array(df.index, float)
            tval_grid = np.array(df.columns, float)
            albedo_grid = df.values
            sz_mesh, tval_mesh = np.meshgrid(sz_grid, tval_grid, indexing='ij')
            points = np.column_stack([sz_mesh.ravel(), tval_mesh.ravel()])
            values = albedo_grid.ravel()
            valid_mask = ~np.isnan(values)
            sza_arr = np.full_like(cot, sza, float)
            target_points = np.column_stack([sza_arr, np.atleast_1d(cot)])
            albedo = griddata(points[valid_mask], values[valid_mask], target_points, method='linear', fill_value=np.nan)
        return albedo
    elif method == 'l74':
        g = 0.85
        b = (1 - g)
        return b * cot / (1 + b * cot)
    elif method == 'quadrature':
        cot, g, miu = np.asarray(cot), 0.85, np.cos(np.radians(sza))
        b = np.sqrt(3)/2*(1-g)
        return (b * cot + (1/2 - np.sqrt(3)/2*miu) * (1 - np.exp(-cot/miu))) / (1 + b * cot)
    elif method == 'eddington':
        cot, g, miu = np.asarray(cot), 0.85, np.cos(np.radians(sza))
        return ((1-g) * cot + (2/3 - miu) * (1 - np.exp(-cot/miu))) / (4/3 + (1-g) * cot)
    else:
        print(f"Supported methods: {METHODS + ['l74']}")
        return np.nan

def calculate_albedo_curves():
    albedo_results = {}
    for method in METHODS:
        for sza in sza_list:
            albedo_results[f'{method}_{sza}'] = cot_to_albedo(cot_range, method=method, sza=sza)
    return albedo_results

def ransac_linear_fit(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask].reshape(-1, 1)
    y_clean = y[mask]
    if len(x_clean) < 5:
        return np.nan, np.nan
    ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=5, residual_threshold=0.5, max_trials=100)
    try:
        ransac.fit(x_clean, y_clean)
        return ransac.estimator_.coef_[0], ransac.estimator_.intercept_
    except:
        try:
            return stats.linregress(x_clean.ravel(), y_clean)[:2]
        except:
            return np.nan, np.nan

def calc_global_slope(x1, y1, season, x2):
    slopes, intercepts, weights = [], [], []
    for s in season_dict:
        mask = ~np.isnan(y1) & (season == s)
        if not np.any(mask) or len(x1[mask]) <5:
            continue
        k, b = ransac_linear_fit(x1[mask], y1[mask])
        if not np.isnan(k):
            slopes.append(k)
            intercepts.append(b)
            weights.append(len(x1[mask]))
    if len(weights) == 0:
        return np.nan, np.nan, np.full_like(x2, np.nan)
    norm_w = np.array(weights)/np.sum(weights)
    global_k = np.sum(np.array(slopes)*norm_w)
    global_b = np.sum(np.array(intercepts)*norm_w)
    return global_k, global_b, global_k*x2 + global_b

def split_data_by_percentile(df, col_name, n_bins):
    percentiles = np.linspace(0, 100, n_bins +1)
    bin_edges = np.unique(np.percentile(df[col_name].dropna(), percentiles))
    n_bins = len(bin_edges)-1 if len(bin_edges) <n_bins+1 else n_bins
    df['group_label'] = pd.cut(df[col_name], bins=bin_edges, labels=False, include_lowest=True)
    return df['group_label'].values, bin_edges

def process_all_oceans(n_bins=2):
    all_results = []
    cot_ref = np.exp(np.linspace(np.log(3),4.5,15))
    x2 = cot_to_x(cot_ref)
    for ocean in oceans:
        file_path = f"{BASE_PATH}/processed_data/merged_msk_and_ret_csv/{ocean}.csv"
        cols = ['ret_albedo','ret_cot_cer','ret_cotstd_cer','time','lat','sw_all','sw_clr','solar_incoming','cf_liq_ceres','cot_mod08','sza','cf_mod08']
        df = pd.read_csv(file_path, usecols=cols)
        
        df['albedo'] = ((df['sw_all'] - df['sw_clr']*(1-df['cf_liq_ceres']))/df['cf_liq_ceres']/df['solar_incoming'])
        df['month'] = pd.to_datetime(df['time'], format='mixed').dt.month
        
        df['cot_disp'] = df['ret_cotstd_cer']/df['ret_cot_cer']
        df['unr_fra'] = (df['cf_mod08'] - df['cf_liq_ceres'])/df['cf_mod08']
        
        mask = (df['cf_mod08']>0.3)&(df['cot_mod08']>3)&(df['ret_cot_cer']>3)&(df['ret_albedo'].between(0,1))&(df['albedo'].between(0,1))
        df = df[mask].dropna()
        
        for s, ms in season_dict.items():
            df.loc[df['month'].isin(ms), 'season'] = s
        df['season'] = df['season'].astype('object')
        
        cot_disp_res = []
        g0_label, _ = split_data_by_percentile(df.copy(), 'cot_disp', n_bins)
        for idx in range(n_bins):
            bin_df = df[g0_label==idx].copy()
            if len(bin_df)<5:
                continue
            albedo_sbd = cot_to_albedo(bin_df['ret_cot_cer'].values, 'sbdart', sza=bin_df['sza'].values, season=bin_df['season'].values, ocean=ocean, table_folder='cp')
            x1 = cot_to_x(bin_df['ret_cot_cer'].values)
            y1_ret = albedo_to_y(bin_df['ret_albedo'].values)
            y1_sbd = albedo_to_y(albedo_sbd)
            k_ret, _, _ = calc_global_slope(x1, y1_ret, bin_df['season'].values, x2)
            k_sbd, _, _ = calc_global_slope(x1, y1_sbd, bin_df['season'].values, x2)
            cot_disp_res.append({'Ocean':ocean, 'Bin':idx, 'Slope_Diff':k_sbd - k_ret})
        
        unr_fra_res = []
        g1_label, _ = split_data_by_percentile(df.copy(), 'unr_fra', n_bins)
        for idx in range(n_bins):
            bin_df = df[g1_label==idx].copy()
            if len(bin_df)<5:
                continue
            x1_msk = cot_to_x(bin_df['cot_mod08'].values)
            y1_msk = albedo_to_y(bin_df['albedo'].values)
            k_msk, _, _ = calc_global_slope(x1_msk, y1_msk, bin_df['season'].values, x2)
            x1_ret = cot_to_x(bin_df['ret_cot_cer'].values)
            y1_ret = albedo_to_y(bin_df['ret_albedo'].values)
            k_ret, _, _ = calc_global_slope(x1_ret, y1_ret, bin_df['season'].values, x2)
            unr_fra_res.append({'Ocean':ocean, 'Bin':idx, 'Slope_Diff':k_ret - k_msk})
        
        all_results.append({'ocean': ocean, 'cot_disp':cot_disp_res, 'unr_fra':unr_fra_res})
    return all_results

def plot_combined_4panels():
    slope_results = process_all_oceans(n_bins=2)
    
    cot_disp_list, unr_fra_list = [], []
    for res in slope_results:
        cot_disp_list.extend(res['cot_disp'])
        unr_fra_list.extend(res['unr_fra'])
    df_cot = pd.DataFrame(cot_disp_list)
    df_unr = pd.DataFrame(unr_fra_list)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4), dpi=300)
    bar_width = 0.35
    x_ocean = np.arange(len(oceans))
    COLORS_SLOPE = {'low': 'steelblue', 'high': 'coral'}
    labels = ['Sbdart', 'Quadrature ($k$=1)', 'Eddington']
    x_raw = cot_range
    sza_target = 54.5
    for m_idx, method in enumerate(METHODS):
        albedo_vals = cot_to_albedo(x_raw, method=method, sza=sza_target)
        ax1.plot(cot_to_x(x_raw), albedo_to_y(albedo_vals), 
                color=COLORS_ALBEDO[m_idx], lw=2,
                label=labels[m_idx],
                alpha=0.5 if COLORS_ALBEDO[m_idx] == 'purple' else 1.0)
    
    ax1.set_xlabel('ln(COT)', fontsize=16, fontweight='medium')
    ax1.set_ylabel(r'ln[$A_{\mathrm{c}}/(1-A_{\mathrm{c}})]$', fontsize=16, fontweight='medium')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_title(r'$\mathbf{(b)}$', fontsize=21, loc='left')
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)

    lookup_labels = [
        'Decoupled',
        'With Surface Albedo',
        'With Gas',
        'With Observed SZA'
    ]
    lookup_colors = ['#50A2E9', '#FF8C00', '#9370DB', '#2CA02C']
    
    for idx, (lookup_type, table_folder) in enumerate([('dcp', 'dcp'), ('gasdcp_surcp', 'gasdcp_surcp'), ('surdcp_gascp', 'surdcp_gascp')]):
        albedo_vals = cot_to_albedo(x_raw, method='sbdart', sza=54.5, 
                                    lookup_type=lookup_type, use_tpo_mam_only=True, table_folder=table_folder)
        ax2.plot(cot_to_x(x_raw), albedo_to_y(albedo_vals), 
                color=lookup_colors[idx], lw=2, label=lookup_labels[idx])
    
    all_sza_values = []
    for ocean in oceans:
        file_path = f"{BASE_PATH}/processed_data/merged_msk_and_ret_csv/{ocean}.csv"
        cols = ['sza', 'cf_mod08', 'cot_mod08', 'ret_cot_cer', 'ret_albedo', 'sw_all', 'sw_clr', 'solar_incoming', 'cf_liq_ceres']
        df = pd.read_csv(file_path, usecols=cols)
        mask = (df['cf_mod08']>0.3)&(df['cot_mod08']>3)&(df['ret_cot_cer']>3)&(df['ret_albedo'].between(0,1))
        all_sza_values.extend(df[mask]['sza'].dropna().values)
    
    mean_sza = np.mean(all_sza_values)
    
    albedo_mean_sza = cot_to_albedo(x_raw, method='sbdart', sza=mean_sza, 
                                    lookup_type='dcp', use_tpo_mam_only=True, table_folder='dcp')
    
    ax2.plot(cot_to_x(x_raw), albedo_to_y(albedo_mean_sza), 
            color=lookup_colors[3], lw=2, label=lookup_labels[3])
    
    ax2.set_xlabel('ln(COT)', fontsize=16, fontweight='medium')
    ax2.set_ylabel(r'ln[$A_{\mathrm{c}}/(1-A_{\mathrm{c}})]$', fontsize=16, fontweight='medium')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_title(r'$\mathbf{(c)}$', fontsize=21, loc='left')
    ax2.legend(loc='lower right', fontsize=11, framealpha=0.5)

    cot_disp_low_values = []
    cot_disp_high_values = []
    for ocean in oceans:
        ocean_df = df_cot[df_cot['Ocean']==ocean].sort_values('Bin')
        low_val = np.nan
        high_val = np.nan
        if len(ocean_df[ocean_df['Bin']==0])>0:
            low_val = ocean_df[ocean_df['Bin']==0]['Slope_Diff'].values[0]
        if len(ocean_df[ocean_df['Bin']==1])>0:
            high_val = ocean_df[ocean_df['Bin']==1]['Slope_Diff'].values[0]
        cot_disp_low_values.append(low_val)
        cot_disp_high_values.append(high_val)
    
    ax3.bar(x_ocean - bar_width/2, cot_disp_low_values, bar_width, label=r'low $d_{\mathrm{COT}}$', color=COLORS_SLOPE['low'], alpha=0.8, edgecolor=None)
    ax3.bar(x_ocean + bar_width/2, cot_disp_high_values, bar_width, label=r'high $d_{\mathrm{COT}}$', color=COLORS_SLOPE['high'], alpha=0.8, edgecolor=None)
    ax3.set_title(r'$\mathbf{(d)}$', fontsize=21, loc='left')
    ax3.set_ylabel(r'$k_{\mathrm{cp}}-k_{\mathrm{ret}}$', fontsize=18)
    ax3.set_ylim(0, 0.205)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.set_xticks(x_ocean)
    ax3.set_xticklabels(oceans, fontsize=14, ha='right', rotation=45)
    ax3.legend(fontsize=13, loc='best')
    ax3.grid(axis='y', linestyle='--', alpha=0.3)

    unr_fra_low_values = []
    unr_fra_high_values = []
    for ocean in oceans:
        ocean_df = df_unr[df_unr['Ocean']==ocean].sort_values('Bin')
        low_val = np.nan
        high_val = np.nan
        if len(ocean_df[ocean_df['Bin']==0])>0:
            low_val = ocean_df[ocean_df['Bin']==0]['Slope_Diff'].values[0]
        if len(ocean_df[ocean_df['Bin']==1])>0:
            high_val = ocean_df[ocean_df['Bin']==1]['Slope_Diff'].values[0]
        unr_fra_low_values.append(low_val)
        unr_fra_high_values.append(high_val)
    
    ax4.bar(x_ocean - bar_width/2, unr_fra_low_values, bar_width, label='low RUR', color=COLORS_SLOPE['low'], alpha=0.8, edgecolor=None)
    ax4.bar(x_ocean + bar_width/2, unr_fra_high_values, bar_width, label='high RUR', color=COLORS_SLOPE['high'], alpha=0.8, edgecolor=None)
    ax4.set_title(r'$\mathbf{(e)}$', fontsize=21, loc='left')
    ax4.set_ylabel('$k_{\mathrm{ret}}-k_{\mathrm{msk}}$', fontsize=18)
    ax4.tick_params(axis='y', labelsize=12)
    ax4.set_xticks(x_ocean)
    ax4.set_xticklabels(oceans, fontsize=14, ha='right', rotation=45)
    ax4.legend(fontsize=13, bbox_to_anchor=(0.18, 0.73))
    ax4.grid(axis='y', linestyle='--', alpha=0.3)

    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.22, wspace=0.3)
    plt.savefig(FIG_SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Final combined figure saved to: {os.path.abspath(FIG_SAVE_PATH)}")

if __name__ == "__main__":
    plot_combined_4panels()