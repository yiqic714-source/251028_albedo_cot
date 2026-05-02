import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import odr, stats
from scipy.interpolate import griddata

np.random.seed(0)

BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIG_SAVE_PATH = f'{BASE_PATH}/figs/fig3_bias_attribution.png'
os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)

season_dict = {'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11], 'DJF': [12, 1, 2]}
oceans = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']
sza_list = [55, 15]
cot_range = np.exp(np.linspace(np.log(3), 3, 15))

METHODS = ['sbdart', 'quadrature']
LINECOLOR = ['steelblue', 'orange', 'coral', 'red']
LINESTYLE = ['-', '-', '--', ':']

# ============================================================
# Inlined fitting functions (from Ac_cot_fitting_utils.py)
# ============================================================

def _linear_func(beta, x):
    return beta[0] * x + beta[1]


def _as_sigma_array(sigma, n):
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        return np.full(n, float(sigma), dtype=float)
    sigma = sigma.ravel()
    if sigma.size != n:
        raise ValueError("sigma must be scalar or have the same length as the input data.")
    return sigma.astype(float)


def _raw_to_fit_arrays(cot, albedo, cot_sigma, albedo_sigma):
    cot = np.asarray(cot, dtype=float).ravel()
    albedo = np.asarray(albedo, dtype=float).ravel()
    cot_sigma = np.asarray(cot_sigma, dtype=float).ravel()
    albedo_sigma = np.asarray(albedo_sigma, dtype=float).ravel()

    cot = np.clip(cot, 1e-6, None)
    albedo = np.clip(albedo, 1e-6, 1 - 1e-6)

    x = cot_to_x(cot)
    y = albedo_to_y(albedo)

    sx = cot_sigma / cot
    sy = albedo_sigma / (albedo * (1 - albedo))

    x_scale = np.nanmax(np.abs(x)) if np.any(np.isfinite(x)) else 1.0
    y_scale = np.nanmax(np.abs(y)) if np.any(np.isfinite(y)) else 1.0
    sx_floor = max(x_scale * 1e-12, 1e-12)
    sy_floor = max(y_scale * 1e-12, 1e-12)

    sx = np.where(np.isfinite(sx) & (sx > 0), sx, sx_floor)
    sy = np.where(np.isfinite(sy) & (sy > 0), sy, sy_floor)

    return x, y, sx, sy


def _fit_odr_once(x, y, sx, sy, beta0=None):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    sx = np.asarray(sx, dtype=float).ravel()
    sy = np.asarray(sy, dtype=float).ravel()

    if x.size < 2:
        raise ValueError("At least 2 points are required.")

    if beta0 is None:
        k0, b0 = np.polyfit(x, y, 1)
        beta0 = [k0, b0]

    data = odr.RealData(x, y, sx=sx, sy=sy)
    model = odr.Model(_linear_func)
    out = odr.ODR(data, model, beta0=beta0).run()
    return out.beta[0], out.beta[1]


def mc_fit(cot, albedo, cot_std=0.0, albedo_std=0.0, n_mc=300, bootstrap=True, random_seed=42):
    cot = np.asarray(cot, dtype=float).ravel()
    albedo = np.asarray(albedo, dtype=float).ravel()

    if cot.size != albedo.size:
        raise ValueError("cot and albedo must have the same length.")

    cot_sigma = _as_sigma_array(cot_std, cot.size)
    albedo_sigma = _as_sigma_array(albedo_std, albedo.size)

    mask = (
        np.isfinite(cot) & np.isfinite(albedo) &
        np.isfinite(cot_sigma) & np.isfinite(albedo_sigma) &
        (cot > 0) & (albedo > 0) & (albedo < 1)
    )

    cot = cot[mask]
    albedo = albedo[mask]
    cot_sigma = cot_sigma[mask]
    albedo_sigma = albedo_sigma[mask]

    if cot.size < 3:
        return np.nan, np.nan, np.nan, np.nan

    x, y, sx, sy = _raw_to_fit_arrays(cot, albedo, cot_sigma, albedo_sigma)

    try:
        k_best, b_best = _fit_odr_once(x, y, sx, sy)
    except Exception:
        try:
            lr = stats.linregress(x, y)
            k_best, b_best = lr.slope, lr.intercept
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_seed)
    n = cot.size
    base_idx = np.arange(n)

    count = 0
    mean_k = 0.0
    mean_b = 0.0
    M2_k = 0.0
    M2_b = 0.0

    for _ in range(n_mc):
        idx = rng.integers(0, n, size=n) if bootstrap else base_idx

        cot_i = cot[idx]
        albedo_i = albedo[idx]
        cot_sigma_i = cot_sigma[idx]
        albedo_sigma_i = albedo_sigma[idx]

        cot_pert = cot_i + rng.normal(0.0, cot_sigma_i)
        albedo_pert = albedo_i + rng.normal(0.0, albedo_sigma_i)

        cot_pert = np.clip(cot_pert, 1e-6, None)
        albedo_pert = np.clip(albedo_pert, 1e-6, 1 - 1e-6)

        if np.unique(cot_pert).size < 2:
            continue

        x_i, y_i, sx_i, sy_i = _raw_to_fit_arrays(
            cot_pert, albedo_pert, cot_sigma_i, albedo_sigma_i
        )

        try:
            ki, bi = _fit_odr_once(x_i, y_i, sx_i, sy_i, beta0=[k_best, b_best])
        except Exception:
            try:
                lr = stats.linregress(x_i, y_i)
                ki, bi = lr.slope, lr.intercept
            except Exception:
                continue

        count += 1
        dk = ki - mean_k
        mean_k += dk / count
        M2_k += dk * (ki - mean_k)

        db = bi - mean_b
        mean_b += db / count
        M2_b += db * (bi - mean_b)

    if count < 2:
        return k_best, b_best, np.nan, np.nan

    k_unc = np.sqrt(M2_k / (count - 1))
    b_unc = np.sqrt(M2_b / (count - 1))

    return k_best, b_best, k_unc, b_unc



def format_panel_tag(panel_idx, icon_style):
    if icon_style == 'science':
        letter = chr(ord('A') + panel_idx)
        return rf'$\mathbf{{{letter}}}$'

    letter = chr(ord('a') + panel_idx)
    return rf'$\mathbf{{({letter})}}$'


def cot_to_x(cot):
    return np.log(cot)


def albedo_to_y(albedo):
    albedo = np.clip(albedo, 1e-6, 1 - 1e-6)
    return np.log(albedo / (1 - albedo))


def cot_to_albedo(cot, method, sza=None, table_folder='dcp'):
    """
    Calculate cloud albedo from cloud optical thickness.

    Parameters
    ----------
    table_folder : str
        Folder containing lookup tables.
        Supported examples: 'dcp', 'cp', 'gasdcp_surcp', 'surdcp_gascp', 'dcp_mono'
    """
    if method == 'sbdart':
        albedo = np.full(np.asarray(cot).shape, np.nan)
        seasons_to_use = ['MAM']
        ocean_to_use = 'TPO'

        for season_p in seasons_to_use:
            file_path = (
                f'{BASE_PATH}/build_sbdart_lookup_table/'
                f'cot_sza_to_albedo_lookup_table_{table_folder}/'
                f'cot_sza_to_albedo_lookup_table_{ocean_to_use}_{season_p}.csv'
            )

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

            cot_arr = np.atleast_1d(cot)
            if np.ndim(sza) == 0:
                sza_arr = np.full_like(cot_arr, sza, dtype=float)
            else:
                sza_arr = np.asarray(sza, dtype=float)

            target_points = np.column_stack([sza_arr, cot_arr])
            albedo_interp = griddata(
                points[valid_mask],
                values[valid_mask],
                target_points,
                method='linear',
                fill_value=np.nan
            )
            albedo = albedo_interp

        return albedo

    elif method == 'l74':
        g = 0.85
        b = 1 - g
        cot = np.asarray(cot)
        return b * cot / (1 + b * cot)

    elif method == 'quadrature':
        cot = np.asarray(cot)
        g = 0.85
        miu = np.cos(np.radians(sza))
        b = np.sqrt(3) / 2 * (1 - g)
        return (b * cot + (1 / 2 - np.sqrt(3) / 2 * miu) * (1 - np.exp(-cot / miu))) / (1 + b * cot)

    elif method == 'eddington':
        cot = np.asarray(cot)
        g = 0.85
        miu = np.cos(np.radians(sza))
        return ((1 - g) * cot + (2 / 3 - miu) * (1 - np.exp(-cot / miu))) / (4 / 3 + (1 - g) * cot)

    else:
        print(f"Supported methods: {METHODS + ['l74']}")
        return np.nan


def calculate_albedo_curves():
    albedo_results = {}
    for method in METHODS:
        for sza in sza_list:
            albedo_results[f'{method}_{sza}'] = cot_to_albedo(cot_range, method=method, sza=sza)
    return albedo_results


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights)
    if not np.any(mask):
        return np.nan

    values = values[mask]
    weights = weights[mask]

    if weights.sum() <= 0:
        return np.nan

    return np.sum(values * weights) / np.sum(weights)


def calc_global_slope_from_raw(cot, albedo, season, x2, cot_std=0.0, albedo_std=0.0, n_mc=100):
    """
    Compute season-weighted global slope/intercept using acfu.mc_fit
    on raw COT and raw albedo.
    """
    cot = np.asarray(cot, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    season = np.asarray(season)

    slopes, intercepts, weights = [], [], []

    valid_mask = (
        np.isfinite(cot) &
        np.isfinite(albedo) &
        (cot > 0) &
        (albedo > 0) &
        (albedo < 1)
    )

    for s in season_dict:
        mask = valid_mask & (season == s)
        if np.sum(mask) < 5:
            continue

        k, b, _, _ = mc_fit(
            cot[mask],
            albedo[mask],
            cot_std=cot_std,
            albedo_std=albedo_std,
            n_mc=n_mc,
            bootstrap=True,
            random_seed=42
        )


        if np.isfinite(k):
            slopes.append(k)
            intercepts.append(b)
            weights.append(np.sum(mask))

    if len(weights) == 0:
        return np.nan, np.nan, np.full_like(x2, np.nan, dtype=float)

    global_k = weighted_mean(slopes, weights)
    global_b = weighted_mean(intercepts, weights)
    return global_k, global_b, global_k * x2 + global_b


def split_data_by_percentile(df, col_name, n_bins):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(df[col_name].dropna(), percentiles))
    n_bins = len(bin_edges) - 1 if len(bin_edges) < n_bins + 1 else n_bins
    df['group_label'] = pd.cut(df[col_name], bins=bin_edges, labels=False, include_lowest=True)
    return df['group_label'].values, bin_edges


def process_all_oceans(n_bins=2):
    all_results = []
    cot_ref = np.exp(np.linspace(np.log(3), 4.5, 15))
    x2 = cot_to_x(cot_ref)

    for ocean in oceans:
        file_path = f"{BASE_PATH}/processed_data/merged_msk_and_ret_csv/{ocean}.csv"
        cols = [
            'ret_albedo', 'ret_cot_cer', 'ret_cotstd_cer', 'time', 'lat',
            'sw_all', 'sw_clr', 'solar_incoming', 'cf_ceres', 'cot_mod08',
            'sza', 'cf_ret_liq_mod08', 'clr_fra'
        ]
        df = pd.read_csv(file_path, usecols=cols)

        df['albedo'] = (
            (df['sw_all'] - df['sw_clr'] * (1 - df['cf_ceres'])) /
            df['cf_ceres'] / df['solar_incoming']
        )
        df['month'] = pd.to_datetime(df['time'], format='mixed').dt.month

        df['cot_disp'] = df['ret_cotstd_cer'] / df['ret_cot_cer']
        df['unr_fra'] = (1 - df['cf_ret_liq_mod08'] - df['clr_fra'])

        mask = (
            (df['cf_ceres'] > 0.1) &
            (df['cot_mod08'] > 2.5) &
            (df['ret_cot_cer'] > 2.5) &
            (df['ret_albedo'].between(0, 1)) &
            (df['albedo'].between(0, 1))
        )
        df = df[mask].dropna()

        for s, ms in season_dict.items():
            df.loc[df['month'].isin(ms), 'season'] = s
        df['season'] = df['season'].astype('object')

        cot_disp_res = []
        g0_label, _ = split_data_by_percentile(df.copy(), 'cot_disp', n_bins)

        for idx in range(n_bins):
            bin_df = df[g0_label == idx].copy()
            if len(bin_df) < 5:
                continue

            ret_cot = bin_df['ret_cot_cer'].values
            ret_albedo = bin_df['ret_albedo'].values
            season_vals = bin_df['season'].values

            albedo_sbd = cot_to_albedo(
                ret_cot,
                'sbdart',
                sza=bin_df['sza'].values,
                table_folder='cp'
            )

            k_ret, _, _ = calc_global_slope_from_raw(
                ret_cot, ret_albedo, season_vals, x2,
                cot_std=0.0, albedo_std=0.03, n_mc=1000
            )
            k_sbd, _, _ = calc_global_slope_from_raw(
                ret_cot, albedo_sbd, season_vals, x2,
                cot_std=0.0, albedo_std=0.03, n_mc=1000
            )

            cot_disp_res.append({
                'Ocean': ocean,
                'Bin': idx,
                'Slope_Diff': k_sbd - k_ret
            })

        unr_fra_res = []
        g1_label, edges = split_data_by_percentile(df.copy(), 'unr_fra', n_bins)
        df['unr_bin'] = pd.cut(df['unr_fra'], bins=edges, labels=False, include_lowest=True)

        for idx in range(len(edges) - 1):
            bin_df = df[g1_label == idx].copy()
            if len(bin_df) < 5:
                continue

            msk_cot = bin_df['cot_mod08'].values
            msk_albedo = bin_df['albedo'].values
            ret_cot = bin_df['ret_cot_cer'].values
            ret_albedo = bin_df['ret_albedo'].values
            season_vals = bin_df['season'].values

            k_msk, _, _ = calc_global_slope_from_raw(
                msk_cot, msk_albedo, season_vals, x2,
                cot_std=0.10, albedo_std=0.13, n_mc=1000
            )
            k_ret, _, _ = calc_global_slope_from_raw(
                ret_cot, ret_albedo, season_vals, x2,
                cot_std=0.10, albedo_std=0.10, n_mc=1000
            )

            unr_fra_res.append({
                'Ocean': ocean,
                'Bin': idx,
                'Slope_Diff': k_ret - k_msk
            })

        all_results.append({
            'ocean': ocean,
            'cot_disp': cot_disp_res,
            'unr_fra': unr_fra_res
        })

    return all_results


def plot_combined_4panels(icon_style='nature'):
    if icon_style not in ('nature', 'science'):
        raise ValueError("icon_style must be 'nature' or 'science'.")

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
    x_raw = cot_range
    sza_target = 54.5

    albedo_sbdart = cot_to_albedo(x_raw, method='sbdart', sza=sza_target, table_folder='dcp')
    ax1.plot(
        cot_to_x(x_raw), albedo_to_y(albedo_sbdart),
        color=LINECOLOR[1], lw=2,
        label='Sbdart, Shortwave', alpha=1.0
    )

    albedo_sbdart_mono = cot_to_albedo(x_raw, method='sbdart', sza=sza_target, table_folder='dcp_mono')
    ax1.plot(
        cot_to_x(x_raw), albedo_to_y(albedo_sbdart_mono),
        color=LINECOLOR[3], lw=2, linestyle='--',
        label='Sbdart, Visible', alpha=1.0
    )

    albedo_quad = cot_to_albedo(x_raw, method='quadrature', sza=sza_target)
    ax1.plot(
        cot_to_x(x_raw), albedo_to_y(albedo_quad),
        color=LINECOLOR[0], lw=2,
        label='Quadrature (T91)', alpha=1.0
    )

    ax1.set_xlabel('ln(COT)', fontsize=16, fontweight='medium')
    ax1.set_ylabel(r'ln[$A_{\mathrm{c}}/(1-A_{\mathrm{c}})]$', fontsize=16, fontweight='medium')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_title(format_panel_tag(1, icon_style), fontsize=21, loc='left')
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)

    lookup_labels = [
        'Decoupled',
        r'With Observed $A_{\mathrm{sfc}}$',
        'With Gas',
        r'With Observed SZA'
    ]

    for idx, table_folder in enumerate(['dcp', 'surdcp_gascp', 'gasdcp_surcp']):
        albedo_vals = cot_to_albedo(x_raw, method='sbdart', sza=54.4, table_folder=table_folder)
        ax2.plot(
            cot_to_x(x_raw), albedo_to_y(albedo_vals),
            color=LINECOLOR[idx], lw=2, linestyle=LINESTYLE[idx],
            label=lookup_labels[idx]
        )

    all_sza_values = []
    for ocean in oceans:
        file_path = f"{BASE_PATH}/processed_data/merged_msk_and_ret_csv/{ocean}.csv"
        cols = ['sza', 'cf_ceres', 'cot_mod08', 'ret_cot_cer', 'ret_albedo',
                'sw_all', 'sw_clr', 'solar_incoming', 'cf_ret_liq_mod08', 'clr_fra']
        df = pd.read_csv(file_path, usecols=cols)
        mask = (
            (df['cf_ceres'] > 0.3) &
            (df['cot_mod08'] > 3) &
            (df['ret_cot_cer'] > 3) &
            (df['ret_albedo'].between(0, 1))
        )
        all_sza_values.extend(df[mask]['sza'].dropna().values)

    mean_sza = np.mean(all_sza_values)
    albedo_mean_sza = cot_to_albedo(x_raw, method='sbdart', sza=mean_sza, table_folder='dcp')

    ax2.plot(
        cot_to_x(x_raw), albedo_to_y(albedo_mean_sza),
        color=LINECOLOR[3], lw=2, linestyle=LINESTYLE[3],
        label=lookup_labels[3]
    )

    ax2.set_xlabel('ln(COT)', fontsize=16, fontweight='medium')
    ax2.set_ylabel(r'ln[$A_{\mathrm{c}}/(1-A_{\mathrm{c}})]$', fontsize=16, fontweight='medium')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_title(format_panel_tag(2, icon_style), fontsize=21, loc='left')
    ax2.legend(loc='lower right', fontsize=11, framealpha=0.5)

    cot_disp_low_values = []
    cot_disp_high_values = []
    for ocean in oceans:
        ocean_df = df_cot[df_cot['Ocean'] == ocean].sort_values('Bin')
        low_val = np.nan
        high_val = np.nan
        if len(ocean_df[ocean_df['Bin'] == 0]) > 0:
            low_val = ocean_df[ocean_df['Bin'] == 0]['Slope_Diff'].values[0]
        if len(ocean_df[ocean_df['Bin'] == 1]) > 0:
            high_val = ocean_df[ocean_df['Bin'] == 1]['Slope_Diff'].values[0]
        cot_disp_low_values.append(low_val)
        cot_disp_high_values.append(high_val)

    ax3.bar(
        x_ocean - bar_width / 2, cot_disp_low_values, bar_width,
        label=r'low $d_{\mathrm{COT}}$', color=COLORS_SLOPE['low'],
        alpha=0.8, edgecolor=None
    )
    ax3.bar(
        x_ocean + bar_width / 2, cot_disp_high_values, bar_width,
        label=r'high $d_{\mathrm{COT}}$', color=COLORS_SLOPE['high'],
        alpha=0.8, edgecolor=None
    )
    ax3.set_title(format_panel_tag(3, icon_style), fontsize=21, loc='left')
    ax3.set_ylabel(r'$k_{\mathrm{cp}}-k_{\mathrm{ret}}$', fontsize=18)
    ax3.set_ylim(0, 0.22)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.set_xticks(x_ocean)
    ax3.set_xticklabels(oceans, fontsize=13, ha='right', rotation=45)
    ax3.legend(fontsize=13, loc='best')
    ax3.grid(axis='y', linestyle='--', alpha=0.3)

    unr_fra_low_values = []
    unr_fra_high_values = []
    for ocean in oceans:
        ocean_df = df_unr[df_unr['Ocean'] == ocean].sort_values('Bin')
        if len(ocean_df) == 0:
            unr_fra_low_values.append(np.nan)
            unr_fra_high_values.append(np.nan)
            continue

        low_val = np.nan
        high_val = np.nan
        if len(ocean_df[ocean_df['Bin'] == 0]) > 0:
            low_val = ocean_df[ocean_df['Bin'] == 0]['Slope_Diff'].values[0]
        if len(ocean_df[ocean_df['Bin'] == 1]) > 0:
            high_val = ocean_df[ocean_df['Bin'] == 1]['Slope_Diff'].values[0]
        unr_fra_low_values.append(low_val)
        unr_fra_high_values.append(high_val)

    ax4.bar(
        x_ocean - bar_width / 2, unr_fra_low_values, bar_width,
        label='low TZF', color=COLORS_SLOPE['low'],
        alpha=0.8, edgecolor=None
    )
    ax4.bar(
        x_ocean + bar_width / 2, unr_fra_high_values, bar_width,
        label='high TZF', color=COLORS_SLOPE['high'],
        alpha=0.8, edgecolor=None
    )
    ax4.set_title(format_panel_tag(4, icon_style), fontsize=21, loc='left')
    ax4.set_ylabel(r'$k_{\mathrm{ret}}-k_{\mathrm{msk}}$', fontsize=18)
    ax4.tick_params(axis='y', labelsize=12)
    ax4.set_xticks(x_ocean)
    ax4.set_xticklabels(oceans, fontsize=13, ha='right', rotation=45)
    ax4.legend(fontsize=13, bbox_to_anchor=(0.18, 0.73))
    ax4.grid(axis='y', linestyle='--', alpha=0.3)

    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.22, wspace=0.3)
    plt.savefig(FIG_SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Final combined figure saved to: {os.path.abspath(FIG_SAVE_PATH)}")


if __name__ == "__main__":
    # Choose panel tag style here: 'nature' -> (a)(b)(c), 'science' -> A B C.
    plot_combined_4panels(icon_style='nature')