import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import odr, stats
from scipy.interpolate import griddata

# ============================================================
# Inlined constants and helper functions from Ac_cot_fitting_utils.py
# ============================================================

np.random.seed(0)

oceans = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']
season_dict = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}
input_dir = "/home/chenyiqi/251028_albedo_cot/processed_data/merged_msk_and_ret_csv/"
MIN_COT = 2.5
MIN_CF = 0.1
columns = [
    'ret_albedo', 'ret_cot_mod', 'ret_cotstd_mod',
    'ret_cot_cer', 'ret_cotstd_cer',
    'time', 'lat', 'sw_all', 'sw_clr', 'solar_incoming',
    'cf_liq_ceres', 'cot_mod08', 'cotstd_mod08', 'sza', 'aod_mod08'
]
cot_range = np.exp(np.linspace(np.log(2), 4.50, 15))


def cot_to_albedo(cot, method, sza=None, season=None, ocean_name=None):
    if method == 'sbdart_cp':
        albedo = np.full(cot.shape, np.nan)
        for season_processed in season_dict.keys():
            csv_path = (
                f'/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table/'
                f'cot_sza_to_albedo_lookup_table_cp/'
                f'cot_sza_to_albedo_lookup_table_{ocean_name}_{season_processed}.csv'
            )
            try:
                df = pd.read_csv(csv_path, index_col=0)
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

            mask = (season == season_processed)
            target_points = np.column_stack([
                np.atleast_1d(sza[mask]),
                np.atleast_1d(cot[mask])
            ])

            interpolated = griddata(
                points_valid, values_valid, target_points,
                method='linear', fill_value=np.nan
            )
            albedo[mask] = interpolated
        return albedo

    elif method == 'sbdart_dcp':
        albedo = np.full(cot.shape, np.nan)
        fixed_sza = 54.7
        csv_path = (
            '/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table/'
            'cot_sza_to_albedo_lookup_table_dcp/'
            'cot_sza_to_albedo_lookup_table_TPO_MAM.csv'
        )
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except FileNotFoundError:
            print("Warning: Lookup table not found for dcp. Skipping.")
            return albedo

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
            return albedo

        target_points = np.column_stack([
            np.full(cot.shape, fixed_sza),
            np.atleast_1d(cot)
        ])

        interpolated = griddata(
            points_valid, values_valid, target_points,
            method='linear', fill_value=np.nan
        )
        albedo[:] = interpolated
        return albedo

    elif method == 'l74':
        cot = np.asarray(cot)
        b = 0.13
        return b * cot / (1 + b * cot)
    else:
        print("Supported methods: ['sbdart_cp', 'sbdart_dcp', 'l74']")
        return np.nan


def cot_to_x(cot):
    return np.log(cot)


def albedo_to_y(albedo):
    albedo = np.clip(albedo, 1e-6, 1 - 1e-6)
    return np.log(albedo / (1 - albedo))


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


def preprocess_ocean_data():
    """
    Preprocess data for all ocean regions.
    Returns a dict: ocean_name -> processed_data dict
    """
    all_processed_ocean_data = {}

    print("Starting one-time preprocessing for all ocean data...")

    for ocean in oceans:
        file_path = os.path.join(input_dir, f"{ocean}.csv")

        try:
            df = pd.read_csv(file_path, usecols=columns)

            df['albedo'] = (
                (df['sw_all'] - df['sw_clr'] * (1 - df['cf_liq_ceres'])) /
                df['cf_liq_ceres'] / df['solar_incoming']
            )
            df['month'] = pd.to_datetime(df['time'], format='mixed').dt.month

            for season_name, months in season_dict.items():
                df.loc[df['month'].isin(months), 'season'] = season_name

            mask = (
                (df['cot_mod08'] > MIN_COT) &
                (df['ret_cot_cer'] > MIN_COT) &
                (df['ret_albedo'] > 0) & (df['ret_albedo'] < 1) &
                (df['albedo'] > 0) & (df['albedo'] < 1)
            )
            mask = mask & (df['cf_liq_ceres'] > MIN_CF)
            df_filtered = df[mask].dropna().reset_index(drop=True)

            if len(df_filtered) == 0:
                print(f"{ocean} has no valid data, skipping.")
                all_processed_ocean_data[ocean] = None
                continue

            ret_cot = df_filtered['ret_cot_cer'].values
            ret_albedo_obs = df_filtered['ret_albedo'].values
            msk_cot = df_filtered['cot_mod08'].values
            msk_albedo = df_filtered['albedo'].values

            albedo_cp = cot_to_albedo(
                ret_cot, 'sbdart_cp',
                sza=df_filtered['sza'].values,
                season=df_filtered['season'].values,
                ocean_name=ocean
            )
            albedo_dcp = cot_to_albedo(
                ret_cot, 'sbdart_dcp',
                sza=df_filtered['sza'].values,
                season=df_filtered['season'].values,
                ocean_name=ocean
            )

            aod_mod08 = df_filtered['aod_mod08'].values

            ocean_processed_data = {
                'ret_cot': ret_cot,
                'ret_albedo_list': [ret_albedo_obs, albedo_cp, albedo_dcp],
                'msk_cot': msk_cot,
                'msk_albedo': msk_albedo,
                'sza': df_filtered['sza'].values,
                'season': df_filtered['season'].values,
                'aod_mod08': aod_mod08,
                'data_count': len(df_filtered)
            }
            all_processed_ocean_data[ocean] = ocean_processed_data
            print(f"{ocean} preprocessing completed, valid data count: {len(df_filtered)}")

        except Exception as e:
            print(f"Error processing {ocean}: {e}")
            import traceback
            traceback.print_exc()
            all_processed_ocean_data[ocean] = None

    return all_processed_ocean_data


# ============================================================
# Plotting logic: 2 rows x 4 columns
#   Row 1: by ocean (one line per ocean, all seasons combined)
#          For each ocean.
#   Row 2: by AOD bin (8 bins, all oceans+seasons combined)
#          For each AOD bin.
# ============================================================

# Color map for oceans
ocean_colors = {
    'NPO': '#1f77b4',  # blue
    'NAO': '#ff7f0e',  # orange
    'TPO': '#2ca02c',  # green
    'TAO': '#d62728',  # red
    'TIO': '#9467bd',  # purple
    'SPO': '#8c564b',  # brown
    'SAO': '#e377c2',  # pink
    'SIO': '#7f7f7f',  # gray
}

# Method display names and their fit parameters
method_specs = {
    'dcp': {'idx': 2, 'color': 'red',     'label': 'dcp', 'cot_std': 0.0,  'albedo_std': 0.03},
    'cp':  {'idx': 1, 'color': 'orange',  'label': 'cp',  'cot_std': 0.0,  'albedo_std': 0.03},
    'ret': {'idx': 0, 'color': 'blue',    'label': 'ret', 'cot_std': 0.1,  'albedo_std': 0.13},
    'msk': {'idx': None, 'color': 'magenta', 'label': 'msk', 'cot_std': 0.1, 'albedo_std': 0.20},
}

def format_panel_tag(panel_idx, icon_style):
    """Format panel tag: 'science' -> A, B, C...; 'nature' -> (a), (b), (c)..."""
    if icon_style == 'science':
        letter = chr(ord('A') + panel_idx)
        return rf'$\mathbf{{{letter}}}$'
    letter = chr(ord('a') + panel_idx)
    return rf'$\mathbf{{({letter})}}$'


# Panel labels (a-h for 8 subplots)
panel_labels = [
    ['dcp by ocean', 'cp by ocean', 'ret by ocean', 'msk by ocean'],
    ['dcp by AOD', 'cp by AOD', 'ret by AOD', 'msk by AOD'],
]

method_keys = ['dcp', 'cp', 'ret', 'msk']

# Number of AOD bins and SZA bins
N_AOD_BINS = 8

# Color map for AOD bins (a gradient from light to dark)
aod_bin_colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, N_AOD_BINS))


def _get_raw_arrays(ocean_data, method_key, spec):
    """Extract raw cot, albedo, sza, and aod arrays for a given method."""
    if method_key == 'msk':
        cot_raw = ocean_data['msk_cot']
        albedo_raw = ocean_data['msk_albedo']
    else:
        cot_raw = ocean_data['ret_cot']
        albedo_raw = ocean_data['ret_albedo_list'][spec['idx']]

    valid = (
        np.isfinite(cot_raw) & np.isfinite(albedo_raw) &
        (cot_raw > 0) & (albedo_raw > 0) & (albedo_raw < 1)
    )

    cot_raw = cot_raw[valid]
    albedo_raw = albedo_raw[valid]
    sza_arr = ocean_data['sza'][valid]
    aod_arr = ocean_data['aod_mod08'][valid]

    return cot_raw, albedo_raw, sza_arr, aod_arr


def _compute_mean_curve(cot, albedo):
    """
    Bin data by x (ln(COT)) with step 0.25 and compute mean y in each bin.
    Returns (x_centers, y_mean) arrays.
    """
    n_total = len(cot)
    if n_total < 5:
        return None, None

    # Define x-bin edges
    x_bin_edges = np.arange(0.85, 3.5, 0.3)
    x_centers = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2.0

    # Transform to transformed space
    x = cot_to_x(cot)
    y = albedo_to_y(albedo)

    # Bin x values into the x-bin edges
    bin_indices = np.digitize(x, x_bin_edges) - 1
    valid = (bin_indices >= 0) & (bin_indices < len(x_centers))

    y_mean = np.full(len(x_centers), np.nan)
    for j in range(len(x_centers)):
        mask_j = valid & (bin_indices == j)
        if np.any(mask_j):
            y_mean[j] = np.mean(y[mask_j])

    return x_centers, y_mean



def compute_curve_by_ocean(all_processed_ocean_data, method_key, spec):
    """
    Compute mean curve for each ocean separately (all seasons combined).
    Returns dict: ocean_name -> (x_centers, y_mean)
    """
    results = {}
    for ocean in oceans:
        od = all_processed_ocean_data[ocean]
        if od is None:
            results[ocean] = (None, None)
            continue
        cot_raw, albedo_raw, _, _ = _get_raw_arrays(od, method_key, spec)
        if len(cot_raw) < 5:
            results[ocean] = (None, None)
            continue
        x_c, y_m = _compute_mean_curve(cot_raw, albedo_raw)
        results[ocean] = (x_c, y_m)
    return results


def compute_curve_by_aod_bins(all_processed_ocean_data, method_key, spec):
    """
    Concatenate data from all oceans, then split by AOD quantile bins.
    For each AOD bin, compute mean curve.
    Returns dict: bin_index -> (x_centers, y_mean, bin_label)
    """
    all_cot = []
    all_albedo = []
    all_aod = []

    for ocean in oceans:
        od = all_processed_ocean_data[ocean]
        if od is None:
            continue
        cot_raw, albedo_raw, _, aod_arr = _get_raw_arrays(od, method_key, spec)
        all_cot.append(cot_raw)
        all_albedo.append(albedo_raw)
        all_aod.append(aod_arr)

    if len(all_cot) == 0:
        return {}

    cot_all = np.concatenate(all_cot)
    albedo_all = np.concatenate(all_albedo)
    aod_all = np.concatenate(all_aod)

    # Remove any remaining invalid AOD
    valid_aod = np.isfinite(aod_all) & (aod_all >= 0)
    cot_all = cot_all[valid_aod]
    albedo_all = albedo_all[valid_aod]
    aod_all = aod_all[valid_aod]

    if len(cot_all) < N_AOD_BINS * 5:
        return {}

    # Compute quantile edges for N_AOD_BINS bins
    percentiles = np.linspace(0, 100, N_AOD_BINS + 1)
    edges = np.percentile(aod_all, percentiles)

    results = {}
    for i in range(N_AOD_BINS):
        low, high = edges[i], edges[i + 1]
        if i == N_AOD_BINS - 1:
            mask = (aod_all >= low) & (aod_all <= high)
        else:
            mask = (aod_all >= low) & (aod_all < high)

        n_pts = np.sum(mask)
        if n_pts < 5:
            results[i] = (None, None, f'bin{i}')
            continue

        x_c, y_m = _compute_mean_curve(cot_all[mask], albedo_all[mask])
        bin_label = f'AOD [{low:.2f}, {high:.2f})' if i < N_AOD_BINS - 1 else f'AOD [{low:.2f}, {high:.2f}]'
        results[i] = (x_c, y_m, bin_label)

    return results



def plot_figure(icon_style='nature'):
    if icon_style not in ('nature', 'science'):
        raise ValueError("icon_style must be 'nature' or 'science'.")

    all_processed_ocean_data = preprocess_ocean_data()

    # LH74 reference line
    x2 = cot_to_x(cot_range)
    albedo_l74 = cot_to_albedo(cot_range, 'l74')
    y22 = albedo_to_y(albedo_l74)

    # ---- Pre-compute mean curves and global fits ----

    # Row 0: by ocean
    all_curves_ocean = {}
    all_global_fits = {}
    for method_key in method_keys:
        spec = method_specs[method_key]
        all_curves_ocean[method_key] = compute_curve_by_ocean(all_processed_ocean_data, method_key, spec)

        # Global fit: concatenate all oceans' data and fit once
        all_cot, all_albedo = [], []
        for ocean in oceans:
            od = all_processed_ocean_data[ocean]
            if od is None:
                continue
            cot_raw, albedo_raw, _, _ = _get_raw_arrays(od, method_key, spec)
            all_cot.append(cot_raw)
            all_albedo.append(albedo_raw)
        if len(all_cot) > 0:
            cot_all = np.concatenate(all_cot)
            albedo_all = np.concatenate(all_albedo)
            k, b, _, _ = mc_fit(cot_all, albedo_all,
                                cot_std=spec['cot_std'],
                                albedo_std=spec['albedo_std'])
            all_global_fits[method_key] = (k, b)
        else:
            all_global_fits[method_key] = (np.nan, np.nan)

    # Row 1: by AOD bins
    all_curves_aod = {}
    for method_key in method_keys:
        spec = method_specs[method_key]
        all_curves_aod[method_key] = compute_curve_by_aod_bins(all_processed_ocean_data, method_key, spec)

    # ---- Create 2x4 figure ----
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for col_idx, method_key in enumerate(method_keys):
        spec = method_specs[method_key]

        # ===== Row 0: by ocean =====
        ax0 = axes[0, col_idx]
        ax0.plot(x2, y22, color='black', linestyle='--', lw=1.8,
                 label=r'$k_{\mathrm{T91}}=1.0$')

        # Global fit line with formula (plotted right after LH74, before ocean lines)
        k, b = all_global_fits[method_key]
        if np.isfinite(k):
            y_fit = k * x2 + b
            formula = rf'$k_{{\mathrm{{{method_key}}}}}={k:.2f}$'
            ax0.plot(x2, y_fit, color='black', linestyle='-', lw=1.8, label=formula)

        for ocean in oceans:
            x_c, y_m = all_curves_ocean[method_key].get(ocean, (None, None))
            if x_c is None or y_m is None:
                continue
            ax0.plot(x_c, y_m, color=ocean_colors[ocean],
                     linestyle='-', lw=1.8, label=ocean)


        ax0.set_xlim([0.8, 3.25])
        ax0.set_ylim([-1.6, 0.8])
        panel_tag = format_panel_tag(col_idx, icon_style)
        ax0.set_title(f'{panel_tag}  {panel_labels[0][col_idx]}', fontsize=18, loc='left')
        ax0.grid(True, linestyle='--', alpha=0.3)
        ax0.tick_params(axis='both', labelsize=14)
        if col_idx == 0:
            ax0.legend(fontsize=12, loc='lower right', ncol=2)
        else:
            # Only show LH74 and global fit formula in legend
            handles, labels = ax0.get_legend_handles_labels()
            ax0.legend(handles[0:2], labels[0:2],
                       fontsize=12, loc='lower right')


        # ===== Row 1: AOD bins =====
        ax1 = axes[1, col_idx]
        # Plot LH74 and global fit lines but without legend labels (redundant with row 0)
        ax1.plot(x2, y22, color='black', linestyle='--', lw=1.5, label='_nolegend_')

        aod_curves = all_curves_aod[method_key]
        for bin_i in range(N_AOD_BINS):
            if bin_i not in aod_curves:
                continue
            x_c, y_m, bin_label = aod_curves[bin_i]
            if x_c is None or y_m is None:
                continue
            ax1.plot(x_c, y_m, color=aod_bin_colors[bin_i],
                     linestyle='-', lw=1.5, label=bin_label)

        # Global fit line without legend label
        if np.isfinite(k):
            ax1.plot(x2, y_fit, color='black', linestyle='-', lw=1.8, label='_nolegend_')

        ax1.set_xlim([0.8, 3.25])
        ax1.set_ylim([-1.6, 0.8])
        panel_tag = format_panel_tag(4 + col_idx, icon_style)
        ax1.set_title(f'{panel_tag} {panel_labels[1][col_idx]}', fontsize=18, loc='left')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.tick_params(axis='both', labelsize=14)
        if col_idx == 0:
            ax1.legend(fontsize=12, loc='lower right', ncol=1)
        else:
            ax1.legend().set_visible(False)




    fig.text(0.5, 0.02, r'ln(COT)', ha='center', fontsize=20)
    fig.text(0.02, 0.5, r'$\ln\left[A_{\mathrm{c}}/(1-A_{\mathrm{c}})\right]$',
             va='center', rotation='vertical', fontsize=20)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.98])

    os.makedirs('figs', exist_ok=True)
    output_fig_path = 'figs/fig2_curves_by_ocean_and_aod.png'
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"2x4 figure saved to: {output_fig_path}")


if __name__ == "__main__":
    # Choose panel tag style here: 'nature' -> (a)(b)(c)..., 'science' -> A B C...
    plot_figure(icon_style='nature')


