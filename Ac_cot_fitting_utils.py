# Ac_cot_fitting_utils.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import odr, stats
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde

# Fixed seed for reproducibility of density sampling
np.random.seed(0)

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
    'time', 'lat', 'sw_all', 'sw_clr', 'solar_incoming',
    'cf_liq_ceres', 'cot_mod08', 'cotstd_mod08', 'sza'
]
cot_range = np.exp(np.linspace(np.log(2), 4.50, 15))


def cot_to_albedo(cot, method, sza=None, season=None, ocean_name=None):
    """
    Convert COT to albedo using the specified method.
    """
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

        for season_processed in season_dict.keys():
            csv_path = (
                '/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table/'
                'cot_sza_to_albedo_lookup_table_dcp/'
                'cot_sza_to_albedo_lookup_table_TPO_MAM.csv'
            )
            try:
                df = pd.read_csv(csv_path, index_col=0)
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

            mask = (season == season_processed)
            target_points = np.column_stack([
                np.full(cot[mask].shape, fixed_sza),
                np.atleast_1d(cot[mask])
            ])

            interpolated = griddata(
                points_valid, values_valid, target_points,
                method='linear', fill_value=np.nan
            )
            albedo[mask] = interpolated

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
    """
    Convert raw COT/albedo and their absolute 1-sigma uncertainties
    into transformed x/y and approximate transformed uncertainties.
    """
    cot = np.asarray(cot, dtype=float).ravel()
    albedo = np.asarray(albedo, dtype=float).ravel()
    cot_sigma = np.asarray(cot_sigma, dtype=float).ravel()
    albedo_sigma = np.asarray(albedo_sigma, dtype=float).ravel()

    cot = np.clip(cot, 1e-6, None)
    albedo = np.clip(albedo, 1e-6, 1 - 1e-6)

    x = cot_to_x(cot)
    y = albedo_to_y(albedo)

    # First-order error propagation:
    # x = ln(cot)                  -> dx/dcot = 1 / cot
    # y = ln[a / (1-a)]           -> dy/da = 1 / [a (1-a)]
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


def mc_fit(
    cot,
    albedo,
    cot_std=0.0,
    albedo_std=0.0,
    n_mc=300,
    bootstrap=True,
    random_seed=42
):
    """
    Fit y = k*x + b in transformed space, but perturb raw COT and raw albedo
    in Monte Carlo space.

    Parameters
    ----------
    cot : array-like
        Raw cloud optical thickness.
    albedo : array-like
        Raw albedo.
    cot_std : scalar or array-like
        Absolute 1-sigma uncertainty of raw COT.
    albedo_std : scalar or array-like
        Absolute 1-sigma uncertainty of raw albedo.

    Returns
    -------
    k, b, k_unc, b_unc
    """
    cot = np.asarray(cot, dtype=float).ravel()
    albedo = np.asarray(albedo, dtype=float).ravel()

    if cot.size != albedo.size:
        raise ValueError("cot and albedo must have the same length.")

    cot_sigma = _as_sigma_array(cot_std, cot.size)
    albedo_sigma = _as_sigma_array(albedo_std, albedo.size)

    mask = (
        np.isfinite(cot) &
        np.isfinite(albedo) &
        np.isfinite(cot_sigma) &
        np.isfinite(albedo_sigma) &
        (cot > 0) &
        (albedo > 0) &
        (albedo < 1)
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


def _weighted_mean(values, weights):
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


def _build_sza_groups(sza, season_mask, n_sza_groups):
    sza_valid = sza[season_mask]
    if sza_valid.size < n_sza_groups:
        return []

    edges = np.percentile(sza_valid, np.linspace(0, 100, n_sza_groups + 1))
    groups = []

    for i in range(n_sza_groups):
        low, high = edges[i], edges[i + 1]
        if i == n_sza_groups - 1:
            groups.append((sza >= low) & (sza <= high))
        else:
            groups.append((sza >= low) & (sza < high))

    return groups


def plot_weighted_fit_line(
    cot_raw, albedo_raw, sza, season, x2, color, label,
    line_handles, line_labels, ax,
    linestyle=None, n_sza_groups=2,
    cot_std=0.0, albedo_std=0.0
):
    """
    Plot a weighted fit line based on season + SZA groups.
    The Monte Carlo perturbation is applied in raw COT/albedo space.
    """
    fit_records = []
    season_records = {s: [] for s in season_dict.keys()}

    valid_raw = (
        np.isfinite(cot_raw) &
        np.isfinite(albedo_raw) &
        (cot_raw > 0) &
        (albedo_raw > 0) &
        (albedo_raw < 1)
    )

    for s in season_dict.keys():
        season_mask = valid_raw & (season == s)
        if not np.any(season_mask):
            continue

        for sza_mask in _build_sza_groups(sza, season_mask, n_sza_groups):
            final_mask = season_mask & sza_mask
            n_points = np.sum(final_mask)

            if n_points < 5:
                continue

            k, b, k_unc, b_unc = mc_fit(
                cot_raw[final_mask],
                albedo_raw[final_mask],
                cot_std=cot_std,
                albedo_std=albedo_std
            )
            if np.isnan(k):
                continue

            fit_records.append((k, b, k_unc, b_unc, n_points))
            season_records[s].append((k, b, k_unc, b_unc, n_points))

    if fit_records:
        weights = np.array([r[4] for r in fit_records], dtype=float)
        k_ann = _weighted_mean([r[0] for r in fit_records], weights)
        b_ann = _weighted_mean([r[1] for r in fit_records], weights)
        k_unc_ann = _weighted_mean([r[2] for r in fit_records], weights)
        b_unc_ann = _weighted_mean([r[3] for r in fit_records], weights)
        global_line = k_ann * x2 + b_ann
    else:
        k_ann = np.nan
        b_ann = np.nan
        k_unc_ann = np.nan
        b_unc_ann = np.nan
        global_line = np.full_like(x2, np.nan, dtype=float)

    sign = '+' if b_ann >= 0 else ''
    eq = f'y={k_ann:.2f}x{sign}{b_ann:.1f}'
    line = ax.plot(
        x2, global_line,
        color=color, linestyle=linestyle, lw=1.5,
        label=f'{label}: {eq}'
    )
    line_handles.append(line[0])
    line_labels.append(f'{label}: {eq}')



    k_season = {}
    b_season = {}
    k_unc_season = {}
    b_unc_season = {}

    for s in season_dict.keys():
        recs = season_records[s]
        if recs:
            weights = np.array([r[4] for r in recs], dtype=float)
            k_season[s] = _weighted_mean([r[0] for r in recs], weights)
            b_season[s] = _weighted_mean([r[1] for r in recs], weights)
            k_unc_season[s] = _weighted_mean([r[2] for r in recs], weights)
            b_unc_season[s] = _weighted_mean([r[3] for r in recs], weights)
        else:
            k_season[s] = np.nan
            b_season[s] = np.nan
            k_unc_season[s] = np.nan
            b_unc_season[s] = np.nan

    return (
        k_ann, b_ann, k_unc_ann, b_unc_ann,
        k_season, b_season, k_unc_season, b_unc_season,
        line_handles, line_labels
    )


def plot_axes_content(data, ax, title=None):
    """
    Populate a single subplot with density overlay and fit lines.
    Return slope/intercept results for further analysis.
    """
    ax.set_xlim([0.5, 3.5])
    ax.set_ylim([-2, 1.5])

    plot_density_overlay(
        data['x1_ret'], data['y1_list_ret'][0],
        data['x1_msk'], data['y1_msk'], ax
    )

    line_handles = []
    line_labels = []
    all_results = {}

    # LH74
    k, b, _, _, _ = stats.linregress(data['x2'], data['y22'])
    sign = '+' if b >= 0 else ''
    eq = f'y={k:.2f}x{sign}{b:.1f}'
    line = ax.plot(data['x2'], data['y22'], color='black', lw=1.5, label=f'LH74: {eq}')
    line_handles.append(line[0])
    line_labels.append(f'LH74: {eq}')
    all_results['LH74'] = (k, b, 0.0, 0.0, {}, {}, {}, {})

    fit_specs = [
        # name, raw_cot, raw_albedo, color, linestyle, cot_std, albedo_std
        ('dcp', data['ret_cot'], data['ret_albedo_list'][2], 'red',     '--', 0.0, 0.03),
        ('cp',  data['ret_cot'], data['ret_albedo_list'][1], 'orange',  '-',  0.0, 0.03),
        ('ret', data['ret_cot'], data['ret_albedo_list'][0], 'blue',    '--', 0.1, 0.13),
        ('msk', data['msk_cot'], data['msk_albedo'],         'magenta', '-',  0.1, 0.20),
    ]

    for name, cot_raw, albedo_raw, color, linestyle, cot_std, albedo_std in fit_specs:
        result = plot_weighted_fit_line(
            cot_raw, albedo_raw,
            data['sza'], data['season'], data['x2'],
            color, name, line_handles, line_labels, ax,
            linestyle=linestyle,
            cot_std=cot_std,
            albedo_std=albedo_std
        )
        (
            k_fit, b_fit, k_unc, b_unc,
            k_season, b_season, k_unc_season, b_unc_season,
            line_handles, line_labels
        ) = result

        all_results[name] = (
            k_fit, b_fit, k_unc, b_unc,
            k_season, b_season, k_unc_season, b_unc_season
        )

    ax.legend(handles=line_handles, labels=line_labels, fontsize=9, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)

    if title:
        ax.set_title(title, fontsize=13, loc='left')

    return all_results


def plot_density_overlay(x_ret, y_ret, x_msk, y_msk, ax, sample_size=5000):
    """
    Plot density overlay for ret (blue filled contour) and
    msk (magenta contour) data.
    """
    mask_ret = ~(np.isnan(x_ret) | np.isnan(y_ret))
    x_u = x_ret[mask_ret]
    y_u = y_ret[mask_ret]

    if len(x_u) > sample_size:
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

    mask_msk = ~(np.isnan(x_msk) | np.isnan(y_msk))
    x_f = x_msk[mask_msk]
    y_f = y_msk[mask_msk]

    if len(x_f) > sample_size:
        idx = np.random.choice(len(x_f), sample_size, replace=False)
        x_f = x_f[idx]
        y_f = y_f[idx]

    if len(x_f) >= 20:
        xi, yi = np.mgrid[0.5:4.25:100j, -2.0:1.5:100j]
        positions = np.vstack([xi.ravel(), yi.ravel()])
        values = np.vstack([x_f, y_f])
        kernel = gaussian_kde(values)
        zi = np.reshape(kernel(positions).T, xi.shape)

        ax.contour(
            xi, yi, zi,
            levels=5,
            colors='magenta',
            alpha=0.6,
            linewidths=0.8
        )


def preprocess_ocean_data(
    min_cot_mod08=2.0,
    min_ret_cot_cer=2.0,
    min_cf_liq_ceres=None
):
    """
    Preprocess data for all ocean regions.
    """
    albedo_l74 = cot_to_albedo(cot_range, 'l74')

    all_processed_ocean_data = {}
    global_data_collector = {
        'ret_cot': [],
        'ret_albedo_list': [[], [], []],
        'msk_cot': [],
        'msk_albedo': [],
        'x1_ret': [],
        'y1_list_ret': [[], [], []],
        'x1_msk': [],
        'y1_msk': [],
        'sza': [],
        'season': [],
        'x2': cot_to_x(cot_range),
        'y22': albedo_to_y(albedo_l74)
    }

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
                (df['cot_mod08'] > min_cot_mod08) &
                (df['ret_cot_cer'] > min_ret_cot_cer) &
                (df['ret_albedo'] > 0) & (df['ret_albedo'] < 1) &
                (df['albedo'] > 0) & (df['albedo'] < 1)
            )
            if min_cf_liq_ceres is not None:
                mask = mask & (df['cf_liq_ceres'] > min_cf_liq_ceres)
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

            x1_ret = cot_to_x(ret_cot)
            y1_ret_obs = albedo_to_y(ret_albedo_obs)
            y1_ret_cp = albedo_to_y(albedo_cp)
            y1_ret_dcp = albedo_to_y(albedo_dcp)

            x1_msk = cot_to_x(msk_cot)
            y1_msk = albedo_to_y(msk_albedo)

            ocean_processed_data = {
                'ret_cot': ret_cot,
                'ret_albedo_list': [ret_albedo_obs, albedo_cp, albedo_dcp],
                'msk_cot': msk_cot,
                'msk_albedo': msk_albedo,
                'x1_ret': x1_ret,
                'y1_list_ret': [y1_ret_obs, y1_ret_cp, y1_ret_dcp],
                'x1_msk': x1_msk,
                'y1_msk': y1_msk,
                'x2': global_data_collector['x2'],
                'y22': global_data_collector['y22'],
                'sza': df_filtered['sza'].values,
                'season': df_filtered['season'].values,
                'data_count': len(df_filtered)
            }
            all_processed_ocean_data[ocean] = ocean_processed_data

            valid_global_mask = ~(
                np.isnan(y1_ret_obs) &
                np.isnan(y1_ret_cp) &
                np.isnan(y1_ret_dcp)
            )

            global_data_collector['ret_cot'].extend(ret_cot[valid_global_mask])
            global_data_collector['ret_albedo_list'][0].extend(ret_albedo_obs[valid_global_mask])
            global_data_collector['ret_albedo_list'][1].extend(albedo_cp[valid_global_mask])
            global_data_collector['ret_albedo_list'][2].extend(albedo_dcp[valid_global_mask])

            global_data_collector['msk_cot'].extend(msk_cot[valid_global_mask])
            global_data_collector['msk_albedo'].extend(msk_albedo[valid_global_mask])

            global_data_collector['x1_ret'].extend(x1_ret[valid_global_mask])
            global_data_collector['y1_list_ret'][0].extend(y1_ret_obs[valid_global_mask])
            global_data_collector['y1_list_ret'][1].extend(y1_ret_cp[valid_global_mask])
            global_data_collector['y1_list_ret'][2].extend(y1_ret_dcp[valid_global_mask])

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

    global_processed_data = {
        'ret_cot': np.array(global_data_collector['ret_cot']),
        'ret_albedo_list': [np.array(lst) for lst in global_data_collector['ret_albedo_list']],
        'msk_cot': np.array(global_data_collector['msk_cot']),
        'msk_albedo': np.array(global_data_collector['msk_albedo']),
        'x1_ret': np.array(global_data_collector['x1_ret']),
        'y1_list_ret': [np.array(lst) for lst in global_data_collector['y1_list_ret']],
        'x1_msk': np.array(global_data_collector['x1_msk']),
        'y1_msk': np.array(global_data_collector['y1_msk']),
        'x2': global_data_collector['x2'],
        'y22': global_data_collector['y22'],
        'sza': np.array(global_data_collector['sza']),
        'season': np.array(global_data_collector['season']),
        'data_count': len(global_data_collector['ret_cot'])
    }

    print(f"Global data construction completed, integrated valid data count: {global_processed_data['data_count']}")
    return all_processed_ocean_data, global_processed_data