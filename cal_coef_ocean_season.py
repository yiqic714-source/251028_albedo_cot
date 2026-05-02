import os
import numpy as np
import pandas as pd
from scipy import odr, stats
from scipy.interpolate import griddata

# ============================================================
# Constants and helper functions (inlined from Ac_cot_fitting_utils.py)
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


# ============================================================
# Main: compute and save coefficients for each Method x Ocean x Season
# ============================================================

# Method specifications
method_specs = {
    'dcp': {'idx': 2, 'cot_std': 0.0,  'albedo_std': 0.03},
    'cp':  {'idx': 1, 'cot_std': 0.0,  'albedo_std': 0.03},
    'ret': {'idx': 0, 'cot_std': 0.1,  'albedo_std': 0.13},
    'msk': {'idx': None, 'cot_std': 0.1, 'albedo_std': 0.20},
}
method_keys = ['dcp', 'cp', 'ret', 'msk']


def preprocess_ocean_data(min_cot_mod08=2.0, min_ret_cot_cer=2.0, min_cf_liq_ceres=None):
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

            ocean_processed_data = {
                'ret_cot': ret_cot,
                'ret_albedo_list': [ret_albedo_obs, albedo_cp, albedo_dcp],
                'msk_cot': msk_cot,
                'msk_albedo': msk_albedo,
                'season': df_filtered['season'].values,
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


def fit_ocean_season(ocean_data, method_key, spec):
    """
    Fit one ocean x season combination for a given method.
    Returns dict: season_name -> (k, b, k_unc, b_unc)
    """
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
    season_arr = ocean_data['season'][valid]

    results = {}
    for s_name in season_dict.keys():
        mask = (season_arr == s_name)
        n_pts = np.sum(mask)
        if n_pts < 5:
            results[s_name] = (np.nan, np.nan, np.nan, np.nan)
            continue

        k, b, k_unc, b_unc = mc_fit(
            cot_raw[mask], albedo_raw[mask],
            cot_std=spec['cot_std'],
            albedo_std=spec['albedo_std']
        )
        results[s_name] = (k, b, k_unc, b_unc)

    return results


def main():
    all_processed_ocean_data = preprocess_ocean_data(
        min_cot_mod08=2.5,
        min_ret_cot_cer=2.5
    )

    # Compute fits for each Method x Ocean x Season
    fit_records = []

    for method_key in method_keys:
        spec = method_specs[method_key]
        for ocean in oceans:
            od = all_processed_ocean_data[ocean]
            if od is None:
                for s_name in season_dict.keys():
                    fit_records.append({
                        'Method': method_key,
                        'Ocean': ocean,
                        'Season': s_name,
                        'Slope': np.nan,
                        'Intercept': np.nan,
                        'Slope_Unc': np.nan,
                        'Intercept_Unc': np.nan,
                    })
                continue

            fits = fit_ocean_season(od, method_key, spec)
            for s_name in season_dict.keys():
                k, b, k_unc, b_unc = fits[s_name]
                fit_records.append({
                    'Method': method_key,
                    'Ocean': ocean,
                    'Season': s_name,
                    'Slope': k,
                    'Intercept': b,
                    'Slope_Unc': k_unc,
                    'Intercept_Unc': b_unc,
                })

    # Save to CSV
    output_dir = '/home/chenyiqi/251028_albedo_cot/processed_data'
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, 'coef_ocean_season.csv')

    output_df = pd.DataFrame(fit_records)
    output_df.to_csv(output_csv_path, index=False)
    print(f"\nCoefficients saved to: {output_csv_path}")
    print(f"Total records: {len(output_df)}")
    print(f"\nSummary of non-NaN fits:")
    summary = output_df.dropna(subset=['Slope']).groupby(['Method', 'Ocean']).size().unstack(fill_value=0)
    print(summary)


if __name__ == "__main__":
    main()
