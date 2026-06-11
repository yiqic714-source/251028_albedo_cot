import os
import numpy as np
import pandas as pd

from utils_fitting import (
    oceans, season_dict,
    cot_to_albedo, mc_fit, cot_to_x, albedo_to_y, _fit_odr_once
)

input_dir = "/home/chenyiqi/251028_albedo_cot/processed_data/merged_data/"
MIN_COT = 2.5
MIN_CF = 0.1

# Paths for SZA correction
WEIGHTED_FILE = '/home/chenyiqi/251028_albedo_cot/processed_data/ocean_season_sza_weighted.csv'
HEATMAP_DATA_DIR = '/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table/cot_sza_to_albedo_lookup_table_cp/'

# Core parameters
SEASONS = ['MAM', 'JJA', 'SON', 'DJF']
SEASON_MONTHS = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}
LN_COT_LOW = 1.
LN_COT_HIGH = 3.
MIN_POINTS_FOR_FIT = 2

# Method specifications
method_specs = {
    'dcp': {'idx': 2, 'cot_std': 0.0,  'albedo_std': 0.03},
    'cp':  {'idx': 1, 'cot_std': 0.0,  'albedo_std': 0.03},
    'ret': {'idx': 0, 'cot_std': 0.1,  'albedo_std': 0.13},
    'msk': {'idx': None, 'cot_std': 0.1, 'albedo_std': 0.20},
}
method_keys = ['dcp', 'cp', 'ret', 'msk']


def preprocess_ocean_data():
    """
    Preprocess data for all ocean regions.
    Returns a dict: ocean_name -> processed_data dict
    """
    all_processed_ocean_data = {}

    print("Starting one-time preprocessing for all ocean data...")

    for ocean in oceans:
        # Read all season files for this ocean and concatenate
        dfs = []
        for season_name in season_dict.keys():
            file_path = os.path.join(input_dir, f"{ocean}_{season_name}.csv")
            if not os.path.exists(file_path):
                continue
            df_season = pd.read_csv(file_path)
            df_season['season'] = season_name
            dfs.append(df_season)

        if not dfs:
            print(f"{ocean} has no data files, skipping.")
            all_processed_ocean_data[ocean] = None
            continue

        df = pd.concat(dfs, ignore_index=True)

        try:
            df['albedo'] = (
                (df['sw_all'] - df['sw_clr'] * (1 - df['cf_liq_ceres'])) /
                df['cf_liq_ceres'] / df['solar_incoming']
            )

            mask = (
                (df['cf_ceres'] > MIN_CF) &
                (df['cf_liq_ceres'] / df['cf_ceres'] > 0.99) &
                (df['cot_mod08'] > MIN_COT) &
                (df['ret_cot_cer'] > MIN_COT) &
                (df['ret_albedo'].between(0, 1)) &
                (df['albedo'].between(0, 1))
            )
            df_filtered = df[mask].dropna().reset_index(drop=True)

            if len(df_filtered) == 0:
                print(f"{ocean} has no valid data, skipping.")
                all_processed_ocean_data[ocean] = None
                continue

            ret_cot = df_filtered['ret_cot_cer'].values
            ret_albedo_obs = df_filtered['ret_albedo'].values
            msk_cot = df_filtered['cot_mod08'].values
            msk_albedo = df_filtered['albedo'].values

            # Compute cp albedo per season.
            # Important: cot_to_albedo expects season to be a scalar season name here.
            # Passing the full season array can make cp_albedo become all NaN.
            albedo_cp = np.full(len(df_filtered), np.nan, dtype=float)
            for s_name in season_dict.keys():
                s_mask = (df_filtered['season'] == s_name)
                if s_mask.sum() == 0:
                    continue

                albedo_cp[s_mask.values] = cot_to_albedo(
                    ret_cot[s_mask.values], 'sbdart',
                    sza=df_filtered.loc[s_mask, 'sza'].values,
                    table_folder='cp',
                    ocean=ocean,
                    season=s_name
                )

            # Compute dcp albedo using shared cot_to_albedo (fixed SZA=54.7)
            albedo_dcp = cot_to_albedo(
                ret_cot, 'sbdart',
                sza=54.7,
                table_folder='dcp'
            )

            ocean_processed_data = {
                'ret_cot': ret_cot,
                'ret_albedo_list': [ret_albedo_obs, albedo_cp, albedo_dcp],
                'msk_cot': msk_cot,
                'msk_albedo': msk_albedo,
                'season': df_filtered['season'].values,
                'sza': df_filtered['sza'].values,
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


# ============================================================
# SZA correction functions (from 6_sza_adjust.py)
# ============================================================

def load_weighted_angles(file_path):
    """Load weighted SZA data. Return dict: {(ocean, season): weighted_angle_deg}"""
    df = pd.read_csv(file_path)
    df = df[~df['season'].isin(['Global'])]
    return {(row['ocean'], row['season']): row['weighted_angle_deg'] for _, row in df.iterrows()}


def calculate_seasonal_stats(ocean_list, data_dir):
    """Calculate mean SZA per ocean-season. Return dict: {(ocean, season): mean_sza}"""
    seasonal_stats = {}
    for ocean in ocean_list:
        dfs = []
        for season in SEASONS:
            fp = os.path.join(data_dir, f'{ocean}_{season}.csv')
            if os.path.exists(fp):
                dfs.append(pd.read_csv(fp))
        if not dfs:
            for season in SEASONS:
                seasonal_stats[(ocean, season)] = np.nan
            continue
        df = pd.concat(dfs, ignore_index=True)
        df['time'] = pd.to_datetime(df['time'], format='mixed')
        df['month'] = df['time'].dt.month
        for season_name, months in SEASON_MONTHS.items():
            df.loc[df['month'].isin(months), 'season'] = season_name
        seasonal_avg = df.dropna(subset=['season']).groupby('season')['sza'].mean()
        for season in SEASONS:
            seasonal_stats[(ocean, season)] = seasonal_avg.loc[season] if season in seasonal_avg.index else np.nan
    return seasonal_stats


def compute_k_and_intercept(cot_vals, albedo_vals):
    """Calculate slope (k) and intercept (lnb) for ln(A/(1-A)) vs ln(COT) in range [1.5, 3.0]."""
    mask = ((cot_vals > 0.0) & np.isfinite(cot_vals) & np.isfinite(albedo_vals) &
            (albedo_vals > 0.0) & (albedo_vals < 1.0))
    if np.sum(mask) < MIN_POINTS_FOR_FIT:
        return np.nan, np.nan
    x = cot_to_x(cot_vals[mask])
    y = albedo_to_y(albedo_vals[mask])
    range_mask = (x >= LN_COT_LOW) & (x <= LN_COT_HIGH)
    if np.sum(range_mask) < MIN_POINTS_FOR_FIT:
        return np.nan, np.nan
    try:
        k, b = _fit_odr_once(x[range_mask], y[range_mask],
                              np.full(np.sum(range_mask), 1e-12),
                              np.full(np.sum(range_mask), 1e-12))
        return float(k), float(b)
    except Exception:
        return np.nan, np.nan


def get_lookup_data(ocean, season):
    """Load lookup table and return cos(SZA), slope (k), and intercept (lnb)."""
    file_path = os.path.join(HEATMAP_DATA_DIR, f'cot_sza_to_albedo_lookup_table_{ocean}_{season}.csv')
    df = pd.read_csv(file_path, index_col=0)
    sza = np.array(df.index.astype(float))
    cot = np.array(df.columns.astype(float))
    albedo_grid = df.values.astype(float)
    sort_sza_idx = np.argsort(sza)
    sza_sorted = sza[sort_sza_idx]
    cos_sza_sorted = np.cos(np.radians(sza_sorted))
    albedo_sorted = albedo_grid[sort_sza_idx, :]
    slope_list, intercept_list = [], []
    for i in range(len(sza_sorted)):
        slope, intercept = compute_k_and_intercept(cot, albedo_sorted[i, :])
        slope_list.append(slope)
        intercept_list.append(intercept)
    return cos_sza_sorted, np.array(slope_list), np.array(intercept_list)


def get_value_at_sza(cos_sza_vals, target_cos_sza, value_array):
    """Get the closest value from value_array at target cos(SZA)."""
    if cos_sza_vals is None or value_array is None:
        return np.nan
    valid_mask = np.isfinite(cos_sza_vals) & np.isfinite(value_array)
    if not np.any(valid_mask):
        return np.nan
    cos_sza_valid = cos_sza_vals[valid_mask]
    value_valid = value_array[valid_mask]
    closest_idx = np.argmin(np.abs(cos_sza_valid - target_cos_sza))
    return value_valid[closest_idx]


def compute_sza_correction(seasonal_stats, weight_dict, all_processed_ocean_data):
    """
    Compute the SZA correction (daytime-mean minus 10:30) for k and lnb
    from the SBDART lookup table.

    Returns:
        diff_k, diff_b: DataFrames (index=Ocean, columns=Season) with SZA corrections (additive)
        ratio_k, ratio_b: DataFrames (index=Ocean, columns=Season) with SZA ratios (multiplicative)
    """
    diff_k_data = pd.DataFrame(index=oceans, columns=SEASONS, dtype=float)
    diff_b_data = pd.DataFrame(index=oceans, columns=SEASONS, dtype=float)
    ratio_k_data = pd.DataFrame(index=oceans, columns=SEASONS, dtype=float)
    ratio_b_data = pd.DataFrame(index=oceans, columns=SEASONS, dtype=float)

    for ocean in oceans:
        od = all_processed_ocean_data[ocean]
        if od is None:
            continue

        for season in SEASONS:
            mean_sza_deg = seasonal_stats[(ocean, season)]
            weighted_sza_deg = weight_dict.get((ocean, season), np.nan)
            cos_sza, slope_vals, intercept_vals = get_lookup_data(ocean, season)

            if not np.isnan(mean_sza_deg) and not np.isnan(weighted_sza_deg):
                cos_mean_sza = np.cos(np.radians(mean_sza_deg))
                cos_weighted_sza = np.cos(np.radians(weighted_sza_deg))
                k_mean = get_value_at_sza(cos_sza, cos_mean_sza, slope_vals)
                k_weighted = get_value_at_sza(cos_sza, cos_weighted_sza, slope_vals)
                b_mean = get_value_at_sza(cos_sza, cos_mean_sza, intercept_vals)
                b_weighted = get_value_at_sza(cos_sza, cos_weighted_sza, intercept_vals)
                if np.isfinite(k_weighted) and np.isfinite(k_mean):
                    diff_k_data.loc[ocean, season] = k_weighted - k_mean
                    ratio_k_data.loc[ocean, season] = k_weighted / k_mean
                if np.isfinite(b_weighted) and np.isfinite(b_mean):
                    diff_b_data.loc[ocean, season] = b_weighted - b_mean
                    ratio_b_data.loc[ocean, season] = b_weighted / b_mean

    return diff_k_data, diff_b_data, ratio_k_data, ratio_b_data


def main():
    # Step 1: Preprocess data and compute 10:30 coefficients
    all_processed_ocean_data = preprocess_ocean_data()

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

    coef_df = pd.DataFrame(fit_records)

    # Step 2: Compute SZA correction only for methods that need daytime-mean coefficients.
    # cp and dcp will keep only the 10:30 coefficients in the output table.
    print("\nComputing SZA correction for ret/msk daytime-mean coefficients...")
    seasonal_stats = calculate_seasonal_stats(oceans, input_dir)
    weight_dict = load_weighted_angles(WEIGHTED_FILE)
    diff_k, diff_b, ratio_k, ratio_b = \
        compute_sza_correction(seasonal_stats, weight_dict, all_processed_ocean_data)

    # Switch: True = use multiplicative ratio, False = use additive difference
    USE_RATIO = False

    # Step 3: Build output CSV.
    # ret and msk: write both 10:30 and daytime-mean coefficients.
    # cp and dcp: write only 10:30 coefficients; daytime columns remain NaN.
    output_records = []
    for method in method_keys:
        mdf = coef_df[coef_df['Method'] == method]
        for _, row in mdf.iterrows():
            ocean = row['Ocean']
            season = row['Season']
            k_1030 = row['Slope']
            b_1030 = row['Intercept']
            k_1030_unc = row['Slope_Unc']
            b_1030_unc = row['Intercept_Unc']

            if method in ['cp', 'dcp']:
                k_daytime = np.nan
                b_daytime = np.nan
                k_daytime_unc = np.nan
                b_daytime_unc = np.nan

            else:
                if USE_RATIO:
                    # Multiplicative correction: multiply by ratio
                    rk = ratio_k.loc[ocean, season] if np.isfinite(ratio_k.loc[ocean, season]) else 1
                    rb = ratio_b.loc[ocean, season] if np.isfinite(ratio_b.loc[ocean, season]) else 1
                    k_daytime = k_1030 * rk if np.isfinite(k_1030) else np.nan
                    b_daytime = b_1030 * rb if np.isfinite(b_1030) else np.nan
                    # Uncertainty also multiplied by ratio
                    k_daytime_unc = k_1030_unc * rk if np.isfinite(k_daytime) else np.nan
                    b_daytime_unc = b_1030_unc * rb if np.isfinite(b_daytime) else np.nan
                else:
                    # Additive correction: add difference
                    dk = diff_k.loc[ocean, season] if np.isfinite(diff_k.loc[ocean, season]) else 0
                    db = diff_b.loc[ocean, season] if np.isfinite(diff_b.loc[ocean, season]) else 0
                    k_daytime = k_1030 + dk if np.isfinite(k_1030) else np.nan
                    b_daytime = b_1030 + db if np.isfinite(b_1030) else np.nan
                    # Uncertainty unchanged (dk/db from lookup table without uncertainty)
                    k_daytime_unc = k_1030_unc if np.isfinite(k_daytime) else np.nan
                    b_daytime_unc = b_1030_unc if np.isfinite(b_daytime) else np.nan

            output_records.append({
                'Method': method,
                'Ocean': ocean,
                'Season': season,
                'Slope_1030': k_1030,
                'Intercept_1030': b_1030,
                'Slope_1030_Unc': k_1030_unc,
                'Intercept_1030_Unc': b_1030_unc,
                'Slope_Daytime': k_daytime,
                'Intercept_Daytime': b_daytime,
                'Slope_Daytime_Unc': k_daytime_unc,
                'Intercept_Daytime_Unc': b_daytime_unc,
            })

    output_df = pd.DataFrame(output_records).round(4)

    # Save to CSV
    output_dir = '/home/chenyiqi/251028_albedo_cot/processed_data'
    os.makedirs(output_dir, exist_ok=True)
    if USE_RATIO:
        output_csv_path = os.path.join(output_dir, 'sensitivity_albedo_vs_cot_ratio.csv')
    else:
        output_csv_path = os.path.join(output_dir, 'sensitivity_albedo_vs_cot_diff.csv')
    output_df.to_csv(output_csv_path, index=False)
    print(f"\nMerged coefficients saved to: {output_csv_path}")
    print(f"Total records: {len(output_df)}")
    print(f"\nColumns: {list(output_df.columns)}")
    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()
