"""
SZA adjustment method 2:
For each ocean-season, compute the ratio albedo_daytime / albedo_1030 from the cp lookup table
(at each cot value), then multiply observed albedo by this ratio before fitting k and lnb
using the same method as 5_cal_k_b_by_ocean_season.py.
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from utils_fitting import oceans, season_dict, mc_fit

# Paths
BASE_DATA_DIR = '/home/chenyiqi/251028_albedo_cot/processed_data/merged_data/'
WEIGHTED_FILE = '/home/chenyiqi/251028_albedo_cot/processed_data/ocean_season_sza_weighted.csv'
CP_LOOKUP_DIR = '/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table/cot_sza_to_albedo_lookup_table_cp/'
OUTPUT_CSV = '/home/chenyiqi/251028_albedo_cot/processed_data/coef_k_b_szacorr_method2.csv'

MIN_COT = 2.5
MIN_CF = 0.1

# Method specs (same as 5_cal_k_b_by_ocean_season.py)
method_specs = {
    'ret': {'cot_std': 0.1, 'albedo_std': 0.13},
    'msk': {'cot_std': 0.1, 'albedo_std': 0.20},
}
method_keys = ['ret', 'msk']


def load_weighted_angles(file_path):
    """Load weighted SZA data. Return dict: {(ocean, season): weighted_angle_deg}"""
    df = pd.read_csv(file_path)
    df = df[~df['season'].isin(['Global'])]
    return {(row['ocean'], row['season']): row['weighted_angle_deg'] for _, row in df.iterrows()}


def get_albedo_ratio(ocean, season, sza_1030, sza_daytime):
    """
    From the cp lookup table, for each cot value, compute:
        ratio = albedo(at sza_daytime) / albedo(at sza_1030)
    Returns (cot_values, ratio_values) where both are 1D arrays.
    """
    file_path = os.path.join(CP_LOOKUP_DIR, f'cot_sza_to_albedo_lookup_table_{ocean}_{season}.csv')
    if not os.path.exists(file_path):
        return None, None

    df = pd.read_csv(file_path, index_col=0)
    sza_grid = np.array(df.index.astype(float))
    cot_grid = np.array(df.columns.astype(float))
    albedo_grid = df.values.astype(float)

    # Find closest SZA indices
    idx_1030 = np.argmin(np.abs(sza_grid - sza_1030))
    idx_daytime = np.argmin(np.abs(sza_grid - sza_daytime))

    albedo_1030 = albedo_grid[idx_1030, :]
    albedo_daytime = albedo_grid[idx_daytime, :]

    # Avoid division by zero
    valid = albedo_1030 > 0
    ratio = np.full_like(albedo_1030, np.nan)
    ratio[valid] = albedo_daytime[valid] / albedo_1030[valid]

    return cot_grid, ratio


def preprocess_ocean_data(weight_dict):
    """
    Preprocess data for all ocean regions, applying SZA adjustment.
    Returns a dict: ocean_name -> processed_data dict (with adjusted albedo)
    """
    all_processed = {}

    print("Starting one-time preprocessing with SZA adjustment (method 2)...")

    for ocean in oceans:
        # Read all season files for this ocean
        dfs = []
        for season_name in season_dict.keys():
            file_path = os.path.join(BASE_DATA_DIR, f"{ocean}_{season_name}.csv")
            if not os.path.exists(file_path):
                continue
            df_season = pd.read_csv(file_path)
            df_season['season'] = season_name
            dfs.append(df_season)

        if not dfs:
            print(f"{ocean} has no data files, skipping.")
            all_processed[ocean] = None
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
                all_processed[ocean] = None
                continue

            ret_cot = df_filtered['ret_cot_cer'].values
            ret_albedo_obs = df_filtered['ret_albedo'].values
            msk_cot = df_filtered['cot_mod08'].values
            msk_albedo = df_filtered['albedo'].values
            season_arr = df_filtered['season'].values
            sza_arr = df_filtered['sza'].values

            # Apply SZA adjustment per season
            ret_albedo_adj = ret_albedo_obs.copy()
            msk_albedo_adj = msk_albedo.copy()

            for season_name in season_dict.keys():
                s_mask = (season_arr == season_name)
                if not np.any(s_mask):
                    continue

                # Get 10:30 mean SZA and daytime weighted SZA for this ocean-season
                sza_1030 = np.mean(sza_arr[s_mask])
                key = (ocean, season_name)
                sza_daytime = weight_dict.get(key, np.nan)

                if np.isnan(sza_daytime):
                    continue

                # Get ratio from cp lookup table
                cot_ratio, ratio_vals = get_albedo_ratio(ocean, season_name, sza_1030, sza_daytime)
                if cot_ratio is None or ratio_vals is None:
                    continue

                # Interpolate ratio to each observation's cot value
                valid_ratio = np.isfinite(ratio_vals)
                if not np.any(valid_ratio):
                    continue

                # For ret: interpolate ratio at ret_cot
                ret_cot_s = ret_cot[s_mask]
                ratio_interp_ret = np.interp(ret_cot_s, cot_ratio[valid_ratio], ratio_vals[valid_ratio],
                                             left=np.nan, right=np.nan)
                ret_albedo_adj[s_mask] = ret_albedo_obs[s_mask] * ratio_interp_ret

                # For msk: interpolate ratio at msk_cot
                msk_cot_s = msk_cot[s_mask]
                ratio_interp_msk = np.interp(msk_cot_s, cot_ratio[valid_ratio], ratio_vals[valid_ratio],
                                             left=np.nan, right=np.nan)
                msk_albedo_adj[s_mask] = msk_albedo[s_mask] * ratio_interp_msk

            # Clip adjusted albedo to valid range
            ret_albedo_adj = np.clip(ret_albedo_adj, 1e-6, 1 - 1e-6)
            msk_albedo_adj = np.clip(msk_albedo_adj, 1e-6, 1 - 1e-6)

            ocean_processed = {
                'ret_cot': ret_cot,
                'ret_albedo_adj': ret_albedo_adj,
                'msk_cot': msk_cot,
                'msk_albedo_adj': msk_albedo_adj,
                'season': season_arr,
                'data_count': len(df_filtered),
            }
            all_processed[ocean] = ocean_processed
            print(f"{ocean} preprocessing completed, valid data count: {len(df_filtered)}")

        except Exception as e:
            print(f"Error processing {ocean}: {e}")
            import traceback
            traceback.print_exc()
            all_processed[ocean] = None

    return all_processed


def fit_ocean_season(ocean_data, method_key, spec):
    """
    Fit one ocean x season combination for a given method using adjusted albedo.
    Returns dict: season_name -> (k, b, k_unc, b_unc)
    """
    if method_key == 'msk':
        cot_raw = ocean_data['msk_cot']
        albedo_raw = ocean_data['msk_albedo_adj']
    else:
        cot_raw = ocean_data['ret_cot']
        albedo_raw = ocean_data['ret_albedo_adj']

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
    weight_dict = load_weighted_angles(WEIGHTED_FILE)
    all_processed = preprocess_ocean_data(weight_dict)

    fit_records = []

    for method_key in method_keys:
        spec = method_specs[method_key]
        for ocean in oceans:
            od = all_processed[ocean]
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

    output_dir = os.path.dirname(OUTPUT_CSV)
    os.makedirs(output_dir, exist_ok=True)

    output_df = pd.DataFrame(fit_records)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSZA-corrected (method 2) coefficients saved to: {OUTPUT_CSV}")
    print(f"Total records: {len(output_df)}")
    print(f"\nSummary of non-NaN fits:")
    summary = output_df.dropna(subset=['Slope']).groupby(['Method', 'Ocean']).size().unstack(fill_value=0)
    print(summary)


if __name__ == "__main__":
    main()
