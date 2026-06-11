import os
import numpy as np
import pandas as pd

from utils_fitting import (
    oceans, season_dict,
    cot_to_albedo, mc_fit
)

input_dir = "/home/chenyiqi/251028_albedo_cot/processed_data/merged_data/"
MIN_COT = 2.5
MIN_CF = 0.1

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

    output_df = pd.DataFrame(fit_records).round(4)

    # Save to CSV
    output_dir = '/home/chenyiqi/251028_albedo_cot/processed_data'
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, 'sensitivity_albedo_vs_cot_1030.csv')
    output_df.to_csv(output_csv_path, index=False)
    print(f"\nMerged coefficients saved to: {output_csv_path}")


if __name__ == "__main__":
    main()
