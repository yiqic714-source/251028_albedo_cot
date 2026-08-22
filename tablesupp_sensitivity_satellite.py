# -*- coding: utf-8 -*-
"""
tablesupp_sensitivity_satellite.py

Replicate the k_ret/lnb_ret/k_msk/lnb_msk computation from
fig2_fittings_global_and_reasons.py panel (a), but with alternative
filtering conditions instead of the default (MIN_COT=2.5, MIN_CF=0.1).

Conditions tested:
  1. cot_gt_4:         min_cot_mod08=4.0, min_ret_cot_cer=4.0, min_cf=0.1
  2. cot_gt_0.5:       min_cot_mod08=0.5, min_ret_cot_cer=0.5, min_cf=0.1
  3. cf_liq_gt_0.25:   min_cot_mod08=2.5, min_ret_cot_cer=2.5, min_cf=0.25
  4. cf_liq_gt_0.05:   min_cot_mod08=2.5, min_ret_cot_cer=2.5, min_cf=0.05
"""

import os
import numpy as np
import pandas as pd

from utils_fitting import (
    oceans, season_dict, mc_fit
)


BASE_PATH = '/home/chenyiqi/251028_albedo_cot'


def load_global_data(min_cot_mod08=2.5, min_ret_cot_cer=2.5, min_cf=0.1):
    """Load and filter merged data, replicating fig2's load_global_data logic."""
    dfs = []
    for ocean in oceans:
        for season_name in season_dict:
            file_path = f'{BASE_PATH}/processed_data/merged_data/{ocean}_{season_name}.csv'
            if not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)
            df['season'] = season_name
            df['ocean'] = ocean
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError('No data files found.')

    df = pd.concat(dfs, ignore_index=True)

    df['albedo'] = (
        (df['sw_all'] - df['sw_clr'] * (1 - df['cf_ceres'])) /
        df['cf_ceres'] / df['solar_incoming']
    )

    mask = (
        (df['cf_ceres'] > min_cf) &
        (df['cf_liq_ceres'] / df['cf_ceres'] > 0.99) &
        (df['cot_mod08'] > min_cot_mod08) &
        (df['ret_cot_cer'] > min_ret_cot_cer) &
        (df['ret_albedo'].between(0, 1)) &
        (df['albedo'].between(0, 1))
    )

    return df[mask].dropna()


def run_once(pass_name, min_cot_mod08=2.5, min_ret_cot_cer=2.5, min_cf=0.1):
    """Load data with given filters and compute k_ret/lnb_ret/k_msk/lnb_msk."""
    print(f'\n=== {pass_name} ===')
    print(f'  Filters: min_cot_mod08={min_cot_mod08}, '
          f'min_ret_cot_cer={min_ret_cot_cer}, min_cf={min_cf}')

    df = load_global_data(
        min_cot_mod08=min_cot_mod08,
        min_ret_cot_cer=min_ret_cot_cer,
        min_cf=min_cf,
    )
    n_data = len(df)
    print(f'  Data points after filtering: {n_data}')

    result = {
        'condition': pass_name,
        'min_cot_mod08': min_cot_mod08,
        'min_ret_cot_cer': min_ret_cot_cer,
        'min_cf': min_cf,
        'n_data': n_data,
        'k_ret': np.nan,
        'lnb_ret': np.nan,
        'k_msk': np.nan,
        'lnb_msk': np.nan,
    }

    if n_data < 10:
        print('  Too few data points, skipping.')
        return result

    # --- k_ret, lnb_ret (same as fig2 panel a) ---
    k_ret, lnb_ret, _, _ = mc_fit(
        df['ret_cot_cer'].values,
        df['ret_albedo'].values,
        cot_std=0.10,
        albedo_std=0.13,
        n_mc=300,
        bootstrap=True
    )

    # --- k_msk, lnb_msk (same as fig2 panel a) ---
    k_msk, lnb_msk, _, _ = mc_fit(
        df['cot_mod08'].values,
        df['albedo'].values,
        cot_std=0.10,
        albedo_std=0.20,
        n_mc=300,
        bootstrap=True
    )

    result.update({
        'k_ret': k_ret,
        'lnb_ret': lnb_ret,
        'k_msk': k_msk,
        'lnb_msk': lnb_msk,
    })

    print(f'  k_ret  = {k_ret:.2f},  lnb_ret  = {lnb_ret:.2f}')
    print(f'  k_msk  = {k_msk:.2f},  lnb_msk  = {lnb_msk:.2f}')

    return result


def main():
    experiments = [
        {
            'pass_name': 'Default',
            'min_cot_mod08': 2.5,
            'min_ret_cot_cer': 2.5,
            'min_cf': 0.1,
        },
        {
            'pass_name': 'cot_gt_4',
            'min_cot_mod08': 4.0,
            'min_ret_cot_cer': 4.0,
            'min_cf': 0.1,
        },
        {
            'pass_name': 'cot_gt_0.5',
            'min_cot_mod08': 0.5,
            'min_ret_cot_cer': 0.5,
            'min_cf': 0.1,
        },
        {
            'pass_name': 'cf_liq_gt_0.25',
            'min_cot_mod08': 2.5,
            'min_ret_cot_cer': 2.5,
            'min_cf': 0.25,
        },
        {
            'pass_name': 'cf_liq_gt_0.05',
            'min_cot_mod08': 2.5,
            'min_ret_cot_cer': 2.5,
            'min_cf': 0.05,
        },
    ]

    results = [run_once(**experiment) for experiment in experiments]

    summary = pd.DataFrame(results)
    print('\n=== Summary: sample size and fitted parameters ===')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.2f}'))


if __name__ == '__main__':
    main()
