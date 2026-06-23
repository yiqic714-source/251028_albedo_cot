# -*- coding: utf-8 -*-
"""
tablesupp_sensitivity_sbdart.py

Replicate the k_dcp/lnb_dcp/k_cp/lnb_cp computation from
fig2_fittings_global_and_reasons.py panel (a), but using different
SBDART lookup-table configurations.

Cases tested:
  cbh2, cth2, cth4, cer7, cer13
"""

import os
import numpy as np
import pandas as pd

from utils_fitting import (
    oceans, season_dict, cot_to_albedo, cot_to_x, albedo_to_y,
    mc_fit
)


BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
MIN_COT = 2.5
MIN_CF = 0.1

CASES = ["", "_cbh2", "_cth2", "_cth4", "_cer7", "_cer13"]


def load_global_data():
    """Load and filter merged data, same as fig2's load_global_data."""
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
        (df['cf_ceres'] > MIN_CF) &
        (df['cf_liq_ceres'] / df['cf_ceres'] > 0.99) &
        (df['cot_mod08'] > MIN_COT) &
        (df['ret_cot_cer'] > MIN_COT) &
        (df['ret_albedo'].between(0, 1)) &
        (df['albedo'].between(0, 1))
    )

    return df[mask].dropna()


def fit_k_b_in_logit_space(cot, albedo):
    x = cot_to_x(np.asarray(cot, dtype=float))
    y = albedo_to_y(np.asarray(albedo, dtype=float))

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan

    k, b = np.polyfit(x[mask], y[mask], 1)
    return k, b

def compute_sbdart_albedo_per_point(df, table_folder):
    """Per-point SBDART albedo using each point's own sza, ocean, and season.
    Same as fig2's compute_sbdart_albedo_per_point."""
    result = np.full(len(df), np.nan)

    for ocean in oceans:
        for season_name in season_dict:
            mask = (df['ocean'] == ocean) & (df['season'] == season_name)
            if mask.sum() == 0:
                continue

            result[mask.values] = cot_to_albedo(
                df.loc[mask, 'ret_cot_cer'].values,
                'sbdart',
                sza=df.loc[mask, 'sza'].values,
                table_folder=table_folder,
                ocean=ocean,
                season=season_name
            )

    return result


def run_once(case_name):
    """Compute k_dcp/lnb_dcp/k_cp/lnb_cp for one lookup-table case."""
    print(f'\n=== {case_name} ===')

    df = load_global_data()
    print(f'  Data points: {len(df)}')

    if len(df) < 10:
        print('  Too few data points, skipping.')
        return

    # --- k_dcp, lnb_dcp (same as fig2: fixed sza=54.4, simple polyfit) ---
    alb_dcp = cot_to_albedo(
        df['ret_cot_cer'].values,
        'sbdart',
        sza=54.4,
        table_folder='dcp' + case_name,
    )
    k_dcp, lnb_dcp = fit_k_b_in_logit_space(df['ret_cot_cer'].values, alb_dcp)

    # --- k_cp, lnb_cp (same as fig2: per-point sza/ocean/season, mc_fit) ---
    alb_cp = compute_sbdart_albedo_per_point(df, 'cp' + case_name)
    k_cp, lnb_cp, _, _ = mc_fit(
        df['ret_cot_cer'].values,
        alb_cp,
        cot_std=0.0,
        albedo_std=0.03,
        n_mc=300,
        bootstrap=True
    )

    print(f'  k_dcp = {k_dcp:.2f},  lnb_dcp = {lnb_dcp:.2f}')
    print(f'  k_cp  = {k_cp:.2f},  lnb_cp  = {lnb_cp:.2f}')


def main():
    for case in CASES:
        run_once(case)


if __name__ == '__main__':
    main()
