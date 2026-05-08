import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_fitting import (
    oceans, season_dict, cot_range, albedo_to_y,
    cot_to_x, cot_to_albedo, mc_fit
)


BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIG_SAVE_PATH = f'{BASE_PATH}/figs/fig2_global_5curves.png'
os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)

MIN_COT = 2.5
MIN_CF = 0.1


def load_global_data():
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


def bin_data_by_cot(df, cot_col, albedo_col, bin_edges):
    labels = pd.cut(df[cot_col], bins=bin_edges, labels=False, include_lowest=True)

    bin_means_cot = []
    bin_means_alb = []
    bin_stds_alb = []

    for i in range(len(bin_edges) - 1):
        mask = labels == i
        if mask.sum() < 5:
            continue

        cot_vals = df.loc[mask, cot_col].values
        alb_vals = df.loc[mask, albedo_col].values

        bin_means_cot.append(np.mean(cot_vals))
        bin_means_alb.append(np.mean(alb_vals))
        bin_stds_alb.append(np.std(alb_vals))

    return np.array(bin_means_cot), np.array(bin_means_alb), np.array(bin_stds_alb)


def fit_k_b_in_logit_space(cot, albedo):
    x = cot_to_x(np.asarray(cot, dtype=float))
    y = albedo_to_y(np.asarray(albedo, dtype=float))

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan

    k, b = np.polyfit(x[mask], y[mask], 1)
    return k, b


def logit_fit_to_albedo(cot, k, b):
    x = cot_to_x(np.asarray(cot, dtype=float))
    y = k * x + b
    return np.exp(y) / (1 + np.exp(y))


def compute_sbdart_albedo_per_point(df, table_folder):
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


def main():
    print('Loading global data...')
    df = load_global_data()
    print(f'Total data points: {len(df)}')

    bin_edges = cot_range

    print('Binning observational and SBDART data...')
    ret_cot_bins, ret_alb_bins, ret_alb_std = bin_data_by_cot(
        df, 'ret_cot_cer', 'ret_albedo', bin_edges
    )

    msk_cot_bins, msk_alb_bins, msk_alb_std = bin_data_by_cot(
        df, 'cot_mod08', 'albedo', bin_edges
    )

    alb_t91 = cot_to_albedo(cot_range, 'quadrature', sza=54.4)
    alb_dcp = cot_to_albedo(df['ret_cot_cer'], 'sbdart', sza=54.4, table_folder='dcp')

    print('Computing coupled SBDART albedo...')
    df['cp_albedo'] = compute_sbdart_albedo_per_point(df, 'cp')

    cp_cot_bins, cp_alb_bins, cp_alb_std = bin_data_by_cot(
        df, 'ret_cot_cer', 'cp_albedo', bin_edges
    )

    print('Fitting k values...')
    k_t91, b_t91 = fit_k_b_in_logit_space(cot_range, alb_t91)
    k_dcp, b_dcp = fit_k_b_in_logit_space(df['ret_cot_cer'], alb_dcp)

    k_cp, b_cp, _, _ = mc_fit(
        df['ret_cot_cer'].values,
        df['cp_albedo'].values,
        cot_std=0.0,
        albedo_std=0.03,
        n_mc=300,
        bootstrap=True
    )

    k_ret, b_ret, _, _ = mc_fit(
        df['ret_cot_cer'].values,
        df['ret_albedo'].values,
        cot_std=0.10,
        albedo_std=0.13,
        n_mc=300,
        bootstrap=True
    )

    k_msk, b_msk, _, _ = mc_fit(
        df['cot_mod08'].values,
        df['albedo'].values,
        cot_std=0.10,
        albedo_std=0.20,
        n_mc=300,
        bootstrap=True
    )

    alb_t91_fit = logit_fit_to_albedo(cot_range, k_t91, b_t91)
    alb_dcp_fit = logit_fit_to_albedo(cot_range, k_dcp, b_dcp)
    alb_cp_fit = logit_fit_to_albedo(cot_range, k_cp, b_cp)
    alb_ret_fit = logit_fit_to_albedo(cot_range, k_ret, b_ret)
    alb_msk_fit = logit_fit_to_albedo(cot_range, k_msk, b_msk)

    print('Plotting...')
    fig, ax = plt.subplots(figsize=(6.8, 5.5), dpi=300)

    colors = plt.cm.tab10(np.linspace(0, 1, 5))

    solid_labels = [
        'T91',
        'Decoupled SBDART',
        'Coupled SBDART',
        'Retrieval-Domain Obs.',
        'Mask-Domain Obs.'
    ]

    dashed_labels = [
        rf'$k_{{\mathrm{{T91}}}}$={k_t91:.2f}',
        rf'$k_{{\mathrm{{dcp}}}}$={k_dcp:.2f}',
        rf'$k_{{\mathrm{{cp}}}}$={k_cp:.2f}',
        rf'$k_{{\mathrm{{ret}}}}$={k_ret:.2f}',
        rf'$k_{{\mathrm{{msk}}}}$={k_msk:.2f}'
    ]

    solid_handles = []
    dashed_handles = []

    h, = ax.plot(cot_range, alb_t91, color=colors[0], lw=2, ls='-')
    solid_handles.append(h)
    h, = ax.plot(cot_range, alb_t91_fit, color=colors[0], lw=1.5, ls='--', alpha=0.7)
    dashed_handles.append(h)

    sorted_idx = np.argsort(df['ret_cot_cer'])
    h, = ax.plot(df['ret_cot_cer'].values[sorted_idx], alb_dcp[sorted_idx], color=colors[1], lw=2, ls='-')
    solid_handles.append(h)
    h, = ax.plot(cot_range, alb_dcp_fit, color=colors[1], lw=1.5, ls='--', alpha=0.7)
    dashed_handles.append(h)

    h = ax.errorbar(
        cp_cot_bins, cp_alb_bins, yerr=cp_alb_std,
        color=colors[2], fmt='o-', lw=2, capsize=3, capthick=1
    )
    solid_handles.append(h)
    h, = ax.plot(cot_range, alb_cp_fit, color=colors[2], lw=1.5, ls='--', alpha=0.7)
    dashed_handles.append(h)

    h = ax.errorbar(
        ret_cot_bins, ret_alb_bins, yerr=ret_alb_std,
        color=colors[3], fmt='o-', lw=2, capsize=3, capthick=1
    )
    solid_handles.append(h)
    h, = ax.plot(cot_range, alb_ret_fit, color=colors[3], lw=1.5, ls='--', alpha=0.7)
    dashed_handles.append(h)

    h = ax.errorbar(
        msk_cot_bins, msk_alb_bins, yerr=msk_alb_std,
        color=colors[4], fmt='s-', lw=2, capsize=3, capthick=1
    )
    solid_handles.append(h)
    h, = ax.plot(cot_range, alb_msk_fit, color=colors[4], lw=1.5, ls='--', alpha=0.7)
    dashed_handles.append(h)

    ax.set_xlabel('COT', fontsize=15, fontweight='medium')
    ax.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=15, fontweight='medium')
    ax.tick_params(axis='both', labelsize=12)

    handles = solid_handles + dashed_handles
    labels = solid_labels + dashed_labels

    ax.legend(
        handles, labels,
        loc='lower right',
        fontsize=10.5,
        framealpha=0.9,
        ncol=2,
        columnspacing=1.4,
        handlelength=2.6,
        handletextpad=0.6
    )

    ax.set_xlim(0, cot_range.max() * 1.05)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(FIG_SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Figure saved to: {os.path.abspath(FIG_SAVE_PATH)}')


if __name__ == '__main__':
    main()