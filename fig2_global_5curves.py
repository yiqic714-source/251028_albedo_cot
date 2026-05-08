import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from utils_fitting import (
    oceans, season_dict, cot_range, albedo_to_y,
    cot_to_x, mc_fit
)


BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIG_SAVE_PATH = f'{BASE_PATH}/figs/fig2_global_5curves.png'
os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)

MIN_COT = 2.5
MIN_CF = 0.1
N_COT_BINS = 15


def cot_to_albedo(cot, method, sza=None, table_folder='dcp', ocean=None, season=None):
    cot = np.asarray(cot, dtype=float)

    if method == 'sbdart':
        if ocean is not None and season is not None:
            file_name = f'cot_sza_to_albedo_lookup_table_{ocean}_{season}.csv'
        else:
            file_name = 'cot_sza_to_albedo_lookup_table_TPO_MAM.csv'

        file_path = (
            f'{BASE_PATH}/build_sbdart_lookup_table/'
            f'cot_sza_to_albedo_lookup_table_{table_folder}/'
            f'{file_name}'
        )

        if not os.path.exists(file_path):
            return np.full(cot.shape, np.nan)

        df = pd.read_csv(file_path, index_col=0)
        sza_grid = np.array(df.index, dtype=float)
        cot_grid = np.array(df.columns, dtype=float)
        albedo_grid = df.values

        sza_mesh, cot_mesh = np.meshgrid(sza_grid, cot_grid, indexing='ij')
        points = np.column_stack([sza_mesh.ravel(), cot_mesh.ravel()])
        values = albedo_grid.ravel()

        valid = np.isfinite(values)

        cot_arr = np.atleast_1d(cot)
        if np.ndim(sza) == 0:
            sza_arr = np.full_like(cot_arr, sza, dtype=float)
        else:
            sza_arr = np.asarray(sza, dtype=float)

        target = np.column_stack([sza_arr, cot_arr])

        albedo = griddata(
            points[valid],
            values[valid],
            target,
            method='linear',
            fill_value=np.nan
        )

        return albedo.reshape(cot_arr.shape)

    if method == 'l74':
        g = 0.85
        b = 1 - g
        return b * cot / (1 + b * cot)

    if method == 'quadrature':
        g = 0.85
        mu = np.cos(np.radians(sza))
        b = np.sqrt(3) / 2 * (1 - g)
        return (
            b * cot + (1 / 2 - np.sqrt(3) / 2 * mu) * (1 - np.exp(-cot / mu))
        ) / (1 + b * cot)

    if method == 'eddington':
        g = 0.85
        mu = np.cos(np.radians(sza))
        return (
            (1 - g) * cot + (2 / 3 - mu) * (1 - np.exp(-cot / mu))
        ) / (4 / 3 + (1 - g) * cot)

    raise ValueError(f'Unsupported method: {method}')


def load_global_data():
    """Load and merge all ocean-season data, apply filters."""
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

    # Compute derived quantities
    df['albedo'] = (
        (df['sw_all'] - df['sw_clr'] * (1 - df['cf_ceres'])) /
        df['cf_ceres'] / df['solar_incoming']
    )

    # Apply filters
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
    """Bin data by COT and compute mean/std for each bin."""
    labels = pd.cut(df[cot_col], bins=bin_edges, labels=False, include_lowest=True)
    bin_means_cot = []
    bin_means_alb = []
    bin_stds_alb = []
    bin_centers = []

    for i in range(len(bin_edges) - 1):
        mask = labels == i
        if mask.sum() < 5:
            continue
        cot_vals = df.loc[mask, cot_col].values
        alb_vals = df.loc[mask, albedo_col].values
        bin_means_cot.append(np.mean(cot_vals))
        bin_means_alb.append(np.mean(alb_vals))
        bin_stds_alb.append(np.std(alb_vals))
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

    return (np.array(bin_means_cot), np.array(bin_means_alb),
            np.array(bin_stds_alb), np.array(bin_centers))


def fit_k_b_in_logit_space(cot, albedo):
    """
    Fit in ln(COT) vs ln(Ac/(1-Ac)) space.
    Returns (k, b) where y = k*x + b, with x = ln(COT), y = ln(Ac/(1-Ac)).
    """
    x = cot_to_x(np.asarray(cot, dtype=float))
    y = albedo_to_y(np.asarray(albedo, dtype=float))
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    k, b = np.polyfit(x[mask], y[mask], 1)
    return k, b


def logit_fit_to_albedo(cot, k, b):
    """
    Convert fitted line back to albedo space.
    y = k*ln(COT) + b
    Ac = exp(y) / (1 + exp(y))
    """
    x = cot_to_x(np.asarray(cot, dtype=float))
    y = k * x + b
    return np.exp(y) / (1 + np.exp(y))


def compute_sbdart_albedo_per_point(df, table_folder):
    """Compute SBDART albedo for each data point using its own COT, SZA, ocean, season."""
    result = np.full(len(df), np.nan)
    for ocean in oceans:
        for season_name in season_dict:
            mask = (df['ocean'] == ocean) & (df['season'] == season_name)
            if mask.sum() == 0:
                continue
            cot_vals = df.loc[mask, 'ret_cot_cer'].values
            sza_vals = df.loc[mask, 'sza'].values
            alb = cot_to_albedo(
                cot_vals, 'sbdart', sza=sza_vals,
                table_folder=table_folder, ocean=ocean, season=season_name
            )
            result[mask.values] = alb
    return result


def main():
    print('Loading global data...')
    df = load_global_data()
    print(f'Total data points: {len(df)}')

    # Define COT bin edges using cot_range
    bin_edges = cot_range  # 15 bins

    # ---- Bin ret data ----
    print('Binning ret data...')
    ret_cot_bins, ret_alb_bins, ret_alb_std, _ = bin_data_by_cot(
        df, 'ret_cot_cer', 'ret_albedo', bin_edges
    )

    # ---- Bin msk data ----
    print('Binning msk data...')
    msk_cot_bins, msk_alb_bins, msk_alb_std, _ = bin_data_by_cot(
        df, 'cot_mod08', 'albedo', bin_edges
    )

    # ---- Compute theoretical curves ----
    print('Computing theoretical curves...')

    # T91: quadrature with SZA=54.5
    alb_t91 = cot_to_albedo(cot_range, 'quadrature', sza=54.5)

    # dcp: decoupled LUT on cot_range with SZA=54.5
    alb_dcp = cot_to_albedo(cot_range, 'sbdart', sza=54.5, table_folder='dcp')

    # ---- Compute cp per data point ----
    print('Computing cp albedo per data point...')
    df['cp_albedo'] = compute_sbdart_albedo_per_point(df, 'cp')

    # ---- Bin cp data ----
    print('Binning cp data...')
    cp_cot_bins, cp_alb_bins, cp_alb_std, _ = bin_data_by_cot(
        df, 'ret_cot_cer', 'cp_albedo', bin_edges
    )

    # ---- Fit k values in logit space ----
    print('Fitting k values...')

    # T91: fit on theoretical curve
    k_t91, b_t91 = fit_k_b_in_logit_space(cot_range, alb_t91)
    alb_t91_fit = logit_fit_to_albedo(cot_range, k_t91, b_t91)

    # dcp: fit on cot_range curve
    k_dcp, b_dcp = fit_k_b_in_logit_space(cot_range, alb_dcp)
    alb_dcp_fit = logit_fit_to_albedo(cot_range, k_dcp, b_dcp)

    # cp: fit on all data points
    k_cp, b_cp = fit_k_b_in_logit_space(
        df['ret_cot_cer'].values, df['cp_albedo'].values
    )
    alb_cp_fit = logit_fit_to_albedo(cot_range, k_cp, b_cp)

    # ret: fit on all data points
    k_ret, b_ret = fit_k_b_in_logit_space(
        df['ret_cot_cer'].values, df['ret_albedo'].values
    )
    alb_ret_fit = logit_fit_to_albedo(cot_range, k_ret, b_ret)

    # msk: fit on all data points
    k_msk, b_msk = fit_k_b_in_logit_space(
        df['cot_mod08'].values, df['albedo'].values
    )
    alb_msk_fit = logit_fit_to_albedo(cot_range, k_msk, b_msk)

    # ---- Plot ----
    print('Plotting...')
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)

    # Use a built-in colormap for consistent colors
    colors = plt.cm.tab10(np.linspace(0, 1, 5))

    # Solid line labels (no k value)
    solid_labels = [
        'T91',
        'Decoupled SBDART',
        'Coupled SBDART',
        'Retrieval-Domain Obs.',
        'Mask-Domain Obs.',
    ]

    # Dashed line labels (with k value)
    k_labels = [
        f'$k_\\mathrm{{T91}}$={k_t91:.2f}',
        f'$k_\\mathrm{{dcp}}$={k_dcp:.2f}',
        f'$k_\\mathrm{{cp}}$={k_cp:.2f}',
        f'$k_\\mathrm{{ret}}$={k_ret:.2f}',
        f'$k_\\mathrm{{msk}}$={k_msk:.2f}',
    ]

    # T91: solid = theoretical, dashed = fit
    ax.plot(cot_range, alb_t91, color=colors[0], lw=2, linestyle='-',
            label=solid_labels[0])
    ax.plot(cot_range, alb_t91_fit, color=colors[0], lw=1.5, linestyle='--',
            alpha=0.7, label=k_labels[0])

    # dcp: solid = theoretical on cot_range, dashed = fit
    ax.plot(cot_range, alb_dcp, color=colors[1], lw=2, linestyle='-',
            label=solid_labels[1])
    ax.plot(cot_range, alb_dcp_fit, color=colors[1], lw=1.5, linestyle='--',
            alpha=0.7, label=k_labels[1])

    # cp: solid = binned, dashed = fit
    ax.errorbar(cp_cot_bins, cp_alb_bins, yerr=cp_alb_std,
                color=colors[2], fmt='o-', lw=2, capsize=3, capthick=1,
                label=solid_labels[2])
    ax.plot(cot_range, alb_cp_fit, color=colors[2], lw=1.5, linestyle='--',
            alpha=0.7, label=k_labels[2])

    # ret: solid = binned, dashed = fit
    ax.errorbar(ret_cot_bins, ret_alb_bins, yerr=ret_alb_std,
                color=colors[3], fmt='o-', lw=2, capsize=3, capthick=1,
                label=solid_labels[3])
    ax.plot(cot_range, alb_ret_fit, color=colors[3], lw=1.5, linestyle='--',
            alpha=0.7, label=k_labels[3])

    # msk: solid = binned, dashed = fit
    ax.errorbar(msk_cot_bins, msk_alb_bins, yerr=msk_alb_std,
                color=colors[4], fmt='s-', lw=2, capsize=3, capthick=1,
                label=solid_labels[4])
    ax.plot(cot_range, alb_msk_fit, color=colors[4], lw=1.5, linestyle='--',
            alpha=0.7, label=k_labels[4])

    ax.set_xlabel('COT', fontsize=15, fontweight='medium')
    ax.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=15, fontweight='medium')
    ax.tick_params(axis='both', labelsize=12)

    # Two-column legend: first column = solid lines, second column = dashed lines
    handles, labels = ax.get_legend_handles_labels()
    # Handles are ordered: T91_solid, T91_dash, dcp_solid, dcp_dash, ...
    # Reorder to: [T91_solid, dcp_solid, ..., T91_dash, dcp_dash, ...]
    solid_handles = handles[0::2]
    dash_handles = handles[1::2]
    solid_labels_out = labels[0::2]
    dash_labels_out = labels[1::2]
    ordered_handles = solid_handles + dash_handles
    ordered_labels = solid_labels_out + dash_labels_out
    ax.legend(ordered_handles, ordered_labels, loc='lower right', fontsize=11,
              framealpha=0.9, ncol=2)

    ax.set_xlim(0, cot_range.max() * 1.05)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(FIG_SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Figure saved to: {os.path.abspath(FIG_SAVE_PATH)}')


if __name__ == '__main__':
    main()
