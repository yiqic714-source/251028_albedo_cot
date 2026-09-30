# -*- coding: utf-8 -*-
"""
figsupp_min_cf_impact.py

Impact of the cloud-fraction screening thresholds on the "Grid" k
(albedo-based COT-albedo fit), for the global ocean and for each ocean basin.

For every (cf_ceres, cf_ret_tot) threshold pair the L3 data are filtered with
the same conditions as fig1_fitting_8oceans.py (plus cf_ceres > x_thr and
cf_ret_tot > y_thr) and the resulting Grid k is used as the filled colour.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import mc_fit, format_panel_tag

BASE_DIR = Path(__file__).resolve().parent
L3_DIR = BASE_DIR / 'L3_product'
FIG_DIR = BASE_DIR / 'figs'

MIN_COT = 3
COT_EDGES = np.geomspace(MIN_COT, 76, 17)
MIN_GROUP_SIZE = 5

# threshold grid: 0 to 0.6 with a 0.05 step
THRESHOLDS = np.arange(0.05, 0.15 + 1e-9, 0.01)

LAYOUT = [['NPO', 'NAO', None], ['TPO', 'TAO', 'TIO'], ['SPO', 'SAO', 'SIO']]


def load_l3_data():
    """L3 data with the fig1 fixed screening conditions (variable thresholds
    for cf_ceres / cf_ret_tot are applied later)."""
    frames = []
    for path in sorted(L3_DIR.glob('*.csv')):
        parts = path.stem.rsplit('_', 1)
        if len(parts) != 2:
            continue
        data = pd.read_csv(path)
        data['ocean'] = parts[0]
        data['season'] = parts[1]
        frames.append(data)

    data = pd.concat(frames, ignore_index=True)
    data['albedo'] = ((data['sw_all'] - data['sw_clr'] * (1 - data['cf_ceres'])) /
                      data['cf_ceres'] / data['solar_incoming'])

    mask = (
        (data['cf_liq_ceres'] / data['cf_ceres'] > 0.99) &
        (data['cot'] > MIN_COT) &
        data['albedo'].between(0, 1) &
        (data['cttmin'] >= 270)
    )
    return data[mask].dropna(subset=['cot', 'albedo']).copy()


def grid_k(data):
    """Grid k: bin observed albedo in COT, then fit k to the binned points
    (same as fig1's Grid curve)."""
    labels = pd.cut(data['cot'], COT_EDGES, labels=False, include_lowest=True)
    cot_values, albedo_values = [], []
    for index in range(len(COT_EDGES) - 1):
        subset = data[labels == index]
        if len(subset) < MIN_GROUP_SIZE:
            continue
        cot_values.append(subset['cot'].mean())
        albedo_values.append(subset['albedo'].mean())

    if len(cot_values) < 3:
        return np.nan

    k, _ = mc_fit(
        np.asarray(cot_values), np.asarray(albedo_values),
        cot_std=0.10, albedo_std=0.20, n_mc=300, bootstrap=True,
        calculate_uncertainty=False,
    )[:2]
    return k


def compute_k_grid(data):
    """(cf_ret_tot threshold, cf_ceres threshold) grid of Grid k."""
    cf_ceres = data['cf_ceres'].to_numpy(dtype=float)
    cf_ret_tot = data['cf_ret_tot'].to_numpy(dtype=float)

    n = len(THRESHOLDS)
    k_grid = np.full((n, n), np.nan)

    for i, cf_ceres_thr in enumerate(THRESHOLDS):
        ceres_ok = cf_ceres > cf_ceres_thr
        for j, cf_ret_thr in enumerate(THRESHOLDS):
            k_grid[j, i] = grid_k(data[ceres_ok & (cf_ret_tot > cf_ret_thr)])

    return k_grid



def main():
    data = load_l3_data()
    print(f'Total footprints (fixed conditions): {len(data)}')

    grids = {}
    for ocean_row in LAYOUT:
        for ocean in ocean_row:
            if ocean is None:
                continue
            subset = data[data['ocean'] == ocean]
            print(f'Computing {ocean} ({len(subset)} footprints)...')
            grids[ocean] = compute_k_grid(subset)

    vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(3, 3, figsize=(12, 11),
                             sharex=True, sharey=True, squeeze=False)

    mesh = None
    panel_index = 0
    for row, ocean_row in enumerate(LAYOUT):
        for column, ocean in enumerate(ocean_row):
            ax = axes[row, column]
            if ocean is None:
                ax.axis('off')
                continue
            mesh = ax.contourf(
                THRESHOLDS, THRESHOLDS, grids[ocean],
                levels=np.linspace(vmin, vmax, 21), cmap='viridis',
                vmin=vmin, vmax=vmax,
            )
            ax.set_title(ocean)
            ax.set_aspect('equal')
            # mark the point (0.1, 0.1)
            ax.plot(0.1, 0.1, marker='o', ms=4, color='k',
                    markeredgecolor='white', markeredgewidth=0.6, zorder=5)
            ax.text(-0.03, 1.02, format_panel_tag(panel_index, 'science'),
                    transform=ax.transAxes, fontsize=12, va='bottom', ha='left')
            panel_index += 1
            if row == 2:
                ax.set_xlabel('Lower Bound of CF')
            if column == 0:
                ax.set_ylabel('Lower Bound of CRF')

    # colorbar on the right side of the whole figure
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(r'Grid $k$')

    FIG_DIR.mkdir(exist_ok=True)
    out_path = FIG_DIR / 'figsupp_min_cf_impact.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
