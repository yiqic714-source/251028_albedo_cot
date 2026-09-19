# -*- coding: utf-8 -*-
"""
figsupp_3methods_global.py

Global (all-ocean) comparison of three effective-COT representations used to
fit the COT -> albedo relationship from RFOV data:
    Method 1: full COT distribution
    Method 2: arithmetic mean COT
    Method 3: log-mean COT

Footprints from every ocean/season are pooled and split into three equal-size
groups by std(lgCOT) (small / medium / large); the three subplots show the
three methods fitted within each group.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

from utils_fitting import oceans, format_panel_tag

BASE_DIR = Path(__file__).resolve().parent
RFOV_DIR = BASE_DIR / 'RFOV_product' / 'ocean_season'
FIG_DIR = BASE_DIR / 'figs'

SEASONS = ('MAM', 'JJA', 'SON', 'DJF')

LOG10_B_MIN, LOG10_B_MAX = -5.0, 5.0
K_MIN, K_MAX = 0.05, 10.0

COT_PLOT = np.linspace(0.01, 80, 500)

# Style (three subplots: small / medium / large std(lgCOT))
FIG_FIGSIZE = (10, 3.6)
LABEL_SIZE = 11
TICK_SIZE = 7
LEGEND_SIZE = 7.5
TAG_SIZE = 12
TITLE_SIZE = 9
XLIM = (0, 60)
YLIM = (0.05, 0.95)
OUTPUT_PATH = FIG_DIR / 'figsupp_3methods_global.png'


# ============================================================
# Model
# ============================================================

def albedo_from_cot(cot, b, k):
    """Ac = b*COT^k / (1 + b*COT^k)."""
    cot_k = np.power(cot, k)
    return (b * cot_k) / (1.0 + b * cot_k)


# ============================================================
# Data
# ============================================================

def cot_bins_from_columns(columns):
    """Return (freq column names, bin midpoints) sorted by COT bin."""
    pattern = re.compile(r'^cot_freq_(\d+)_(\d+)$')
    bins = []
    for column in columns:
        match = pattern.match(column)
        if match:
            lower, upper = map(int, match.groups())
            bins.append((lower, upper, column))
    bins.sort()
    cot_cols = [column for _, _, column in bins]
    midpoints = np.array([(lower + upper) / 2.0 for lower, upper, _ in bins])
    return cot_cols, midpoints


def load_all_data():
    """Load every ocean/season file and return (prob, obs_albedo, cot_mid, std_lgcot)."""
    frames = [pd.read_csv(path) for path in sorted(RFOV_DIR.glob('*.csv'))]
    if not frames:
        raise FileNotFoundError(f'No RFOV data in {RFOV_DIR}')
    data = pd.concat(frames, ignore_index=True)

    cot_cols, cot_mid = cot_bins_from_columns(data.columns)
    freq = (
        data[cot_cols]
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )
    obs_albedo = pd.to_numeric(data['ret_albedo'], errors='coerce').to_numpy(dtype=float)

    freq_sum = freq.sum(axis=1)
    valid = (
        np.isfinite(obs_albedo)
        & np.isfinite(freq).all(axis=1)
        & (freq_sum > 0)
        & (obs_albedo >= 0)
        & (obs_albedo <= 1)
    )
    freq = freq[valid]
    obs_albedo = obs_albedo[valid]
    prob = freq / freq_sum[valid][:, None]

    # per-footprint std of lg(COT) from its own COT distribution
    lgcot = np.log(cot_mid)
    mean_lg = prob @ lgcot
    var_lg = prob @ (lgcot ** 2) - mean_lg ** 2
    std_lgcot = np.sqrt(np.maximum(var_lg, 0.0))

    return prob, obs_albedo, cot_mid, std_lgcot


# ============================================================
# Optimization framework
# ============================================================

def unpack(x):
    log10_b, k = x
    return 10 ** log10_b, k


def optimize_model(predict_function, obs_albedo):
    """Fit (b, k) by differential evolution, then refine with least squares."""

    def residual(x):
        b, k = unpack(x)
        return predict_function(b, k) - obs_albedo

    def mse(x):
        r = residual(x)
        return np.mean(r * r)

    result_global = differential_evolution(
        mse,
        [(LOG10_B_MIN, LOG10_B_MAX), (K_MIN, K_MAX)],
        seed=42,
        popsize=20,
        maxiter=500,
        tol=1e-10,
    )

    result_ls = least_squares(
        residual,
        result_global.x,
        bounds=([LOG10_B_MIN, K_MIN], [LOG10_B_MAX, K_MAX]),
        max_nfev=10000,
    )

    b, k = unpack(result_ls.x)
    pred = predict_function(b, k)
    rmse = np.sqrt(np.mean((pred - obs_albedo) ** 2))
    return b, k, rmse


# ============================================================
# Draw one ocean panel (style follows fig1_fitting_8oceans.py)
# ============================================================

def draw_methods(ax, title, fits):
    """fits: list of (label, b, k); draw the three method curves on ax."""
    linestyles = ('-', ':', '--')
    for (label, b, k), linestyle in zip(fits, linestyles):
        albedo = albedo_from_cot(COT_PLOT, b, k)
        ax.plot(
            COT_PLOT, albedo, linestyle, lw=1.6,
            label=f'{label}\nb={b:.2g}, k={k:.2f}',
        )

    ax.set(xlim=XLIM, ylim=YLIM)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(loc='lower right', fontsize=LEGEND_SIZE, framealpha=0.8)


# ============================================================
# Main
# ============================================================

def main():
    FIG_DIR.mkdir(exist_ok=True)

    prob, obs_albedo, cot_mid, std_lgcot = load_all_data()
    print(f'Total valid footprints: {len(obs_albedo)}')

    # split every footprint into three equal-size groups by std(lgCOT)
    q1, q2 = np.quantile(std_lgcot, [1.0 / 3.0, 2.0 / 3.0])
    groups = [
        ('small std(lgCOT)', std_lgcot <= q1,
         float(std_lgcot.min()), float(q1)),
        ('medium std(lgCOT)', (std_lgcot > q1) & (std_lgcot <= q2),
         float(q1), float(q2)),
        ('large std(lgCOT)', std_lgcot > q2,
         float(q2), float(std_lgcot.max())),
    ]

    fig, axes = plt.subplots(1, 3, figsize=FIG_FIGSIZE, sharex=True, sharey=True)

    for panel_index, (ax, (name, mask, lo, hi)) in enumerate(zip(axes, groups)):
        prob_g = prob[mask]
        obs_g = obs_albedo[mask]
        cot_mean = prob_g @ cot_mid
        log_cot_mean = np.exp(prob_g @ np.log(cot_mid))

        # Method 1: full COT distribution
        b1, k1, rmse1 = optimize_model(
            lambda b, k: prob_g @ albedo_from_cot(cot_mid, b, k), obs_g)
        # Method 2: arithmetic mean COT
        b2, k2, rmse2 = optimize_model(
            lambda b, k: albedo_from_cot(cot_mean, b, k), obs_g)
        # Method 3: log-mean COT
        b3, k3, rmse3 = optimize_model(
            lambda b, k: albedo_from_cot(log_cot_mean, b, k), obs_g)

        print(f'\n==== {name} (n={len(obs_g)}, std(lgCOT) in [{lo:.3f}, {hi:.3f}]) ====')
        print(f'Method 1 COT Distribution : b={b1:.4g}  k={k1:.4f}  RMSE={rmse1:.4f}')
        print(f'Method 2 Mean COT         : b={b2:.4g}  k={k2:.4f}  RMSE={rmse2:.4f}')
        print(f'Method 3 Log-mean COT     : b={b3:.4g}  k={k3:.4f}  RMSE={rmse3:.4f}')

        draw_methods(ax, f'{name}\nstd(lgCOT): {lo:.2f}-{hi:.2f}', [
            ('COT Distribution', b1, k1),
            ('Log mean COT', b3, k3),
            ('Mean COT', b2, k2),
        ])

        ax.text(-0.03, 1.02, format_panel_tag(panel_index, 'science'),
                transform=ax.transAxes, fontsize=TAG_SIZE,
                va='bottom', ha='left')
        ax.set_xlabel('COT', fontsize=LABEL_SIZE)
        if panel_index == 0:
            ax.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=LABEL_SIZE)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()

