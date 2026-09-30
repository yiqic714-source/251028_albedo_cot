import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_fitting import mc_fit, cot_to_albedo

# Paths
BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
TABLE_FOLDER = 'gascp_aodcp_sfccp_sw'  # 3D SBDART LUTs (sza x cer x cot)
RFOV_DIR = Path(f'{BASE_PATH}/project0904/RFOV_product/ocean_season')
COT_EDGES = np.geomspace(2.5, 76, 17)
MIN_GROUP_SIZE = 5
FIG_DIR = f'{BASE_PATH}/project0904/figs'
os.makedirs(FIG_DIR, exist_ok=True)

MIN_COT = 2.5
MAIN_FACE_COLOR = (1, 1, 1, 1.0)


def apply_main_background(fig, axes=None):
    fig.patch.set_facecolor(MAIN_FACE_COLOR)
    fig.patch.set_alpha(MAIN_FACE_COLOR[-1])

    if axes is None:
        axes = fig.axes
    elif not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    for ax in axes:
        ax.patch.set_facecolor(MAIN_FACE_COLOR)
        ax.patch.set_alpha(MAIN_FACE_COLOR[-1])


def save_png(fig, out_path, dpi=300, bbox_inches='tight'):
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        facecolor=fig.get_facecolor(),
        edgecolor='none',
        transparent=False
    )


# ============================================================
# Ac vs. COT at fixed SZA (global RFOV footprints)
# ============================================================

def add_rfov_cot(data):
    bins = []
    pattern = re.compile(r'^cot_freq_(\d+)_(\d+)$')
    for column in data.columns:
        match = pattern.match(column)
        if match:
            lower, upper = map(int, match.groups())
            bins.append((lower, upper, column))
    bins.sort()
    columns = [column for _, _, column in bins]
    midpoints = np.array([(lower + upper) / 2 for lower, upper, _ in bins])
    weights = (data[columns]
               .apply(pd.to_numeric, errors='coerce')
               .fillna(0).clip(lower=0).to_numpy(float))
    total = weights.sum(axis=1)
    mean_log_cot = np.divide(
        weights @ np.log(midpoints), total,
        out=np.full(len(data), np.nan), where=total > 0,
    )
    result = data.copy()
    result['cot_rfov'] = np.exp(mean_log_cot)
    return result


def load_rfov_global():
    """Global RFOV footprints: cot_rfov and cer_ret_mean from all oceans/seasons."""
    frames = []
    for path in sorted(RFOV_DIR.glob('*.csv')):
        ocean, season = path.stem.rsplit('_', 1)
        frame = add_rfov_cot(pd.read_csv(path))
        frame['cer_ret_mean'] = pd.to_numeric(frame['cer_ret_mean'], errors='coerce')
        frame['ocean'] = ocean
        frame['season'] = season
        frames.append(frame[['cot_rfov', 'cer_ret_mean', 'ocean', 'season']])
    if not frames:
        raise FileNotFoundError(f'No RFOV files in {RFOV_DIR}')
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=['cot_rfov', 'cer_ret_mean'])
    data = data[(data['cot_rfov'] >= MIN_COT) & (data['cer_ret_mean'] > 0)]
    return data.reset_index(drop=True)


def sbdart_albedo_at_sza(data, target_sza, table_folder=TABLE_FOLDER):
    """SBDART albedo for every RFOV footprint at a fixed SZA (3D cot/cer lookup)."""
    result = np.full(len(data), np.nan)
    for (ocean, season), idx in data.groupby(['ocean', 'season']).indices.items():
        rows = data.iloc[idx]
        result[idx] = cot_to_albedo(
            rows['cot_rfov'].to_numpy(), 'sbdart', sza=float(target_sza),
            cer=rows['cer_ret_mean'].to_numpy(),
            table_folder=table_folder, ocean=ocean, season=season,
        )
    return result


def bin_relation(cot, albedo, edges):
    labels = pd.cut(cot, edges, labels=False, include_lowest=True)
    cot_bins, albedo_bins, albedo_std = [], [], []
    for index in range(len(edges) - 1):
        mask = labels == index
        if np.count_nonzero(mask) < MIN_GROUP_SIZE:
            continue
        cot_bins.append(np.nanmean(cot[mask]))
        albedo_bins.append(np.nanmean(albedo[mask]))
        albedo_std.append(np.nanstd(albedo[mask]))
    return (np.asarray(cot_bins), np.asarray(albedo_bins), np.asarray(albedo_std))


def draw_ac_cot_curves(ax, data):
    sza_targets = np.arange(0.0, 75.0 + 0.1, 15.0)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(sza_targets)))
    legend_labels = []
    cot = data['cot_rfov'].to_numpy(dtype=float)

    for target_sza, color in zip(sza_targets, colors):
        albedo = sbdart_albedo_at_sza(data, target_sza)
        cot_bins, albedo_bins, albedo_std = bin_relation(cot, albedo, COT_EDGES)
        if len(cot_bins) < 3:
            continue
        k_val, _, _, _ = mc_fit(
            cot_bins, albedo_bins,
            cot_std=0.0, albedo_std=0.03, n_mc=300, bootstrap=True,
        )
        legend_labels.append(rf'SZA={target_sza:.0f}°: $k$={k_val:.2f}')
        ax.plot(cot_bins, albedo_bins, lw=2, color=color,
                label=f'SZA={target_sza:.0f}°')

    for line, label in zip(ax.lines, legend_labels):
        line.set_label(label)

    # ax.set_xlim(2.5, 60)
    ax.set_xlabel('COT', fontsize=13)
    ax.set_ylabel(r'$A_{\mathrm{c,cp}}$', fontsize=13)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)


def main():
    print('Loading global RFOV data...')
    data = load_rfov_global()
    print(f'Total footprints: {len(data)}')

    fig = plt.figure(figsize=(5.0, 4.5))
    apply_main_background(fig)
    ax = fig.add_subplot(1, 1, 1)
    apply_main_background(fig, ax)

    draw_ac_cot_curves(ax, data)

    out_path = os.path.join(FIG_DIR, 'figsupp_sza_k_relation.png')
    save_png(fig, out_path, dpi=300)
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
