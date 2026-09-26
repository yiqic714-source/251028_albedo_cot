# -*- coding: utf-8 -*-
"""
fig3_sbd_daytime_relation_8oceans.py

Daytime SBDART albedo -> COT relation for each ocean, built from the RFOV
footprint product (RFOV_product/ocean_season/{ocean}_{season}.csv):

  * the RFOV COT distribution gives cot_rfov, and cer_ret_mean gives the cloud
    effective radius; for every footprint the daytime hours of that footprint's
    latitude/date (SZA from the solar geometry) are used, each hour albedo is
    looked up from the 3D SBDART LUT (cot, sza, cer) via
    utils_fitting.cot_to_albedo, and the footprint's daytime albedo is the
    cos(SZA)-weighted mean over those hours;
  * the fit follows fig2_sbd_relation_formation_global.py: the footprints are
    binned in cot_rfov, and k is fitted to the binned points (with the binned
    within-bin std as error bars).
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import cot_k_b_to_albedo, cot_to_albedo, mc_fit, oceans
from utils_solar import calc_grid_cell_area, get_daytime_sza
from util_ocean_season_division import oceans_def

BASE_DIR = Path(__file__).resolve().parent
RFOV_DIR = BASE_DIR / 'RFOV_product' / 'ocean_season'

OUTPUT_PATH = './figs/fig3_sbd_daytime_relation_8oceans.png'
FITS_CSV_PATH = './processed_data/fig3_sbd_daytime_relation_8oceans_fits.csv'

SBDART_LUT_FOLDER = 'gascp_aodcp_sfccp_sw'
COT = np.geomspace(2.5, 60, 80)
COT_EDGES = np.geomspace(2.5, 76, 17)
SEASONS = ('MAM', 'JJA', 'SON', 'DJF')
MIN_COT = 2.5
MIN_GROUP_SIZE = 5
MAX_SZA = 70

TROPICAL_OCEANS = {'TPO', 'TAO', 'TIO'}
TROPICAL_COLORS = ("#090EA5", "#35f3d0", '#2c7fb8')
EXTRATROPICAL_COLORS = ("#f1c515", "#8b745e", "#ef7809", "#f93939", "#ec7fe6")
LINESTYLES = (':', '--', '-')


def add_rfov_cot(data):
    """Add cot_rfov = exp(log-mean COT) from the cot_freq_* distribution."""
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


def ocean_area_km2(ocean, resolution=1.0):
    """Geometric surface area (km2) of an ocean basin."""
    area = 0.0
    for (west, south, east, north) in oceans_def[ocean]:
        n_lon = (east - west) / resolution
        lat_centers = np.arange(south + resolution / 2.0, north, resolution)
        area += n_lon * sum(
            calc_grid_cell_area(lat, lon_res=resolution, lat_res=resolution)
            for lat in lat_centers
        )
    return area


def ocean_daytime_albedo(ocean, table_folder=SBDART_LUT_FOLDER, max_sza=MAX_SZA):
    """Per-footprint daytime SBDART albedo for one ocean from the RFOV rows.

    Returns (cot_rfov, albedo, lat).  albedo[i] is the cos(SZA)-weighted mean
    over the daytime hours of footprint i's latitude/date, looked up from the
    3D (cot, sza, cer) SBDART LUT.
    """
    frames = []
    for season in SEASONS:
        path = RFOV_DIR / f'{ocean}_{season}.csv'
        if not path.exists():
            continue
        frame = add_rfov_cot(pd.read_csv(path))
        frame['season'] = season
        frames.append(frame[['cot_rfov', 'cer_ret_mean', 'lat', 'time', 'season']])
    if not frames:
        return None

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=['cot_rfov', 'cer_ret_mean', 'lat', 'time'])
    data = data[(data['cot_rfov'] >= MIN_COT) & (data['cer_ret_mean'] > 0)]
    data = data.reset_index(drop=True)
    if data.empty:
        return None

    data['doy'] = pd.to_datetime(data['time']).dt.dayofyear
    lat = data['lat'].to_numpy(dtype=float)
    cot = data['cot_rfov'].to_numpy(dtype=float)
    cer = data['cer_ret_mean'].to_numpy(dtype=float)
    season_col = data['season'].to_numpy()

    albedo = np.full(len(data), np.nan)
    # group rows sharing the same latitude and day (same daytime SZA set)
    for (lat_v, doy_v), idx in data.groupby(['lat', 'doy']).indices.items():
        sza_values = get_daytime_sza(lat_v, doy_v, max_sza=max_sza)
        if sza_values.size == 0:
            continue
        weights = np.cos(np.deg2rad(sza_values))
        season = season_col[idx[0]]
        acc = np.zeros(len(idx), dtype=float)
        for sza_h, weight_h in zip(sza_values, weights):
            acc += weight_h * cot_to_albedo(
                cot[idx], 'sbdart', sza=sza_h, cer=cer[idx],
                table_folder=table_folder, ocean=ocean, season=season,
            )
        albedo[idx] = acc / np.sum(weights)

    return cot, albedo, lat


def bin_relation(cot, albedo, edges):
    """Binned mean cot / albedo / within-bin std (same as fig2)."""
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


def main():
    fig, ax = plt.subplots(figsize=(5, 4.1))
    analy_miu13 = cot_to_albedo(COT, 'analy', miu=3 ** (-0.5))
    ax.plot(COT, analy_miu13, color='k', lw=1.8,
            label=r'Ana, $\mu=3^{-1/2}: k=1$')
    analy = cot_to_albedo(COT, 'analy', miu=1)
    ax.plot(COT, analy, color='k', lw=1.8, ls='--', label=r'Ana, $\mu=1: k=1$')

    tropical_index = 0
    extratropical_index = 0
    records = []
    ratio_by_ocean = {}

    for index, ocean in enumerate(oceans):
        result = ocean_daytime_albedo(ocean)

        if result is None:
            records.append({'Ocean': ocean, 'k': np.nan, 'lnb': np.nan})
            ratio_by_ocean[ocean] = np.nan
            continue

        cot, albedo, lat = result
        valid = np.isfinite(albedo) & np.isfinite(cot)
        cot, albedo = cot[valid], albedo[valid]

        # ratio Ac(1-Ac) daytime / LH74 using each row's cot
        analy_rows = cot_to_albedo(cot, 'analy', miu=1)
        valid_ratio = analy_rows > 0
        ratio_by_ocean[ocean] = np.nanmean(
            albedo[valid_ratio] * (1 - albedo[valid_ratio]) /
            (analy_rows[valid_ratio] * (1 - analy_rows[valid_ratio]))
        )

        if ocean in TROPICAL_OCEANS:
            color = TROPICAL_COLORS[tropical_index]
            tropical_index += 1
        else:
            color = EXTRATROPICAL_COLORS[extratropical_index]
            extratropical_index += 1

        # fit like fig2: bin in cot_rfov, then fit the binned points
        cot_bins, albedo_bins, albedo_std = bin_relation(cot, albedo, COT_EDGES)
        if len(cot_bins) >= 3:
            k, lnb, _, _ = mc_fit(
                cot_bins, albedo_bins,
                cot_std=0.0, albedo_std=0.03, n_mc=300, bootstrap=True,
            )
        else:
            k, lnb = np.nan, np.nan

        records.append({'Ocean': ocean, 'k': k, 'lnb': lnb})

        fit_albedo = cot_k_b_to_albedo(COT, k, np.exp(lnb))
        ax.plot(
            COT, fit_albedo, color=color, lw=1.6,
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            label=rf'{ocean}: $k$={k:.2f}',
        )

    # --- Global ocean-area-weighted mean k ----------------------
    area = {ocean: ocean_area_km2(ocean) for ocean in oceans}
    total_area = float(sum(area.values()))
    global_k = np.nansum(
        [record['k'] * area[record['Ocean']] for record in records]
    ) / total_area

    print('\nGlobal ocean-area-weighted mean k: {:.4f}'.format(global_k))
    for record in records:
        print(
            '  {:<4s} k={:.4f}  area={:.6e} km2  weight={:.3%}'.format(
                record['Ocean'],
                record['k'],
                area[record['Ocean']],
                area[record['Ocean']] / total_area,
            )
        )
    # ------------------------------------------------------------

    # --- Global ocean-area-weighted ratio daytime / LH74 --------
    global_ratio = np.nansum(
        [ratio_by_ocean[ocean] * area[ocean] for ocean in oceans]
    ) / total_area
    print('\nGlobal ocean-area-weighted ratio Ac(1-Ac) daytime / LH74: {:.3f}'.format(global_ratio))
    for ocean in oceans:
        print(
            '  {:<4s} ratio={:.3f}  area={:.6e} km2  weight={:.3%}'.format(
                ocean,
                ratio_by_ocean[ocean],
                area[ocean],
                area[ocean] / total_area,
            )
        )
    # ------------------------------------------------------------

    ax.set(xlim=(0, 60), ylim=(0.1, 0.95), xlabel='COT', ylabel=r'$A_{\mathrm{c}}$')
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.85)
    fig.tight_layout()
    pd.DataFrame(records).to_csv(FITS_CSV_PATH, index=False)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_PATH}')
    print(f'Saved: {FITS_CSV_PATH}')


if __name__ == '__main__':
    main()

