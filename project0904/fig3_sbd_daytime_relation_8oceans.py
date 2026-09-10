import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import cot_k_b_to_albedo, cot_to_albedo, mc_fit, oceans
from utils_solar import calc_grid_cell_area, get_daytime_sza
from util_ocean_season_division import oceans_def
from scipy.interpolate import RegularGridInterpolator

OUTPUT_PATH = './figs/fig3_sbd_daytime_relation_8oceans.png'
FITS_CSV_PATH = './processed_data/fig3_sbd_daytime_relation_8oceans_fits.csv'
COT = np.geomspace(2.5, 60, 80)
MONTH_TO_SEASON = {
    1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
    6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON',
    11: 'SON', 12: 'DJF',
}
TROPICAL_OCEANS = {'TPO', 'TAO', 'TIO'}
TROPICAL_COLORS = ("#090EA5", "#35f3d0", '#2c7fb8')
EXTRATROPICAL_COLORS = ("#f1c515", "#8b745e", "#ef7809", "#f93939", "#ec7fe6")
LINESTYLES = (':', '--', '-')


# SBDART lookup tables live in the parent project
LUT_BASE = '/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table'
_SBDART_INTERP_CACHE = {}


def sbdart_interpolator(ocean, season, table_folder='cp'):
    """Cached linear interpolator over the (SZA, COT) SBDART albedo table."""
    key = (ocean, season, table_folder)
    if key in _SBDART_INTERP_CACHE:
        return _SBDART_INTERP_CACHE[key]

    path = (
        f'{LUT_BASE}/cot_sza_to_albedo_lookup_table_{table_folder}/'
        f'cot_sza_to_albedo_lookup_table_{ocean}_{season}.csv'
    )
    table = pd.read_csv(path, index_col=0)
    sza_grid = table.index.to_numpy(dtype=float)
    cot_grid = table.columns.to_numpy(dtype=float)
    grid = table.to_numpy(dtype=float)

    order_sza = np.argsort(sza_grid)
    order_cot = np.argsort(cot_grid)
    interp = RegularGridInterpolator(
        (sza_grid[order_sza], cot_grid[order_cot]),
        grid[np.ix_(order_sza, order_cot)],
        method='linear', bounds_error=False, fill_value=np.nan,
    )
    _SBDART_INTERP_CACHE[key] = interp
    return interp


def annual_ocean_albedo(ocean, table_folder='cp', max_sza=70):
    """
    Annual-mean daytime & latitude weighted SBDART albedo for one ocean.

    Instead of splitting the year into seasons, every COT value is averaged
    over every daytime hour of the whole year, with cos(SZA) as the weight
    (plus cos(latitude) as the ocean-area weight).  Each day uses the
    SBDART lookup table of its season.
    """
    latitudes = np.unique(np.concatenate([
        np.arange(south + 0.5, north, 1.0)
        for _, south, _, north in oceans_def[ocean]
    ]))

    interpolators = {
        season: sbdart_interpolator(ocean, season, table_folder)
        for season in ('MAM', 'JJA', 'SON', 'DJF')
    }

    days = pd.date_range('2021-01-01', '2021-12-31', freq='D')

    numerator = np.zeros_like(COT, dtype=float)
    denominator = 0.0

    for day in days:
        season = MONTH_TO_SEASON[day.month]
        itp = interpolators[season]
        doy = day.dayofyear

        for latitude in latitudes:
            sza_values = get_daytime_sza(latitude, doy, max_sza=max_sza)
            if sza_values.size == 0:
                continue

            time_weights = np.cos(np.deg2rad(sza_values))

            # evaluate the SBDART table for every daytime hour at once
            points = np.column_stack([
                np.repeat(sza_values, COT.size),
                np.tile(COT, sza_values.size),
            ])
            albedo = itp(points).reshape(sza_values.size, COT.size)

            weighted_albedo = (time_weights[:, None] * albedo).sum(axis=0)

            latitude_weight = np.cos(np.deg2rad(latitude))
            numerator += latitude_weight * weighted_albedo
            denominator += latitude_weight * np.sum(time_weights)

    if denominator == 0:
        return np.full_like(COT, np.nan, dtype=float)
    return numerator / denominator


def ocean_area_km2(ocean, resolution=1.0):
    """
    Geometric surface area (km2) of an ocean basin.

    Sums cos(latitude)-weighted 1-degree grid-cell areas over every
    latitude band in each rectangular region box that defines the basin.
    """
    area = 0.0
    for (west, south, east, north) in oceans_def[ocean]:
        n_lon = (east - west) / resolution
        lat_centers = np.arange(south + resolution / 2.0, north, resolution)
        area += n_lon * sum(
            calc_grid_cell_area(
                lat, lon_res=resolution, lat_res=resolution
            )
            for lat in lat_centers
        )
    return area


def main():
    fig, ax = plt.subplots(figsize=(5, 4.1))
    lh74 = cot_to_albedo(COT, 'l74')
    ax.plot(COT, lh74, color='#222222', lw=1.8, label=r'LH74: $k$=1.00')

    tropical_index = 0
    extratropical_index = 0
    records = []
    ratio_by_ocean = {}
    for index, ocean in enumerate(oceans):
        albedo = annual_ocean_albedo(ocean)
        # per-ocean ratio LH74 Ac / daytime Ac averaged over the COT grid
        valid_ratio = albedo > 0
        if valid_ratio.any():
            ratio_by_ocean[ocean] = np.nanmean(albedo[valid_ratio]*(1-albedo[valid_ratio]) / (lh74[valid_ratio]*(1-lh74[valid_ratio])))
        else:
            ratio_by_ocean[ocean] = np.nan
        k, lnb, _, _ = mc_fit(
            COT, albedo, cot_std=0.0, albedo_std=0.0,
            calculate_uncertainty=False,
        )
        fit_albedo = cot_k_b_to_albedo(COT, k, np.exp(lnb))
        records.append({'Ocean': ocean, 'k': k, 'lnb': lnb})
        if ocean in TROPICAL_OCEANS:
            color = TROPICAL_COLORS[tropical_index]
            tropical_index += 1
        else:
            color = EXTRATROPICAL_COLORS[extratropical_index]
            extratropical_index += 1
        ax.plot(
            COT, fit_albedo, color=color, lw=1.6,
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            label=rf'{ocean}: $k$={k:.2f}',
        )

    # --- Global ocean-area-weighted mean k ----------------------
    area = {ocean: ocean_area_km2(ocean) for ocean in oceans}
    total_area = float(sum(area.values()))
    global_k = sum(
        record['k'] * area[record['Ocean']]
        for record in records
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

    # --- Global ocean-area-weighted ratio LH74 Ac / daytime Ac --
    global_ratio = sum(
        ratio_by_ocean[ocean] * area[ocean]
        for ocean in oceans
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

    ax.set(xlim=(0, 60), ylim=(0.05, 0.95), xlabel='COT', ylabel=r'$A_{\mathrm{c}}$')
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
