import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from utils_fitting import (
    oceans, season_dict, cot_range, cot_to_albedo,
    cot_to_x, albedo_to_y, mc_fit, format_panel_tag
)
from utils_solar import (
    compute_daytime_fit_data, cot_k_b_to_albedo,
    calc_monthly_swdown, calc_grid_cell_area
)

# Paths
BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
TABLE_FOLDER = 'cp'  # coupled SBDART lookup tables (per ocean-season)
TABLE_DIR = f'{BASE_PATH}/build_sbdart_lookup_table/cot_sza_to_albedo_lookup_table_{TABLE_FOLDER}'
FIG_DIR = f'{BASE_PATH}/figs'
FIT_DATA_PATH = f'{BASE_PATH}/processed_data/fig4_panel_b_fit_data.npz'
SENSITIVITY_1030_CSV = f'{BASE_PATH}/processed_data/sensitivity_albedo_vs_cot_1030.csv'
SENSITIVITY_DAY_CSV = f'{BASE_PATH}/processed_data/sensitivity_albedo_vs_cot_day.csv'
LNND_VS_LNAOD_CSV = f'{BASE_PATH}/processed_data/sensitivity_lnnd_vs_lnaod.csv'
CF_RET_CSV = f'{BASE_PATH}/processed_data/sensitivity_cfret_vs_lnnd.csv'
CF_MSK_CSV = f'{BASE_PATH}/processed_data/sensitivity_cfmsk_vs_lnnd.csv'
os.makedirs(FIG_DIR, exist_ok=True)

MIN_COT = 2.5
MIN_CF = 0.1

# Colors for the 5 lines (order: t91, ret, ret_day, msk, msk_day)
# Keep the current T91 color; use one representative color from each panel (c)/(d) palette for ret/msk.
T91_COLOR = '#222222'
RET_DAY_COLOR = "#D49102"  # orange used for retrieval-domain curves in panel (b)
MSK_DAY_COLOR = '#8B1E3F'  # red-brown used for mask-domain curves in panel (b)
RET_1030_COLOR = '#ff852e'
MSK_1030_COLOR = '#f20d38'
LINE_COLORS = [T91_COLOR, RET_1030_COLOR, RET_DAY_COLOR, MSK_1030_COLOR, MSK_DAY_COLOR]
LINE_STYLES = ['-', '-', '--', '-', '--']
LINE_LABELS = [
    r'T91: $k$=',
    r'Ret ($A_{\mathrm{c,1030}}): k$=',
    r'Ret (COT$_{\mathrm{1030}}): k$=',
    r'Msk ($A_{\mathrm{c,1030}}): k$=',
    r'Msk (COT$_{\mathrm{1030}}): k$=',
]

# T91/uncorrected parameters used consistently in panel (b), (c), and (d).
k_t91 = 1.0
lnb_t91 = np.log(0.13)

# Panel (c)/(d) palettes are tied to the corresponding panel (b) line colors.
# ret: orange-yellow family; msk: red-brown family.
BAR_PALETTES = {
    'ret': {
        'irf': ['#D55E00', '#E69F00', '#F0C808'],
    },
    'msk': {
        'irf': ['#C46A5A', '#8B1E3F', '#E7C2A3'],
    },
}

PANEL_CD_BASE_TITLE = 'Radiative Forcings'
PANEL_CD_TITLE_SUFFIX = {
    'ret': ' Using Retrieval-Domain Obs.',
    'msk': ' Using Mask-Domain Obs.',
}

# ============================================================
# Helper functions (from fig2_reason.py)
# ============================================================

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


# ============================================================
# Panel (a): contourf of mean lookup table
# ============================================================

def read_lookup_table(ocean, season):
    """Read the cot-sza-to-albedo lookup table for a given ocean and season."""
    file_name = f'cot_sza_to_albedo_lookup_table_{ocean}_{season}.csv'
    file_path = os.path.join(TABLE_DIR, file_name)

    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path, index_col=0)
    sza_grid = np.array(df.index, dtype=float)
    cot_grid = np.array(df.columns, dtype=float)
    albedo_grid = df.values

    return sza_grid, cot_grid, albedo_grid


def compute_mean_lookup_table():
    """Average lookup tables across all ocean-season combinations."""
    albedo_sum = None
    count = 0
    common_sza_grid = None
    common_cot_grid = None

    for ocean in oceans:
        for season in season_dict.keys():
            result = read_lookup_table(ocean, season)
            if result is None:
                continue

            sza_grid, cot_grid, albedo_grid = result

            if common_sza_grid is None:
                common_sza_grid = sza_grid
                common_cot_grid = cot_grid
                albedo_sum = np.zeros_like(albedo_grid)
            elif not (np.array_equal(sza_grid, common_sza_grid) and
                      np.array_equal(cot_grid, common_cot_grid)):
                continue

            albedo_sum += albedo_grid
            count += 1

    if count == 0:
        return None, None, None, 0

    albedo_mean = albedo_sum / count
    return common_sza_grid, common_cot_grid, albedo_mean, count


def draw_contourf(ax, sza_grid, cot_grid, albedo_mean, count):
    """Draw contourf plot on given axes."""
    levels = np.arange(0, 0.85, 0.05)
    pcm = ax.contourf(
        cot_grid, sza_grid, albedo_mean,
        levels=levels,
        cmap='Blues',
        # extend='both',
    )
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 70)
    ax.set_xlabel('COT', fontsize=13)
    ax.set_ylabel('SZA (deg)', fontsize=13)
    return pcm


# ============================================================
# Panel (b): 5 fit lines (t91, ret, ret_day, msk, msk_day)
# ============================================================

def draw_fit_lines(ax, recompute=False):
    """Draw t91, ret, msk, and daytime-adjusted ret/msk fit lines."""
    print('Loading global data for panel (b)...')
    df = load_global_data()
    print(f'Total data points: {len(df)}')

    # --- T91 / uncorrected ---
    alb_t91_fit = cot_k_b_to_albedo(cot_range, k_t91, np.exp(lnb_t91))

    # --- ret (retrieval-domain obs.) ---
    k_ret, lnb_ret, _, _ = mc_fit(
        df['ret_cot_cer'].values,
        df['ret_albedo'].values,
        cot_std=0.10,
        albedo_std=0.13,
        n_mc=300,
        bootstrap=True
    )
    alb_ret_fit = cot_k_b_to_albedo(cot_range, k_ret, np.exp(lnb_ret))

    # --- msk (mask-domain obs.) ---
    k_msk, lnb_msk, _, _ = mc_fit(
        df['cot_mod08'].values,
        df['albedo'].values,
        cot_std=0.10,
        albedo_std=0.20,
        n_mc=300,
        bootstrap=True
    )
    alb_msk_fit = cot_k_b_to_albedo(cot_range, k_msk, np.exp(lnb_msk))

    # --- Daytime-adjusted ret and msk ---
    if recompute or not os.path.exists(FIT_DATA_PATH):
        alb_ret_day_fit, alb_msk_day_fit, k_ret_day, k_msk_day = compute_daytime_fit_data(df)
    else:
        print(f'Loading saved fit data from {FIT_DATA_PATH}')
        data = np.load(FIT_DATA_PATH)
        alb_ret_day_fit = data['alb_ret_day_fit']
        alb_msk_day_fit = data['alb_msk_day_fit']
        k_ret_day = float(data['k_ret_day'])
        k_msk_day = float(data['k_msk_day'])

    # --- Plot all 5 lines ---
    fit_curves = [alb_t91_fit, alb_ret_fit, alb_ret_day_fit, alb_msk_fit, alb_msk_day_fit]
    k_values = [k_t91, k_ret, k_ret_day, k_msk, k_msk_day]

    for i in range(5):
        ax.plot(cot_range, fit_curves[i],
                color=LINE_COLORS[i], lw=2, ls=LINE_STYLES[i],
                label=rf'{LINE_LABELS[i]}{k_values[i]:.2f}')

    ax.set_xlim(0, 60)
    ax.set_xlabel('COT', fontsize=13)
    ax.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=13)
    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.9)


# ============================================================
# Panel (c) and (d): IRF + CFA stacked bar charts
# ============================================================

def compute_irf_cfa_data():
    """
    Compute IRF and CFA (= ERF - IRF) for each ocean, for:
      - ret_day, ret_1030, ret_orig
      - msk_day, msk_1030, msk_orig

    Returns
    -------
    dict with keys 'ret' and 'msk', each a dict {ocean: {variant: (irf, cfa)}}.
    The ocean-level values are area-weighted means over all valid seasonal grid cells.
    """
    print('Computing IRF and CFA data...')

    variants_order = ['day', '1030', 'orig']
    methods = ['ret', 'msk']

    # Load merged data
    merged_df = load_global_data()

    # Load sensitivity coefficients (wide format)
    coef_1030 = pd.read_csv(SENSITIVITY_1030_CSV)
    coef_day = pd.read_csv(SENSITIVITY_DAY_CSV)

    # Load lnnd_o_lnaod
    lnnd_df = pd.read_csv(LNND_VS_LNAOD_CSV)
    lnnd_df.columns = [c.strip() for c in lnnd_df.columns]
    lnnd_df['Ocean'] = lnnd_df['Ocean'].str.strip()
    lnnd_df['Season'] = lnnd_df['Season'].str.strip()
    lnnd_lookup = lnnd_df.set_index(['Ocean', 'Season'])['Slope'].to_dict()

    # Load CF sensitivity
    cf_ret_df = pd.read_csv(CF_RET_CSV)
    cf_ret_df.columns = [c.strip() for c in cf_ret_df.columns]
    cf_ret_df['Ocean'] = cf_ret_df['Ocean'].str.strip()
    cf_ret_df['Season'] = cf_ret_df['Season'].str.strip()
    cf_ret_lookup = cf_ret_df.set_index(['Ocean', 'Season'])['Slope'].to_dict()

    cf_msk_df = pd.read_csv(CF_MSK_CSV)
    cf_msk_df.columns = [c.strip() for c in cf_msk_df.columns]
    cf_msk_df['Ocean'] = cf_msk_df['Ocean'].str.strip()
    cf_msk_df['Season'] = cf_msk_df['Season'].str.strip()
    cf_msk_lookup = cf_msk_df.set_index(['Ocean', 'Season'])['Slope'].to_dict()

    # Build coefficient lookups
    def build_coef_lookup(coef_df, suffix=''):
        """Build lookup dict. suffix is e.g. '_day' for day CSV columns like k_ret_day."""
        lookup = {}
        for _, row in coef_df.iterrows():
            key = (str(row['Ocean']).strip(), str(row['Season']).strip())
            lookup[('k_ret', key)] = row[f'k_ret{suffix}']
            lookup[('lnb_ret', key)] = row[f'lnb_ret{suffix}']
            lookup[('k_msk', key)] = row[f'k_msk{suffix}']
            lookup[('lnb_msk', key)] = row[f'lnb_msk{suffix}']
        return lookup

    coef_1030_lookup = build_coef_lookup(coef_1030, suffix='')
    coef_day_lookup = build_coef_lookup(coef_day, suffix='_day')

    def get_coef(lookup, method, ocean, season, param):
        return lookup.get((f'{param}_{method}', (ocean, season)), np.nan)

    # Compute SWdown and grid area
    merged_df['month'] = pd.to_datetime(merged_df['time']).dt.month
    unique_lat_month = merged_df[['lat', 'month']].drop_duplicates()
    unique_lat_month['swdown'] = unique_lat_month.apply(
        lambda r: calc_monthly_swdown(r['lat'], month=r['month']), axis=1
    )
    merged_df = merged_df.merge(unique_lat_month, on=['lat', 'month'], how='left')
    merged_df['grid_area_km2'] = merged_df['lat'].apply(calc_grid_cell_area)

    # Clear-sky albedo
    merged_df['A_clr'] = merged_df['sw_clr'] / merged_df['solar_incoming']

    # Aggregate to seasonal means at each lat/lon
    agg_cols = {
        'swdown': 'mean',
        'log_aod_diff': 'mean',
        'cf_liq_ceres': 'mean',       # CF_msk
        'cf_ret_liq_mod08': 'mean',   # CF_ret
        'cot_mod08': 'mean',
        'A_clr': 'mean',
        'grid_area_km2': 'first',
    }
    seasonal_grid = merged_df.groupby(['ocean', 'season', 'lat', 'lon']).agg(agg_cols).reset_index()

    # Initialize accumulators for final ocean-level bars.
    results = {method: {ocean: {} for ocean in oceans} for method in methods}
    accum = {
        method: {
            ocean: {
                variant: {'irf_sum': 0.0, 'irf_area': 0.0,
                          'cfa_sum': 0.0, 'cfa_area': 0.0}
                for variant in variants_order
            }
            for ocean in oceans
        }
        for method in methods
    }

    for ocean in oceans:
        for season in season_dict.keys():
            mask = (seasonal_grid['ocean'] == ocean) & (seasonal_grid['season'] == season)
            if not mask.any():
                continue

            sub = seasonal_grid[mask]
            area = sub['grid_area_km2'].values.astype(float)
            total_area = np.nansum(area[np.isfinite(area) & (area > 0)])
            if total_area <= 0:
                continue

            # Get coefficients. Use 'lnb', not 'b', because build_coef_lookup stores lnb_ret/lnb_msk.
            k_ret_1030 = get_coef(coef_1030_lookup, 'ret', ocean, season, 'k')
            lnb_ret_1030 = get_coef(coef_1030_lookup, 'ret', ocean, season, 'lnb')
            k_ret_day = get_coef(coef_day_lookup, 'ret', ocean, season, 'k')
            lnb_ret_day = get_coef(coef_day_lookup, 'ret', ocean, season, 'lnb')

            k_msk_1030 = get_coef(coef_1030_lookup, 'msk', ocean, season, 'k')
            lnb_msk_1030 = get_coef(coef_1030_lookup, 'msk', ocean, season, 'lnb')
            k_msk_day = get_coef(coef_day_lookup, 'msk', ocean, season, 'k')
            lnb_msk_day = get_coef(coef_day_lookup, 'msk', ocean, season, 'lnb')

            # Per-grid-cell data
            cot_sub = sub['cot_mod08'].values.astype(float)
            cf_ret_vals = sub['cf_ret_liq_mod08'].values.astype(float)
            cf_msk_vals = sub['cf_liq_ceres'].values.astype(float)
            lnnd_val = lnnd_lookup.get((ocean, season), np.nan)
            dcf_ret = cf_ret_lookup.get((ocean, season), np.nan)
            dcf_msk = cf_msk_lookup.get((ocean, season), np.nan)

            # IRF_base = swdown * lnnd_o_lnaod * log_aod_diff
            swdown = sub['swdown'].values.astype(float)
            log_aod_diff = sub['log_aod_diff'].values.astype(float)
            irf_base = swdown * lnnd_val * log_aod_diff
            Aclr = sub['A_clr'].values.astype(float)

            variants = {
                'ret': {
                    'day': (k_ret_day, lnb_ret_day, cf_ret_vals, dcf_ret),
                    '1030': (k_ret_1030, lnb_ret_1030, cf_ret_vals, dcf_ret),
                    'orig': (k_t91, lnb_t91, cf_ret_vals, dcf_ret),
                },
                'msk': {
                    'day': (k_msk_day, lnb_msk_day, cf_msk_vals, dcf_msk),
                    '1030': (k_msk_1030, lnb_msk_1030, cf_msk_vals, dcf_msk),
                    'orig': (k_t91, lnb_t91, cf_msk_vals, dcf_msk),
                },
            }

            for method in methods:
                for variant, (k_val, lnb_val, cf_vals, dcf_val) in variants[method].items():
                    if np.isnan(k_val) or np.isnan(lnb_val) or np.isnan(dcf_val) or np.isnan(lnnd_val):
                        continue

                    Ac = cot_k_b_to_albedo(cot_sub, k_val, np.exp(lnb_val))

                    # IRF = swdown * dlnNd/dlnAOD * dlnAOD / 3 * k * Ac * (1-Ac) * CF
                    irf_vals = (irf_base / 3.0) * k_val * Ac * (1 - Ac) * cf_vals

                    # CFA = swdown * dlnNd/dlnAOD * dlnAOD * (Ac - Aclr) * dCF/dlnNd
                    cfa_vals = irf_base * (Ac - Aclr) * dcf_val

                    good_irf = np.isfinite(irf_vals) & np.isfinite(area) & (area > 0)
                    good_cfa = np.isfinite(cfa_vals) & np.isfinite(area) & (area > 0)

                    if np.any(good_irf):
                        accum[method][ocean][variant]['irf_sum'] += np.nansum(irf_vals[good_irf] * area[good_irf])
                        accum[method][ocean][variant]['irf_area'] += np.nansum(area[good_irf])
                    if np.any(good_cfa):
                        accum[method][ocean][variant]['cfa_sum'] += np.nansum(cfa_vals[good_cfa] * area[good_cfa])
                        accum[method][ocean][variant]['cfa_area'] += np.nansum(area[good_cfa])

    # Final ocean-level results used by panel (c)/(d).
    for method in methods:
        for ocean in oceans:
            for variant in variants_order:
                item = accum[method][ocean][variant]
                irf_mean = item['irf_sum'] / item['irf_area'] if item['irf_area'] > 0 else np.nan
                cfa_mean = item['cfa_sum'] / item['cfa_area'] if item['cfa_area'] > 0 else np.nan
                results[method][ocean][variant] = (irf_mean, cfa_mean)

    return results


def draw_stacked_bar(ax, data, method, panel_tag):
    """
    Draw stacked bar chart of IRF + CFA for a given method ('ret' or 'msk').

    For each ocean, 3 groups of bars:
      Day, 10:30, Uncorr
    Each group has IRF (bottom) + CFA (top) stacked.
    """
    ocean_names = oceans
    x = np.arange(len(ocean_names))
    width = 0.18
    gap = 0.06   # 第2个柱和第3个柱之间的小间距
    offsets = [-width - gap / 2, -gap / 2, width + gap / 2]

    variants = ['1030', 'day', 'orig']
    variant_labels = ['Corrected, $A_{\mathrm{c,1030}}$', r'Corrected, COT$_{\mathrm{1030}}$', 'Uncorrected']
    irf_colors = BAR_PALETTES[method]['irf']

    legend_handles = []
    legend_labels = []

    for i, (var, vlabel) in enumerate(zip(variants, variant_labels)):
        irf_raw = []
        cfa_raw = []
        for ocean in ocean_names:
            if ocean in data and var in data[ocean]:
                irf, cfa = data[ocean][var]
                irf_raw.append(irf if np.isfinite(irf) else np.nan)
                cfa_raw.append(cfa if np.isfinite(cfa) else np.nan)
            else:
                irf_raw.append(np.nan)
                cfa_raw.append(np.nan)

        irf_raw = np.asarray(irf_raw, dtype=float)
        cfa_raw = np.asarray(cfa_raw, dtype=float)
        irf_vals = np.nan_to_num(irf_raw, nan=0.0)
        cfa_vals = np.nan_to_num(cfa_raw, nan=0.0)
        irf_ocean_mean = np.nanmean(irf_raw) if np.any(np.isfinite(irf_raw)) else np.nan
        cfa_ocean_mean = np.nanmean(cfa_raw) if np.any(np.isfinite(cfa_raw)) else np.nan

        offset = offsets[i]

        # IRF (bottom)
        h_irf = ax.bar(
            x + offset, irf_vals, width,
            color=irf_colors[i],
            edgecolor=irf_colors[i],
            linewidth=1.3,
            hatch=None,
            label=f'IRF, {vlabel}'
        )

        # CFA (top, stacked on IRF)
        h_cfa = ax.bar(
            x + offset, cfa_vals, width,
            bottom=irf_vals,
            color='none',
            edgecolor=irf_colors[i],
            linewidth=1.3,
            hatch='///',
            label=f'CFA, {vlabel}'
        )

        # Keep all six legend entries and append the mean across oceans for each component.
        legend_handles.extend([h_irf, h_cfa])
        legend_labels.extend([
            rf'IRF$_{{\mathrm{{aci}}}}$ ({vlabel}): {irf_ocean_mean:.2f} W m$^{-2}$',
            rf'CFA ({vlabel}): {cfa_ocean_mean:.2f} W m$^{{-2}}$',
        ])

    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35)
    ax.set_title(PANEL_CD_BASE_TITLE + PANEL_CD_TITLE_SUFFIX[method], fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(ocean_names, fontsize=11)
    ax.set_ylabel('Radiative Forcing (W m$^{-2}$)', fontsize=12)
    ax.legend(legend_handles, legend_labels, fontsize=9.5, loc='upper right',
              framealpha=0.8, ncol=3, columnspacing=0.8, handlelength=1.2)
    ax.text(-0.01, 1.01, panel_tag,
            transform=ax.transAxes, fontsize=17, va='bottom', ha='left')


# ============================================================
# Main
# ============================================================

def main(recompute=False):
    # ---- Panel (a): contourf ----
    print('Computing mean lookup table...')
    sza_grid, cot_grid, albedo_mean, count = compute_mean_lookup_table()
    if count == 0:
        print('No lookup tables found!')
        return
    print(f'Averaged {count} ocean-season lookup tables.')

    # ---- Compute IRF/CFA data for panels (c) and (d) ----
    irf_cfa_data = compute_irf_cfa_data()

    # ---- Create figure: 3 rows, 2 columns ----
    # Row 1: (a) contourf, (b) fit lines
    # Row 2: (c) Ret stacked bar
    # Row 3: (d) Msk stacked bar
    fig = plt.figure(figsize=(12, 11))

    # GridSpec: row heights proportional to content
    gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.35,
                          height_ratios=[4.8, 3.2, 3.2],
                          bottom=0.06, top=0.96, left=0.06, right=0.97)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])  # Span full width
    ax_d = fig.add_subplot(gs[2, :])  # Span full width

    # Panel (a): contourf
    pcm = draw_contourf(ax_a, sza_grid, cot_grid, albedo_mean, count)
    cbar = fig.colorbar(pcm, ax=ax_a)
    cbar.set_label('$A_\mathrm{c,dcp}$', fontsize=13)
    ax_a.text(-0.01, 1.01, format_panel_tag(0, 'nature'),
              transform=ax_a.transAxes, fontsize=17, va='bottom', ha='left')

    # Panel (b): fit lines
    draw_fit_lines(ax_b, recompute=recompute)
    ax_b.text(-0.01, 1.01, format_panel_tag(1, 'nature'),
              transform=ax_b.transAxes, fontsize=17, va='bottom', ha='left')

    # Panel (c): Ret stacked bar
    draw_stacked_bar(ax_c, irf_cfa_data['ret'], 'ret', format_panel_tag(2, 'nature'))

    # Panel (d): Msk stacked bar
    draw_stacked_bar(ax_d, irf_cfa_data['msk'], 'msk', format_panel_tag(3, 'nature'))

    out_path = os.path.join(FIG_DIR, 'fig4_radiative_forcings.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    import sys
    recompute = '--recompute' in sys.argv
    main(recompute=recompute)
