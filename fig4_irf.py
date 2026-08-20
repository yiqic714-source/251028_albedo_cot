import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from utils_fitting import oceans, season_dict, format_panel_tag
from utils_solar import cot_k_b_to_albedo, calc_monthly_swdown, calc_grid_cell_area

# Paths
BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIG_DIR = f'{BASE_PATH}/figs'
SENSITIVITY_AC1030_CSV = f'{BASE_PATH}/processed_data/sensitivity_albedo_vs_cot_1030.csv'
SENSITIVITY_COT1030_CSV = f'{BASE_PATH}/processed_data/sensitivity_albedo_vs_cot_day.csv'
BELLOUIN2013_CSV = f'{BASE_PATH}/processed_data/Bellouin2013.csv'
BAR_DIFF_CSV = f'{BASE_PATH}/processed_data/fig4_bar_diff_orig_minus_ac1030.csv'
os.makedirs(FIG_DIR, exist_ok=True)

# Output folder for the split separate bar PNGs and legend PNGs
BAR_EXPORT_DIR = os.path.join(FIG_DIR, 'fig4_ocean_irf_bars')
os.makedirs(BAR_EXPORT_DIR, exist_ok=True)

# Backgrounds
MAIN_FACE_COLOR = (1, 1, 1, 0.55)
TRANSPARENT_FACE_COLOR = (1, 1, 1, 0.0)
LEGEND_FACE_COLOR = (1, 1, 1, 0.25)

# Map style
CONTOUR_COLOR = '#7B3294'  # purple contour lines for both panels
MAP_EXTENT = [-180, 180, -60, 60]

PANEL_TITLES = {
    'ret': r'IRF$_{\mathrm{aci}}$ (Cloud-Retrieval)',
    'msk': r'IRF$_{\mathrm{aci}}$ (Cloud-Mask)',
}

# T91/uncorrected parameters used for the third bar in the separate ocean PNGs
k_t91 = 1.0
lnb_t91 = np.log(0.13)

METHODS = ('ret', 'msk')

# Ocean-bar settings.
BAR_VARIANTS = ['ac1030', 'cot1030', 'orig']
BAR_SPLITS = {
    'ac1030': ('ac1030', 'orig'),
    'cot1030': ('cot1030', 'orig'),
}
BAR_LABELS = {
    'ac1030': r'Obs. Relation',
    'cot1030': r'Obs.  Relation',
    'orig': 'LT Relation',
}
BAR_COLORS = {
    'ret': {'ac1030': '#D55E00', 'cot1030': '#D55E00', 'orig': '#F0BA08'},
    'msk': {'ac1030': '#B2182B', 'cot1030': '#B2182B', 'orig': '#EF8A62'},
}
BAR_ALPHA = 0.65
BAR_YLIMS = {
    'ret': (0, 1.5),
    'msk': (0, 1.5),
}
BAR_AX_POS = [0.28, 0.22, 0.66, 0.66]

# Overestimate decomposition for panel (c).
OVER_GROUPS = [
    ('ret_ac1030', 'ret', 'ac1030', r'Ret'),
    ('ret_cot1030', 'ret', 'cot1030', r'Ret'),
    ('msk_ac1030', 'msk', 'ac1030', r'Msk'),
    ('msk_cot1030', 'msk', 'cot1030', r'Msk'),
]
OVER_BAR_LABELS = [
    r'$k$-caused overestimate',
    r'$A_{\mathrm{c}}$-caused overestimate',
    'Total overestimate',
]
OVER_BAR_COLORS = ["#818181", "#54CB5C", "#7B3294"]
OVER_STACK_KEYS = ('k_caused', 'ac_caused')

UNDERLY_FIGURES = [
    ('ac1030', ['ret_ac1030', 'msk_ac1030'], 'fig4_irf_underly_ac1030.png'),
    ('cot1030', ['ret_cot1030', 'msk_cot1030'], 'fig4_irf_underly_cot1030.png'),
]


# ============================================================
# Background and saving helpers
# ============================================================

def apply_background(fig, axes=None, fig_face_color=MAIN_FACE_COLOR, axes_face_color=None):
    """
    Apply background colors with alpha.

    fig_face_color controls the area outside axes.
    axes_face_color controls the area inside axes; if None, it follows fig_face_color.
    """
    fig.patch.set_facecolor(fig_face_color)
    fig.patch.set_alpha(fig_face_color[-1])

    if axes_face_color is None:
        axes_face_color = fig_face_color

    if axes is None:
        axes = fig.axes
    elif not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    for ax in axes:
        ax.patch.set_facecolor(axes_face_color)
        ax.patch.set_alpha(axes_face_color[-1])


def save_png(fig, out_path, dpi=300, bbox_inches='tight'):
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        facecolor=fig.get_facecolor(),
        edgecolor='none',
        transparent=False
    )


def get_split_output_dir(split_key):
    out_dir = os.path.join(BAR_EXPORT_DIR, split_key)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ============================================================
# Data loading and coefficients
# ============================================================

def load_global_data():
    """Load merged data without applying the cloud/retrieval mask."""
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

    return pd.concat(dfs, ignore_index=True)


def build_coef_lookup(coef_df, suffix=''):
    """Build lookup dict for k and lnb parameters."""
    lookup = {}
    for _, row in coef_df.iterrows():
        key = (str(row['Ocean']).strip(), str(row['Season']).strip())
        lookup[('k_ret', key)] = row[f'k_ret{suffix}']
        lookup[('lnb_ret', key)] = row[f'lnb_ret{suffix}']
        lookup[('k_msk', key)] = row[f'k_msk{suffix}']
        lookup[('lnb_msk', key)] = row[f'lnb_msk{suffix}']
    return lookup


def get_coef(lookup, method, ocean, season, param):
    return lookup.get((f'{param}_{method}', (ocean, season)), np.nan)


# ============================================================
# IRF calculation for maps and separate ocean-bar PNGs
# ============================================================

def compute_irf_data():
    """
    Compute IRF_aci only.

    Returns
    -------
    ocean_irf : dict
        ocean_irf[method][ocean][variant] = area-weighted IRF_aci.
        method is 'ret' or 'msk'; variant is 'ac1030', 'cot1030', or 'orig'.
    ocean_area : dict
        ocean_area[method][ocean][variant] = valid grid-cell area used for
        each ocean-level IRF_aci average.
    grid_irf : pandas.DataFrame
        Grid-level corrected Ac1030 and COT1030 IRF_aci for contour-line maps.
        Columns: method, variant, lat, lon, irf.
    overestimate : dict
        All-ocean area-weighted overestimate ratios for panel (c).
    """
    print('Computing IRF data...')

    merged_df = load_global_data()

    coef_ac1030 = pd.read_csv(SENSITIVITY_AC1030_CSV)
    coef_cot1030 = pd.read_csv(SENSITIVITY_COT1030_CSV)
    coef_ac1030_lookup = build_coef_lookup(coef_ac1030, suffix='')
    coef_cot1030_lookup = build_coef_lookup(coef_cot1030, suffix='_day')

    # Bellouin2013.csv is assumed to be wide format: Ocean, DJF, MAM, JJA, SON.
    lnnd_df = pd.read_csv(BELLOUIN2013_CSV)
    lnnd_df.columns = [c.strip() for c in lnnd_df.columns]
    lnnd_df['Ocean'] = lnnd_df['Ocean'].str.strip()
    lnnd_long = lnnd_df.melt(id_vars=['Ocean'], var_name='Season', value_name='lnnd')
    lnnd_long['Season'] = lnnd_long['Season'].str.strip()
    lnnd_lookup = lnnd_long.set_index(['Ocean', 'Season'])['lnnd'].to_dict()

    # SWdown and grid area
    merged_df['month'] = pd.to_datetime(merged_df['time']).dt.month
    unique_lat_month = merged_df[['lat', 'month']].drop_duplicates()
    unique_lat_month['swdown'] = unique_lat_month.apply(
        lambda r: calc_monthly_swdown(r['lat'], month=r['month']), axis=1
    )
    merged_df = merged_df.merge(unique_lat_month, on=['lat', 'month'], how='left')
    merged_df['grid_area_km2'] = merged_df['lat'].apply(calc_grid_cell_area)

    agg_cols = {
        'swdown': 'mean',
        'log_aod_diff': 'mean',
        'cf_liq_ceres': 'mean',       # CF_msk
        'cf_ret_liq_mod08': 'mean',   # CF_ret
        'cot_mod08': 'mean',
        'grid_area_km2': 'mean',
    }
    seasonal_grid = merged_df.groupby(['ocean', 'season', 'lat', 'lon']).agg(agg_cols).reset_index()

    ocean_irf = {method: {ocean: {} for ocean in oceans} for method in METHODS}
    ocean_area = {method: {ocean: {} for ocean in oceans} for method in METHODS}
    accum = {
        method: {
            ocean: {
                variant: {'sum': 0.0, 'area': 0.0}
                for variant in BAR_VARIANTS
            }
            for ocean in oceans
        }
        for method in METHODS
    }

    # All-ocean accumulators for panel (c).
    # Each denominator is an area-weighted IRF under a partially or fully corrected formula.
    over_accum = {
        group_key: {
            'uncorrected': {'sum': 0.0, 'area': 0.0},
            'k_caused_den': {'sum': 0.0, 'area': 0.0},
            'ac_caused_den': {'sum': 0.0, 'area': 0.0},
            'total_den': {'sum': 0.0, 'area': 0.0},
        }
        for group_key, _, _, _ in OVER_GROUPS
    }

    grid_records = []

    for ocean in oceans:
        for season in season_dict.keys():
            mask = (seasonal_grid['ocean'] == ocean) & (seasonal_grid['season'] == season)
            if not mask.any():
                continue

            sub = seasonal_grid[mask].copy()
            area = sub['grid_area_km2'].values.astype(float)
            if np.nansum(area[np.isfinite(area) & (area > 0)]) <= 0:
                continue

            lnnd_val = lnnd_lookup.get((ocean, season), np.nan)
            if np.isnan(lnnd_val):
                continue

            k_ret_ac1030 = get_coef(coef_ac1030_lookup, 'ret', ocean, season, 'k')
            lnb_ret_ac1030 = get_coef(coef_ac1030_lookup, 'ret', ocean, season, 'lnb')
            k_ret_cot1030 = get_coef(coef_cot1030_lookup, 'ret', ocean, season, 'k')
            lnb_ret_cot1030 = get_coef(coef_cot1030_lookup, 'ret', ocean, season, 'lnb')

            k_msk_ac1030 = get_coef(coef_ac1030_lookup, 'msk', ocean, season, 'k')
            lnb_msk_ac1030 = get_coef(coef_ac1030_lookup, 'msk', ocean, season, 'lnb')
            k_msk_cot1030 = get_coef(coef_cot1030_lookup, 'msk', ocean, season, 'k')
            lnb_msk_cot1030 = get_coef(coef_cot1030_lookup, 'msk', ocean, season, 'lnb')

            cot_vals = sub['cot_mod08'].values.astype(float)
            swdown = sub['swdown'].values.astype(float)
            log_aod_diff = sub['log_aod_diff'].values.astype(float)
            irf_base = swdown * lnnd_val * log_aod_diff / 3.0

            cf_ret_vals = sub['cf_ret_liq_mod08'].values.astype(float)
            cf_msk_vals = sub['cf_liq_ceres'].values.astype(float)

            variants = {
                'ret': {
                    'ac1030': (k_ret_ac1030, lnb_ret_ac1030, cf_ret_vals),
                    'cot1030': (k_ret_cot1030, lnb_ret_cot1030, cf_ret_vals),
                    'orig': (k_t91, lnb_t91, cf_ret_vals),
                },
                'msk': {
                    'ac1030': (k_msk_ac1030, lnb_msk_ac1030, cf_msk_vals),
                    'cot1030': (k_msk_cot1030, lnb_msk_cot1030, cf_msk_vals),
                    'orig': (k_t91, lnb_t91, cf_msk_vals),
                },
            }

            for method in METHODS:
                for variant in BAR_VARIANTS:
                    k_val, lnb_val, cf_vals = variants[method][variant]
                    if np.isnan(k_val) or np.isnan(lnb_val):
                        continue

                    Ac = cot_k_b_to_albedo(cot_vals, k_val, np.exp(lnb_val))
                    irf_vals = irf_base * k_val * Ac * (1 - Ac) * cf_vals

                    good = np.isfinite(irf_vals) & np.isfinite(area) & (area > 0)
                    if np.any(good):
                        accum[method][ocean][variant]['sum'] += np.nansum(irf_vals[good] * area[good])
                        accum[method][ocean][variant]['area'] += np.nansum(area[good])

                    # Grid-level data for Fig. 5 maps: corrected Ac1030 and COT1030.
                    if variant in ('ac1030', 'cot1030'):
                        grid_good = (
                            np.isfinite(irf_vals) &
                            np.isfinite(sub['lat'].values) &
                            np.isfinite(sub['lon'].values)
                        )
                        if np.any(grid_good):
                            grid_records.append(pd.DataFrame({
                                'method': method,
                                'variant': variant,
                                'lat': sub['lat'].values[grid_good].astype(float),
                                'lon': sub['lon'].values[grid_good].astype(float),
                                'irf': irf_vals[grid_good].astype(float),
                            }))

            # Panel (c): all-ocean overestimate decomposition.
            # User-defined formulas:
            #   uncorrected: k_t91 + Ac(k_t91, lnb_t91)
            #   Ac-caused denominator: k_corrected + Ac(k_t91, lnb_t91)
            #   k-caused denominator: k_t91 + Ac(k_corrected, lnb_corrected)
            #   total denominator: k_corrected + Ac(k_corrected, lnb_corrected)
            Ac_t91 = cot_k_b_to_albedo(cot_vals, k_t91, np.exp(lnb_t91))

            for group_key, method, variant, _ in OVER_GROUPS:
                k_corr, lnb_corr, cf_vals = variants[method][variant]
                if np.isnan(k_corr) or np.isnan(lnb_corr):
                    continue

                Ac_corr = cot_k_b_to_albedo(cot_vals, k_corr, np.exp(lnb_corr))

                irf_uncorrected = irf_base * k_t91 * Ac_t91 * (1 - Ac_t91) * cf_vals
                irf_k_caused_den = irf_base * k_t91 * Ac_corr * (1 - Ac_corr) * cf_vals
                irf_ac_caused_den = irf_base * k_corr * Ac_t91 * (1 - Ac_t91) * cf_vals
                irf_total_den = irf_base * k_corr * Ac_corr * (1 - Ac_corr) * cf_vals

                scenario_values = {
                    'uncorrected': irf_uncorrected,
                    'k_caused_den': irf_k_caused_den,
                    'ac_caused_den': irf_ac_caused_den,
                    'total_den': irf_total_den,
                }

                for scenario_name, irf_vals in scenario_values.items():
                    good = np.isfinite(irf_vals) & np.isfinite(area) & (area > 0)
                    if np.any(good):
                        over_accum[group_key][scenario_name]['sum'] += np.nansum(irf_vals[good] * area[good])
                        over_accum[group_key][scenario_name]['area'] += np.nansum(area[good])

    for method in METHODS:
        for ocean in oceans:
            for variant in BAR_VARIANTS:
                item = accum[method][ocean][variant]
                ocean_irf[method][ocean][variant] = (
                    item['sum'] / item['area'] if item['area'] > 0 else np.nan
                )
                ocean_area[method][ocean][variant] = item['area']

    overestimate = {}
    for group_key, _, _, _ in OVER_GROUPS:
        group_item = over_accum[group_key]

        def area_mean(name):
            item = group_item[name]
            return item['sum'] / item['area'] if item['area'] > 0 else np.nan

        uncorrected_mean = area_mean('uncorrected')
        k_caused_den_mean = area_mean('k_caused_den')
        ac_caused_den_mean = area_mean('ac_caused_den')
        total_den_mean = area_mean('total_den')

        # New definition:
        #   overestimate = all-ocean mean uncorrected-or-partly-uncorrected IRF
        #                  / all-ocean mean fully corrected IRF - 1
        #
        # Numerators:
        #   k_caused  : k_t91 + Ac(k_corrected, lnb_corrected)
        #   ac_caused : k_corrected + Ac(k_t91, lnb_t91)
        #   total     : k_t91 + Ac(k_t91, lnb_t91)
        # Denominator:
        #   fully corrected: k_corrected + Ac(k_corrected, lnb_corrected)
        if np.isfinite(total_den_mean) and not np.isclose(total_den_mean, 0):
            k_caused_over = k_caused_den_mean / total_den_mean - 1.0
            ac_caused_over = ac_caused_den_mean / total_den_mean - 1.0
            total_over = uncorrected_mean / total_den_mean - 1.0
        else:
            k_caused_over = np.nan
            ac_caused_over = np.nan
            total_over = np.nan

        overestimate[group_key] = {
            'k_caused': k_caused_over,
            'ac_caused': ac_caused_over,
            'total': total_over,
            'uncorrected_mean': uncorrected_mean,
            'k_partly_uncorrected_mean': k_caused_den_mean,
            'ac_partly_uncorrected_mean': ac_caused_den_mean,
            'corrected_mean': total_den_mean,
        }

    if grid_records:
        grid_irf = pd.concat(grid_records, ignore_index=True)
        grid_irf = grid_irf.groupby(['method', 'variant', 'lat', 'lon'], as_index=False)['irf'].mean()
    else:
        grid_irf = pd.DataFrame(columns=['method', 'variant', 'lat', 'lon', 'irf'])

    return ocean_irf, ocean_area, grid_irf, overestimate


# ============================================================
# Fig. 5: corrected Ac1030/COT1030 IRF contour-line maps
# ============================================================

def get_common_contour_levels(grid_irf, variant):
    vals = grid_irf.loc[grid_irf['variant'] == variant, 'irf'].values.astype(float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([])

    vmin = np.nanpercentile(vals, 5)
    vmax = np.nanpercentile(vals, 95)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)

    if np.isclose(vmin, vmax):
        return np.array([vmin])

    return np.unique(np.round(np.linspace(vmin, vmax, 3), 2))


def draw_irf_contour_map(ax, grid_irf, method, variant, panel_tag, levels):
    df = grid_irf[(grid_irf['method'] == method) & (grid_irf['variant'] == variant)].copy()

    ax.set_global()
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='black', linewidth=0.35, zorder=2)
    ax.coastlines(linewidth=0.45, color='black', zorder=3)

    gl = ax.gridlines(draw_labels=True, color='none')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}

    if df.empty or len(levels) == 0:
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
    else:
        lat_vals = np.sort(df['lat'].unique())
        lon_vals = np.sort(df['lon'].unique())
        z = (
            df.pivot_table(index='lat', columns='lon', values='irf', aggfunc='mean')
              .reindex(index=lat_vals, columns=lon_vals)
        )
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
        zvals = z.values.astype(float)

        if np.sum(np.isfinite(zvals)) >= 4 and len(levels) >= 2:
            cs = ax.contour(
                lon2d, lat2d, zvals,
                levels=levels,
                colors=CONTOUR_COLOR,
                linewidths=0.9,
                transform=ccrs.PlateCarree(),
                zorder=4
            )
            if len(cs.levels) > 0:
                ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f', colors=CONTOUR_COLOR)
        else:
            ax.scatter(
                df['lon'], df['lat'], s=4,
                color=CONTOUR_COLOR,
                transform=ccrs.PlateCarree(), zorder=4
            )
            ax.text(0.5, 0.04, 'Too few gridded points for contour lines',
                    transform=ax.transAxes, ha='center', va='bottom', fontsize=9)

    ax.set_title(PANEL_TITLES[method], fontsize=13, pad=7)
    ax.text(-0.01, 1.01, panel_tag,
            transform=ax.transAxes, fontsize=17, va='bottom', ha='left')


def draw_overestimate_bars(ax, overestimate, group_keys, panel_tag, title):
    """Draw panel (c): horizontal stacked partial overestimates and total overestimate."""
    label_lookup = {item[0]: item[3] for item in OVER_GROUPS}
    group_labels = [label_lookup[g] for g in group_keys]

    stack_vals = [
        np.array([overestimate[g].get(key, np.nan) for g in group_keys], dtype=float)
        for key in OVER_STACK_KEYS
    ]
    total_vals = np.array([overestimate[g].get('total', np.nan) for g in group_keys], dtype=float)

    y = np.arange(len(group_keys))
    height = 0.28
    left = np.zeros(len(group_keys), dtype=float)

    for vals, label, color in zip(stack_vals, OVER_BAR_LABELS[:2], OVER_BAR_COLORS[:2]):
        ax.barh(
            y, vals,
            height=height,
            left=left,
            color=color,
            edgecolor=color,
            alpha=0.78,
            linewidth=1.0,
            label=label
        )
        left = left + vals

    # The total overestimate is shown as a diamond scatter at the same y position.
    ax.scatter(
        total_vals, y,
        marker='D',
        s=62,
        color=OVER_BAR_COLORS[2],
        edgecolor=OVER_BAR_COLORS[2],
        linewidth=0.6,
        zorder=5,
        label=OVER_BAR_LABELS[2]
    )
        
    # Annotate each diamond with its percentage value.
    for yi, tv in zip(y, total_vals):
        if np.isfinite(tv):
            ax.annotate(
                f'{tv*100:.0f}%',
                xy=(tv, yi),
                xytext=(7, 0),
                textcoords='offset points',
                ha='left',
                va='center',
                fontsize=9,
                color=OVER_BAR_COLORS[2],
                fontweight='bold'
            )

    ax.axvline(0.0, color='0.25', linewidth=1.0, linestyle='--')
    ax.set_yticks(y)
    ax.set_yticklabels(group_labels, fontsize=11)
    ax.set_title(title, fontsize=13, pad=7)
    ax.text(-0.01, 1.01, panel_tag,
            transform=ax.transAxes, fontsize=17, va='bottom', ha='left')

    finite_vals = np.concatenate([
        left[np.isfinite(left)],
        total_vals[np.isfinite(total_vals)],
        *[vals[np.isfinite(vals)] for vals in stack_vals]
    ])
    if finite_vals.size > 0:
        xmin = min(0.0, np.nanmin(finite_vals) * 1.15)
        xmax = max(0.05, np.nanmax(finite_vals) * 1.15)
        if np.isclose(xmin, xmax):
            xmin, xmax = -0.05, 0.05
        ax.set_xlim(xmin, xmax)

    ax.set_axisbelow(True)
    ax.grid(axis='x', linestyle='--', linewidth=0.6, alpha=0.35)

    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='both', direction='out', length=3, width=0.8)
    legend = ax.legend(
        frameon=False,
        fontsize=11,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        labelspacing=1.25,
        ncol=1
    )
    return legend


def shrink_axes_for_right_legend(fig, ax, legend, target_right, anchor_x=1.02):
    """Shrink an axes so its outside-right legend ends at target_right."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    pos = ax.get_position()

    new_width = (target_right - legend_bbox.width - pos.x0) / anchor_x
    if new_width > 0:
        ax.set_position([pos.x0, pos.y0, new_width, pos.height])


# ============================================================
# Separate PNGs: split ocean-level bar charts + split legends
# ============================================================

def set_bar_axes_style(ax, show_ylabel=False, ylim=(0, 1.5)):
    # Only left and bottom spines.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)

    # No xticks and no x tick labels.
    ax.set_xticks([])
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', labelsize=8.5, direction='out', length=3, width=0.8)
    ax.set_ylim(*ylim)
    if show_ylabel:
        ax.set_ylabel(r'IRF$_{\mathrm{aci}}$ (W m$^{-2}$)', fontsize=11)
    else:
        ax.set_ylabel('')
        # Keep y ticks and y tick labels for non-NPO panels; only remove the ylabel.
        ax.tick_params(axis='y', labelleft=True)


def draw_single_ocean_bar(ax, ocean_irf, method, ocean, variants):
    color_lookup = BAR_COLORS[method]
    vals = np.asarray([ocean_irf[method][ocean].get(var, np.nan) for var in variants], dtype=float)
    plot_vals = np.nan_to_num(vals, nan=0.0)
    x = np.arange(len(variants))

    ax.bar(
        x, plot_vals,
        width=0.62,
        color=[color_lookup[var] for var in variants],
        edgecolor='k',
        linewidth=1.2,
        alpha=BAR_ALPHA
    )
    ax.axhline(0, color='0.25', linewidth=0.8)
    ax.set_axisbelow(True)
    # ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.30)

    # Title inside the axes, horizontally centered.
    ax.text(0.1, 0.93, ocean, transform=ax.transAxes,
            ha='left', va='top', fontsize=15)

    set_bar_axes_style(ax, show_ylabel=(ocean == 'NPO'), ylim=BAR_YLIMS[method])


def save_bar_differences(ocean_irf):
    """Save orig - ac1030 bar differences for both retrieval methods."""
    records = []
    for method in METHODS:
        for ocean in oceans:
            ac1030 = ocean_irf[method][ocean].get('ac1030', np.nan)
            orig = ocean_irf[method][ocean].get('orig', np.nan)
            records.append({
                'ocean': ocean,
                'method': method,
                'bar_diff': orig - ac1030,
            })
    os.makedirs(os.path.dirname(BAR_DIFF_CSV), exist_ok=True)
    pd.DataFrame(records).to_csv(BAR_DIFF_CSV, index=False)


def save_ocean_bar_pngs(ocean_irf):
    for method in METHODS:
        for split_key, variants in BAR_SPLITS.items():
            out_dir = get_split_output_dir(split_key)
            for ocean in oceans:
                fig = plt.figure(figsize=(1.85, 2.0))
                ax = fig.add_axes(BAR_AX_POS)
                apply_background(
                    fig, ax,
                    fig_face_color=TRANSPARENT_FACE_COLOR,
                    axes_face_color=(1, 1, 1, BAR_ALPHA)
                )
                draw_single_ocean_bar(ax, ocean_irf, method, ocean, variants)
                ax.set_position(BAR_AX_POS)
                out_path = os.path.join(out_dir, f'fig4_{method}_{ocean}_{split_key}_irf_bars.png')
                save_png(fig, out_path, dpi=300, bbox_inches=None)
                plt.close(fig)
                # print(f'Saved: {out_path}')


def area_weighted_ocean_mean(ocean_irf, ocean_area, method, variant):
    vals = np.asarray([
        ocean_irf[method][ocean].get(variant, np.nan)
        for ocean in oceans
    ], dtype=float)
    weights = np.asarray([
        ocean_area[method][ocean].get(variant, np.nan)
        for ocean in oceans
    ], dtype=float)

    good = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.nan

    return np.sum(vals[good] * weights[good]) / np.sum(weights[good])


def save_bar_legend_pngs(ocean_irf, ocean_area):
    for method in METHODS:
        color_lookup = BAR_COLORS[method]

        for split_key, variants in BAR_SPLITS.items():
            out_dir = get_split_output_dir(split_key)
            labels = []
            for variant in variants:
                irf_mean = area_weighted_ocean_mean(ocean_irf, ocean_area, method, variant)
                labels.append(
                    rf'{BAR_LABELS[variant]}: {irf_mean:.3f} W m$^{{-2}}$'
                )

            handles = [
                Patch(
                    facecolor=color_lookup[variant],
                    edgecolor='k',
                    alpha=BAR_ALPHA,
                    label=label
                )
                for variant, label in zip(variants, labels)
            ]

            fig = plt.figure(figsize=(5.8, 0.62))
            apply_background(fig, fig_face_color=LEGEND_FACE_COLOR)
            fig.legend(
                handles=handles,
                labels=labels,
                loc='center',
                ncol=1,
                frameon=False,
                fontsize=9,
                title_fontsize=10.5,
                handlelength=1.6,
                columnspacing=1.2
            )

            out_path = os.path.join(out_dir, f'fig4_{method}_{split_key}_irf_bar_legend.png')
            save_png(fig, out_path, dpi=300)
            plt.close(fig)
            # print(f'Saved: {out_path}')


def save_underly_figure(grid_irf, overestimate, variant, group_keys, out_name):
    levels = get_common_contour_levels(grid_irf, variant)

    fig = plt.figure(figsize=(12, 11.0))

    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1.0, 1.0, 0.35],
        hspace=0.25,
        bottom=0.065,
        top=0.965,
        left=0.06,
        right=0.97
    )

    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_b = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax_c = fig.add_subplot(gs[2, 0])

    draw_irf_contour_map(ax_a, grid_irf, 'ret', variant, format_panel_tag(0, 'nature'), levels)
    draw_irf_contour_map(ax_b, grid_irf, 'msk', variant, format_panel_tag(1, 'nature'), levels)
    legend_c = draw_overestimate_bars(
        ax_c,
        overestimate,
        group_keys,
        format_panel_tag(2, 'nature'),
        r'Relative Overestimation of IRF$_{\mathrm{aci}}$'
    )
    shrink_axes_for_right_legend(fig, ax_c, legend_c, target_right=ax_a.get_position().x1)

    out_path = os.path.join(get_split_output_dir(variant), out_name)
    save_png(fig, out_path, dpi=300)
    plt.close(fig)
    # print(f'Saved: {out_path}')

# ============================================================
# Main
# ============================================================

def main():
    ocean_irf, ocean_area, grid_irf, overestimate = compute_irf_data()
    for variant, group_keys, out_name in UNDERLY_FIGURES:
        save_underly_figure(grid_irf, overestimate, variant, group_keys, out_name)

    save_bar_differences(ocean_irf)

    # Separate outputs: split ocean bar PNGs + split legend PNGs.
    save_ocean_bar_pngs(ocean_irf)
    save_bar_legend_pngs(ocean_irf, ocean_area)


if __name__ == '__main__':
    main()
