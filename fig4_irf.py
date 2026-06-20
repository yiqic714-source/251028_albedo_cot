import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from utils_fitting import oceans, season_dict, format_panel_tag
from utils_solar import cot_k_b_to_albedo, calc_monthly_swdown, calc_grid_cell_area

# Paths
BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIG_DIR = f'{BASE_PATH}/figs'
SENSITIVITY_1030_CSV = f'{BASE_PATH}/processed_data/sensitivity_albedo_vs_cot_1030.csv'
SENSITIVITY_DAY_CSV = f'{BASE_PATH}/processed_data/sensitivity_albedo_vs_cot_day.csv'
BELLOUIN2013_CSV = f'{BASE_PATH}/processed_data/Bellouin2013.csv'
os.makedirs(FIG_DIR, exist_ok=True)

# Output folder for the 16 separate bar PNGs and 2 separate legend PNGs
BAR_EXPORT_DIR = os.path.join(FIG_DIR, 'fig4_ocean_irf_bars')
os.makedirs(BAR_EXPORT_DIR, exist_ok=True)

# Backgrounds
MAIN_FACE_COLOR = (1, 1, 1, 0.55)          # main Fig. 5
TRANSPARENT_FACE_COLOR = (1, 1, 1, 0.0)   # outside the axes for 16 bar PNGs
LEGEND_FACE_COLOR = (1, 1, 1, 0.25)       # 2 exported legend PNGs

# Map style
CONTOUR_COLOR = '#7B3294'  # purple contour lines for both panels
MAP_EXTENT = [-180, 180, -60, 60]
MAP_ASPECT = 3.0          # physical width / height for 360 deg x 120 deg

PANEL_TITLES = {
    'ret': r'Corrected IRF$_{\mathrm{aci}}$ Using Retrieval-Domain Obs.',
    'msk': r'Corrected IRF$_{\mathrm{aci}}$ Using Mask-Domain Obs.',
}

# T91/uncorrected parameters used for the third bar in the separate ocean PNGs
k_t91 = 1.0
lnb_t91 = np.log(0.13)

# Ocean-bar settings. The order is fixed as in the exported bar PNGs.
BAR_VARIANTS = ['1030', 'day', 'orig']
BAR_LABELS = [
    r'Corrected, $A_{\mathrm{c,1030}}$',
    r'Corrected, COT$_{\mathrm{1030}}$',
    'Uncorrected'
]
BAR_XTICK_LABELS = [r'$A_{c,1030}$', r'COT$_{1030}$', 'Uncorr.']
BAR_PALETTES = {
    'ret': {
        'irf': ['#D55E00', '#E69F00', '#F0C808'],
        'legend_title': 'Retrieval-domain observations'
    },
    'msk': {
        'irf': ['#C46A5A', '#8B1E3F', '#E7C2A3'],
        'legend_title': 'Mask-domain observations'
    },
}
BAR_ALPHA = 0.65
BAR_YLIMS = {
    'ret': (0, 1.1),
    'msk': (0, 1.5),
}
BAR_AX_POS = [0.28, 0.22, 0.66, 0.66]  # fixed axes box for all 16 bar PNGs


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
        method is 'ret' or 'msk'; variant is '1030', 'day', or 'orig'.
    grid_irf : pandas.DataFrame
        Grid-level corrected Ac_1030 IRF_aci for contour-line maps.
        Columns: method, lat, lon, irf.
    """
    print('Computing IRF data...')

    methods = ['ret', 'msk']
    merged_df = load_global_data()

    coef_1030 = pd.read_csv(SENSITIVITY_1030_CSV)
    coef_day = pd.read_csv(SENSITIVITY_DAY_CSV)
    coef_1030_lookup = build_coef_lookup(coef_1030, suffix='')
    coef_day_lookup = build_coef_lookup(coef_day, suffix='_day')

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
        'grid_area_km2': 'first',
    }
    seasonal_grid = merged_df.groupby(['ocean', 'season', 'lat', 'lon']).agg(agg_cols).reset_index()

    ocean_irf = {method: {ocean: {} for ocean in oceans} for method in methods}
    accum = {
        method: {
            ocean: {
                variant: {'sum': 0.0, 'area': 0.0}
                for variant in BAR_VARIANTS
            }
            for ocean in oceans
        }
        for method in methods
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

            k_ret_1030 = get_coef(coef_1030_lookup, 'ret', ocean, season, 'k')
            lnb_ret_1030 = get_coef(coef_1030_lookup, 'ret', ocean, season, 'lnb')
            k_ret_day = get_coef(coef_day_lookup, 'ret', ocean, season, 'k')
            lnb_ret_day = get_coef(coef_day_lookup, 'ret', ocean, season, 'lnb')

            k_msk_1030 = get_coef(coef_1030_lookup, 'msk', ocean, season, 'k')
            lnb_msk_1030 = get_coef(coef_1030_lookup, 'msk', ocean, season, 'lnb')
            k_msk_day = get_coef(coef_day_lookup, 'msk', ocean, season, 'k')
            lnb_msk_day = get_coef(coef_day_lookup, 'msk', ocean, season, 'lnb')

            cot_vals = sub['cot_mod08'].values.astype(float)
            swdown = sub['swdown'].values.astype(float)
            log_aod_diff = sub['log_aod_diff'].values.astype(float)
            irf_base = swdown * lnnd_val * log_aod_diff

            cf_ret_vals = sub['cf_ret_liq_mod08'].values.astype(float)
            cf_msk_vals = sub['cf_liq_ceres'].values.astype(float)

            variants = {
                'ret': {
                    '1030': (k_ret_1030, lnb_ret_1030, cf_ret_vals),
                    'day': (k_ret_day, lnb_ret_day, cf_ret_vals),
                    'orig': (k_t91, lnb_t91, cf_ret_vals),
                },
                'msk': {
                    '1030': (k_msk_1030, lnb_msk_1030, cf_msk_vals),
                    'day': (k_msk_day, lnb_msk_day, cf_msk_vals),
                    'orig': (k_t91, lnb_t91, cf_msk_vals),
                },
            }

            for method in methods:
                for variant in BAR_VARIANTS:
                    k_val, lnb_val, cf_vals = variants[method][variant]
                    if np.isnan(k_val) or np.isnan(lnb_val):
                        continue

                    Ac = cot_k_b_to_albedo(cot_vals, k_val, np.exp(lnb_val))
                    irf_vals = (irf_base / 3.0) * k_val * Ac * (1 - Ac) * cf_vals

                    good = np.isfinite(irf_vals) & np.isfinite(area) & (area > 0)
                    if np.any(good):
                        accum[method][ocean][variant]['sum'] += np.nansum(irf_vals[good] * area[good])
                        accum[method][ocean][variant]['area'] += np.nansum(area[good])

                    # Grid-level data for Fig. 5 maps: corrected Ac_1030 only.
                    if variant == '1030':
                        grid_good = (
                            np.isfinite(irf_vals) &
                            np.isfinite(sub['lat'].values) &
                            np.isfinite(sub['lon'].values)
                        )
                        if np.any(grid_good):
                            grid_records.append(pd.DataFrame({
                                'method': method,
                                'lat': sub['lat'].values[grid_good].astype(float),
                                'lon': sub['lon'].values[grid_good].astype(float),
                                'irf': irf_vals[grid_good].astype(float),
                            }))

    for method in methods:
        for ocean in oceans:
            for variant in BAR_VARIANTS:
                item = accum[method][ocean][variant]
                ocean_irf[method][ocean][variant] = (
                    item['sum'] / item['area'] if item['area'] > 0 else np.nan
                )

    if grid_records:
        grid_irf = pd.concat(grid_records, ignore_index=True)
        grid_irf = grid_irf.groupby(['method', 'lat', 'lon'], as_index=False)['irf'].mean()
    else:
        grid_irf = pd.DataFrame(columns=['method', 'lat', 'lon', 'irf'])

    return ocean_irf, grid_irf


# ============================================================
# Fig. 5: corrected Ac_1030 IRF contour-line maps
# ============================================================

def get_common_contour_levels(grid_irf):
    vals = grid_irf['irf'].values.astype(float)
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


def draw_irf_contour_map(ax, grid_irf, method, panel_tag, levels):
    df = grid_irf[grid_irf['method'] == method].copy()

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


def align_map_axes_to_full_width(fig, ax_top, ax_bottom, x0=0.06, x1=0.97,
                                 y0=0.045, y1=0.975, gap=0.095,
                                 map_aspect=MAP_ASPECT):
    """
    Make the two maps span the same left/right limits while preserving the
    PlateCarree aspect for [-180, 180] x [-60, 60], and force a visible gap
    between the two map panels.
    """
    fig_w, fig_h = fig.get_size_inches()
    width = x1 - x0
    height = (width * fig_w / map_aspect) / fig_h

    available = y1 - y0
    total = 2 * height + gap
    if total > available:
        height = (available - gap) / 2.0

    y_bottom = y0 + (available - (2 * height + gap)) / 2.0
    ax_bottom.set_position([x0, y_bottom, width, height])
    ax_top.set_position([x0, y_bottom + height + gap, width, height])


# ============================================================
# Separate PNGs: 16 ocean-level bar charts + 2 legends
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


def draw_single_ocean_bar(ax, ocean_irf, method, ocean):
    colors = BAR_PALETTES[method]['irf']
    vals = np.asarray([ocean_irf[method][ocean].get(var, np.nan) for var in BAR_VARIANTS], dtype=float)
    plot_vals = np.nan_to_num(vals, nan=0.0)
    x = np.arange(len(BAR_VARIANTS))

    ax.bar(x, plot_vals, width=0.62, color=colors, edgecolor=colors, linewidth=1.2, alpha=BAR_ALPHA)
    ax.axhline(0, color='0.25', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.30)

    # Title inside the axes, horizontally centered.
    ax.text(0.5, 0.93, ocean, transform=ax.transAxes,
            ha='center', va='top', fontsize=15)

    set_bar_axes_style(ax, show_ylabel=(ocean == 'NPO'), ylim=BAR_YLIMS[method])


def save_ocean_bar_pngs(ocean_irf):
    for method in ['ret', 'msk']:
        for ocean in oceans:
            fig = plt.figure(figsize=(2.0, 1.75))
            ax = fig.add_axes(BAR_AX_POS)
            apply_background(
                fig, ax,
                fig_face_color=TRANSPARENT_FACE_COLOR,
                axes_face_color=(1, 1, 1, BAR_ALPHA)
            )
            draw_single_ocean_bar(ax, ocean_irf, method, ocean)
            ax.set_position(BAR_AX_POS)

            out_path = os.path.join(BAR_EXPORT_DIR, f'fig4_{method}_{ocean}_irf_bars.png')
            save_png(fig, out_path, dpi=300, bbox_inches=None)
            plt.close(fig)
            print(f'Saved: {out_path}')


def save_bar_legend_pngs():
    for method in ['ret', 'msk']:
        colors = BAR_PALETTES[method]['irf']
        handles = [
            Patch(facecolor=colors[i], edgecolor=colors[i], alpha=BAR_ALPHA, label=BAR_LABELS[i])
            for i in range(len(BAR_VARIANTS))
        ]

        fig = plt.figure(figsize=(5.8, 0.8))
        apply_background(fig, fig_face_color=LEGEND_FACE_COLOR)
        fig.legend(
            handles=handles,
            labels=BAR_LABELS,
            loc='center',
            ncol=1,
            frameon=False,
            fontsize=10,
            title_fontsize=10.5,
            handlelength=1.6,
            columnspacing=1.2
        )
        out_path = os.path.join(BAR_EXPORT_DIR, f'fig4_{method}_irf_bar_legend.png')
        save_png(fig, out_path, dpi=300)
        plt.close(fig)
        print(f'Saved: {out_path}')


# ============================================================
# Main
# ============================================================

def main():
    ocean_irf, grid_irf = compute_irf_data()
    levels = get_common_contour_levels(grid_irf)

    fig = plt.figure(figsize=(12, 8.9))

    gs = fig.add_gridspec(
        2, 1,
        hspace=0.75,
        bottom=0.055,
        top=0.965,
        left=0.06,
        right=0.97
    )

    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_b = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    

    draw_irf_contour_map(ax_a, grid_irf, 'ret', format_panel_tag(0, 'nature'), levels)
    draw_irf_contour_map(ax_b, grid_irf, 'msk', format_panel_tag(1, 'nature'), levels)

    # Explicitly preserve the map aspect and enforce a visible gap between panels.
    align_map_axes_to_full_width(fig, ax_a, ax_b, x0=0.06, x1=0.97, y0=0.045, y1=0.975, gap=0.095)

    out_path = os.path.join(FIG_DIR, 'fig4_corrected_ac1030_irf_maps.png')
    save_png(fig, out_path, dpi=300)
    plt.close(fig)
    print(f'Saved: {out_path}')

    # Separate outputs: 16 ocean bar PNGs + 2 legend PNGs.
    save_ocean_bar_pngs(ocean_irf)
    save_bar_legend_pngs()


if __name__ == '__main__':
    main()
