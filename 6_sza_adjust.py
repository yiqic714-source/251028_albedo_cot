import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_fitting import cot_to_x, albedo_to_y, _fit_odr_once, format_panel_tag

# Path configuration
BASE_DATA_DIR = '/home/chenyiqi/251028_albedo_cot/processed_data/merged_data/'
WEIGHTED_FILE = '/home/chenyiqi/251028_albedo_cot/processed_data/ocean_season_sza_weighted.csv'
HEATMAP_DATA_DIR = '/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table/cot_sza_to_albedo_lookup_table_cp/'
COEF_KB_FILE = '/home/chenyiqi/251028_albedo_cot/processed_data/coef_k_b.csv'

# Core parameters
OCEANS = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']
SEASONS = ['MAM', 'JJA', 'SON', 'DJF']
SEASON_MONTHS = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}

# Calculation parameters
MIN_POINTS_FOR_FIT = 2
LN_COT_LOW = 1.
LN_COT_HIGH = 3.

# Output configuration
OUTPUT_PNG = 'figs/k_lnb_plot.png'
OUTPUT_CSV = 'processed_data/coef_k_b_szacorr.csv'

# Plot configuration
HEATMAP_CMAP = plt.cm.GnBu
LNB_CMAP = plt.cm.pink_r
K_VMIN, K_VMAX = 0.25, 0.9
LNB_VMIN, LNB_VMAX = -2.7, -0.6
SIZE_PARAMS = {
    'large_tick': 12,
    'small_tick': 9.5,
    'xylabel': 15,
    'title': 17,
    'legend': 11,
    'cbar_tick': 10.5,
}


def load_weighted_angles(file_path):
    """Load weighted SZA data. Return dict: {(ocean, season): weighted_angle_deg}"""
    df = pd.read_csv(file_path)
    df = df[~df['season'].isin(['Global'])]
    return {(row['ocean'], row['season']): row['weighted_angle_deg'] for _, row in df.iterrows()}


def calculate_seasonal_stats(ocean_list, data_dir):
    """Calculate mean SZA per ocean-season. Return dict: {(ocean, season): mean_sza}"""
    seasonal_stats = {}
    for ocean in ocean_list:
        dfs = []
        for season in SEASONS:
            fp = os.path.join(data_dir, f'{ocean}_{season}.csv')
            if os.path.exists(fp):
                dfs.append(pd.read_csv(fp))
        if not dfs:
            for season in SEASONS:
                seasonal_stats[(ocean, season)] = np.nan
            continue
        df = pd.concat(dfs, ignore_index=True)
        df['time'] = pd.to_datetime(df['time'], format='mixed')
        df['month'] = df['time'].dt.month
        for season_name, months in SEASON_MONTHS.items():
            df.loc[df['month'].isin(months), 'season'] = season_name
        seasonal_avg = df.dropna(subset=['season']).groupby('season')['sza'].mean()
        for season in SEASONS:
            seasonal_stats[(ocean, season)] = seasonal_avg.loc[season] if season in seasonal_avg.index else np.nan
    return seasonal_stats


def compute_k_and_intercept(cot_vals, albedo_vals):
    """Calculate slope (k) and intercept (lnb) for ln(A/(1-A)) vs ln(COT) in range [1.5, 3.0]."""
    mask = ((cot_vals > 0.0) & np.isfinite(cot_vals) & np.isfinite(albedo_vals) &
            (albedo_vals > 0.0) & (albedo_vals < 1.0))
    if np.sum(mask) < MIN_POINTS_FOR_FIT:
        return np.nan, np.nan
    x = cot_to_x(cot_vals[mask])
    y = albedo_to_y(albedo_vals[mask])
    range_mask = (x >= LN_COT_LOW) & (x <= LN_COT_HIGH)
    if np.sum(range_mask) < MIN_POINTS_FOR_FIT:
        return np.nan, np.nan
    try:
        k, b = _fit_odr_once(x[range_mask], y[range_mask],
                              np.full(np.sum(range_mask), 1e-12),
                              np.full(np.sum(range_mask), 1e-12))
        return float(k), float(b)
    except Exception:
        return np.nan, np.nan


def get_lookup_data(ocean, season):
    """Load lookup table and return cos(SZA), slope (k), and intercept (lnb)."""
    file_path = os.path.join(HEATMAP_DATA_DIR, f'cot_sza_to_albedo_lookup_table_{ocean}_{season}.csv')
    df = pd.read_csv(file_path, index_col=0)
    sza = np.array(df.index.astype(float))
    cot = np.array(df.columns.astype(float))
    albedo_grid = df.values.astype(float)
    sort_sza_idx = np.argsort(sza)
    sza_sorted = sza[sort_sza_idx]
    cos_sza_sorted = np.cos(np.radians(sza_sorted))
    albedo_sorted = albedo_grid[sort_sza_idx, :]
    slope_list, intercept_list = [], []
    for i in range(len(sza_sorted)):
        slope, intercept = compute_k_and_intercept(cot, albedo_sorted[i, :])
        slope_list.append(slope)
        intercept_list.append(intercept)
    return cos_sza_sorted, np.array(slope_list), np.array(intercept_list)


def get_value_at_sza(cos_sza_vals, target_cos_sza, value_array):
    """Get the closest value from value_array at target cos(SZA)."""
    if cos_sza_vals is None or value_array is None:
        return np.nan
    valid_mask = np.isfinite(cos_sza_vals) & np.isfinite(value_array)
    if not np.any(valid_mask):
        return np.nan
    cos_sza_valid = cos_sza_vals[valid_mask]
    value_valid = value_array[valid_mask]
    closest_idx = np.argmin(np.abs(cos_sza_valid - target_cos_sza))
    return value_valid[closest_idx]


def plot_heatmap(ax, df, title, cmap=HEATMAP_CMAP, vmin=None, vmax=None):
    """Plot heatmap with oceans (x-axis), seasons (y-axis)."""
    for col in SEASONS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan
    heatmap_data = np.where(np.isinf(df[SEASONS].values.astype(np.float64)), np.nan, df[SEASONS].values.astype(np.float64)).T
    oceans = df['ocean'].tolist()
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(oceans)))
    ax.set_yticks(np.arange(len(SEASONS)))
    ax.set_xticklabels(oceans, fontsize=SIZE_PARAMS['large_tick'], rotation=90, ha='right')
    ax.set_yticklabels(SEASONS, fontsize=SIZE_PARAMS['large_tick'])
    for i in range(len(SEASONS)):
        for j in range(len(oceans)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha="center", va="center", color='k', fontsize=9.5, fontweight='bold')
    ax.set_title(title, fontsize=SIZE_PARAMS['title'], pad=7, loc='left')
    return im


def plot_main_ax(ax, seasonal_stats, weight_dict, is_k_plot=True, cmap=HEATMAP_CMAP,
                 panel_idx=None, icon_style='nature', title_y=None):
    """Plot main axis with SZA range 20-75° and legend at top-left."""
    all_sza = []
    for ocean in OCEANS:
        for season in SEASONS:
            cos_sza, _, _ = get_lookup_data(ocean, season)
            if cos_sza is not None:
                all_sza.extend(np.degrees(np.arccos(cos_sza)))
    unique_sza = np.sort(np.unique(all_sza)) if all_sza else np.array([])
    n_y = len(unique_sza)
    n_x = len(OCEANS) * len(SEASONS)
    main_data = np.full((n_y, n_x), np.nan)
    x_ticks, ocean_label_pos, ocean_labels = [], [], []
    mean_sza_x, mean_sza_y = [], []
    weighted_sza_x, weighted_sza_y = [], []
    for o_idx, ocean in enumerate(OCEANS):
        x_start = o_idx * len(SEASONS)
        x_end = x_start + len(SEASONS)
        ocean_label_pos.append((x_start + x_end - 1) / 2)
        ocean_labels.append(ocean)
        for s_idx, season in enumerate(SEASONS):
            x_pos = x_start + s_idx
            x_ticks.append(season)
            mean_sza_deg = seasonal_stats[(ocean, season)]
            weighted_sza_deg = weight_dict.get((ocean, season), np.nan)
            cos_sza, slope_vals, intercept_vals = get_lookup_data(ocean, season)
            if cos_sza is not None:
                sza_vals = np.degrees(np.arccos(cos_sza))
                for y_idx, target_sza in enumerate(unique_sza):
                    closest_idx = np.argmin(np.abs(sza_vals - target_sza))
                    if np.isclose(sza_vals[closest_idx], target_sza):
                        main_data[y_idx, x_pos] = slope_vals[closest_idx] if is_k_plot else intercept_vals[closest_idx]
            if not np.isnan(mean_sza_deg):
                mean_sza_x.append(x_pos)
                mean_sza_y.append(mean_sza_deg)
            if not np.isnan(weighted_sza_deg):
                weighted_sza_x.append(x_pos)
                weighted_sza_y.append(weighted_sza_deg)
    if not np.all(np.isnan(main_data)):
        vmin = K_VMIN if is_k_plot else LNB_VMIN
        vmax = K_VMAX if is_k_plot else LNB_VMAX
        ax.imshow(main_data, aspect='auto', cmap=cmap, extent=[-0.5, n_x - 0.5, 20, 75], vmin=vmin, vmax=vmax)
        mean_mask = (np.array(mean_sza_y) >= 20) & (np.array(mean_sza_y) <= 75)
        weighted_mask = (np.array(weighted_sza_y) >= 20) & (np.array(weighted_sza_y) <= 75)
        ax.scatter(np.array(mean_sza_x)[mean_mask], np.array(mean_sza_y)[mean_mask],
                   color='red', s=50, marker='o', label='10:30', zorder=5, edgecolors='black')
        ax.scatter(np.array(weighted_sza_x)[weighted_mask], np.array(weighted_sza_y)[weighted_mask],
                   color='blue', s=60, marker='^', label='Daytime', zorder=5, edgecolors='black')
        for i in range(len(OCEANS) - 1):
            ax.axvline(x=len(SEASONS) * (i + 1) - 0.5, color='lightgray', linestyle='-', linewidth=1, zorder=4)
        ax.set_ylabel(r'SZA ($^\circ$)', fontsize=SIZE_PARAMS['large_tick'])
        ax.set_xlim(-0.5, n_x - 0.5)
        ax.set_ylim(20, 75)
        ax.set_yticks(np.arange(20, 76, 10))
        ax.set_yticklabels([f'{int(x)}' for x in np.arange(20, 76, 10)], fontsize=SIZE_PARAMS['large_tick'])
        ax.set_xticks(range(n_x))
        ax.set_xticklabels(x_ticks, fontsize=SIZE_PARAMS['small_tick'], va='top', rotation=90, ha='right')
        for pos, label in zip(ocean_label_pos, ocean_labels):
            ax.text(pos, 76, label, ha='center', va='bottom', fontsize=SIZE_PARAMS['large_tick'])
        ax.legend(loc='upper center', fontsize=SIZE_PARAMS['legend'], frameon=True)
        if panel_idx is None:
            panel_idx = 3 if is_k_plot else 4
        panel_tag = format_panel_tag(panel_idx, icon_style)
        if is_k_plot:
            ax.set_title(f'{panel_tag}   $k_{{\mathrm{{cp}}}}$ vs. SZA', loc='left', fontsize=SIZE_PARAMS['title'], pad=1, y=title_y)
        else:
            ax.set_title(f'{panel_tag}   ln$b_{{\mathrm{{cp}}}}$ vs. SZA', loc='left', fontsize=SIZE_PARAMS['title'], pad=1, y=title_y)


def create_main_plot(icon_style='nature'):
    """Create combined plot with k_ret (row 1), lnb_ret (row 2), k_msk (row 3)."""
    if icon_style not in ('nature', 'science'):
        raise ValueError("icon_style must be 'nature' or 'science'.")

    seasonal_stats = calculate_seasonal_stats(OCEANS, BASE_DATA_DIR)
    weight_dict = load_weighted_angles(WEIGHTED_FILE)

    diff_k_data = pd.DataFrame(index=OCEANS, columns=SEASONS, dtype=float)
    diff_b_data = pd.DataFrame(index=OCEANS, columns=SEASONS, dtype=float)

    for ocean in OCEANS:
        for season in SEASONS:
            mean_sza_deg = seasonal_stats[(ocean, season)]
            weighted_sza_deg = weight_dict.get((ocean, season), np.nan)
            cos_sza, slope_vals, intercept_vals = get_lookup_data(ocean, season)
            if not np.isnan(mean_sza_deg) and not np.isnan(weighted_sza_deg):
                cos_mean_sza = np.cos(np.radians(mean_sza_deg))
                cos_weighted_sza = np.cos(np.radians(weighted_sza_deg))
                k_mean = get_value_at_sza(cos_sza, cos_mean_sza, slope_vals)
                k_weighted = get_value_at_sza(cos_sza, cos_weighted_sza, slope_vals)
                b_mean = get_value_at_sza(cos_sza, cos_mean_sza, intercept_vals)
                b_weighted = get_value_at_sza(cos_sza, cos_weighted_sza, intercept_vals)
                if np.isfinite(k_weighted) and np.isfinite(k_mean):
                    diff_k_data.loc[ocean, season] = k_weighted - k_mean
                if np.isfinite(b_weighted) and np.isfinite(b_mean):
                    diff_b_data.loc[ocean, season] = b_weighted - b_mean

    # Load uncorrected values from coef_k_b.csv and apply SZA corrections
    coef_df = pd.read_csv(COEF_KB_FILE)
    coef_df = coef_df[coef_df['Ocean'] != 'Global'].copy()

    uncor_dfs = {}  # method -> {ocean -> {season -> (slope, intercept)}}
    for method in ['ret', 'msk']:
        mdf = coef_df[coef_df['Method'] == method]
        tbl = {}
        for _, row in mdf.iterrows():
            tbl.setdefault(row['Ocean'], {})[row['Season']] = (row['Slope'], row['Intercept'])
        uncor_dfs[method] = tbl

    szacorr_dfs = {}  # method -> DataFrame with columns [ocean, MAM, JJA, SON, DJF] for slope
    szacorr_lnb_dfs = {}
    for method in ['ret', 'msk']:
        tbl = uncor_dfs[method]
        k_rows, b_rows = [], []
        for ocean in OCEANS:
            k_row, b_row = {'ocean': ocean}, {'ocean': ocean}
            for season in SEASONS:
                k0, b0 = tbl.get(ocean, {}).get(season, (np.nan, np.nan))
                dk = diff_k_data.loc[ocean, season] if np.isfinite(diff_k_data.loc[ocean, season]) else 0
                db = diff_b_data.loc[ocean, season] if np.isfinite(diff_b_data.loc[ocean, season]) else 0
                k_row[season] = k0 + dk
                b_row[season] = b0 + db
            k_rows.append(k_row)
            b_rows.append(b_row)
        szacorr_dfs[method] = pd.DataFrame(k_rows).round(4)
        szacorr_lnb_dfs[method] = pd.DataFrame(b_rows).round(4)

    uncor_k2_df = pd.DataFrame([{'ocean': o, **{s: uncor_dfs['ret'][o][s][0] for s in SEASONS}} for o in OCEANS]).round(4)
    uncor_k1_df = pd.DataFrame([{'ocean': o, **{s: uncor_dfs['msk'][o][s][0] for s in SEASONS}} for o in OCEANS]).round(4)
    uncor_lnb2_df = pd.DataFrame([{'ocean': o, **{s: uncor_dfs['ret'][o][s][1] for s in SEASONS}} for o in OCEANS]).round(4)

    # Save single CSV with Method, Ocean, Season, Slope, Intercept
    records = []
    for method in ['msk', 'ret']:
        k_df = szacorr_dfs[method]
        b_df = szacorr_lnb_dfs[method]
        for _, row in k_df.iterrows():
            ocean = row['ocean']
            for season in SEASONS:
                records.append({
                    'Method': method,
                    'Ocean': ocean,
                    'Season': season,
                    'Slope': row[season],
                    'Intercept': b_df.loc[b_df['ocean'] == ocean, season].values[0],
                })
    output_df = pd.DataFrame(records).round(4)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"SZA-corrected coefficients saved to: {OUTPUT_CSV}")

    fig = plt.figure(figsize=(18, 17), dpi=100)

    # Row 1 (3 panels): 10:30
    ax_k_a = fig.add_axes([0.06, 0.70, 0.26, 0.22])
    im_k_a = plot_heatmap(ax_k_a, uncor_k2_df,
                          f'{format_panel_tag(0, icon_style)}   $k_{{\mathrm{{ret}}}}$, 10:30',
                          vmin=K_VMIN, vmax=K_VMAX)
    ax_lnb_a = fig.add_axes([0.38, 0.70, 0.26, 0.22])
    im_lnb_a = plot_heatmap(ax_lnb_a, uncor_lnb2_df,
                            f'{format_panel_tag(1, icon_style)}   ln$b_{{\mathrm{{ret}}}}$, 10:30',
                            cmap=LNB_CMAP, vmin=LNB_VMIN, vmax=LNB_VMAX)
    ax_k1_g = fig.add_axes([0.70, 0.70, 0.26, 0.22])
    im_k1_g = plot_heatmap(ax_k1_g, uncor_k1_df,
                           f'{format_panel_tag(2, icon_style)}   $k_{{\mathrm{{msk}}}}$, 10:30',
                           vmin=K_VMIN, vmax=K_VMAX)

    # Row 2 (2 panels): cp vs SZA
    ax_k_b = fig.add_axes([0.06, 0.40, 0.26, 0.22])
    plot_main_ax(ax_k_b, seasonal_stats, weight_dict, is_k_plot=True,
                 panel_idx=3, icon_style=icon_style, title_y=1.10)
    ax_lnb_b = fig.add_axes([0.38, 0.40, 0.26, 0.22])
    plot_main_ax(ax_lnb_b, seasonal_stats, weight_dict, is_k_plot=False,
                 cmap=LNB_CMAP, panel_idx=4, icon_style=icon_style, title_y=1.10)

    # Row 3 (2 panels): Daytime Mean
    ax_k_c = fig.add_axes([0.06, 0.12, 0.26, 0.22])
    im_k_c = plot_heatmap(ax_k_c, szacorr_dfs['ret'],
                          f'{format_panel_tag(5, icon_style)}   $k_{{\mathrm{{ret}}}}$, Daytime Mean',
                          vmin=K_VMIN, vmax=K_VMAX)
    ax_lnb_c = fig.add_axes([0.38, 0.12, 0.26, 0.22])
    im_lnb_c = plot_heatmap(ax_lnb_c, szacorr_lnb_dfs['ret'],
                            f'{format_panel_tag(6, icon_style)}   ln$b_{{\mathrm{{ret}}}}$, Daytime Mean',
                            cmap=LNB_CMAP, vmin=LNB_VMIN, vmax=LNB_VMAX)

    # Color bars
    cbar_c_ax = fig.add_axes([0.70, 0.65, 0.26, 0.014])
    cbar_c = fig.colorbar(im_k1_g, cax=cbar_c_ax, orientation='horizontal')
    cbar_c.set_label('$k$', fontsize=SIZE_PARAMS['xylabel'])
    cbar_c.set_ticks(np.arange(0.25, 0.91, 0.1))
    cbar_c.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])

    cbar_f_ax = fig.add_axes([0.06, 0.07, 0.26, 0.014])
    cbar_f = fig.colorbar(im_k_c, cax=cbar_f_ax, orientation='horizontal')
    cbar_f.set_label('$k$', fontsize=SIZE_PARAMS['xylabel'])
    cbar_f.set_ticks(np.arange(0.25, 0.91, 0.1))
    cbar_f.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])

    cbar_g_ax = fig.add_axes([0.38, 0.07, 0.26, 0.014])
    cbar_g = fig.colorbar(im_lnb_c, cax=cbar_g_ax, orientation='horizontal')
    cbar_g.set_label('ln$b$', fontsize=SIZE_PARAMS['xylabel'])
    cbar_g.set_ticks(np.arange(-2.7, -0.59, 0.3))
    cbar_g.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('figs', exist_ok=True)
    icon_style = 'science'
    print("=== Generating combined k, lnb and k1 plot ===")
    create_main_plot(icon_style=icon_style)
