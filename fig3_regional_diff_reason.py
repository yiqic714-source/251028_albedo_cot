import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from utils_fitting import oceans, season_dict, format_panel_tag

# Paths
BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
SENSITIVITY_CSV = f'{BASE_PATH}/processed_data/sensitivity_albedo_vs_cot_1030.csv'
MERGED_DIR = f'{BASE_PATH}/processed_data/merged_data'
FIG_DIR = f'{BASE_PATH}/figs'
os.makedirs(FIG_DIR, exist_ok=True)

MIN_COT = 2.5
MIN_CF = 0.1


def filter_merged_data(df):
    """Apply the same screening logic as preprocess_ocean_data."""
    df = df.copy()

    df['albedo'] = (
        (df['sw_all'] - df['sw_clr'] * (1 - df['cf_liq_ceres'])) /
        df['cf_liq_ceres'] / df['solar_incoming']
    )

    mask = (
        (df['cf_ceres'] > MIN_CF) &
        (df['cf_liq_ceres'] / df['cf_ceres'] > 0.99) &
        (df['cot_mod08'] > MIN_COT) &
        (df['ret_cot_cer'] > MIN_COT) &
        (df['ret_albedo'].between(0, 1)) &
        (df['albedo'].between(0, 1))
    )

    return df[mask].dropna().reset_index(drop=True)


def build_points():
    """
    Build 8*4 ocean-season points with:
      - k values from sensitivity CSV (wide format)
      - unr_fra, cot_disp, sza from merged_data CSVs
    """
    # Read sensitivity coefficients (wide format: Ocean, Season, k_dcp, b_dcp, ...)
    coef_df = pd.read_csv(SENSITIVITY_CSV)

    rows = []

    for ocean in oceans:
        for season_name in season_dict.keys():
            # --- Get k values from sensitivity CSV ---
            ocean_season = coef_df[(coef_df['Ocean'] == ocean) & (coef_df['Season'] == season_name)]

            if ocean_season.empty:
                continue

            k_dcp = float(ocean_season['k_dcp'].values[0])
            k_cp = float(ocean_season['k_cp'].values[0])
            k_ret = float(ocean_season['k_ret'].values[0])
            k_msk = float(ocean_season['k_msk'].values[0])

            if not (np.isfinite(k_dcp) and np.isfinite(k_cp) and np.isfinite(k_ret) and np.isfinite(k_msk)):
                continue

            k_cp_minus_k_ret = k_cp - k_ret
            k_dcp_minus_k_cp = k_dcp - k_cp
            k_ret_minus_k_msk = k_ret - k_msk

            # --- Get unr_fra, cot_disp, sza from merged data ---
            file_path = os.path.join(MERGED_DIR, f'{ocean}_{season_name}.csv')
            if not os.path.exists(file_path):
                continue

            df_raw = pd.read_csv(file_path)
            df_filtered = filter_merged_data(df_raw)

            if len(df_filtered) == 0:
                continue

            df_filtered['unr_fra'] = df_filtered['cf_ceres'] - df_filtered['cf_ret_liq_mod08']
            df_filtered['cot_disp'] = df_filtered['ret_cotstd_cer'] / df_filtered['ret_cot_cer']

            unr_fra_mean = df_filtered['unr_fra'].mean()
            cot_disp_mean = df_filtered['cot_disp'].mean()
            sza_mean = df_filtered['sza'].mean()

            rows.append({
                'ocean': ocean,
                'season': season_name,
                'unr_fra': unr_fra_mean,
                'cot_disp': cot_disp_mean,
                'sza': sza_mean,
                'k_cp_minus_k_ret': k_cp_minus_k_ret,
                'k_dcp_minus_k_cp': k_dcp_minus_k_cp,
                'k_ret_minus_k_msk': k_ret_minus_k_msk,
            })

    return pd.DataFrame(rows)


def plot_scatter_points(ax, points_df, x_col, y_col):
    """Draw scatter points (no fit line, no legend) on given axes."""
    season_markers = {
        'MAM': 'o',
        'JJA': 's',
        'SON': '^',
        'DJF': 'D',
    }
    ocean_colors = {
        o: plt.cm.tab10(i % 10) for i, o in enumerate(oceans)
    }

    for ocean in oceans:
        for season in season_dict.keys():
            sub = points_df[(points_df['ocean'] == ocean) & (points_df['season'] == season)]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=58,
                alpha=0.9,
                marker=season_markers.get(season, 'o'),
                color=ocean_colors[ocean],
            )


def add_fit_line_and_r(ax, points_df, x_col, y_col):
    """Add linear regression fit line and R/p text to axes."""
    valid = points_df[[x_col, y_col]].dropna()
    if len(valid) < 2:
        return

    slope, intercept, r_value, p_value, _ = stats.linregress(
        valid[x_col].values,
        valid[y_col].values,
    )

    x_line = np.linspace(valid[x_col].min(), valid[x_col].max(), 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='black', lw=2, label='fit line')

    ax.text(
        0.56, 0.10,
        f'R={r_value:.2f}, p={p_value:.3f}',
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=11.5,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'edgecolor': 'black', 'linewidth': 0.8},
    )


def create_shared_legend():
    """Create legend handles for seasons and oceans."""
    season_markers = {
        'MAM': 'o',
        'JJA': 's',
        'SON': '^',
        'DJF': 'D',
    }
    ocean_colors = {
        o: plt.cm.tab10(i % 10) for i, o in enumerate(oceans)
    }

    season_labels = list(season_dict.keys())
    ocean_labels = list(oceans)
    season_handles = {
        s: plt.Line2D([0], [0], marker=season_markers[s], color='black', linestyle='', markersize=7)
        for s in season_labels
    }
    ocean_handles = {
        o: plt.Line2D([0], [0], marker='o', color=ocean_colors[o], linestyle='', markersize=7)
        for o in ocean_labels
    }

    half = len(ocean_labels) // 2
    left_oceans = ocean_labels[:half]
    right_oceans = ocean_labels[half:]
    combined_handles = [
        *[season_handles[s] for s in season_labels],
        *[ocean_handles[o] for o in left_oceans],
        *[ocean_handles[o] for o in right_oceans],
    ]
    combined_labels = [
        *season_labels,
        *left_oceans,
        *right_oceans,
    ]
    return combined_handles, combined_labels


def main():
    print('Building points from sensitivity CSV and merged data...')
    points_df = build_points()
    print(f'Total points: {len(points_df)}')

    # ================================================================
    # Figure 1: Two subplots sharing one legend on the right
    #   Left:  k_dcp - k_cp  vs  sza
    #   Right: k_cp - k_ret  vs  cot_disp
    # ================================================================
    fig1, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    fig1.subplots_adjust(wspace=0.29)

    # Left subplot: k_dcp - k_cp vs sza
    plot_scatter_points(ax_left, points_df, x_col='sza', y_col='k_dcp_minus_k_cp')
    add_fit_line_and_r(ax_left, points_df, x_col='sza', y_col='k_dcp_minus_k_cp')
    ax_left.set_xlabel('SZA (deg)', fontsize=14)
    ax_left.set_ylabel(r'$k_{\mathrm{dcp}} - k_{\mathrm{cp}}$', fontsize=14)
    ax_left.set_yticks([-0.03, 0, 0.03, 0.06, 0.09, 0.12, 0.15])
    ax_left.grid(True, linestyle='--', alpha=0.3)
    ax_left.text(-0.01, 1.01, format_panel_tag(1, 'nature'),
                 transform=ax_left.transAxes, fontsize=17, va='bottom', ha='left')

    # Right subplot: k_cp - k_ret vs cot_disp
    plot_scatter_points(ax_right, points_df, x_col='cot_disp', y_col='k_cp_minus_k_ret')
    add_fit_line_and_r(ax_right, points_df, x_col='cot_disp', y_col='k_cp_minus_k_ret')
    ax_right.set_xlabel(r'$d_{\mathrm{COT}}$', fontsize=14)
    ax_right.set_ylabel(r'$k_{\mathrm{cp}} - k_{\mathrm{ret}}$', fontsize=14)
    ax_right.grid(True, linestyle='--', alpha=0.3)
    ax_right.text(-0.01, 1.01, format_panel_tag(2, 'nature'),
                  transform=ax_right.transAxes, fontsize=17, va='bottom', ha='left')

    # Shared legend on the right side of both subplots
    handles, labels = create_shared_legend()
    fig1.legend(
        handles, labels,
        ncol=1,
        loc='center left',
        bbox_to_anchor=(0.94, 0.5),
        borderaxespad=0.0,
        framealpha=0.5,
        title='Season | Ocean',
        fontsize=11
    )
    out_path1 = os.path.join(FIG_DIR, 'fig3_scatter_cotdisp_and_sza.png')
    fig1.savefig(out_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f'Saved: {out_path1}')

    # ================================================================
    # Figure 2: Single plot: k_ret - k_msk vs unr_fra (with legend)
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(6, 4.8))

    plot_scatter_points(ax2, points_df, x_col='unr_fra', y_col='k_ret_minus_k_msk')
    add_fit_line_and_r(ax2, points_df, x_col='unr_fra', y_col='k_ret_minus_k_msk')
    ax2.set_xlabel('Twilight Zone Fraction', fontsize=12)
    ax2.set_ylabel(r'$k_{\mathrm{ret}} - k_{\mathrm{msk}}$', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Legend on the right outside
    handles, labels = create_shared_legend()
    ax2.legend(
        handles, labels,
        ncol=3,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        framealpha=0.5,
        title='Season | Ocean',
    )

    out_path2 = os.path.join(FIG_DIR, 'fig3_scatter_unr_vs_kret_minus_kmsk.png')
    fig2.savefig(out_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {out_path2}')


if __name__ == '__main__':
    main()
