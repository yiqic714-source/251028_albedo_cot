import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


oceans = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']
season_names = ['MAM', 'JJA', 'SON', 'DJF']

BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
FIG_DIR = f'{BASE_PATH}/figs'
OUTPUT_DIR = f'{BASE_PATH}/processed_data'

AOD_TRIM = 0.10  # fraction to trim from each tail of ln_aod
ND_TRIM = 0.10   # fraction to trim from each tail of ln_nd
CF_MIN = 0.
BINS = 20
COT_MIN = 2.5


def load_ocean_season_data():
    """Load all L3_product/{ocean}_{season}.csv files and return a dict
    with keys: 'ln_nd', 'ln_aod', 'cf_ret', 'cf_msk',
    'ocean_idx', 'season_idx'.
    """
    data = {k: [] for k in ['ln_nd', 'ln_aod', 'cf_ret',
                             'cf_msk', 'ocean_idx', 'season_idx']}

    for oi, ocean in enumerate(oceans):
        for si, season in enumerate(season_names):
            path = f"{BASE_PATH}/processed_data/merged_data/{ocean}_{season}.csv"

            df = pd.read_csv(path)
            # Compute required variables
            nd = df['nd'].values
            aod = df['aod_mod08'].values
            cf_ret = df['cf_ret_liq_mod08'].values
            cf_msk = df['cf_liq_ceres'].values
            cot = df['cot_mod08'].values
            valid_mask = (nd > 0) & (aod > 0) & (cf_ret > CF_MIN) & (cf_msk > CF_MIN) & (cot > COT_MIN)

            data['ln_nd'].append(np.log(nd[valid_mask]))
            data['ln_aod'].append(np.log(aod[valid_mask]))
            data['cf_ret'].append(cf_ret[valid_mask])
            data['cf_msk'].append(cf_msk[valid_mask])
            data['ocean_idx'].append(np.full(len(nd[valid_mask]), oi, dtype=int))
            data['season_idx'].append(np.full(len(nd[valid_mask]), si, dtype=int))

    return {k: np.concatenate(v) if v else np.array([]) for k, v in data.items()}


def get_array(ocean_data, ocean_i, season_i, key):
    """Extract data for a given ocean and season."""
    mask = (ocean_data['ocean_idx'] == ocean_i) & (ocean_data['season_idx'] == season_i)
    return ocean_data[key][mask]


def trim_data(x, y, trim_pct):

    lower = np.percentile(x, trim_pct * 100)
    upper = np.percentile(x, (1 - trim_pct) * 100)
    trim = (x >= lower) & (x <= upper)
    x, y = x[trim], y[trim]

    return x, y


def plot_density_fit(
    ocean_data, x_key, y_key,
    x_label, y_label, title, fig_name, csv_name,
    trim_pct, legend_loc='lower right'
):
    fig, axes = plt.subplots(4, 8, figsize=(32, 16))
    fig.suptitle(title, fontsize=20, y=0.98)

    records = []
    pcm = None

    for ri, season_i in enumerate(range(4)):
        for ci, ocean_i in enumerate(range(8)):
            ax = axes[ri, ci]

            x = get_array(ocean_data, ocean_i, season_i, x_key)
            y = get_array(ocean_data, ocean_i, season_i, y_key)
            ln_aod = get_array(ocean_data, ocean_i, season_i, 'ln_aod')
            cf_ret = get_array(ocean_data, ocean_i, season_i, 'cf_ret')
            cf_msk = get_array(ocean_data, ocean_i, season_i, 'cf_msk')

            ax.set_title(f'{oceans[ci]} {season_names[ri]}')

            if len(x) < 10:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                records.append({
                    'Ocean': oceans[ci], 'Season': season_names[ri],
                    'Slope': np.nan, 'Intercept': np.nan,
                    'R2': np.nan, 'N': len(x)
                })
                continue

            x_f, y_f = trim_data(x, y, trim_pct)

            if len(x_f) < 10:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                records.append({
                    'Ocean': oceans[ci], 'Season': season_names[ri],
                    'Slope': np.nan, 'Intercept': np.nan,
                    'R2': np.nan, 'N': len(x_f)
                })
                continue

            # --- Equal-data-quantity bins in x direction ---
            sort_idx = np.argsort(x_f)
            x_sorted = x_f[sort_idx]
            y_sorted = y_f[sort_idx]

            n_per_bin = len(x_sorted) // BINS
            x_centers = []
            y_medians = []
            for i in range(BINS):
                start = i * n_per_bin
                end = (i + 1) * n_per_bin if i < BINS - 1 else len(x_sorted)
                if end - start < 3:
                    continue
                x_centers.append(np.median(x_sorted[start:end]))
                y_medians.append(np.median(y_sorted[start:end]))

            x_centers = np.array(x_centers)
            y_medians = np.array(y_medians)

            # --- Plot 2D histogram: x = equal-data-quantity bins, y = uniform bins ---
            x_bin_edges = np.percentile(x_f, np.linspace(0, 100, BINS + 1))
            x_bin_edges = np.unique(x_bin_edges)
            y_bin_edges = np.linspace(y_f.min(), y_f.max(), BINS + 1)
            hist, _, _ = np.histogram2d(x_f, y_f, bins=[x_bin_edges, y_bin_edges])
            pcm = ax.pcolormesh(x_bin_edges, y_bin_edges, hist.T,
                                cmap='viridis', shading='auto')

            # --- Fit on binned median points ---
            if len(x_centers) >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_centers, y_medians)
                x_fit = np.linspace(x_f.min(), x_f.max(), 100)
                ax.plot(x_fit, slope * x_fit + intercept, 'r-', lw=2,
                        label=f'k={slope:.4f}\n$R^2$={r_value**2:.4f}')
                ax.scatter(x_centers, y_medians, c='red', s=20, marker='o', zorder=3)
            else:
                slope, intercept, r_value = np.nan, np.nan, np.nan
                ax.text(0.5, 0.5, 'Not enough bins', ha='center', va='center',
                        transform=ax.transAxes)

            ax.legend(fontsize=8, loc=legend_loc)

            if ci == 0:
                ax.set_ylabel(y_label)
            if ri == 3:
                ax.set_xlabel(x_label)

            records.append({
                'Ocean': oceans[ci], 'Season': season_names[ri],
                'Slope': slope, 'Intercept': intercept,
                'R2': r_value ** 2, 'P_value': p_value,
                'Std_err': std_err, 'N': len(x_f)
            })

    fig.subplots_adjust(right=0.92, hspace=0.3, wspace=0.3)

    if pcm is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        fig.colorbar(pcm, cax=cbar_ax).set_label('Count', fontsize=14)

    fig.savefig(os.path.join(FIG_DIR, fig_name), dpi=300, bbox_inches='tight')
    plt.close(fig)

    pd.DataFrame(records).to_csv(os.path.join(OUTPUT_DIR, csv_name), index=False)
    print(f'Figure saved: {fig_name}')
    print(f'CSV saved: {csv_name}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print('Loading L3_product data...')
    ocean_data = load_ocean_season_data()
    print(f'Total valid points: {len(ocean_data["ln_nd"])}')

    plot_density_fit(
        ocean_data=ocean_data,
        x_key='ln_aod', y_key='ln_nd',
        x_label='ln(AOD)', y_label='ln(nd)',
        title='ln(nd) vs ln(AOD) with Linear Fit',
        fig_name='lnnd_vs_lnaod.png',
        csv_name='coef_lnnd_vs_lnaod.csv',
        trim_pct=AOD_TRIM,
        legend_loc='lower right'
    )

    plot_density_fit(
        ocean_data=ocean_data,
        x_key='ln_nd', y_key='cf_ret',
        x_label='ln(nd)', y_label='cf_ret',
        title='cf_ret vs ln(nd) with Linear Fit',
        fig_name='cf_ret_vs_lnnd.png',
        csv_name='coef_cf_ret_vs_lnnd.csv',
        trim_pct=ND_TRIM,
        legend_loc='upper right'
    )

    plot_density_fit(
        ocean_data=ocean_data,
        x_key='ln_nd', y_key='cf_msk',
        x_label='ln(nd)', y_label='cf_msk',
        title='cf_msk vs ln(nd) with Linear Fit',
        fig_name='cf_msk_vs_lnnd.png',
        csv_name='coef_cf_msk_vs_lnnd.csv',
        trim_pct=ND_TRIM,
        legend_loc='upper right',
    )


if __name__ == '__main__':
    main()
