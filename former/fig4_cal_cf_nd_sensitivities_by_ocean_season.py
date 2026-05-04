import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from generate_lncf_lnnd_lnaod_npz import generate_npz_for_year


oceans = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']
season_names = ['MAM', 'JJA', 'SON', 'DJF']

OUTPUT_DIR = '/home/chenyiqi/251028_albedo_cot/processed_data'
FIG_DIR = '/home/chenyiqi/251028_albedo_cot/figs'

YEAR_START = 2020
YEAR_END = 2020

AOD_TRIM = 0.15  # fraction to trim from each tail of ln_aod
CWP_MIN = 20
CF_MIN = 0.05
BINS = 20


def load_or_generate_year(year):
    path = os.path.join(OUTPUT_DIR, f'cf_lnnd_lnaod_{year}.npz')

    if not os.path.isfile(path):
        print(f'Generating NPZ for {year}...')
        generate_npz_for_year(year)

    if not os.path.isfile(path):
        print(f'Failed to load {year}')
        return None

    return np.load(path)


def merge_years(yearly_data):
    keys = ['ln_nd', 'ln_aod', 'ln_cf_ret', 'ln_cwp', 'ln_cf_msk',
            'ocean_idx', 'season_idx']
    merged = {k: [] for k in keys}

    for data in yearly_data:
        if data is None:
            continue
        for k in keys:
            merged[k].append(data[k])

    return {k: np.concatenate(v) if v else np.array([]) for k, v in merged.items()}


def get_array(ocean_data, ocean_i, season_i, key):
    """Extract data for a given ocean and season."""
    mask = (ocean_data['ocean_idx'] == ocean_i) & (ocean_data['season_idx'] == season_i)
    return ocean_data[key][mask]


def filter_data(x, y, ln_aod, ln_cwp, ln_cf_ret, ln_cf_msk, filter_type):
    """Common filtering: cf_ret > CF_MIN, cf_msk > CF_MIN, CWP > CWP_MIN,
    plus type-specific filter (AOD range for 'aod')."""
    good = (np.isfinite(x) & np.isfinite(y) & np.isfinite(ln_cwp) &
            np.isfinite(ln_cf_ret) & np.isfinite(ln_cf_msk))
    if filter_type == 'aod':
        good &= np.isfinite(ln_aod)

    x, y = x[good], y[good]
    ln_cwp, ln_cf_ret, ln_cf_msk = ln_cwp[good], ln_cf_ret[good], ln_cf_msk[good]
    if filter_type == 'aod':
        ln_aod = ln_aod[good]

    # CWP filter
    cwp_mask = np.exp(ln_cwp) > CWP_MIN
    x, y = x[cwp_mask], y[cwp_mask]
    ln_cf_ret, ln_cf_msk = ln_cf_ret[cwp_mask], ln_cf_msk[cwp_mask]
    if filter_type == 'aod':
        ln_aod = ln_aod[cwp_mask]

    # cf_ret > CF_MIN and cf_msk > CF_MIN (for all analyses)
    if filter_type == 'cf':
        cf_mask = (np.exp(ln_cf_ret) > CF_MIN) & (np.exp(ln_cf_msk) > CF_MIN)
        x, y = x[cf_mask], y[cf_mask]
        # ln_aod = ln_aod[cf_mask]

    # AOD trim filter (only for 'aod' analysis): remove smallest and largest AOD_TRIM fraction
    if filter_type == 'aod':
        lower_bound = np.percentile(ln_aod, AOD_TRIM * 100)
        upper_bound = np.percentile(ln_aod, (1 - AOD_TRIM) * 100)
        aod_mask = (ln_aod >= lower_bound) & (ln_aod <= upper_bound)
        x, y = x[aod_mask], y[aod_mask]

    return x, y


def plot_density_fit(
    ocean_data, x_key, y_key,
    x_label, y_label, title, fig_name, csv_name,
    filter_type, legend_loc='lower right'
):
    fig, axes = plt.subplots(4, 8, figsize=(32, 16))
    fig.suptitle(title, fontsize=20, y=0.98)

    records = []
    im = None

    for ri, season_i in enumerate(range(4)):
        for ci, ocean_i in enumerate(range(8)):
            ax = axes[ri, ci]

            x = get_array(ocean_data, ocean_i, season_i, x_key)
            y = get_array(ocean_data, ocean_i, season_i, y_key)
            ln_aod = get_array(ocean_data, ocean_i, season_i, 'ln_aod')
            ln_cwp = get_array(ocean_data, ocean_i, season_i, 'ln_cwp')
            ln_cf_ret = get_array(ocean_data, ocean_i, season_i, 'ln_cf_ret')
            ln_cf_msk = get_array(ocean_data, ocean_i, season_i, 'ln_cf_msk')

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

            x_f, y_f = filter_data(x, y, ln_aod, ln_cwp, ln_cf_ret, ln_cf_msk, filter_type)

            if len(x_f) < 10:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                records.append({
                    'Ocean': oceans[ci], 'Season': season_names[ri],
                    'Slope': np.nan, 'Intercept': np.nan,
                    'R2': np.nan, 'N': len(x_f)
                })
                continue

            # Bin x data and compute mean y in each bin
            bin_edges = np.linspace(x_f.min(), x_f.max(), BINS + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            y_means = []
            x_centers_used = []
            for i in range(BINS):
                mask = (x_f >= bin_edges[i]) & (x_f < bin_edges[i + 1])
                if i == BINS - 1:
                    mask = (x_f >= bin_edges[i]) & (x_f <= bin_edges[i + 1])
                if np.sum(mask) >= 3:
                    y_means.append(np.nanmean(y_f[mask]))
                    x_centers_used.append(bin_centers[i])

            x_centers_used = np.array(x_centers_used)
            y_means = np.array(y_means)

            # Plot 2D histogram
            hist, x_edges, y_edges = np.histogram2d(x_f, y_f, bins=BINS)
            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
            im = ax.imshow(hist.T, origin='lower', extent=extent,
                           aspect='auto', cmap='viridis')

            # Fit on binned mean points
            if len(x_centers_used) >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_centers_used, y_means)
                x_fit = np.linspace(x_f.min(), x_f.max(), 100)
                ax.plot(x_fit, slope * x_fit + intercept, 'r-', lw=2,
                        label=f'k={slope:.4f}\n$R^2$={r_value**2:.4f}')
                ax.scatter(x_centers_used, y_means, c='red', s=20, marker='o', zorder=3)
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

    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax).set_label('Count', fontsize=14)

    fig.savefig(os.path.join(FIG_DIR, fig_name), dpi=300, bbox_inches='tight')
    plt.close(fig)

    pd.DataFrame(records).to_csv(os.path.join(OUTPUT_DIR, csv_name), index=False)
    print(f'Figure saved: {fig_name}')
    print(f'CSV saved: {csv_name}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    yearly_data = [
        load_or_generate_year(year)
        for year in range(YEAR_START, YEAR_END + 1)
    ]

    ocean_data = merge_years(yearly_data)

    plot_density_fit(
        ocean_data=ocean_data,
        x_key='ln_aod', y_key='ln_nd',
        x_label='ln(AOD)', y_label='ln(nd)',
        title='ln(nd) vs ln(AOD) with Linear Fit',
        fig_name='lnnd_vs_lnaod.png',
        csv_name='coef_lnnd_vs_lnaod0503.csv',
        filter_type='aod',
        legend_loc='lower right'
    )

    plot_density_fit(
        ocean_data=ocean_data,
        x_key='ln_nd', y_key='ln_cf_ret',
        x_label='ln(nd)', y_label='ln(cf_ret)',
        title='ln(cf_ret) vs ln(nd) with Linear Fit',
        fig_name='ln_cf_ret_vs_lnnd.png',
        csv_name='coef_ln_cf_ret_vs_lnnd.csv',
        filter_type='cf',
        legend_loc='upper right'
    )

    plot_density_fit(
        ocean_data=ocean_data,
        x_key='ln_nd', y_key='ln_cf_msk',
        x_label='ln(nd)', y_label='ln(cf_msk)',
        title='ln(cf_msk) vs ln(nd) with Linear Fit',
        fig_name='ln_cf_msk_vs_lnnd.png',
        csv_name='coef_ln_cf_msk_vs_lnnd.csv',
        filter_type='cf',
        legend_loc='upper right',
    )


if __name__ == '__main__':
    main()
