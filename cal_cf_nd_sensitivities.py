import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from generate_cf_lnnd_lnaod_npz import generate_npz_for_year


oceans = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']

season_dict = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}

OUTPUT_DIR = '/home/chenyiqi/251028_albedo_cot/processed_data'
FIG_DIR = '/home/chenyiqi/251028_albedo_cot/figs'

YEAR_START = 2020
YEAR_END = 2020

AOD_MIN = 0.07
AOD_MAX = 0.8
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

    return np.load(path, allow_pickle=True)['ocean_data'].item()


def merge_years(yearly_data):
    keys = ['ln_nd', 'ln_aod', 'cf_ret', 'ln_cwp']
    merged = {
        ocean: {s: {k: [] for k in keys} for s in season_dict}
        for ocean in oceans
    }

    for data in yearly_data:
        if data is None:
            continue
        for ocean in oceans:
            for season in season_dict:
                for k in keys:
                    merged[ocean][season][k].extend(data[ocean][season][k])

    return merged


def get_array(ocean_data, ocean, season, key):
    values = ocean_data[ocean][season][key]
    if len(values) == 0:
        return np.array([])
    return np.concatenate(values)


def filter_data(x, y, ln_aod, ln_cwp, cf_ret, filter_type):
    """
    Apply filters based on analysis type.
    filter_type='aod':  AOD + CWP filters
    filter_type='cf':   cf_ret + CWP filters
    """
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(ln_cwp)
    if filter_type == 'aod':
        good &= np.isfinite(ln_aod)
    elif filter_type == 'cf':
        good &= np.isfinite(cf_ret)
    x, y, ln_cwp = x[good], y[good], ln_cwp[good]
    if filter_type == 'aod':
        ln_aod = ln_aod[good]
    elif filter_type == 'cf':
        cf_ret = cf_ret[good]

    # CWP filter (common to both)
    cwp_mask = np.exp(ln_cwp) > CWP_MIN
    x, y = x[cwp_mask], y[cwp_mask]
    if filter_type == 'aod':
        ln_aod = ln_aod[cwp_mask]
    elif filter_type == 'cf':
        cf_ret = cf_ret[cwp_mask]

    # Type-specific filter
    if filter_type == 'aod':
        aod_mask = (np.exp(ln_aod) > AOD_MIN) & (np.exp(ln_aod) < AOD_MAX)
        x, y = x[aod_mask], y[aod_mask]
    elif filter_type == 'cf':
        cf_mask = cf_ret > CF_MIN
        x, y = x[cf_mask], y[cf_mask]

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

    for ri, season in enumerate(['MAM', 'JJA', 'SON', 'DJF']):
        for ci, ocean in enumerate(oceans):
            ax = axes[ri, ci]

            x = get_array(ocean_data, ocean, season, x_key)
            y = get_array(ocean_data, ocean, season, y_key)
            ln_aod = get_array(ocean_data, ocean, season, 'ln_aod')
            ln_cwp = get_array(ocean_data, ocean, season, 'ln_cwp')
            cf_ret = get_array(ocean_data, ocean, season, 'cf_ret')

            x, y = filter_data(x, y, ln_aod, ln_cwp, cf_ret, filter_type)

            ax.set_title(f'{ocean} {season}')

            if len(x) < 10:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                records.append({
                    'Ocean': ocean, 'Season': season,
                    'Slope': np.nan, 'Intercept': np.nan,
                    'R2': np.nan, 'N': len(x)
                })
                continue

            hist, x_edges, y_edges = np.histogram2d(x, y, bins=BINS)
            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

            im = ax.imshow(hist.T, origin='lower', extent=extent,
                           aspect='auto', cmap='viridis')

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            x_fit = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, 'r-', lw=2,
                    label=f'k={slope:.4f}\n$R^2$={r_value**2:.4f}')
            ax.legend(fontsize=8, loc=legend_loc)

            if ci == 0:
                ax.set_ylabel(y_label)
            if ri == 3:
                ax.set_xlabel(x_label)

            records.append({
                'Ocean': ocean, 'Season': season,
                'Slope': slope, 'Intercept': intercept,
                'R2': r_value ** 2, 'P_value': p_value,
                'Std_err': std_err, 'N': len(x)
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
        csv_name='coef_lnnd_vs_lnaod.csv',
        filter_type='aod',
        legend_loc='lower right'
    )

    plot_density_fit(
        ocean_data=ocean_data,
        x_key='ln_nd', y_key='cf_ret',
        x_label='ln(nd)', y_label='cf_ret',
        title='cf_ret vs ln(nd) with Linear Fit',
        fig_name='cf_ret_vs_lnnd.png',
        csv_name='coef_cf_ret_vs_lnnd.csv',
        filter_type='cf',
        legend_loc='upper right'
    )


if __name__ == '__main__':
    main()
