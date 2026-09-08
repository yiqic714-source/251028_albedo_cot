import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import cot_k_b_to_albedo, cot_to_albedo, mc_fit, oceans
from utils_solar import daytime_latitude_weighted_albedo

OUTPUT_PATH = './figs/fig3_sbd_daytime_relation_8oceans.png'
FITS_CSV_PATH = './processed_data/fig3_sbd_daytime_relation_8oceans_fits.csv'
COT = np.geomspace(2.5, 60, 80)
SEASONS = ('MAM', 'JJA', 'SON', 'DJF')
TROPICAL_OCEANS = {'TPO', 'TAO', 'TIO'}
TROPICAL_COLORS = ("#090EA5", '#00a6a6', '#2c7fb8')
EXTRATROPICAL_COLORS = ("#f1c515", "#8b745e", "#ef7809", "#f93939", "#ec7fe6")
LINESTYLES = (':', '--', '-')


def daytime_ocean_albedo(ocean):
    seasonal_albedo = [
        daytime_latitude_weighted_albedo(COT, ocean, season, table_folder='cp')
        for season in SEASONS
    ]
    return np.nanmean(seasonal_albedo, axis=0)


def main():
    fig, ax = plt.subplots(figsize=(5, 4.1))
    lh74 = cot_to_albedo(COT, 'l74')
    ax.plot(COT, lh74, color='#222222', lw=1.8, label=r'LH74: $k$=1.00')

    tropical_index = 0
    extratropical_index = 0
    records = []
    for index, ocean in enumerate(oceans):
        albedo = daytime_ocean_albedo(ocean)
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
    ax.set(xlim=(0, 60), ylim=(0.15, 0.9), xlabel='COT', ylabel=r'$A_{\mathrm{c}}$')
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
