import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from utils_fitting import oceans, season_dict

# Paths
BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
TABLE_FOLDER = 'cp'  # coupled SBDART lookup tables (per ocean-season)
TABLE_DIR = f'{BASE_PATH}/build_sbdart_lookup_table/cot_sza_to_albedo_lookup_table_{TABLE_FOLDER}'
FIG_DIR = f'{BASE_PATH}/figs'
os.makedirs(FIG_DIR, exist_ok=True)


def read_lookup_table(ocean, season):
    """
    Read the cot-sza-to-albedo lookup table for a given ocean and season.
    Returns (sza_grid, cot_grid, albedo_grid) or None if file not found.
    """
    file_name = f'cot_sza_to_albedo_lookup_table_{ocean}_{season}.csv'
    file_path = os.path.join(TABLE_DIR, file_name)

    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path, index_col=0)
    sza_grid = np.array(df.index, dtype=float)
    cot_grid = np.array(df.columns, dtype=float)
    albedo_grid = df.values  # shape: (len(sza_grid), len(cot_grid))

    return sza_grid, cot_grid, albedo_grid


def main():
    # Accumulate albedo grids across all ocean-season combinations
    albedo_sum = None
    count = 0
    common_sza_grid = None
    common_cot_grid = None

    for ocean in oceans:
        for season in season_dict.keys():
            result = read_lookup_table(ocean, season)
            if result is None:
                print(f'  Skipping {ocean}_{season}: file not found')
                continue

            sza_grid, cot_grid, albedo_grid = result

            if common_sza_grid is None:
                common_sza_grid = sza_grid
                common_cot_grid = cot_grid
                albedo_sum = np.zeros_like(albedo_grid)
            elif not (np.array_equal(sza_grid, common_sza_grid) and
                      np.array_equal(cot_grid, common_cot_grid)):
                print(f'  Warning: {ocean}_{season} has different grid, skipping')
                continue

            albedo_sum += albedo_grid
            count += 1

    if count == 0:
        print('No lookup tables found!')
        return

    albedo_mean = albedo_sum / count
    print(f'\nAveraged {count} ocean-season lookup tables.')

    # Create pcolor plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Use log scale for COT
    pcm = ax.pcolor(
        common_cot_grid, common_sza_grid, albedo_mean,
        shading='auto',
        cmap='viridis',
        norm=Normalize(vmin=0, vmax=1),
    )

    cbar = fig.colorbar(pcm, ax=ax, label=r'$A_\mathrm{c}$-COT-SZA')

    ax.set_xlim(0, 60)
    ax.set_xlim(0, 70)
    ax.set_xlabel('COT', fontsize=13)
    ax.set_ylabel('SZA (deg)', fontsize=13)
    ax.set_title(r'$A_\mathrm{c}$-COT-SZA', fontsize=12)

    out_path = os.path.join(FIG_DIR, 'fig4_albedo_pcolor.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
