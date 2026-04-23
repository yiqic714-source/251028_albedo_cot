import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Ac_cot_fitting_utils as acfu


def plot_8_oceans():
    all_processed_ocean_data, _ = acfu.preprocess_ocean_data(
		min_cot_mod08=2.5,
		min_ret_cot_cer=2.5)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    position_map = {
        'NAO': 0, 'NPO': 1, 'TIO': 2, 'TAO': 3,
        'TPO': 4, 'SIO': 5, 'SAO': 6, 'SPO': 7
    }

    all_fit_results = []

    for ocean in acfu.oceans:
        if ocean not in position_map or all_processed_ocean_data[ocean] is None:
            continue

        ax_idx = position_map[ocean]
        ocean_data = all_processed_ocean_data[ocean]

        ocean_title = f'{ocean}'
        ocean_results = acfu.plot_axes_content(ocean_data, axes[ax_idx], title=ocean_title)

        ocean_result_row = {'Ocean': ocean}

        if ocean_results:
            for key in ['ret', 'cp', 'dcp', 'msk', 'LH74']:
                (
                    global_k, global_b, global_k_unc, global_b_unc,
                    season_k, season_b, season_k_unc, season_b_unc
                ) = ocean_results[key]

                ocean_result_row[f'Ann_Slope_{key}'] = global_k
                ocean_result_row[f'Ann_Intercept_{key}'] = global_b
                ocean_result_row[f'Ann_SlopeUnc_{key}'] = global_k_unc
                ocean_result_row[f'Ann_InterceptUnc_{key}'] = global_b_unc

                for s_name in acfu.season_dict.keys():
                    ocean_result_row[f'{s_name}_Slope_{key}'] = season_k.get(s_name, np.nan)
                    ocean_result_row[f'{s_name}_Intercept_{key}'] = season_b.get(s_name, np.nan)
                    ocean_result_row[f'{s_name}_SlopeUnc_{key}'] = season_k_unc.get(s_name, np.nan)
                    ocean_result_row[f'{s_name}_InterceptUnc_{key}'] = season_b_unc.get(s_name, np.nan)
        else:
            for key in ['ret', 'cp', 'dcp', 'msk', 'LH74']:
                ocean_result_row[f'Ann_Slope_{key}'] = np.nan
                ocean_result_row[f'Ann_Intercept_{key}'] = np.nan
                ocean_result_row[f'Ann_SlopeUnc_{key}'] = np.nan
                ocean_result_row[f'Ann_InterceptUnc_{key}'] = np.nan

                for s_name in acfu.season_dict.keys():
                    ocean_result_row[f'{s_name}_Slope_{key}'] = np.nan
                    ocean_result_row[f'{s_name}_Intercept_{key}'] = np.nan
                    ocean_result_row[f'{s_name}_SlopeUnc_{key}'] = np.nan
                    ocean_result_row[f'{s_name}_InterceptUnc_{key}'] = np.nan

        all_fit_results.append(ocean_result_row)

    fig.text(0.5, 0.04, r'ln(COT)', ha='center', fontsize=16)
    fig.text(0.04, 0.5, r'$\ln\left[A_{\mathrm{c}}/(1-A_{\mathrm{c}})\right]$', va='center', rotation='vertical', fontsize=16)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.98])

    os.makedirs('figs', exist_ok=True)
    output_fig_path = 'figs/fittings_8_oceans.png'
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"8 oceans figure saved to: {output_fig_path}")

    output_csv_path = '/home/chenyiqi/251028_albedo_cot/processed_data/k_lnb_by_seasons_oceans.csv'
    output_df = pd.DataFrame(all_fit_results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"8 oceans slope/intercept results saved to: {output_csv_path}")


if __name__ == "__main__":
    plot_8_oceans()