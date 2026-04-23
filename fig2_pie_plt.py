import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define ocean identifiers and seasonal month mapping
oceans = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']
season_dict = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}

# Output directory
FIG_DIR = 'figs'
os.makedirs(FIG_DIR, exist_ok=True)

# Color palette for 5-segment pie chart
colors = [
    (1.0, 0.8, 0.8),      # delta1
    (1.0, 0.4, 0.4),      # delta2
    (0.8, 0.0, 0.0),      # delta3
    (0.7, 0.87, 0.98),    # delta4
    (0.0, 0.4, 0.8)       # k_msk
]


def plot_pie_chart(ocean, delta1, delta2, delta3, delta4, slope):
    """
    Generate 5-segment pie chart for annual slope components.
    Show legend only for Global.
    """
    pie_sizes = [delta1, delta2, delta3, delta4, slope]
    labels = [
        r'$1-k_{\mathrm{dcp}}$',
        r'$k_{\mathrm{dcp}}-k_{\mathrm{cp}}$',
        r'$k_{\mathrm{cp}}-k_{\mathrm{ret}}$',
        r'$k_{\mathrm{ret}}-k_{\mathrm{msk}}$',
        r'$k_{\mathrm{msk}}$'
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0.0)

    wedges, _ = ax.pie(
        pie_sizes,
        labels=[None] * 5,
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor='w', linewidth=4)
    )

    for i, w in enumerate(wedges):
        ang = (w.theta2 + w.theta1) / 2
        x = np.cos(np.deg2rad(ang)) * 0.63
        y = np.sin(np.deg2rad(ang)) * 0.63

        ax.text(
            x, y,
            f"{pie_sizes[i]:.2f}",
            ha='center', va='center',
            fontsize=32, weight='bold',
            color='k'
        )

    if ocean == 'Global':
        ax.legend(
            wedges,
            labels,
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            fontsize=15,
            borderaxespad=0.0,
            ncol=len(labels),
            frameon=True,
            facecolor='white',
            edgecolor='lightgray',
            framealpha=0.7,
            columnspacing=5
        )

    ax.text(
        0, -1.25,
        f'{ocean}',
        ha='center', va='center',
        fontsize=45,
        weight='bold'
    )

    ax.axis('equal')

    out_path = os.path.join(FIG_DIR, f'transparent_pie_chart_{ocean}.png')
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches='tight',
        transparent=True
    )
    plt.close()


def compute_annual_components(row):
    """
    Compute annual pie-chart components from one row of slope data.
    """
    k_msk = row['Ann_Slope_msk']
    k_ret = row['Ann_Slope_ret']
    k_cp = row['Ann_Slope_cp']
    k_dcp = row['Ann_Slope_dcp']

    delta1 = 1 - k_dcp
    delta2 = k_dcp - k_cp
    delta3 = k_cp - k_ret
    delta4 = k_ret - k_msk
    slope = k_msk

    return delta1, delta2, delta3, delta4, slope


if __name__ == "__main__":
    df_oceans = pd.read_csv('/home/chenyiqi/251028_albedo_cot/processed_data/k_lnb_by_seasons_oceans.csv')
    df_global = pd.read_csv('/home/chenyiqi/251028_albedo_cot/processed_data/k_lnb_global_by_seasons.csv')

    df = pd.concat([df_oceans, df_global], ignore_index=True)

    for ocean in oceans + ['Global']:
        ocean_rows = df[df['Ocean'] == ocean]
        if ocean_rows.empty:
            print(f"Warning: {ocean} not found in input CSVs, skipping.")
            continue

        row = ocean_rows.iloc[0]
        delta1, delta2, delta3, delta4, slope = compute_annual_components(row)
        plot_pie_chart(ocean, delta1, delta2, delta3, delta4, slope)

    print("Pie charts saved.")