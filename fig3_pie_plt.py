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
    (0.6, 0.78, 0.95),    # delta4
    (0.05, 0.42, 0.85)       # k_msk
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

    ax.text(
        -1.6, 0,
        f'{ocean}',
        ha='center', va='center',
        fontsize=45,
        weight='bold'
    )

    ax.axis('equal')

    out_path = os.path.join(FIG_DIR, f'fig3_pie_chart_{ocean}.png')
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches='tight',
        transparent=True
    )
    plt.close()


def plot_legend_only():
    """
    Draw a figure containing only the legend (no pie chart).
    """
    labels = [
        r'$1-k_{\mathrm{dcp}}$',
        r'$k_{\mathrm{dcp}}-k_{\mathrm{cp}}$',
        r'$k_{\mathrm{cp}}-k_{\mathrm{ret}}$',
        r'$k_{\mathrm{ret}}-k_{\mathrm{msk}}$',
        r'$k_{\mathrm{msk}}$'
    ]

    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_alpha(0.0)
    ax.axis('off')

    # Create dummy wedges for legend
    dummy_wedges = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]

    ax.legend(
        dummy_wedges,
        labels,
        loc='center',
        fontsize=20,
        ncol=len(labels),
        frameon=True,
        facecolor='white',
        edgecolor='lightgray',
        framealpha=0.7,
        columnspacing=5,
        handlelength=2
    )

    out_path = os.path.join(FIG_DIR, 'fig3_pie_legend_only.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


if __name__ == "__main__":
    # Read CSV format: Method, Ocean, Season, Slope_1030, Intercept_1030, Slope_1030_Unc, Intercept_1030_Unc
    df = pd.read_csv('/home/chenyiqi/251028_albedo_cot/processed_data/sensitivity_albedo_vs_cot_ratio.csv')

    # Compute annual slope as mean of 4 seasons for each method-ocean
    df_annual = df.groupby(['Method', 'Ocean'], as_index=False)['Slope_1030'].mean()

    for ocean in oceans:
        ocean_rows = df_annual[df_annual['Ocean'] == ocean]
        if ocean_rows.empty:
            print(f"Warning: {ocean} not found, skipping.")
            continue

        k_dcp = ocean_rows.loc[ocean_rows['Method'] == 'dcp', 'Slope_1030'].values
        k_cp  = ocean_rows.loc[ocean_rows['Method'] == 'cp', 'Slope_1030'].values
        k_ret = ocean_rows.loc[ocean_rows['Method'] == 'ret', 'Slope_1030'].values
        k_msk = ocean_rows.loc[ocean_rows['Method'] == 'msk', 'Slope_1030'].values

        if len(k_dcp) == 0 or len(k_cp) == 0 or len(k_ret) == 0 or len(k_msk) == 0:
            print(f"Warning: {ocean} missing one or more methods, skipping.")
            continue

        k_dcp = k_dcp[0]
        k_cp  = k_cp[0]
        k_ret = k_ret[0]
        k_msk = k_msk[0]

        delta1 = 1 - k_dcp
        delta2 = k_dcp - k_cp
        delta3 = k_cp - k_ret
        delta4 = k_ret - k_msk
        slope = k_msk

        plot_pie_chart(ocean, delta1, delta2, delta3, delta4, slope)

    # Also draw a legend-only figure
    plot_legend_only()

    print("Pie charts saved.")
