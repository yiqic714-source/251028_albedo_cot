import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define ocean identifiers and seasonal month mapping
oceans = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']
seasons = ['MAM', 'JJA', 'SON', 'DJF']

# Output directory
FIG_DIR = 'figs'
os.makedirs(FIG_DIR, exist_ok=True)

# Color palette for 5-segment pie chart (same as fig3_pie_plt.py)
colors = [
    (0.72, 0.78, 0.86),   # delta1: 1 - k_dcp, soft gray-blue
    (0.00, 0.75, 1.00),   # delta2: k_dcp - k_cp, bright sky blue
    (0.34, 0.30, 1.00),   # delta3: k_cp - k_ret, vivid blue-purple
    (1.00, 0.52, 0.18),   # delta4: k_ret - k_msk, bright orange
    (0.95, 0.05, 0.22)    # k_msk, vivid red-pink
]

# Labels for the 5 components
component_labels = [
    r'$1-k_{\mathrm{dcp}}$',
    r'$k_{\mathrm{dcp}}-k_{\mathrm{cp}}$',
    r'$k_{\mathrm{cp}}-k_{\mathrm{ret}}$',
    r'$k_{\mathrm{ret}}-k_{\mathrm{msk}}$',
    r'$k_{\mathrm{msk}}$'
]


def plot_donut_chart(ocean, mean_sizes):
    """
    Generate a single-ring donut chart showing the 4-season mean of the
    5 slope decomposition components. The ocean name is written in the center.
    
    Parameters
    ----------
    ocean : str
        Ocean name (e.g., 'NPO')
    mean_sizes : list
        [delta1, delta2, delta3, delta4, slope] averaged over 4 seasons
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_alpha(0.0)

    # Draw a single pie with a hole in the center (donut)
    wedges, _ = ax.pie(
        mean_sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor='w', linewidth=3)
    )

    # Cover the center with a white circle to create the donut hole
    circle = plt.Circle(
        (0, 0), 0.45,
        transform=ax.transData,
        color='white',
        zorder=3
    )
    ax.add_artist(circle)

    # Write ocean name in the center
    ax.text(
        0, 0,
        f'{ocean}',
        ha='center', va='center',
        fontsize=30,
        weight='bold',
        zorder=4
    )

    ax.axis('equal')

    out_path = os.path.join(FIG_DIR, f'fig3_donut_{ocean}.png')
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches='tight',
        transparent=True
    )
    plt.close()
    print(f"  Saved: {out_path}")


def plot_legend_only():
    """
    Draw a figure containing only the legend (no donut chart).
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_alpha(0.0)
    ax.axis('off')

    # Create dummy wedges for legend
    dummy_wedges = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]

    ax.legend(
        dummy_wedges,
        component_labels,
        loc='center',
        fontsize=20,
        ncol=len(component_labels),
        frameon=True,
        facecolor='white',
        edgecolor='lightgray',
        framealpha=0.7,
        columnspacing=5,
        handlelength=2
    )

    out_path = os.path.join(FIG_DIR, 'fig3_donut_legend_only.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    # Read CSV format (wide): Ocean, Season, k_dcp, b_dcp, k_cp, b_cp, k_ret, b_ret, k_msk, b_msk, ...
    df = pd.read_csv('/home/chenyiqi/251028_albedo_cot/processed_data/sensitivity_albedo_vs_cot_1030.csv')

    for ocean in oceans:
        ocean_rows = df[df['Ocean'] == ocean]
        if ocean_rows.empty:
            print(f"Warning: {ocean} not found, skipping.")
            continue

        season_data = {}
        for season in seasons:
            season_rows = ocean_rows[ocean_rows['Season'] == season]
            if season_rows.empty:
                print(f"Warning: {ocean} {season} not found, skipping this season.")
                continue

            row = season_rows.iloc[0]
            k_dcp = float(row['k_dcp'])
            k_cp  = float(row['k_cp'])
            k_ret = float(row['k_ret'])
            k_msk = float(row['k_msk'])

            if not (np.isfinite(k_dcp) and np.isfinite(k_cp) and np.isfinite(k_ret) and np.isfinite(k_msk)):
                print(f"Warning: {ocean} {season} has NaN k values, skipping this season.")
                continue

            delta1 = 1 - k_dcp
            delta2 = k_dcp - k_cp
            delta3 = k_cp - k_ret
            delta4 = k_ret - k_msk
            slope = k_msk

            season_data[season] = [delta1, delta2, delta3, delta4, slope]

        if len(season_data) < 4:
            print(f"Warning: {ocean} has only {len(season_data)} seasons, skipping.")
            continue

        # Compute 4-season mean of the 5 components
        mean_sizes = np.mean([season_data[s] for s in seasons], axis=0).tolist()

        plot_donut_chart(ocean, mean_sizes)

    # Also draw a legend-only figure
    plot_legend_only()

    print("Donut charts saved.")
