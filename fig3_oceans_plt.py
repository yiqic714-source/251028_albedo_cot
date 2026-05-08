import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Polygon

# Define ocean regions with coordinates (west, south, east, north)
oceans = {
    'NPO': [
        [-170, 20, -100, 60],
        [-180, 20, -170, 60],
        [105, 20, 180, 60]
    ],
    'NAO': [
        [-100, 55, 45, 60],
        [-100, 40, 27, 55],
        [-100, 30, 45, 40],
        [-100, 20, 30, 30]
    ],
    'TPO': [
        [-170, 16, -100, 20],
        [-170, 13, -89, 16],
        [-170, 9, -84, 13],
        [-170, -20, -70, 9],
        [100, 0, 180, 20],
        [130, -20, 180, 0],
        [-180, -20, -170, 20]
    ],
    'TAO': [
        [-100, 16, -15, 20],
        [-84, 9, -13, 16],
        [-60, -20, 15, 9]
    ],
    'TIO': [
        [30, 0, 100, 30],
        [30, -20, 130, 0]
    ],
    'SPO': [
        [-170, -60, -70, -20],
        [130, -60, 180, -20],
        [-180, -60, -170, -20]
    ],
    'SAO': [
        [-70, -60, 20, -20]
    ],
    'SIO': [
        [20, -60, 130, -20]
    ]
}


def format_panel_tag(panel_idx, icon_style):
    if icon_style == 'science':
        letter = chr(ord('A') + panel_idx)
        return rf'$\mathbf{{{letter}}}$'

    letter = chr(ord('a') + panel_idx)
    return rf'$\mathbf{{({letter})}}$'


if __name__ == "__main__":
    # Choose panel tag style here: 'nature' -> (a)(b)(c), 'science' -> A B C.

    # Create figure and axis with PlateCarree projection
    fig = plt.figure(figsize=(10.5, 4))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
    ax.add_feature(cfeature.OCEAN, color='white')

    # pick 11 colors
    cmap_colors = ["#F0F0F0", "#D0D0D0", "#C0C0C0", "#E8E8E8", "#E0E0E0", "#D8D8D8", "#F8F8F8", "#C8C8C8"]
    ocean_color_map = {ocean: cmap_colors[i % len(cmap_colors)] for i, ocean in enumerate(oceans)}

    for ocean, regions in oceans.items():
        color = ocean_color_map.get(ocean, (0.5, 0.5, 0.5))

        for (west, south, east, north) in regions:
            # construct polygon vertices
            poly_xy = [
                (west, south),
                (west, north),
                (east, north),
                (east, south)
            ]

            polygon = Polygon(
                poly_xy,
                facecolor=color,
                edgecolor='none',
                linewidth=0.6,
                alpha=1,
                transform=ccrs.PlateCarree(),
                zorder=1
            )
            ax.add_patch(polygon)

    # land edge
    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='black', linewidth=0.5, zorder=2)
    # Choose panel tag style here: 'nature' -> (a)(b)(c), 'science' -> A B C.
    ax.set_title(format_panel_tag(0, 'nature'), fontsize=16, loc='left')
    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--', alpha=0.0, )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # save figure
    out_dir = 'figs'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig3_division_8oceans.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")
    plt.close(fig)