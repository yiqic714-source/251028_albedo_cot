import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box
from shapely.ops import unary_union

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


def draw_ocean_boundary(ax, regions, color='0.25', linestyle='-', linewidth=1.1):
    """
    Merge all rectangles belonging to one ocean and draw only the outer boundary.
    Internal boundaries between adjacent rectangles are removed by unary_union.
    """
    rects = []
    for west, south, east, north in regions:
        rects.append(box(west, south, east, north))

    merged = unary_union(rects)

    if merged.geom_type == 'Polygon':
        geoms = [merged]
    elif merged.geom_type == 'MultiPolygon':
        geoms = list(merged.geoms)
    else:
        geoms = []

    for geom in geoms:
        x, y = geom.exterior.xy
        ax.plot(
            x, y,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            transform=ccrs.PlateCarree(),
            zorder=3
        )


if __name__ == "__main__":
    fig = plt.figure(figsize=(10.5, 4))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Background
    ax.add_feature(cfeature.OCEAN, facecolor='whitesmoke', zorder=0)

    # Draw only ocean outer boundaries
    for ocean, regions in oceans.items():
        draw_ocean_boundary(
            ax,
            regions,
            color='m',
            linestyle='-',
            linewidth=1
        )

    # Land mask above boundaries, so lines over land are hidden
    ax.add_feature(
        cfeature.LAND,
        facecolor='white',
        edgecolor='black',
        linewidth=0.5,
        zorder=4
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black', zorder=5)

    ax.set_title(format_panel_tag(0, 'nature'), fontsize=16, loc='left')
    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())

    # Gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        linestyle='--',
        alpha=0.0
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

        # Put axes/frame lines above ocean boundaries
    for spine in ax.spines.values():
        spine.set_zorder(20)
        spine.set_linewidth(1.0)
        spine.set_edgecolor('black')

    # For Cartopy GeoAxes, the map frame is usually the "geo" spine
    if 'geo' in ax.spines:
        ax.spines['geo'].set_zorder(20)
        ax.spines['geo'].set_linewidth(1.0)
        ax.spines['geo'].set_edgecolor('black')

    out_dir = 'figs'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig3_division_8oceans.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")
    plt.close(fig)