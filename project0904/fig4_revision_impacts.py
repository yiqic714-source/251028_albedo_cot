"""
fig4_revision_impacts.py
Plotting only.
The per‑ocean "Original" and "Corrected" IRF_aci values are computed by
6_cal_radiative_forcing.py and saved to processed_data/fig4_revision_impacts.csv.
This script reads that table and draws:
    Composite figure:
        (a) filled‑color map of (Original - Corrected) IRF / ERF_CO2
        (b) vertical dumbbell plot (before / after revision)
    Plus individual per‑ocean horizontal bar plots saved separately.
Continents (LAND) are drawn on the top‑most layer so the filled ocean color
boxes never cover the land.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box
from shapely.ops import unary_union
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

from utils_fitting import format_panel_tag

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
# per‑ocean IRF computed by 6_cal_radiative_forcing.py
RESULT_CSV = BASE_DIR / "processed_data" / "fig4_revision_impacts.csv"
FIG_DIR = (
    BASE_DIR /
    "figs" /
    "fig4_revision_impacts"
)
ERF_CO2_FILE = (
    BASE_DIR /
    "processed_data" /
    "ERF_CO2_ocean_regions.csv"
)
OCEANS_MAP = {
    "NPO": [
        [-170,20,-100,60],
        [-180,20,-170,60],
        [105,20,180,60]
    ],
    "NAO": [
        [-100,55,45,60],
        [-100,40,27,55],
        [-100,30,45,40],
        [-100,20,30,30]
    ],
    "TPO": [
        [-170,16,-100,20],
        [-170,13,-89,16],
        [-170,9,-84,13],
        [-170,-20,-70,9],
        [100,0,180,20],
        [130,-20,180,0],
        [-180,-20,-170,20]
    ],
    "TAO": [
        [-100,16,-15,20],
        [-84,9,-13,16],
        [-60,-20,15,9]
    ],
    "TIO": [
        [30,0,100,30],
        [30,-20,130,0]
    ],
    "SPO": [
        [-170,-60,-70,-20],
        [130,-60,180,-20],
        [-180,-60,-170,-20]
    ],
    "SAO": [
        [-70,-60,20,-20]
    ],
    "SIO": [
        [20,-60,130,-20]
    ],
}

# ============================================================
# Load computed IRF
# ============================================================
def load_result():
    return pd.read_csv(RESULT_CSV)

def load_erf_co2():
    df = pd.read_csv(ERF_CO2_FILE)
    required = {"ocean", "ERF_CO2"}
    if not required.issubset(df.columns):
        raise ValueError(f"ERF_CO2 file must contain {required}")
    erf_dict = dict(zip(df["ocean"], df["ERF_CO2"]))
    return erf_dict

# ============================================================
# Plot one ocean (individual small barh plots, unchanged)
# ============================================================
def plot_single_ocean(ocean, original, corrected):
    fig, ax = plt.subplots(
        figsize=(2.8, 1.8),
        facecolor="none"
    )
    ax.set_facecolor("none")
    values = [corrected, original]
    labels = ["Revised", "LH74"]
    colors = ["orangered", "orange"]
    y = np.arange(2)
    ax.barh(y, values, color=colors, height=0.45)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(r"IRF$_{\mathrm{aci}}$ (W m$^{-2}$)")
    ax.set_xlim(0, 3.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_title(ocean, fontsize=17, fontweight='bold', loc='left')
    ax.axvline(0, color="k", linewidth=0.8)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{ocean}_IRF.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()

def region_geometry(regions):
    return unary_union([box(west, south, east, north) for west,south,east,north in regions])

# ============================================================
# Modified map function: accepts external ax for composite figure
# ============================================================
def plot_irf_difference_map(result, ERF_CO2, ax):
    values={}
    for _,row in result.iterrows():
        ocean=row["Ocean"]
        values[ocean]=(row["Original"] - row["Corrected"]) / ERF_CO2[ocean]
    ax.set_extent([-180,180,-60,60])
    ax.add_feature(
        cfeature.LAND,
        facecolor="0.88",
        edgecolor="black",
        linewidth=0.3,
        zorder=4,
    )
    ax.coastlines(linewidth=0.4, zorder=5)

    # ========= 调淡viridis，不使用alpha透明度 =========
    fade = 0.5   # 0=原版；0.2轻微；0.35中等；0.5很淡，按需修改
    base_cmap = cm.viridis
    n_colors = 256
    raw_rgba = base_cmap(np.linspace(0,1,n_colors))
    new_rgb = []
    for r,g,b,a in raw_rgba:
        # 向白色偏移实现变淡，alpha保持1.0完全不透明
        rn = r + (1.0 - r)*fade
        gn = g + (1.0 - g)*fade
        bn = b + (1.0 - b)*fade
        new_rgb.append((rn, gn, bn, 1.0))
    light_viridis = ListedColormap(new_rgb)
    # =================================================

    norm=Normalize(vmin=0, vmax=max(values.values()))

    for ocean,regions in OCEANS_MAP.items():
        geom=region_geometry(regions)
        rgba = light_viridis(norm(values[ocean]))
        ax.add_geometries(
            [geom],
            ccrs.PlateCarree(),
            facecolor=rgba,
            edgecolor="black",
            linewidth=0.4,
            zorder=3
        )
    gl = ax.gridlines(draw_labels=True, color="none")
    gl.top_labels = False
    gl.right_labels = False

    sm=cm.ScalarMappable(norm=norm, cmap=light_viridis)
    cbar=plt.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.5, pad=0.08)
    cbar.set_label(r'$\Delta$IRF$_{\mathrm{aci}}$ / ERF$_{\mathrm{CO2}}$')


# ============================================================
# Main
# ============================================================
def main():
    result = load_result()
    erf_co2 = load_erf_co2()
    print("Loaded:", RESULT_CSV)

    # plot individual per‑ocean bar plots (unchanged behaviour)
    for _, row in result.iterrows():
        ocean = row["Ocean"]
        print("Plotting", ocean)
        plot_single_ocean(ocean, row["Original"], row["Corrected"])

    # single-panel figure: the IRF difference map (panel b removed)
    fig = plt.figure(figsize=(14, 7))
    ax_a = fig.add_axes([0.06, 0.30, 0.81, 0.62], projection=ccrs.PlateCarree())

    plot_irf_difference_map(result, erf_co2, ax_a)

    # --- global area-weighted means (printed in terminal, not annotated) ---
    areas = result.set_index("Ocean")["Area"]
    total_area = areas.sum()
    original_awm = sum(
        row.Original * areas[row.Ocean]
        for _, row in result.iterrows()
    ) / total_area
    corrected_awm = sum(
        row.Corrected * areas[row.Ocean]
        for _, row in result.iterrows()
    ) / total_area
    delta_awm = sum(
        (row.Original - row.Corrected) * areas[row.Ocean]
        for _, row in result.iterrows()
    ) / total_area
    ratio_awm = sum(
        (row.Original - row.Corrected) / erf_co2[row.Ocean] * areas[row.Ocean]
        for _, row in result.iterrows()
    ) / total_area
    print(f"\nGlobal area-weighted mean of original  IRFaci (before revision) : {original_awm:.4f} W m-2")
    print(f"Global area-weighted mean of corrected IRFaci (after revision)  : {corrected_awm:.4f} W m-2")
    print(f"Global area-weighted mean of delta_IRFaci                       : {delta_awm:.4f} W m-2")
    print(f"Global area-weighted mean of delta_IRFaci/ERF_CO2               : {ratio_awm:.4f}")

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "fig4_underlying.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nMap values:")
    values={}
    for _,row in result.iterrows():
        ocean=row["Ocean"]
        values[ocean]=(row["Original"] - row["Corrected"]) / erf_co2[ocean]
    for k,v in values.items():
        print(k, v)

if __name__ == "__main__":
    main()
