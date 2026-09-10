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
    ax.set_title(ocean, fontsize=17, fontweight='bold')
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
    cbar=plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7)
    cbar.set_label(r'$\Delta$IRF$_{\mathrm{aci}}$ / ERF$_{\mathrm{CO2}}$')

# ============================================================
# New: Vertical dumbbell plot
# X: sample labels; Y: numeric values; vertical lines connect old / new
# ============================================================
def plot_dumbbell_vertical(ax):
    labels = ["Q08", "B13", "M14", "Mc17", "G17", "R18", "H19", "T19", "D20", "J21"]
    old_vals = np.array([-0.2, -0.6, -0.34, -1.0, -0.4, -0.8, -1.14, -0.52, -0.69, -0.59])
    new_vals = old_vals.copy()
    new_vals[[4,6,7,8]] = new_vals[[4,6,7,8]] * 0.653
    new_vals[3] = new_vals[3] * 0.653 * 1.265

    x_pos = np.arange(len(labels))

    # ---- ±1σ (inter-study spread) shaded bands per group ------
    mean_b = np.mean(old_vals)
    sd_b = np.std(old_vals, ddof=1)
    mean_a = np.mean(new_vals)
    sd_a = np.std(new_vals, ddof=1)
    # bands drawn first so points/lines stay on top
    ax.axhspan(mean_b - sd_b, mean_b + sd_b,
               color="#1f77b4", alpha=0.13, zorder=1)
    ax.axhspan(mean_a - sd_a, mean_a + sd_a,
               color="#d62728", alpha=0.13, zorder=1)

    # mean lines (dashed) for before / after revision
    line_b = ax.axhline(mean_b, color="#1f77b4", linestyle="--", lw=1.5,
                        label=rf"Mean$_{{\mathrm{{before}}}}$ = {mean_b:.2f}",
                        zorder=3)
    line_a = ax.axhline(mean_a, color="#d62728", linestyle="--", lw=1.5,
                        label=rf"Mean$_{{\mathrm{{after}}}}$ = {mean_a:.2f}",
                        zorder=3)

    # draw vertical connecting lines
    for xi, yo, yn in zip(x_pos, old_vals, new_vals):
        ax.plot([xi, xi], [yo, yn], color="#707070", lw=2.5, zorder=2)
    # scatter points
    h_before = ax.scatter(x_pos, old_vals, color='white', edgecolor="#1f77b4", linewidths=1.5, s=90, zorder=4, label="Before revision")
    h_after = ax.scatter(x_pos, new_vals, color="#d62728", s=40, zorder=5, label="After revision")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r'IRF$_{\mathrm{aci}}$ (W m$^{-2}$)')
    ax.legend(
        handles=[
            h_before,
            h_after,
            line_b,
            line_a,
            Patch(facecolor="#1f77b4", alpha=0.13, edgecolor="none",
                  label=rf"$\pm1\sigma_{{\mathrm{{before}}}}$ = $\pm${sd_b:.2f}"),
            Patch(facecolor="#d62728", alpha=0.13, edgecolor="none",
                  label=rf"$\pm1\sigma_{{\mathrm{{after}}}}$ = $\pm${sd_a:.2f}"),
        ],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=8.5,
    )
    ax.set_ylim(-1.3, 0)
    # panel (b) keeps top & right frame lines (full box frame)
    ax.grid(axis='y', alpha=0.3, linestyle=":")

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

    # composite figure: panel (a) map on top, panel (b) vertical dumbbell below
    # 用 add_axes 精确定位：两张图 left 坐标相同(左对齐)，宽度可以不同
    fig = plt.figure(figsize=(14, 7))
    ax_a = fig.add_axes([0.06, 0.30, 0.81, 0.62], projection=ccrs.PlateCarree())
    ax_b = fig.add_axes([0.06, 0.12, 0.60, 0.18])

    plot_irf_difference_map(result, erf_co2, ax_a)
    plot_dumbbell_vertical(ax_b)

    # panel identifiers a / b via format_panel_tag
    ax_a.text(-0.02, 1.08, format_panel_tag(0, "science"),
              transform=ax_a.transAxes, fontsize=16, va="top")
    ax_b.text(-0.02, 1.23, format_panel_tag(1, "science"),
              transform=ax_b.transAxes, fontsize=16, va="top")

    # --- area-weighted global mean of the mapped fill field (top-right of A) ---
    areas = result.set_index("Ocean")["Area"]
    map_vals = {
        row.Ocean: (row.Original - row.Corrected) / erf_co2[row.Ocean]
        for _, row in result.iterrows()
    }
    gmean = sum(map_vals[o] * areas[o] for o in areas.index) / areas.sum()
    ax_a.text(0.99, 1.06,
              rf"Global area-weighted mean = {gmean:.3f}",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=11)

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "fig4_composite_ab.png", dpi=300, bbox_inches="tight")
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
