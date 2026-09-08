import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import cot_k_b_to_albedo, oceans, season_dict
from utils_solar import calc_grid_cell_area, calc_monthly_swdown

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box
from shapely.ops import unary_union
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# L3 data
L3_DIR = BASE_DIR / "L3_product"

# CMIP6 aerosol forcing
CMIP_FILE = (
    BASE_DIR /
    "processed_data" /
    "cmip6_AodDiff_nat1850to1860_aer2010to2020_HadGEM3.csv"
)

# fitted COT-albedo parameters
K_CSV = (
    BASE_DIR /
    "processed_data" /
    "fig3_sbd_daytime_relation_8oceans_fits.csv"
)

FIG_DIR = (
    BASE_DIR /
    "figs" /
    "fig4_revision_impacts"
)


# dlnNd/dlnAOD
LNND = {
    'NAO': 0.40,
    'TAO': 0.41,
    'SAO': 0.52,
    'TIO': 0.53,
    'SIO': 0.52,
    'NPO': 0.75,
    'TPO': 0.39,
    'SPO': 0.34,
}

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
ERF_CO2 = {

    "NPO":2.16,
    "NAO":2.16,
    "TPO":2.16,
    "TAO":2.16,
    "TIO":2.16,
    "SPO":2.16,
    "SAO":2.16,
    "SIO":2.16,
}
# ============================================================
# Load CMIP6 AOD difference
# ============================================================

def load_cmip():

    df = pd.read_csv(CMIP_FILE)

    df.rename(
        columns={
            "longitude": "lon",
            "latitude": "lat"
        },
        inplace=True
    )

    return df



# ============================================================
# Load fitted parameters
# ============================================================

def load_coefficients():

    return pd.read_csv(
        K_CSV
    ).set_index("Ocean")



# ============================================================
# Compute IRF
# ============================================================

def compute_irf(ocean, coeff):


    cmip = load_cmip()


    # corrected COT-albedo relationship
    k_corrected = float(
        coeff.loc[ocean, "k"]
    )

    b_corrected = np.exp(
        float(coeff.loc[ocean, "lnb"])
    )


    # original relationship
    k_original = 1.0
    b_original = 0.13



    total = {
        "original": 0.0,
        "corrected": 0.0
    }


    area_total = 0.0



    for season in season_dict:


        # ============================================
        # Read L3 data
        # ============================================

        file = (
            L3_DIR /
            f"{ocean}_{season}.csv"
        )


        if not file.exists():
            continue


        data = pd.read_csv(file)


        data["time"] = pd.to_datetime(
            data["time"]
        )

        data["month"] = (
            data["time"]
            .dt.month
        )



        # ============================================
        # Match CMIP6 AOD
        # ============================================

        data["lat_round"] = (
            np.round(data["lat"] * 2)
            / 2
        )

        data["lon_round"] = (
            np.round(data["lon"] * 2)
            / 2
        )


        data = data.merge(
            cmip,
            left_on=[
                "month",
                "lat_round",
                "lon_round"
            ],
            right_on=[
                "month",
                "lat",
                "lon"
            ],
            how="left",
            suffixes=(
                "",
                "_cmip"
            )
        )


        data.drop(
            columns=[
                "lat_round",
                "lon_round",
                "lat_cmip",
                "lon_cmip"
            ],
            errors="ignore",
            inplace=True
        )



        # ============================================
        # Variables from L3
        # ============================================

        cot = data["cot"].to_numpy(
            float
        )

        cf = data["cf_liq_ceres"].to_numpy(
            float
        )

        aod_diff = data["log_aod_diff"].to_numpy(
            float
        )



        # solar radiation
        swdown = np.array(
            [
                calc_monthly_swdown(
                    lat,
                    month=month
                )
                for lat, month in zip(
                    data["lat"],
                    data["month"]
                )
            ]
        )


        # grid area
        area = np.array(
            [
                calc_grid_cell_area(lat)
                for lat in data["lat"]
            ]
        )



        # ============================================
        # Base forcing
        # ============================================

        base = (
            swdown *
            LNND[ocean] *
            aod_diff /
            3.0
        )



        # ============================================
        # Original and corrected
        # ============================================

        for name, k, b in [

            (
                "original",
                k_original,
                b_original
            ),

            (
                "corrected",
                k_corrected,
                b_corrected
            )

        ]:


            albedo = cot_k_b_to_albedo(
                cot,
                k,
                b
            )


            value = (
                base *
                k *
                albedo *
                (1-albedo) *
                cf
            )


            valid = (
                np.isfinite(value)
                &
                np.isfinite(area)
                &
                (area > 0)
            )


            total[name] += np.sum(
                value[valid] *
                area[valid]
            )



        valid_area = (
            np.isfinite(area)
            &
            (area > 0)
        )


        area_total += np.sum(
            area[valid_area]
        )



    if area_total == 0:

        return np.nan, np.nan


    return (
        total["original"] / area_total,
        total["corrected"] / area_total,
        area_total
    )



# ============================================================
# Plot one ocean
# ============================================================

def plot_single_ocean(
        ocean,
        original,
        corrected
):


    fig, ax = plt.subplots(
        figsize=(5, 2.2),
        facecolor="none"
    )

    ax.set_facecolor("none")


    values = [
        corrected,
        original
    ]

    labels = [
        "Corrected",
        "LH74"
    ]


    colors = [
        "orangered",
        "orange"
    ]


    y = np.arange(2)


    ax.barh(
        y,
        values,
        color=colors,
        height=0.45
    )


    ax.set_yticks(
        y,
        labels
    )


    ax.set_xlabel(
        r"IRF$_{\mathrm{aci}}$ (W m$^{-2}$)"
    )
    # fixed x-axis range
    ax.set_xlim(0, 3.1)


    # remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    # keep transparent background
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.set_title(
        ocean
    )


    ax.axvline(
        0,
        color="k",
        linewidth=0.8
    )


    fig.tight_layout()


    FIG_DIR.mkdir(
        exist_ok=True
    )


    fig.savefig(
        FIG_DIR /
        f"{ocean}_IRF.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True
    )


    plt.close()

def region_geometry(regions):

    return unary_union(
        [
            box(
                west,
                south,
                east,
                north
            )
            for west,south,east,north in regions
        ]
    )



def plot_irf_difference_map(result):


    values={}


    for _,row in result.iterrows():

        ocean=row["Ocean"]

        values[ocean]=(
            row["Original"]
            -
            row["Corrected"]
        ) / ERF_CO2[ocean]



    fig,ax=plt.subplots(
        figsize=(10,5),
        subplot_kw={
            "projection":
            ccrs.PlateCarree()
        }
    )


    ax.set_extent(
        [-180,180,-60,60]
    )


    ax.add_feature(
        cfeature.LAND,
        facecolor="0.85",
        edgecolor="black"
    )


    ax.add_feature(
        cfeature.OCEAN,
        facecolor="white"
    )


    cmap=cm.Reds

    norm=Normalize(
        vmin=0,
        vmax=max(values.values())
    )


    for ocean,regions in OCEANS_MAP.items():

        geom=region_geometry(
            regions
        )

        ax.add_geometries(
            [geom],
            ccrs.PlateCarree(),
            facecolor=cmap(
                norm(values[ocean])
            ),
            edgecolor="black",
            linewidth=1
        )


        # ocean label
        centroid=geom.centroid

        ax.text(
            centroid.x,
            centroid.y,
            ocean,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=11
        )


    ax.coastlines()


    sm=cm.ScalarMappable(
        norm=norm,
        cmap=cmap
    )

    cbar=plt.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        shrink=0.7
    )


    cbar.set_label(
        "(Original-Corrected IRF$_{aci}$)/ERF$_{CO2}$"
    )


    fig.tight_layout()


    fig.savefig(
        FIG_DIR /
        "IRF_difference_over_ERF_CO2_map.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



    print(
        "\nMap values:"
    )

    for k,v in values.items():
        print(
            k,
            v
        )

# ============================================================
# Main
# ============================================================

def main():

    coeff = load_coefficients()

    records = []


    for ocean in oceans:


        if ocean not in LNND:
            continue


        print(
            "Processing",
            ocean
        )


        original, corrected, area = compute_irf(
            ocean,
            coeff
        )


        records.append(
            {
                "Ocean": ocean,
                "Original": original,
                "Corrected": corrected,
                "Area": area
            }
        )


        plot_single_ocean(
            ocean,
            original,
            corrected
        )



    result = pd.DataFrame(
        records
    )

    # ============================================================
    # Global ocean area-weighted mean IRF
    # ============================================================

    global_original = (
        np.sum(
            result["Original"] *
            result["Area"]
        )
        /
        np.sum(
            result["Area"]
        )
    )


    global_corrected = (
        np.sum(
            result["Corrected"] *
            result["Area"]
        )
        /
        np.sum(
            result["Area"]
        )
    )


    print("\nGlobal ocean area-weighted IRF:")
    print(
        f"Original:  {global_original:.4f} W m-2"
    )

    print(
        f"Corrected: {global_corrected:.4f} W m-2"
    )

    print(
        f"Difference: {global_corrected-global_original:.4f} W m-2"
    )
    result.to_csv(
        BASE_DIR /
        "processed_data" /
        "fig4_revision_impacts.csv",
        index=False
    )

    print(result)
    
    plot_irf_difference_map(
        result
    )


if __name__ == "__main__":

    main()