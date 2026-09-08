"""
6_cal_radiative_forcing.py

Compute the "Original" and "Corrected" aerosol-cloud radiative forcing
(IRF_aci, W m-2) for each ocean basin and save the per-ocean table.

This is the calculation step only (no plotting).  The saved table
(processed_data/fig4_revision_impacts.csv) is read by
fig4_revision_impacts.py, which handles the plotting.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from utils_fitting import cot_k_b_to_albedo, oceans, season_dict
from utils_solar import calc_grid_cell_area, calc_monthly_swdown

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

# output table (input for fig4_revision_impacts.py)
OUTPUT_CSV = BASE_DIR / "processed_data" / "fig4_revision_impacts.csv"


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

    OUTPUT_CSV.parent.mkdir(
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\nSaved: {OUTPUT_CSV}"
    )

    print(result)


if __name__ == "__main__":

    main()

