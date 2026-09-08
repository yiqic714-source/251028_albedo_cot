"""
calc_all_forcing.py
Combine 5_ERF_CO2.py + 6_cal_radiative_forcing.py
1) compute ERF_CO2 per‑ocean basin
2) compute Original(LH74) / Corrected IRF_aci per‑ocean basin
Output two csv files under processed_data/:
    ERF_CO2_ocean_regions.csv
    fig4_revision_impacts.csv
"""
import glob
import os
import netCDF4 as nc
import numpy as np
import pandas as pd
from pathlib import Path

# ========= config for ERF_CO2 part (from 5_ERF_CO2.py) =========
DATA_DIR = "/home/chenyiqi/251028_albedo_cot/cmip6"
VARIABLES = ("rsd", "rld", "rsu", "rlu")
CORRECTION_FACTOR = 2.16

OCEANS = {
    "NPO": [[-170, 20, -100, 60], [-180, 20, -170, 60], [105, 20, 180, 60]],
    "NAO": [[-100, 55, 45, 60], [-100, 40, 27, 55], [-100, 30, 45, 40], [-100, 20, 30, 30]],
    "TPO": [[-170, 16, -100, 20], [-170, 13, -89, 16], [-170, 9, -84, 13], [-170, -20, -70, 9], [100, 0, 180, 20], [130, -20, 180, 0], [-180, -20, -170, 20]],
    "TAO": [[-100, 16, -15, 20], [-84, 9, -13, 16], [-60, -20, 15, 9]],
    "TIO": [[30, 0, 100, 30], [30, -20, 130, 0]],
    "SPO": [[-170, -60, -70, -20], [130, -60, 180, -20], [-180, -60, -170, -20]],
    "SAO": [[-70, -60, 20, -20]],
    "SIO": [[20, -60, 130, -20]],
}

# ========= functions for ERF_CO2 =========
def load_variable(variable, experiment):
    pattern = os.path.join(DATA_DIR, f"{variable}_CFmon_MIROC6_piClim-{experiment}_r*.nc")
    file_paths = sorted(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError(f"No files found for {variable}, {experiment}: {pattern}")
    monthly_data = []
    lat = lon = None
    file_path = file_paths[-1]
    if os.path.getsize(file_path) == 0:
        raise OSError(f"Empty NetCDF file: {file_path}")
    with nc.Dataset(file_path, "r") as dataset:
        monthly_data.append(np.ma.filled(dataset.variables[variable][:, -1, :, :], np.nan).astype(np.float32))
        if lat is None:
            lat = dataset.variables["lat"][:].astype(np.float32)
            lon = dataset.variables["lon"][:].astype(np.float32)
    return np.concatenate(monthly_data, axis=0), lat, lon

def annual_mean(data):
    if data.shape[0] % 12 != 0:
        raise ValueError(f"Expected complete years, got {data.shape[0]} months")
    years = data.shape[0] // 12
    monthly_data = data.reshape(years, 12, data.shape[1], data.shape[2])
    return np.nanmean(monthly_data, axis=(0, 1))

def area_weighted_mean(data, lat, region_mask=None):
    weights = np.cos(np.deg2rad(lat))[:, np.newaxis]
    valid = np.isfinite(data)
    if region_mask is not None:
        valid &= region_mask
    return np.sum(np.where(valid, data * weights, 0.0)) / np.sum(np.where(valid, weights, 0.0))

def region_mask(lat, lon, regions):
    lon_grid, lat_grid = np.meshgrid(((lon + 180) % 360) - 180, lat)
    mask = np.zeros(lat_grid.shape, dtype=bool)
    for west, south, east, north in regions:
        mask |= (lon_grid >= west) & (lon_grid <= east) & (lat_grid >= south) & (lat_grid <= north)
    return mask

def run_erf_co2(base_dir: Path):
    OUTPUT_CSV = base_dir / "processed_data" / "ERF_CO2_ocean_regions.csv"
    flux_difference = {}
    lat = lon = None
    for variable in VARIABLES:
        four_x_co2, lat, lon = load_variable(variable, "4xCO2")
        control, control_lat, control_lon = load_variable(variable, "control")
        if not (np.array_equal(lat, control_lat) and np.array_equal(lon, control_lon)):
            raise ValueError(f"Coordinate mismatch for {variable}")
        flux_difference[variable] = annual_mean(four_x_co2) - annual_mean(control)
    annual_erf = (
        flux_difference["rsd"] + flux_difference["rld"]
        - flux_difference["rsu"] - flux_difference["rlu"]
    )
    global_mean = area_weighted_mean(annual_erf, lat)
    ERF_CO2 = {}
    for name, regions in OCEANS.items():
        mask = region_mask(lat, lon, regions)
        region_mean = area_weighted_mean(annual_erf, lat, mask)
        ERF_CO2[name] = region_mean / global_mean * CORRECTION_FACTOR
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(
        [{"ocean": name, "ERF_CO2": value} for name, value in ERF_CO2.items()]
    )
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[ERF‑CO2] Saved: {OUTPUT_CSV}")
    return ERF_CO2

# ========= IRF‑ACI calculation part (from 6_cal_radiative_forcing.py) =========
# import external utilities (keep as original)
from utils_fitting import cot_k_b_to_albedo, oceans, season_dict
from utils_solar import calc_grid_cell_area, calc_monthly_swdown

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

def load_cmip(base_dir:Path):
    CMIP_FILE = base_dir / "processed_data" / "cmip6_AodDiff_nat1850to1860_aer2010to2020_HadGEM3.csv"
    df = pd.read_csv(CMIP_FILE)
    df.rename(
        columns={
            "longitude": "lon",
            "latitude": "lat"
        },
        inplace=True
    )
    return df

def load_coefficients(base_dir:Path):
    K_CSV = base_dir / "processed_data" / "fig3_sbd_daytime_relation_8oceans_fits.csv"
    return pd.read_csv(K_CSV).set_index("Ocean")

def compute_irf(ocean, coeff, base_dir:Path):
    L3_DIR = base_dir / "L3_product"
    cmip = load_cmip(base_dir)
    k_corrected = float(coeff.loc[ocean, "k"])
    b_corrected = np.exp(float(coeff.loc[ocean, "lnb"]))
    k_original = 1.0
    b_original = 0.13
    total = {"original": 0.0, "corrected": 0.0}
    area_total = 0.0
    for season in season_dict:
        file = L3_DIR / f"{ocean}_{season}.csv"
        if not file.exists():
            continue
        data = pd.read_csv(file)
        data["time"] = pd.to_datetime(data["time"])
        data["month"] = data["time"].dt.month
        data["lat_round"] = np.round(data["lat"] * 2) / 2
        data["lon_round"] = np.round(data["lon"] * 2) / 2
        data = data.merge(
            cmip,
            left_on=["month", "lat_round", "lon_round"],
            right_on=["month", "lat", "lon"],
            how="left",
            suffixes=("", "_cmip")
        )
        data.drop(
            columns=["lat_round", "lon_round", "lat_cmip", "lon_cmip"],
            errors="ignore", inplace=True
        )
        cot = data["cot"].to_numpy(float)
        cf = data["cf_liq_ceres"].to_numpy(float)
        aod_diff = data["log_aod_diff"].to_numpy(float)
        swdown = np.array([calc_monthly_swdown(lat, month=m) for lat,m in zip(data["lat"], data["month"])])
        area = np.array([calc_grid_cell_area(lat) for lat in data["lat"]])
        base = swdown * LNND[ocean] * aod_diff / 3.0
        for name, k, b in [("original", k_original, b_original), ("corrected", k_corrected, b_corrected)]:
            albedo = cot_k_b_to_albedo(cot, k, b)
            value = base * k * albedo * (1-albedo) * cf
            valid = np.isfinite(value) & np.isfinite(area) & (area>0)
            total[name] += np.sum(value[valid] * area[valid])
        valid_area = np.isfinite(area) & (area>0)
        area_total += np.sum(area[valid_area])
    if area_total == 0:
        return np.nan, np.nan, area_total
    return total["original"]/area_total, total["corrected"]/area_total, area_total

def run_irf_calc(base_dir:Path):
    OUTPUT_CSV = base_dir / "processed_data" / "fig4_revision_impacts.csv"
    coeff = load_coefficients(base_dir)
    records = []
    for ocean in oceans:
        if ocean not in LNND:
            continue
        print("Processing", ocean)
        original, corrected, area = compute_irf(ocean, coeff, base_dir)
        records.append({"Ocean": ocean, "Original": original, "Corrected": corrected, "Area": area})
    result = pd.DataFrame(records)
    global_original = np.sum(result["Original"] * result["Area"]) / np.sum(result["Area"])
    global_corrected = np.sum(result["Corrected"] * result["Area"]) / np.sum(result["Area"])
    print("\nGlobal ocean area‑weighted IRF:")
    print(f"Original:  {global_original:.4f} W m-2")
    print(f"Corrected: {global_corrected:.4f} W m-2")
    print(f"Difference: {global_corrected - global_original:.4f} W m-2")
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[IRF‑ACI] Saved: {OUTPUT_CSV}")
    print(result)
    return result


# ========= main entry: run both calculations =========
def main():
    BASE_DIR = Path(__file__).resolve().parent
    # step 1: compute per‑ocean ERF_CO2
    _ = run_erf_co2(BASE_DIR)
    # step 2: compute per‑ocean Original / Corrected IRF_aci
    _ = run_irf_calc(BASE_DIR)
    print("\n==== All forcing calculation finished. ====")

if __name__ == "__main__":
    main()
