import glob
import os
import netCDF4 as nc
import numpy as np
import pandas as pd

DATA_DIR = "/home/chenyiqi/251028_albedo_cot/cmip6"
OUTPUT_CSV = "./processed_data/ERF_CO2_ocean_regions.csv"
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


def main():
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
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
