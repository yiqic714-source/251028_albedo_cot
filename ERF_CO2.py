import glob
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from cartopy.util import add_cyclic_point
from matplotlib.colors import TwoSlopeNorm


DATA_DIR = "/home/chenyiqi/251028_albedo_cot/cmip6"
OUTPUT_FILE = "/home/chenyiqi/251028_albedo_cot/figs/ERF_CO2_annual.png"
VARIABLES = ("rsd", "rld", "rsu", "rlu")


def load_variable(variable, experiment):
    """Read one experiment at the top-of-atmosphere level (lev=0)."""
    pattern = os.path.join(DATA_DIR, f"{variable}_CFmon_MIROC6_piClim-{experiment}_r*.nc")
    file_paths = sorted(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError(f"No files found for {variable}, {experiment}: {pattern}")

    monthly_data = []
    lat = lon = None
    for file_path in file_paths:
        if os.path.getsize(file_path) == 0:
            raise OSError(f"Empty NetCDF file: {file_path}")
        with nc.Dataset(file_path, "r") as dataset:
            monthly_data.append(
                np.ma.filled(dataset.variables[variable][:, 0, :, :], np.nan).astype(np.float32)
            )
            if lat is None:
                lat = dataset.variables["lat"][:].astype(np.float32)
                lon = dataset.variables["lon"][:].astype(np.float32)
    return np.concatenate(monthly_data, axis=0), lat, lon


def annual_mean(data):
    """Average all complete years and all months within each year."""
    if data.shape[0] % 12 != 0:
        raise ValueError(f"Expected complete years, got {data.shape[0]} months")
    years = data.shape[0] // 12
    monthly_data = data.reshape(years, 12, data.shape[1], data.shape[2])
    return np.nanmean(monthly_data, axis=(0, 1))


def area_weighted_global_mean(data, lat):
    """Calculate a longitude-latitude grid area-weighted mean using cos(latitude)."""
    weights = np.cos(np.deg2rad(lat))[:, np.newaxis]
    valid = np.isfinite(data)
    weighted_sum = np.sum(np.where(valid, data * weights, 0.0))
    weight_sum = np.sum(np.where(valid, weights, 0.0))
    return weighted_sum / weight_sum


def main():
    flux_difference = {}
    lat = lon = None
    for variable in VARIABLES:
        four_x_co2, lat, lon = load_variable(variable, "4xCO2")
        control, control_lat, control_lon = load_variable(variable, "control")
        if not (np.array_equal(lat, control_lat) and np.array_equal(lon, control_lon)):
            raise ValueError(f"Coordinate mismatch for {variable}")
        four_x_co2_annual = annual_mean(four_x_co2)
        control_annual = annual_mean(control)
        flux_difference[variable] = four_x_co2_annual - control_annual

    annual_erf = (
        flux_difference["rsd"]
        + flux_difference["rld"]
        - flux_difference["rsu"]
        - flux_difference["rlu"]
    )
    global_mean = area_weighted_global_mean(annual_erf, lat)
    finite_values = annual_erf[np.isfinite(annual_erf)]
    limit = np.nanpercentile(np.abs(finite_values), 99)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(15, 8),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=180)},
        constrained_layout=True,
    )
    annual_erf_cyclic = add_cyclic_point(annual_erf, coord=lon)[0]
    lon_cyclic = np.append(lon, lon[0] + 360)
    mesh = ax.pcolormesh(
        lon_cyclic,
        lat,
        annual_erf_cyclic,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        norm=norm,
        shading="auto",
    )
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=1)
    ax.coastlines(linewidth=0.45)
    ax.set_global()
    ax.gridlines(linewidth=0.35, color="gray", alpha=0.5, linestyle="--")
    ax.set_title("Annual Global Distribution of ERF CO2", fontsize=15)
    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.75, pad=0.04)
    colorbar.set_label("ERF CO2 (W m$^{-2}$)")
    fig.suptitle(
        "ERF CO2: (4xCO2 - control) of (rsd + rld - rsu - rlu)\n"
        f"Area-weighted global mean: {global_mean:.3f} W m$^{{-2}}",
        fontsize=16,
    )
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Area-weighted annual global mean ERF_CO2: {global_mean:.6f} W m-2")
    print(f"Grid-cell ERF range: {np.nanmin(annual_erf):.3f} to {np.nanmax(annual_erf):.3f} W m-2")


if __name__ == "__main__":
    main()
