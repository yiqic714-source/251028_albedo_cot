import glob
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import netCDF4 as nc
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from shapely.geometry import box
from shapely.ops import unary_union

from utils_fitting import format_panel_tag


DATA_DIR = "/home/chenyiqi/251028_albedo_cot/cmip6"
OUTPUT_FILE = "/home/chenyiqi/251028_albedo_cot/figs/figsupp_ERF_CO2_ocean_regions.png"
BAR_DIFF_CSV = "/home/chenyiqi/251028_albedo_cot/processed_data/fig4_bar_diff_orig_minus_ac1030.csv"
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
    """Read one experiment at the top-of-atmosphere level (lev=0)."""
    pattern = os.path.join(DATA_DIR, f"{variable}_CFmon_MIROC6_piClim-{experiment}_r*.nc")
    file_paths = sorted(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError(f"No files found for {variable}, {experiment}: {pattern}")

    monthly_data = []
    lat = lon = None
    # for file_path in file_paths:
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
    """Average all complete years and all months within each year."""
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


def region_geometry(regions):
    return unary_union([box(west, south, east, north) for west, south, east, north in regions])


def region_mask(lat, lon, regions):
    lon_grid, lat_grid = np.meshgrid(((lon + 180) % 360) - 180, lat)
    mask = np.zeros(lat_grid.shape, dtype=bool)
    for west, south, east, north in regions:
        mask |= (lon_grid >= west) & (lon_grid <= east) & (lat_grid >= south) & (lat_grid <= north)
    return mask


def load_bar_differences():
    if not os.path.exists(BAR_DIFF_CSV):
        raise FileNotFoundError(f"Bar-difference file not found: {BAR_DIFF_CSV}")
    bar_diff = pd.read_csv(BAR_DIFF_CSV)
    required_columns = {"ocean", "method", "bar_diff"}
    if not required_columns.issubset(bar_diff.columns):
        raise ValueError(f"Bar-difference file must contain {required_columns}")
    return bar_diff


def make_discrete_scale(values, bins=9, value_range=None):
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("No finite values available for color scale")
    if value_range is None:
        lower = np.nanmin(finite_values)
        upper = np.nanmax(finite_values)
        if np.isclose(lower, upper):
            lower -= 0.5
            upper += 0.5
    else:
        lower, upper = value_range
    boundaries = np.linspace(lower, upper, bins + 1)
    cmap = matplotlib.colormaps["viridis"].resampled(bins)
    norm = BoundaryNorm(boundaries, cmap.N)
    return norm, cmap, boundaries


def draw_region_map(ax, values, norm, cmap, title, panel_index):
    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(
        cfeature.LAND, facecolor="0.88", edgecolor="black",
        linewidth=0.5, zorder=4,
    )
    for name, regions in OCEANS.items():
        geometry = region_geometry(regions)
        ax.add_geometries(
            [geometry], ccrs.PlateCarree(),
            facecolor=cmap(norm(values[name])),
            edgecolor="0.25", linestyle="--", linewidth=1.0, zorder=2,
        )
    ax.coastlines(linewidth=0.7, zorder=5)
    ax.gridlines(draw_labels=True, color="none")
    ax.set_title(title, fontsize=14)
    ax.text(
        -0.05, 1.05, format_panel_tag(panel_index, "nature"),
        transform=ax.transAxes, fontsize=17, va="bottom", ha="left",
    )


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
    region_values = {}
    all_regions_mask = np.zeros(annual_erf.shape, dtype=bool)
    for name, regions in OCEANS.items():
        mask = region_mask(lat, lon, regions)
        all_regions_mask |= mask
        region_mean = area_weighted_mean(annual_erf, lat, mask)
        region_values[name] = region_mean / global_mean * CORRECTION_FACTOR
    all_regions_mean = area_weighted_mean(annual_erf, lat, all_regions_mask)

    bar_diff = load_bar_differences()
    ratio_by_method = {}
    for method in ("ret", "msk"):
        method_values = bar_diff[bar_diff["method"] == method].set_index("ocean")["bar_diff"]
        missing = set(OCEANS) - set(method_values.index)
        if missing:
            raise ValueError(f"Missing bar differences for {method}: {sorted(missing)}")
        ratio_by_method[method] = {
            name: method_values[name] / region_values[name]
            for name in OCEANS
        }

    erf_norm, erf_cmap, erf_boundaries = make_discrete_scale(
        list(region_values.values()), value_range=(1.90, 2.53)
    )
    ratio_values = np.concatenate([
        np.asarray(list(ratio_by_method[method].values()), dtype=float)
        for method in ("ret", "msk")
    ])
    ratio_norm, ratio_cmap, ratio_boundaries = make_discrete_scale(
        ratio_values, value_range=(0.01, 0.46)
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 13),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    draw_region_map(axes[0], region_values, erf_norm, erf_cmap, r'$\mathrm{ERF}_\mathrm{CO2}$ (W m$^\mathrm{-2}$)', 0)
    draw_region_map(
        axes[1], ratio_by_method["ret"], ratio_norm, ratio_cmap,
        r'$\mathrm{IRF}_\mathrm{aci}/\mathrm{ERF}_\mathrm{CO2}$ (Cloud-Retrieval)', 1,
    )
    draw_region_map(
        axes[2], ratio_by_method["msk"], ratio_norm, ratio_cmap,
        r'$\mathrm{IRF}_\mathrm{aci}/\mathrm{ERF}_\mathrm{CO2}$ (Cloud-Mask)', 2,
    )

    fig.colorbar(
        ScalarMappable(norm=erf_norm, cmap=erf_cmap), ax=axes[0],
        orientation="horizontal", shrink=0.7, aspect=40, pad=0.05,
        boundaries=erf_boundaries, spacing="proportional",
    )
    fig.colorbar(
        ScalarMappable(norm=ratio_norm, cmap=ratio_cmap), ax=axes[1:],
        orientation="horizontal", shrink=0.7, aspect=40, pad=0.025,
        boundaries=ratio_boundaries, spacing="proportional",
    )
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Global area-weighted RF_4xCO2: {global_mean:.6f} W m-2")
    print(f"All ocean regions combined area-weighted RF_4xCO2: {all_regions_mean:.6f} W m-2")
    print(
        "All ocean regions combined area-weighted ERF_CO2: "
        f"{CORRECTION_FACTOR / global_mean * all_regions_mean:.6f} W m-2"
    )
    print("Regional ERF_CO2 (W m-2):")
    for name, value in region_values.items():
        print(f"  {name}: {value:.6f}")
    print("Regional IRF_aci/ERF_CO2 ratios:")
    for method, values in ratio_by_method.items():
        print(f"  {method}:")
        for name, value in values.items():
            print(f"    {name}: {value:.6f}")


if __name__ == "__main__":
    main()
