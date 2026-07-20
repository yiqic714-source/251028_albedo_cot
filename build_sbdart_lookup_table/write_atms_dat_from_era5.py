import glob
import os
import sys
import warnings

import netCDF4 as nc
import numpy as np


warnings.filterwarnings("ignore")

method = "dcp"  # "cp" or "dcp"

# ===================== Ocean regions =====================
oceans = {
    "NPO": [
        [-170, 20, -100, 60],
        [-180, 20, -170, 60],
        [105, 20, 180, 60],
    ],
    "NAO": [
        [-100, 55, 45, 60],
        [-100, 40, 27, 55],
        [-100, 30, 45, 40],
        [-100, 20, 30, 30],
    ],
    "TPO": [
        [-170, 16, -100, 20],
        [-170, 13, -89, 16],
        [-170, 9, -84, 13],
        [-170, -20, -70, 9],
        [100, 0, 180, 20],
        [130, -20, 180, 0],
        [-180, -20, -170, 20],
    ],
    "TAO": [
        [-100, 16, -15, 20],
        [-84, 9, -13, 16],
        [-60, -20, 15, 9],
    ],
    "TIO": [
        [30, 0, 100, 30],
        [30, -20, 130, 0],
    ],
    "SPO": [
        [-170, -60, -70, -20],
        [130, -60, 180, -20],
        [-180, -60, -170, -20],
    ],
    "SAO": [
        [-70, -60, 20, -20],
    ],
    "SIO": [
        [20, -60, 130, -20],
    ],
}

# ===================== Seasons =====================
season_dict = {
    "DJF": [1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

# ERA5 pressure-level files used here have these 18 levels.
pressure = np.array([
    1000, 950, 925, 900, 850,
    800, 750, 700, 650, 600,
    550, 500, 400, 300,
    200, 100, 50, 10,
], dtype=float)

if method == "dcp":
    pressure = pressure / 150.0


# ===================== Utilities =====================
def q2rho(t, q, p):
    m_air = 28.959
    r_gas = 8.31
    return p * 1e2 * m_air * q / r_gas / t


def get_05_grid(lon, lat):
    lon05 = lon[np.isclose(lon % 1, 0.5, atol=1e-4)]
    lat05 = lat[np.isclose(lat % 1, 0.5, atol=1e-4)]
    return np.sort(lon05), np.sort(lat05)


def interp_05(lon, lat, field):
    lon05, lat05 = get_05_grid(lon, lat)
    out = np.full((len(lat05), len(lon05)), np.nan)

    lon_id = np.where(np.isclose(lon % 1, 0.5, atol=1e-4))[0]
    lat_id = np.where(np.isclose(lat % 1, 0.5, atol=1e-4))[0]

    for i, iy in enumerate(lat_id):
        for j, ix in enumerate(lon_id):
            vals = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    y, x = iy + dy, ix + dx
                    if 0 <= y < len(lat) and 0 <= x < len(lon):
                        vals.append(field[y, x])
            out[i, j] = np.nanmean(vals)

    return out


def in_ocean(lon, lat, boxes):
    mask = np.zeros(lon.size, dtype=bool)
    for w, s, e, n in boxes:
        if w > e:
            lon_ok = (lon >= w) | (lon <= e)
        else:
            lon_ok = (lon >= w) & (lon <= e)
        lat_ok = (lat >= s) & (lat <= n)
        mask |= lon_ok & lat_ok
    return mask


# ===================== Land-sea mask =====================
def load_lsmask(lsmask_path):
    with nc.Dataset(lsmask_path) as ds:
        lon1 = ds["lon"][:]
        lat1 = ds["lat"][:]
        lsm = ds["LSMASK"][:]

    lon1 = np.where(lon1 > 180, lon1 - 360, lon1)

    if lsm.shape == (len(lon1), len(lat1)):
        lsm = lsm.T

    ocean = lsm == 0
    return lon1, lat1, ocean


def ocean_mask_05(lat05, lon05, lon1, lat1, ocean1):
    mask = np.zeros((len(lat05), len(lon05)), dtype=bool)
    for i, la in enumerate(lat05):
        ii = np.where(np.isclose(lat1, la))[0]
        if ii.size != 1:
            continue
        for j, lo in enumerate(lon05):
            jj = np.where(np.isclose(lon1, lo))[0]
            if jj.size == 1 and ocean1[ii[0], jj[0]]:
                mask[i, j] = True
    return mask


# ===================== Profile calculation =====================
def calc_profiles(data, lon, lat, month, method):
    results = {oc: {se: {} for se in season_dict} for oc in oceans}

    for oc, boxes in oceans.items():
        oc_mask = in_ocean(lon, lat, boxes)
        if not np.any(oc_mask):
            continue

        for se, mons in season_dict.items():
            se_mask = oc_mask & np.isin(month, mons)
            if not np.any(se_mask):
                continue

            z = np.nanmean(data["z"][se_mask], axis=0) / 9.80665 / 1000.0
            t = np.nanmean(data["t"][se_mask], axis=0)
            p = pressure[:len(z)]

            if method == "cp":
                q = np.nanmean(data["q"][se_mask], axis=0)
                o3 = np.nanmean(data["o3"][se_mask], axis=0)
                wh = q2rho(t, q, p)
                wo = q2rho(t, o3, p)
            elif method == "dcp":
                wh = 1e-30 * np.ones(z.shape)
                wo = 1e-30 * np.ones(z.shape)
                z = z * 6.0
            else:
                raise ValueError("method must be 'cp' or 'dcp'.")

            results[oc][se] = dict(z=z, t=t, wh=wh, wo=wo, p=p)

    return results


# ===================== Main =====================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <year>")
        sys.exit(1)

    year = int(sys.argv[1])
    lsmask_path = "/data/chenyiqi/251007_tropic/landsea.nc"
    era5_dir = "/data/chenyiqi/251201_ERFaci/era5_monthly_hourly_NoCld/Terra_pass_2002to2024_pl/"

    lon1, lat1, ocean1 = load_lsmask(lsmask_path)

    vars_use = ["z", "o3", "q", "t"]
    n_levels = len(pressure)

    all_lon, all_lat, all_month = [], [], []
    data = {v: [] for v in vars_use}

    files = sorted(glob.glob(f"{era5_dir}/era5_pl_monthly_{year}_utc*.nc"))
    if not files:
        sys.exit(f"No ERA5 files for {year}")

    for f in files:
        print(f)
        with nc.Dataset(f) as ds:
            lon = ds["longitude"][:]
            lat = ds["latitude"][:]
            lon = np.where(lon > 180, lon - 360, lon)

            lon05, lat05 = get_05_grid(lon, lat)
            LON, LAT = np.meshgrid(lon05, lat05)
            ocean05 = ocean_mask_05(lat05, lon05, lon1, lat1, ocean1)

            lon_flat = LON[ocean05].ravel()
            lat_flat = LAT[ocean05].ravel()
            ngrid = lon_flat.size

        if ngrid == 0:
            print(f"Skip {f}: no ocean grid cells matched the 0.5-degree mask.")
            continue

        for m in range(1, 13):
            tmp = {v: np.full((ngrid, n_levels), np.nan) for v in vars_use}
            ok = True

            for v in vars_use:
                layers = []
                for lev in range(n_levels):
                    try:
                        with nc.Dataset(f) as ds:
                            field = ds[v][m - 1, lev]
                        f05 = interp_05(lon, lat, field)
                        layers.append(f05[ocean05].ravel())
                    except Exception as e:
                        print(
                            f"Failed: file={f}, month={m}, "
                            f"var={v}, lev={lev}, error={e}"
                        )
                        ok = False
                        break

                if not ok:
                    break

                tmp[v] = np.column_stack(layers)

            if ok:
                all_lon.append(lon_flat)
                all_lat.append(lat_flat)
                all_month.append(np.full(ngrid, m))
                for v in vars_use:
                    data[v].append(tmp[v])

    if not all_lon:
        sys.exit(
            "No valid data collected. Check pressure levels, variable names, "
            "file dimensions, and the land-sea mask."
        )

    all_lon = np.concatenate(all_lon)
    all_lat = np.concatenate(all_lat)
    all_month = np.concatenate(all_month)
    for v in vars_use:
        data[v] = np.vstack(data[v])

    profiles = calc_profiles(data, all_lon, all_lat, all_month, method)

    # ===================== Save ATMS format =====================
    outdir = "atms_dat_" + method
    os.makedirs(outdir, exist_ok=True)

    for oc in profiles:
        for se in profiles[oc]:
            d = profiles[oc][se]
            if not d:
                continue

            fn = os.path.join(outdir, f"atms_{oc}_{se}.dat")
            with open(fn, "w") as f:
                f.write(f"{len(d['p'])}\n")
                for k in range(len(d["p"])):
                    f.write(
                        f"{d['z'][k]:.6f}\t"
                        f"{d['p'][k]:.1f}\t"
                        f"{d['t'][k]:.6f}\t"
                        f"{d['wh'][k]:.6e}\t"
                        f"{d['wo'][k]:.6e}\n"
                    )
            print(f"Saved {fn}")
