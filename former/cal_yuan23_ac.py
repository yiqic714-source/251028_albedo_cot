import glob
import numpy as np
import xarray as xr
from pyhdf.SD import SD, SDC

def read_and_mask_mod_variable(hdf, var_name):
    """
    Read a MODIS HDF SDS variable and apply:
    - _FillValue masking
    - valid_range / valid_min / valid_max masking (if present)
    - scale_factor and add_offset: physical = (raw - add_offset) * scale_factor
    """
    sds = hdf.select(var_name)
    data = sds[:].astype(np.float32)

    attrs = sds.attributes()
    fill_value = attrs.get('_FillValue', None)

    # mask fill
    if fill_value is not None:
        data = np.where(data == fill_value, np.nan, data)

    # mask valid range if present
    if 'valid_range' in attrs:
        vmin, vmax = attrs['valid_range']
        data = np.where((data < vmin) | (data > vmax), np.nan, data)
    else:
        vmin = attrs.get('valid_min', None)
        vmax = attrs.get('valid_max', None)
        if vmin is not None:
            data = np.where(data < vmin, np.nan, data)
        if vmax is not None:
            data = np.where(data > vmax, np.nan, data)

    # scale/offset
    offset = attrs.get('add_offset', None)
    scale_factor = attrs.get('scale_factor', None)
    if offset is not None:
        data = data - np.float32(offset)
    if scale_factor is not None:
        data = data * np.float32(scale_factor)

    return data

# ---------------- CERES ----------------
fn = "CERES_EBAF_TOA_2020/CERES_EBAF-TOA_Ed4.2.1_Subset_202001-202012.nc"
ds = xr.open_dataset(fn)

# ---------------- MODIS ----------------
modis_files = sorted(glob.glob("/data/MODIS/MxD08_M3/MYD08_M3.A2020*.hdf"))
if len(modis_files) != 12:
    raise RuntimeError(f"expected 12 MODIS files, found {len(modis_files)}")

cf_slices = []
for path in modis_files:
    hdf = SD(path, SDC.READ)
    cf = read_and_mask_mod_variable(hdf, "Cloud_Fraction_Mean_Mean")  # [ny, nx]
    cf[cf<0.1] = np.nan
    cf_slices.append(cf)

ny, nx = cf_slices[0].shape
lat_modis = np.linspace(-89.5, 89.5, ny).astype("f4")
lon_modis = np.linspace(-179.5, 179.5, nx).astype("f4")

modis = xr.DataArray(
    np.stack(cf_slices, axis=0),
    dims=("time", "lat", "lon"),
    coords={"time": ds.time, "lat": lat_modis, "lon": lon_modis},
    name="MODIS_CF"
)

modis = modis.where(np.abs(modis["lat"]) <= 60, drop=True)
modis = modis.assign_coords(lat=ds.lat, lon=ds.lon)

modis_frac = modis

expr = (ds.toa_sw_all_mon - (1.0 - modis_frac) * ds.toa_sw_clr_c_mon) / modis_frac / ds.solar_mon
gm = expr.mean(dim=["lat", "lon"], skipna=True)
print(gm)
print("yearly mean:", gm.mean(skipna=True))