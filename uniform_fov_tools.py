import numpy as np
from datetime import datetime, timedelta
from pyhdf.SD import SD, SDC
from scipy.interpolate import RegularGridInterpolator
import geom_utils as gu

def read_and_mask_mod_variable(hdf, var_name):
    """
    read an HDF variable,
    deal with _FillValue、scale_factor, and add_offset
    """
    sds = hdf.select(var_name)
    data = sds[:].astype(float)
    
    attrs = sds.attributes()
    fill_value = attrs.get('_FillValue', None)
    scale_factor = attrs.get('scale_factor')
    offset = attrs.get('add_offset')
    
    if fill_value is not None:
        data[data == fill_value] = np.nan
    if offset is not None: 
        data = data - offset
    if scale_factor is not None:
        data = data * scale_factor
    return data


def calc_delta_beta(latlon_cer, latlon_mod, latlon_subsat, 
         sensor_altitude=705., earth_radius=6367.):
    """ Get equatorial vectors for the satellite, centroid, and imager pixel """
    eq_sat = gu.get_equatorial_vectors(
            latitude=latlon_subsat[...,0],
            longitude=latlon_subsat[...,1],
            )
    eq_cer = gu.get_equatorial_vectors(
            latitude=latlon_cer[...,0],
            longitude=latlon_cer[...,1],
            )
    eq_mod = gu.get_equatorial_vectors(
            latitude=latlon_mod[...,0],
            longitude=latlon_mod[...,1],
            )

    CX,CY,CZ = np.split(np.expand_dims(gu.get_view_vectors(
            sensor_equatorial_vectors = eq_sat,
            pixel_equatorial_vectors = eq_cer,
            sensor_altitude = sensor_altitude,
            earth_radius = earth_radius,
            ), axis=(1,2)), 3, axis=-1)
    MX,MY,MZ= np.split(gu.get_view_vectors(
            sensor_equatorial_vectors = np.expand_dims(eq_sat, axis=(1,2)),
            pixel_equatorial_vectors = eq_mod,
            sensor_altitude = sensor_altitude,
            earth_radius = earth_radius,
            ), 3, axis=-1)

    ## Calculate along (delta) and across (beta) track angles, CERES ATBD subsystem 4.4 eq. 8
    delta = np.rad2deg(np.arcsin(np.sum(MY*CZ, axis=-1)))
    delta = np.squeeze(delta)
    
    tmp = np.cross(CZ, MY)
    tmp /= np.linalg.norm(tmp, axis=-1, keepdims=True)
    beta = np.rad2deg(np.arcsin(-1.*np.sum(tmp*CY, axis=-1)))
    beta = np.squeeze(beta)
        
    return delta, beta

def julian_to_datetime(julian_days):
    """
    Convert Julian day number (used in CERES) to datetime.
    """
    # January 1, 1970 is Julian day 2440587.5
    # time_ssf is the number of days since 1970-01-01
    base_julian = 2440587.5
    base_datetime = datetime(1970, 1, 1, 0, 0, 0)
    days_since_epoch = julian_days - base_julian
    
    # convert to datetime
    return [base_datetime + timedelta(days=days) for days in days_since_epoch]
               
def upscale_and_interpolate(lat, lon, solar_zenith, sensor_zenith, target_shape, obs_window):
    """
    Change from low resolution to several-times-higher resolution 
    """
    lon_min = np.nanmin(lon)
    lon_max = np.nanmax(lon)
    # print(lon_max-lon_min)
    if (lon_max-lon_min) > 180 and obs_window[1][0]>=0 and obs_window[1][1]>0:
        # print('lon[lon<0] = lon[lon<0] + 360')
        lon[lon<0] = lon[lon<0] + 360
    if (lon_max-lon_min) > 180 and obs_window[1][0]<0 and obs_window[1][1]<=0:
        # print('lon[lon>0] = lon[lon>0] - 360')
        lon[lon>0] = lon[lon>0] - 360
        
    Ny, Nx = lat.shape
    Ny_new = target_shape[0]
    Nx_new = target_shape[1]
    
    # Create original row and column indices
    y = np.arange(Ny)
    x = np.arange(Nx)
    
    # Create new indices
    y_new = np.linspace(0, Ny-1, Ny_new)
    x_new = np.linspace(0, Nx-1, Nx_new)
    
    # Generate new meshgrid point coordinates
    mesh_y, mesh_x = np.meshgrid(y_new, x_new, indexing='ij')
    points = np.stack([mesh_y.ravel(), mesh_x.ravel()], axis=-1)
    
    # Use RegularGridInterpolator to Interpolate
    interp_lat = RegularGridInterpolator((y, x), lat)
    interp_lon = RegularGridInterpolator((y, x), lon)
    interp_solar = RegularGridInterpolator((y, x), solar_zenith)
    interp_sensor = RegularGridInterpolator((y, x), sensor_zenith)
    lat_new = interp_lat(points).reshape(Ny_new, Nx_new)
    lon_new = interp_lon(points).reshape(Ny_new, Nx_new)
    solar_zenith_new = interp_solar(points).reshape(Ny_new, Nx_new)
    sensor_zenith_new = interp_sensor(points).reshape(Ny_new, Nx_new)
    
    return lat_new, lon_new, solar_zenith_new, sensor_zenith_new