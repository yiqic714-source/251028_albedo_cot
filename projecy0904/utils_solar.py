"""Solar geometry and daytime SBDART albedo weighting for Fig. 3."""

import math

import numpy as np
import pandas as pd


def declination(day_of_year):
    """Return solar declination in radians."""
    return math.radians(23.45) * math.sin(
        2 * math.pi * (284 + day_of_year) / 365.0
    )


def hourly_sza(lat_deg, day_of_year, hour):
    """Return solar zenith angle for one latitude, day of year, and hour."""
    latitude = math.radians(lat_deg)
    solar_declination = declination(day_of_year)
    hour_angle = math.radians((hour - 12) * 15)
    cos_sza = (
        math.sin(latitude) * math.sin(solar_declination) +
        math.cos(latitude) * math.cos(solar_declination) * math.cos(hour_angle)
    )
    return math.degrees(math.acos(np.clip(cos_sza, -1, 1)))


def get_daytime_sza(lat_deg, day_of_year, max_sza=70):
    """Return hourly SZA values below the daylight threshold."""
    return np.array([
        hourly_sza(lat_deg, day_of_year, hour)
        for hour in range(24)
        if hourly_sza(lat_deg, day_of_year, hour) < max_sza
    ])


def daytime_latitude_weighted_albedo(
    cot, ocean, season, table_folder='cp', max_sza=70
):
    """Return daytime and latitude weighted SBDART albedo for one ocean-season.

    Daytime hours use cos(SZA) weights. Latitude bands use cos(latitude)
    weights, proportional to their area on a regular longitude grid.
    """
    from util_ocean_season_division import oceans_def
    from utils_fitting import cot_to_albedo

    season_doy = {'MAM': 105, 'JJA': 196, 'SON': 288, 'DJF': 15}
    if ocean not in oceans_def:
        raise ValueError(f'Unknown ocean: {ocean}')
    if season not in season_doy:
        raise ValueError(f'Unknown season: {season}')

    cot = np.asarray(cot, dtype=float)
    latitude_bands = oceans_def[ocean]
    latitudes = np.unique(np.concatenate([
        np.arange(south + 0.5, north, 1.0)
        for _, south, _, north in latitude_bands
    ]))
    numerator = np.zeros_like(cot, dtype=float)
    denominator = 0.0

    for latitude in latitudes:
        sza_values = get_daytime_sza(
            latitude, season_doy[season], max_sza=max_sza
        )
        if sza_values.size == 0:
            continue

        time_weights = np.cos(np.deg2rad(sza_values))
        weighted_albedo = np.zeros_like(cot, dtype=float)
        for sza, time_weight in zip(sza_values, time_weights):
            weighted_albedo += time_weight * cot_to_albedo(
                cot, 'sbdart', sza=sza, table_folder=table_folder,
                ocean=ocean, season=season,
            )

        latitude_weight = np.cos(np.deg2rad(latitude))
        numerator += latitude_weight * weighted_albedo
        denominator += latitude_weight * np.sum(time_weights)

    if denominator == 0:
        return np.full_like(cot, np.nan, dtype=float)
    return numerator / denominator


def calc_monthly_swdown(lat, year=2020, month=None):
    """Approximate monthly mean daily-mean insolation for a latitude."""
    if month is None:
        raise ValueError('month is required')
    from calendar import monthrange
    start = pd.Timestamp(year=year, month=month, day=1)
    days = [start + pd.Timedelta(days=i) for i in range(monthrange(year, month)[1])]
    phi = math.radians(float(lat))
    values = []
    for day in days:
        delta = declination(day.dayofyear)
        cos_h0 = -math.tan(phi) * math.tan(delta)
        if cos_h0 >= 1:
            h0 = 0.0
        elif cos_h0 <= -1:
            h0 = math.pi
        else:
            h0 = math.acos(cos_h0)
        distance_factor = 1 + 0.033 * math.cos(2 * math.pi * day.dayofyear / 365.0)
        values.append(1361.0 / math.pi * distance_factor * (
            h0 * math.sin(phi) * math.sin(delta) +
            math.cos(phi) * math.cos(delta) * math.sin(h0)
        ))
    return float(np.mean(values))


def calc_grid_cell_area(lat, lon_res=1.0, lat_res=1.0):
    """Return 1-degree grid-cell area in km2."""
    radius = 6371000.0
    lat1 = math.radians(float(lat) - lat_res / 2)
    lat2 = math.radians(float(lat) + lat_res / 2)
    dlon = math.radians(lon_res)
    return dlon * radius ** 2 * (math.sin(lat2) - math.sin(lat1)) / 1e6
