# -*- coding: utf-8 -*-
"""
ocean_region_analysis.py

- Plot all latitude–longitude rectangular regions defined in the oceans dictionary
 (supports splitting subregions crossing 180°).
- Automatically assign a color to each ocean and display it in the legend.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Polygon
import glob

# Define ocean regions with coordinates (west, south, east, north)
oceans = {
    'NPO': [
        [-170, 20, -100, 60],
        [-180, 20, -170, 60],
        [105, 20, 180, 60]
    ],
    'NAO': [
        [-100, 55, 45, 60],
        [-100, 40, 27, 55],
        [-100, 30, 45, 40],
        [-100, 20, 30, 30]
    ],
    'TPO': [
        [-170, 16, -100, 20],
        [-170, 13, -89, 16],
        [-170, 9, -84, 13],
        [-170, -20, -70, 9],
        [100, 0, 180, 20],
        [130, -20, 180, 0],
        [-180, -20, -170, 20]
    ],
    'TAO': [
        [-100, 16, -15, 20],
        [-84, 9, -13, 16],
        [-60, -20, 15, 9]
    ],
    'TIO': [
        [30, 0, 100, 30],
        [30, -20, 130, 0]
    ],
    'SPO': [
        [-170, -60, -70, -20],
        [130, -60, 180, -20],
        [-180, -60, -170, -20]
    ],
    'SAO': [
        [-70, -60, 20, -20]
    ],
    'SIO': [
        [20, -60, 130, -20]
    ]
}


def is_in_ocean(lat, lon, bounds_list):
    """
    Determine if coordinates are within the provided list of bounds.
    bounds_list: list of [west, south, east, north] boxes
    lon: will be normalized before comparison
    """
    def normalize_lon(lon):
        """
        将任意经度规范化到 [-180, 180) 范围
        """
        return ((lon + 180) % 360) - 180
    
    lon_n = normalize_lon(lon)
    for bound in bounds_list:
        min_lon, min_lat, max_lon, max_lat = bound
        min_lon_n = normalize_lon(min_lon)
        max_lon_n = normalize_lon(max_lon)

        # latitude check
        if not (lat >= min_lat and lat <= max_lat):
            continue

        # longitude check with possible wrap
        if min_lon_n <= max_lon_n:
            if lon_n >= min_lon_n and lon_n <= max_lon_n:
                return True
        else:
            # wrapped: lon in [min_lon_n, 180] or [-180, max_lon_n]
            if lon_n >= min_lon_n or lon_n <= max_lon_n:
                return True
    return False

def process_file(df, output_pattern):
    """
    Process the input CSV file and categorize data by ocean regions.
    """
    ocean_data = {ocean: pd.DataFrame() for ocean in oceans.keys()}

    for ocean, bounds in oceans.items():
        mask = df.apply(lambda row: is_in_ocean(row['lat'], row['lon'], bounds), axis=1)
        ocean_rows = df[mask].copy()
        ocean_data[ocean] = pd.concat([ocean_data[ocean], ocean_rows], ignore_index=True)

    for ocean, odf in ocean_data.items():
        if not odf.empty:
            output_csv = output_pattern + f'{ocean}.csv'
            odf.to_csv(output_csv, index=False)
            print(f"Saved {ocean} data to {output_csv}, total {len(odf)} rows")
        else:
            print(f"No matching data for {ocean}")

    print("File processing completed")


if __name__ == "__main__":
    file_paths = glob.glob("/home/chenyiqi/251028_albedo_cot/SSFproduct/2020.csv")
    df = pd.concat(
        [pd.read_csv(file) 
        for file in file_paths],
        ignore_index=True)
    # df = df[df['ret_cot_cer'].notna()]
    output_pattern = '/home/chenyiqi/251028_albedo_cot/SSFproduct/ocean_data0311/'
    process_file(df, output_pattern)