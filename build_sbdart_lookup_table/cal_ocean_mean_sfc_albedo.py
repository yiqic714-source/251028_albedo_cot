import netCDF4 as nc
import numpy as np
import os
from netCDF4 import Dataset
import pandas as pd

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

season_dict = {
    "DJF": [11, 0, 1],
    "MAM": [2, 3, 4],
    "JJA": [5, 6, 7],
    "SON": [8, 9, 10]
}

def get_ocean_mask(lat, lon, boxes=None, lsmask_path="/data/chenyiqi/251007_tropic/landsea.nc"):
    lon_is_half = np.isclose(lon % 1, 0.5, atol=1e-6)
    lat_is_half = np.isclose(lat % 1, 0.5, atol=1e-6)
    if not (lon_is_half.any() and lat_is_half.any()):
        raise ValueError("No ×.5° grid points found. Check grid definition.")

    region_mask_2d = np.ones((len(lat), len(lon)), dtype=bool) if (boxes is None or len(boxes) == 0) else np.zeros((len(lat), len(lon)), dtype=bool)
    if boxes and len(boxes) > 0:
        for west, south, east, north in boxes:
            lon_in = (lon >= west) | (lon <= east) if west > east else (lon >= west) & (lon <= east)
            lat_in = (lat >= south) & (lat <= north)
            region_mask_2d |= np.outer(lat_in, lon_in)

    if not os.path.exists(lsmask_path):
        raise FileNotFoundError(f"Land-sea mask file not found: {lsmask_path}")
    with Dataset(lsmask_path, 'r') as ds:
        lon_1deg = ds.variables['lon'][:]
        lat_1deg = ds.variables['lat'][:]
        lsmask = ds.variables['LSMASK'][:]
    lon_1deg = np.where(lon_1deg > 180, lon_1deg - 360, lon_1deg)
    if lsmask.shape == (len(lon_1deg), len(lat_1deg)):
        lsmask = lsmask.T
    ocean_1deg = (lsmask == 0)

    final_mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for i, la in enumerate(lat):
        if not lat_is_half[i]:
            continue
        for j, lo in enumerate(lon):
            if not lon_is_half[j]:
                continue
            lon_idx = np.where(np.isclose(lon_1deg, lo))[0]
            lat_idx = np.where(np.isclose(lat_1deg, la))[0]
            if len(lon_idx) == 1 and len(lat_idx) == 1:
                if ocean_1deg[lat_idx[0], lon_idx[0]] and region_mask_2d[i, j]:
                    final_mask[i, j] = True

    return final_mask

def calculate_seasonal_albedo(filepath, oceans, season_dict):
    try:
        with nc.Dataset(filepath) as ds:
            print(f"Successfully opened file: {filepath}")
            
            sfc_sw_up = ds.variables['ini_sfc_sw_up_naer_mon'][:]
            toa_solar = ds.variables['toa_solar_all_mon'][:]
            
            lat = ds.variables['lat'][:]
            lon = ds.variables['lon'][:]
            lon[lon>180] -= 360
            # print(np.min(lon), np.max(lon))
            time = ds.variables['time'][:]

            print(f"Data dimensions (time, latitude, longitude): {sfc_sw_up.shape}")

            albedo = np.where(toa_solar > 0, sfc_sw_up / toa_solar, np.nan)
            
            results = {}
            
            for ocean_name, boxes in oceans.items():
                print(f"--- Processing ocean region: {ocean_name} ---")
                
                ocean_mask_2d = get_ocean_mask(lat, lon, boxes=boxes)
                
                n_valid_points = np.sum(ocean_mask_2d)
                if n_valid_points == 0:
                    print(f"Warning: No valid grid points found in {ocean_name}.")
                    results[ocean_name] = {season: np.nan for season in season_dict.keys()}
                    continue
                
                print(f"Found {n_valid_points} valid grid points in {ocean_name}.")

                ocean_mask_3d = np.tile(ocean_mask_2d, (len(time), 1, 1))
                
                albedo_ocean_flat = albedo[ocean_mask_3d]
                
                albedo_ocean_reshaped = albedo_ocean_flat.reshape(-1, n_valid_points)
                
                mean_albedo_by_time = np.nanmean(albedo_ocean_reshaped, axis=1)
                
                ocean_results = {}
                
                for season_name, month_indices in season_dict.items():
                    season_time_indices = [i for i in range(len(time)) if (i % 12) in month_indices]
                    
                    if not season_time_indices:
                        print(f"Warning: No data found for season {season_name} in {ocean_name}.")
                        ocean_results[season_name] = np.nan
                        continue
                        
                    season_mean = np.nanmean(mean_albedo_by_time[season_time_indices])
                    ocean_results[season_name] = season_mean
                    print(f"{ocean_name} - {season_name} mean albedo: {season_mean:.4f}")
                
                results[ocean_name] = ocean_results

            return results

    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
        return None
    except Exception as e:
        print(f"An unknown error occurred during processing: {e}")
        return None

if __name__ == "__main__":
    ceres_filepath = "/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table/CERES_SYN1deg_Monthly_2001to2023/CERES_SYN1deg-Month_Terra-Aqua-MODIS_Ed4.1_Subset_200101-202312.nc"
    
    if not os.path.exists(ceres_filepath):
        print(f"Error: CERES file does not exist: {ceres_filepath}")
    else:
        seasonal_albedo_results = calculate_seasonal_albedo(ceres_filepath, oceans, season_dict)

        if seasonal_albedo_results:
            print("\n--- Calculation completed, saving results to CSV file ---")
            df_results = pd.DataFrame.from_dict(seasonal_albedo_results, orient='index')
            df_results['Annual'] = df_results.mean(axis=1)
            
            output_csv_path = "/home/chenyiqi/251028_albedo_cot/make_sbd_atms/atms_output/sfc_albedo_results.csv"
            df_results.to_csv(output_csv_path, float_format='%.4f')
            print(f"Results successfully saved to: {output_csv_path}")
            
            print("\nFinal results summary:")
            print(df_results.round(4))