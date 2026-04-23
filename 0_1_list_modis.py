import pickle
import glob
import calendar
import sys
from datetime import datetime, timedelta
from pyhdf.SD import SD, SDC
import numpy as np

def find_mod_granules_terra(year, doy):
    # Calculate the corresponding month from day of year
    # Create a date object for January 1st of the given year
    start_date = datetime(int(year), 1, 1)
    # Add (doy-1) days to get the target date
    target_date = start_date + timedelta(days=doy - 1)
    # Get the month (1-12)
    month = target_date.month
    # Format month as two-digit string
    month_str = f"{month:02d}"
    
    # Dynamically construct folder path containing the month
    mod_dir = f'{mod_folder}/{year}{month_str}S/MOD06_L2.A{year}{doy:03d}'
    mod_lst = sorted(glob.glob(mod_dir + '*.hdf'))
    mod_lst = delete_no_overlap(mod_lst)
    # print(mod_lst)
    return mod_lst

def delete_no_overlap(filelist):
    delfiles=[]
    for filename in filelist:
        try: 
            hdf = SD(filename, SDC.READ)
        except: 
            delfiles.append(filename)
            continue
        lat = hdf.select('Latitude')[:]
        lon = hdf.select('Longitude')[:]
        lat[lat == -999] = np.nan
        lon[lon == -999] = np.nan
        
        lat_min = np.nanmin(lat)
        lat_max = np.nanmax(lat)
        lon_min = np.nanmin(lon)
        lon_max = np.nanmax(lon)
        if (lon_max-lon_min) > 180 and obs_window[1][0]>0 and obs_window[1][1]>0:
            # print(lons.max(), lons.min())
            lon_min = np.nanmin(lon[lon>0])
            lon_max = np.nanmax(lon[lon<0]) + 360
        if (lon_max-lon_min) > 180 and obs_window[1][0]<0 and obs_window[1][1]<0:
            lon_max = np.nanmax(lon[lon<0])
            lon_min = np.nanmin(lon[lon>0]) - 360
            
        if (lat_max < obs_window[0][0]  or 
            lat_min > obs_window[0][1] or 
            lon_max < obs_window[1][0] or 
            lon_min > obs_window[1][1]):
            delfiles.append(filename)
    
    for filename in delfiles:
        filelist.remove(filename)
    return filelist

# ----------------- Find all files within the date range ----------------------
if __name__ == "__main__": 
    year = sys.argv[1]
    month = int(sys.argv[2])
    hemisph = sys.argv[3]

    if hemisph == "east":
        obs_window = [[-60., -23.],[0., 180.]]
        fname_ending = "lon_0_180"
    elif hemisph == "west":
        obs_window = [[-60., -23.],[-180., 0.]]
        fname_ending = "lon_m180_0"
    else:
        print('Only support hemisphere to be east or west')

    mod_folder = '/data/chenyiqi/251028_albedo_cot/mod06'
    pkl_file = mod_folder + f'/MOD06_files_{year}{month:02d}S_{fname_ending}.pkl'

    # Calculate start_doy and end_doy based on months
    start_date = datetime(int(year), month, 1)
    doy_start = (start_date - datetime(int(year), 1, 1)).days + 1

    # Get the last day of the end_month
    _, last_day = calendar.monthrange(int(year), month)
    end_date = datetime(int(year), month, last_day)
    doy_end = (end_date - datetime(int(year), 1, 1)).days + 1
        
    mod_lst = {'aqua': [], 'terra': []}

    # Iterate through the specified range of days of the year
    for doy in range(doy_start, doy_end + 1):
        print(f"Processing day of year: {doy}")
        fnames = find_mod_granules_terra(year, doy)
        mod_lst['terra'].extend(fnames)

    # -------------------------- Save to pkl file ---------------------------------
    # Construct path for the saved file using start_month and end_month directly
    with open(pkl_file, "wb") as f:
        pickle.dump(mod_lst, f)

    print(f"{len(mod_lst['terra'])} filenames saved to {pkl_file}")
        

