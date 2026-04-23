# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 11:53:48 2025

@author: yiqi
"""

import os
from datetime import datetime
import pickle

def list_ceres_files_and_times(folder):
    """get names, start times, and end times of all quantified CERES files in a folder"""
    rsl_lst = []
    for fname in os.listdir(folder):
        if fname.startswith("CERES_SSF_Terra-XTRK_Edition4A_Subset") and fname.endswith(".nc"):
            full_path = os.path.join(ceres_folder, fname)
            start_str, end_str = fname.split('_')[-1].replace('.nc', '').split('-')
            start_time = datetime.strptime(start_str, "%Y%m%d%H")
            end_time = datetime.strptime(end_str, "%Y%m%d%H")
            rsl_lst.append({
                "filename": full_path,
                "start_time": start_time,
                "end_time": end_time
            })
            continue
            
    return rsl_lst

year = 2020
ceres_folder = '/data/chenyiqi/251028_albedo_cot/CERES_L2SSF_2020/'
ssf_files_and_times = list_ceres_files_and_times(ceres_folder)
data = {
    'ssf_files': ssf_files_and_times
}
# print(data)
pkl_file = '/data/chenyiqi/251028_albedo_cot/CERES_L2SSF_2020/CERES_' + str(year) + '_files_and_times.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(data, f)