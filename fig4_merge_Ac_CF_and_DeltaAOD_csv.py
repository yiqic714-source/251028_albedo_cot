import pandas as pd
import numpy as np
import os

# -------------------------- Load & Process CERES 2020 Data --------------------------
# Read raw CERES 2020 dataset
ceres_file_path = '/home/chenyiqi/251028_albedo_cot/SSFproduct/2020.csv'
ceres_df = pd.read_csv(ceres_file_path)

# Convert time to datetime and extract month
ceres_df['time'] = pd.to_datetime(ceres_df['time'], format='mixed')
ceres_df['month'] = ceres_df['time'].dt.month

# Initialize Ac column with NaN (preserve all rows)
ceres_df['Ac'] = np.nan

# Define mask for valid data (only calculate Ac for valid rows)
valid_mask = (
    (ceres_df['cf_ceres'] > 0.1) & 
    (ceres_df['cf_liq_ceres'] / ceres_df['cf_ceres'] > 0.99) & 
    (ceres_df['solar_incoming'] > 1e-10)  # Avoid division by zero
)

# Calculate Ac for valid rows only (invalid rows remain NaN)
ceres_df.loc[valid_mask, 'Ac'] = (
    (ceres_df.loc[valid_mask, 'sw_all'] - ceres_df.loc[valid_mask, 'sw_clr'] * (1 - ceres_df.loc[valid_mask, 'cf_liq_ceres'])) 
    / ceres_df.loc[valid_mask, 'cf_liq_ceres'] 
    / ceres_df.loc[valid_mask, 'solar_incoming']
)

# Filter invalid Ac values (0 < Ac < 1)
ceres_df['Ac'] = ceres_df['Ac'].mask((ceres_df['Ac'] > 1) | (ceres_df['Ac'] < 0))

# -------------------------- Aggregate by Lat/Lon/Month --------------------------
# Group by geographic coordinates + month, compute nanmean (preserve all groups)
group_columns = ['lat', 'lon', 'month']
aggregation_dict = {
    'Ac': lambda x: np.nanmean(x),
    'cot_mod08': lambda x: np.nanmean(x),
    'cf_ret_liq_mod08': lambda x: np.nanmean(x),
    'cf_liq_ceres': lambda x: np.nanmean(x)
}

# Aggregate data (as_index=False keeps group columns as regular columns)
ceres_avg_df = ceres_df.groupby(group_columns, as_index=False).agg(aggregation_dict)

# -------------------------- Load & Merge CMIP6 AOD Data --------------------------
# Read CMIP6 AOD difference data
aod_file_path = '/home/chenyiqi/251028_albedo_cot/processed_data/cmip6_AodDiff_nat1850to1860_aer2010to2020_HadGEM3.csv'
aod_df = pd.read_csv(aod_file_path)

# Standardize column names for merge compatibility
aod_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)

# Merge CERES and AOD data (LEFT JOIN to preserve all CERES rows)
merge_keys = ['lat', 'lon', 'month']
merge_df = pd.merge(
    ceres_avg_df,
    aod_df,
    on=merge_keys,
    how='left'  # Preserve all rows from left table (CERES), NaN for unmatched AOD data
)

# -------------------------- Save Merged Data --------------------------
# Create output directory if not exists
output_dir = '/home/chenyiqi/251028_albedo_cot/processed_data'
os.makedirs(output_dir, exist_ok=True)

# Define output file path
output_file = os.path.join(output_dir, 'Ac_CF_and_DeltaAOD_2020.csv')

# Save merged data (keep NaN values, use 'NaN' for missing values)
merge_df.to_csv(
    output_file,
    index=False,
    encoding='utf-8',
    na_rep='NaN'
)

# -------------------------- Validation Output --------------------------
print("="*60)
print("✅ Data Processing Completed Successfully!")
print("="*60)
print(f"📊 Raw CERES data rows: {len(ceres_df)}")
print(f"📊 Aggregated CERES data rows (Lat/Lon/Month): {len(ceres_avg_df)}")
print(f"📊 CMIP6 AOD difference data rows: {len(aod_df)}")
print(f"📊 Merged data rows (all retained, including NaN rows): {len(merge_df)}")
print(f"💾 Merged file saved to: {output_file}")
print(f"\n📋 Merged data columns: {merge_df.columns.tolist()}")