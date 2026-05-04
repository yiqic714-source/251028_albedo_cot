import netCDF4 as nc
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
import pandas as pd

# -------------------------- Configuration --------------------------
exp = 'HadGEM3'
# File paths
if exp == 'HadGEM3':
    nat_file = '/home/chenyiqi/251201_ERFaci/cmip6/od550aer_AERmon_HadGEM3-GC31-LL_hist-aer_r1i1p1f3_gn_185001-194912.nc'
    aer_file = '/home/chenyiqi/251201_ERFaci/cmip6/od550aer_AERmon_HadGEM3-GC31-LL_hist-aer_r1i1p1f3_gn_195001-202012.nc'
elif exp == 'NorESM2':
    nat_file = '/home/chenyiqi/251201_ERFaci/cmip6/od550aer_AERmon_NorESM2-LM_hist-nat_r1i1p1f1_gn_201501-202012.nc'
    aer_file = '/home/chenyiqi/251201_ERFaci/cmip6/od550aer_AERmon_NorESM2-LM_hist-aer_r1i1p1f1_gn_201501-202012.nc'
lsmask_path = "/data/chenyiqi/251007_tropic/landsea.nc"

# time range of nat/aer
nat_start_year = 1850
nat_end_year = 1860
aer_start_year = 2010
aer_end_year = 2020

# Output configuration
output_dir = '/home/chenyiqi/251028_albedo_cot/processed_data'
os.makedirs(output_dir, exist_ok=True)
# 输出文件名：体现nat(1850-1860)和aer(2010-2020)的时间范围
csv_filename = os.path.join(output_dir, f'cmip6_AodDiff_nat{nat_start_year}to{nat_end_year}_aer{aer_start_year}to{aer_end_year}_{exp}.csv')
plot_filename = f'/home/chenyiqi/251028_albedo_cot/figs/lnAOD_diff_global_nat{nat_start_year}to{nat_end_year}_aer{aer_start_year}to{aer_end_year}_{exp}.png'

# Target grid (1° resolution, center at X.5°)
lat_target = np.arange(-59.5, 60, 1)  # Restrict to ±60° latitude
lon_target = np.arange(-179.5, 180, 1)  # Full longitude range (-180~180°)

# -------------------------- Load Land-Sea Mask --------------------------
def load_land_sea_mask(mask_path):
    with nc.Dataset(mask_path, 'r') as ds:
        lat_lsm = ds.variables['lat'][:].astype(float)
        lsmask = ds.variables['LSMASK'][:].astype(bool) 
    ocean_mask = lsmask==0
    half_width = ocean_mask.shape[1] // 2
    ocean_mask = np.hstack([ocean_mask[:, half_width:], ocean_mask[:, :half_width]])
    ocean_mask = ocean_mask[(lat_lsm < 60) & (lat_lsm > -60)]
    
    return ocean_mask

ocean_mask = load_land_sea_mask(lsmask_path)

# -------------------------- Data Loading --------------------------
def load_nc_data(file_path):
    """Load AOD data and coordinates from NC file"""
    with nc.Dataset(file_path, 'r') as ds:
        od550 = ds.variables['od550aer'][:].astype(np.float32)  # [time, lat, lon]
        lat = ds.variables['lat'][:].astype(np.float32)
        lon = ds.variables['lon'][:].astype(np.float32)
    
    # Standardize longitude to 0-360°
    lon[lon < 0] += 360
    
    # extract file's start year
    file_time_str = file_path.split('_')[-1].split('-')[0]
    file_base_year = int(file_time_str[:4])
    
    return od550, lat, lon, file_base_year

# Load natural (1850-1860) and anthropogenic+natural (2010-2020) AOD data
od550_nat, lat_had, lon_had, nat_base_year = load_nc_data(nat_file)
od550_aer, _, _, aer_base_year = load_nc_data(aer_file)

# -------------------------- Time Slicing --------------------------
# Calculate time indices for nat data
nat_start_idx = (nat_start_year - nat_base_year) * 12
nat_end_idx = (nat_end_year - nat_base_year + 1) * 12
od550_nat_slice = od550_nat[nat_start_idx:nat_end_idx, :, :]

# Calculate time indices for aer data
aer_start_idx = (aer_start_year - aer_base_year) * 12
aer_end_idx = (aer_end_year - aer_base_year + 1) * 12
od550_aer_slice = od550_aer[aer_start_idx:aer_end_idx, :, :]

# -------------------------- Monthly Average Calculation --------------------------
n_months = 12
# initialize
od550_nat_monthly_avg = np.zeros((n_months, od550_nat_slice.shape[1], od550_nat_slice.shape[2]), dtype=np.float32)
od550_aer_monthly_avg = np.zeros((n_months, od550_aer_slice.shape[1], od550_aer_slice.shape[2]), dtype=np.float32)

# take monthly average of nat
for month in range(n_months):
    nat_month_data = od550_nat_slice[month::n_months, :, :]
    od550_nat_monthly_avg[month] = np.nanmean(nat_month_data, axis=0)

# take monthly average of aer
for month in range(n_months):
    aer_month_data = od550_aer_slice[month::n_months, :, :]
    od550_aer_monthly_avg[month] = np.nanmean(aer_month_data, axis=0)

# month labels
months = np.arange(1, n_months+1, dtype=np.int32)

# -------------------------- Interpolation with Longitude Edge Fix --------------------------
def interpolate_aod(od550_data, lat_src, lon_src, lat_tgt, lon_tgt):
    """
    Interpolate AOD data to target grid with fixed longitude edge issue
    Extend longitude to -15~375° (sufficient for spherical continuity)
    """
    # Extend source longitude to -15~375° for edge interpolation
    lon_src_ext = np.concatenate([
        lon_src[lon_src > 345] - 360,  # -15~0° segment
        lon_src,                       # 0~360° original
        lon_src[lon_src < 15] + 360    # 360~375° segment
    ])
    
    # Extend data to match extended longitude
    od550_ext = np.concatenate([
        od550_data[:, :, lon_src > 345],  # Match -15~0°
        od550_data[:, :, :],              # Match 0~360°
        od550_data[:, :, lon_src < 15]    # Match 360~375°
    ], axis=2)
    
    # Initialize interpolation result array
    interpolated = np.zeros((od550_data.shape[0], len(lat_tgt), len(lon_tgt)), dtype=np.float32)
    
    # Interpolate each time step (now monthly avg, 12 steps total)
    for t in range(od550_data.shape[0]):
        # Ensure monotonic grid (critical for interpolation)
        assert np.all(np.diff(lon_src_ext) > 0), "Longitude must be strictly increasing"
        assert np.all(np.diff(lat_src) > 0), "Latitude must be strictly increasing"
        
        # Create interpolation function
        interp_func = RegularGridInterpolator(
            (lat_src, lon_src_ext),
            od550_ext[t],
            bounds_error=False,
            fill_value=np.nan,
            method='linear'
        )
        
        # Convert target longitude to 0-360° for interpolation
        lon_tgt_360 = np.where(lon_tgt < 0, lon_tgt + 360, lon_tgt)
        lat_grid, lon_grid_360 = np.meshgrid(lat_tgt, lon_tgt_360, indexing='ij')
        points = np.stack([lat_grid.ravel(), lon_grid_360.ravel()], axis=-1)
        
        # Perform interpolation and reshape
        interpolated[t] = interp_func(points).reshape(len(lat_tgt), len(lon_tgt))
    
    return interpolated

# Perform interpolation (±60° latitude only)
od550_nat_interp = interpolate_aod(od550_nat_monthly_avg, lat_had, lon_had, lat_target, lon_target)
od550_aer_interp = interpolate_aod(od550_aer_monthly_avg, lat_had, lon_had, lat_target, lon_target)

# -------------------------- Apply Land-Sea Mask --------------------------
def apply_ocean_mask(data, ocean_mask):
    for t in range(data.shape[0]):
        data[t, ~ocean_mask] = np.nan
    return data

# Apply ocean mask to both datasets
od550_nat_interp = apply_ocean_mask(od550_nat_interp, ocean_mask)
od550_aer_interp = apply_ocean_mask(od550_aer_interp, ocean_mask)

# -------------------------- Calculate ln(aer) - ln(nat) --------------------------
# Mask valid values (positive and non-NaN)
valid_mask = (od550_nat_interp > 0) & (od550_aer_interp > 0) & ~np.isnan(od550_nat_interp) & ~np.isnan(od550_aer_interp)

# Calculate log difference
ln_diff = np.full_like(od550_nat_interp, np.nan, dtype=np.float32)
ln_diff[valid_mask] = np.log(od550_aer_interp[valid_mask]) - np.log(od550_nat_interp[valid_mask])

# -------------------------- Visualization: Global ln(AOD) Difference Scatter Plot --------------------------
def plot_ln_aod_difference(ln_diff, lat_mesh, lon_mesh, output_path):
    """
    Create global scatter plot of mean ln(aer) - ln(nat) difference
    Focus on ocean areas (±60° latitude range)
    """
    # Calculate time-averaged ln difference (mean over 12 months)
    mean_ln_diff = np.nanmean(ln_diff, axis=0)
    
    # Flatten data and filter valid values
    mask_valid = ~np.isnan(mean_ln_diff)
    lon_flat = lon_mesh[mask_valid]
    lat_flat = lat_mesh[mask_valid]
    ln_diff_flat = mean_ln_diff[mask_valid]
    
    # Set up plot with Miller projection (global view)
    fig = plt.figure(figsize=(18, 10))
    proj = ccrs.Miller(central_longitude=180)
    ax = plt.axes(projection=proj)
    
    # Set map extent (±60° latitude to match data range)
    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray', alpha=0.7)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.1)
    
    # Add grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True, 
                      linewidth=0.5, 
                      color='gray', 
                      alpha=0.7, 
                      linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(np.arange(-60, 61, 30))
    gl.xlabel_style = {'fontsize': 10}
    gl.ylabel_style = {'fontsize': 10}
    
    # Set symmetric colormap (centered at 0)
    vmin = np.nanpercentile(ln_diff_flat, 1)  # 1st percentile to exclude outliers
    vmax = np.nanpercentile(ln_diff_flat, 99)
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    
    # Create scatter plot
    scatter = ax.scatter(lon_flat, 
                         lat_flat, 
                         c=ln_diff_flat, 
                         s=8,
                         cmap='RdBu_r',  # Red (positive) / Blue (negative)
                         norm=norm,
                         alpha=0.8,
                         edgecolors='none',
                         transform=ccrs.PlateCarree())
    
    # Add colorbar with label
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.05, extend='both')
    cbar.set_label(f'ln(od550aer) - ln(od550nat) (aer: {aer_start_year}-{aer_end_year}, nat: {nat_start_year}-{nat_end_year})', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title(f'Global Distribution of ln(AOD$_{{aer}}$) - ln(AOD$_{{nat}}$)\nAER: {aer_start_year}-{aer_end_year} Avg, NAT: {nat_start_year}-{nat_end_year} Avg', 
                 fontsize=14, pad=20)
    
    # Save high-resolution plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n📊 Plot saved to: {output_path}")
    print(f"📈 ln_diff Mean: {np.nanmean(mean_ln_diff):.4f}")

# Generate the global scatter plot
lon_mesh, lat_mesh = np.meshgrid(lon_target, lat_target)
plot_ln_aod_difference(ln_diff, lat_mesh, lon_mesh, plot_filename)

# -------------------------- Build Feature Array --------------------------
# Create grid coordinates
lon_flat = lon_mesh.ravel()
lat_flat = lat_mesh.ravel()

# Build feature list (filter NaN rows)
feature_list = []
for t in range(len(months)):
    # Create monthly feature array: [lon, lat, month, ln_diff]
    month_feature = np.column_stack([
        lon_flat,
        lat_flat,
        np.full_like(lon_flat, months[t], dtype=np.int32),
        ln_diff[t].ravel()
    ])
    
    # Filter out rows with NaN values
    valid_rows = ~np.isnan(month_feature).any(axis=1)
    feature_list.append(month_feature[valid_rows])

# Combine all valid features
feature_data = np.vstack(feature_list).astype(np.float32)

# Feature column names
feature_column_names = [
    'longitude', 
    'latitude', 
    'month', 
    'log_aod_diff'
]

# -------------------------- Save to CSV --------------------------
# Convert to DataFrame
feature_df = pd.DataFrame(
    data=feature_data,
    columns=feature_column_names
)
# Fix month data type
feature_df['month'] = feature_df['month'].astype(int)
# Save CSV
feature_df.to_csv(csv_filename, index=False, encoding='utf-8')

# -------------------------- Validation Output --------------------------
print("="*60)
print("✅ Data Processing Complete!")
print("="*60)
print(f"Time Range:")
print(f"  - NAT Data: {nat_start_year}-{nat_end_year} (multi-year, 12 months avg)")
print(f"  - AER Data: {aer_start_year}-{aer_end_year} (multi-year, 12 months avg)")
print(f"Target Grid:")
print(f"  - Latitude: {lat_target.min():.1f}° to {lat_target.max():.1f}° (±60° range) | Points: {len(lat_target)}")
print(f"  - Longitude: {lon_target.min():.1f}° to {lon_target.max():.1f}° (full 360°) | Points: {len(lon_target)}")
print(f"Feature Array Shape: {feature_data.shape} (rows × columns)")
print(f"Feature Columns: {feature_column_names}")
print(f"Output File: {csv_filename}")

# Validate interpolation results
lon_vals = feature_data[:, 0]
lat_vals = feature_data[:, 1]
print(f"\nInterpolation Validation:")
print(f"  - Longitude Range: {np.min(lon_vals):.1f}° to {np.max(lon_vals):.1f}° (expected: -179.5° to 178.5°)")
print(f"  - Latitude Range: {np.min(lat_vals):.1f}° to {np.max(lat_vals):.1f}° (expected: -59.5° to 59.5°)")
print(f"  - Valid Samples (non-NaN): {feature_data.shape[0]}")

# Log difference statistics
valid_diff = feature_data[:, 3]
print(f"\nln(aer)-ln(nat) Mean: {np.nanmean(valid_diff):.4f}")