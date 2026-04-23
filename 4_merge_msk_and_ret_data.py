import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from dateutil import parser

# Plot configuration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (4.5, 6.5)
plt.rcParams['axes.unicode_minus'] = False

# Different fields to exclude for different directories
FIELDS_TO_REMOVE_DIR1 = {'cot_ceres', 'clr_fra'}
FIELDS_TO_REMOVE_DIR2 = {
    'ret_re_mod', 'solar_zenith', 'sensor_zenith',
    'ret_fov_fra', 'ret_albedo_uncert',
    'ret_uncorrected_albedo', 'ret_fra', 'unr_fra'
}

def calculate_rmse(x, y):
    """Calculate Root Mean Square Error (RMSE) between two arrays"""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2 or len(y_clean) < 2:
        return np.nan
    
    rmse = np.sqrt(np.mean((x_clean - y_clean) ** 2))
    return round(rmse, 1)

def load_and_clean_csv(file_path, fields_to_remove):
    """
    Load CSV file, remove specified fields, return dict with lat+lon+date as key
    
    Args:
        file_path: Path to CSV file
        fields_to_remove: Set of fields to remove from the CSV
    """
    csv_dict = {}
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        return csv_dict
    
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cleaned_headers = [h for h in reader.fieldnames if h not in fields_to_remove]
        
        for row in reader:
            lat = row.get('lat', '').strip()
            lon = row.get('lon', '').strip()
            time_str = row.get('time', '').strip()
            
            try:
                dt = parser.parse(time_str)
                date_key = dt.date().strftime('%Y-%m-%d')
            except:
                date_key = time_str
            
            key = f"{lat}_{lon}_{date_key}"
            cleaned_row = {h: row[h].strip() for h in cleaned_headers}
            csv_dict[key] = cleaned_row
    
    print(f"Processed {file_path}: {len(reader.fieldnames)} fields → {len(cleaned_headers)} fields")
    return csv_dict

def process_ocean_data(ocean, input_dir1, input_dir2, output_csv_dir):
    """Merge ocean region data, filter valid rows (aod_mod08 non-nan + basic conditions), save results"""
    file1 = os.path.join(input_dir1, f"{ocean}.csv")
    file2 = os.path.join(input_dir2, f"{ocean}.csv")
    output_csv = os.path.join(output_csv_dir, f"{ocean}.csv")
    
    # Load each file with its specific field removal rules
    dict1 = load_and_clean_csv(file1, FIELDS_TO_REMOVE_DIR1)
    dict2 = load_and_clean_csv(file2, FIELDS_TO_REMOVE_DIR2)
    
    if not dict1 or not dict2:
        print(f"Warning: No valid data in {ocean}, skip processing")
        return [], [], []
    
    common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
    valid_rows = []
    header1 = list(dict1.values())[0].keys() if dict1 else []
    header2 = list(dict2.values())[0].keys() if dict2 else []
    merged_header = list(header1) + [h for h in header2 if h not in header1]
    
    total_count = 0
    valid_count = 0
    
    for key in common_keys:
        total_count += 1
        merged_row = {h: '' for h in merged_header}
        
        if key in dict1:
            for k, v in dict1[key].items():
                merged_row[k] = v
        if key in dict2:
            for k, v in dict2[key].items():
                merged_row[k] = v
        
        try:
            # Parse numeric fields and handle empty strings
            cf_ceres_str = merged_row.get('cf_ceres', '').strip()
            cf_liq_ceres_str = merged_row.get('cf_liq_ceres', '').strip()
            cttmin_str = merged_row.get('cttmin', '').strip()
            
            cf_ceres = float(cf_ceres_str) if cf_ceres_str else np.nan
            cf_liq_ceres = float(cf_liq_ceres_str) if cf_liq_ceres_str else np.nan
            cttmin = float(cttmin_str) if cttmin_str else np.nan
            
            # Filter valid rows: non-nan aod_mod08 + basic conditions
            if (not np.isnan(cf_liq_ceres) and not np.isnan(cttmin) and
                cf_liq_ceres / cf_ceres > 0.99 and 
                cf_ceres > 0.1):
                valid_rows.append(merged_row)
                valid_count += 1

        except Exception:
            continue
    
    # Print statistics
    print(f"\n{ocean} Data Statistics:")
    print(f"  Total matched rows: {total_count}")
    print(f"  Valid rows after filtering: {valid_count}")
    
    # Save valid data to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=merged_header)
        writer.writeheader()
        writer.writerows(valid_rows)
    
    print(f"  Output file saved: {output_csv}")
    
    return valid_rows, merged_header, valid_count

def convert_to_plot_dict(rows, header):
    """Convert row list to plot-ready dictionary"""
    plot_dict = {h: [] for h in header}
    for row in rows:
        for h in header:
            plot_dict[h].append(row[h])
    return plot_dict

def plot_density_subplot(ax, plot_data, x_field, y_field, x_label, y_label, title, bins=40, xlim=None, ylim=None):
    """Create 2D density plot with post-filter RMSE annotation (format: RMSE=?.1f)"""
    def to_numeric(values):
        nums = []
        for v in values:
            try:
                num = float(v)
                nums.append(num if not np.isnan(num) and not np.isinf(num) else np.nan)
            except:
                nums.append(np.nan)
        return np.array(nums)
    
    # Prepare data
    x_data = to_numeric(plot_data.get(x_field, []))
    y_data = to_numeric(plot_data.get(y_field, []))
    
    # Filter NaN values
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    
    # Plot 2D density histogram
    if len(x_clean) > 0 and len(y_clean) > 0:
        hist, x_edges, y_edges = np.histogram2d(x_clean, y_clean, bins=bins, density=True)
        X, Y = np.meshgrid((x_edges[:-1] + x_edges[1:])/2, (y_edges[:-1] + y_edges[1:])/2)
        
        # Default colormap
        im = ax.pcolormesh(X, Y, hist.T, alpha=0.8, shading='auto')
        plt.colorbar(im, ax=ax, shrink=1, label='Density')
    
    # Calculate and annotate post-filter RMSE (1 decimal place)
    rmse = calculate_rmse(x_clean, y_clean)
    rmse_text = f'RMSE={rmse:.1f}' if not np.isnan(rmse) else 'RMSE=N/A'
    ax.text(
        0.03, 0.97, rmse_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Plot properties
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=16, pad=5, loc='left')
    ax.grid(alpha=0.3)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

def plot_combined_density_plots(valid_data, header, output_dir):
    """Create combined density plots for all ocean regions"""
    required_fields = ['ret_cot_mod', 'ret_cot_cer', 'cf_ceres', 'cf_liq_ceres', 'aod_mod08']
    missing_fields = [f for f in required_fields if f not in header]
    
    if missing_fields:
        print(f"Warning: Missing fields for plotting - {missing_fields}")
        return
    
    # Convert to plot format
    plot_data = convert_to_plot_dict(valid_data, header)
    
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(4, 5.6))
    
    # Subplot 1: COT (0-50 range)
    plot_density_subplot(
        ax=axes[0],
        plot_data=plot_data,
        x_field='ret_cot_mod',
        y_field='ret_cot_cer',
        x_label='MODIS URF COT',
        y_label='CERES URF COT',
        title='(a)',
        bins=70,
        xlim=(0, 40),
        ylim=(0, 40)
    )
    
    # Subplot 2: CF
    plot_density_subplot(
        ax=axes[1],
        plot_data=plot_data,
        x_field='cf_ceres',
        y_field='cf_liq_ceres',
        x_label='MODIS CF$_{\mathrm{mask}}$',
        y_label='CERES CF$_{\mathrm{mask}}$',
        title='(b)',
        bins=40
    )
    
    # Save plot
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, top=0.95, bottom=0.07, hspace=0.35)
    plot_path = os.path.join(output_dir, 'modis_ceres_consistence.png')
    plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nCombined density plot saved: {plot_path}")

if __name__ == "__main__":
    # Ocean regions
    OCEANS = ['NPO', 'NAO', 'TAO', 'TIO', 'TPO', 'SPO', 'SAO', 'SIO']
    
    # Path configuration
    INPUT_DIR1 = "/home/chenyiqi/251028_albedo_cot/SSFproduct/ocean_data0311/"
    INPUT_DIR2 = "/home/chenyiqi/251028_albedo_cot/uniform_fov_product/ocean_data1227/"
    OUTPUT_CSV_DIR = "/home/chenyiqi/251028_albedo_cot/processed_data/merged_msk_and_ret_csv/"
    OUTPUT_PLOT_DIR = "/home/chenyiqi/251028_albedo_cot/figs/"
    
    # Create output directories
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    
    # Collect valid data
    all_valid_rows = []
    merged_header = []
    total_valid_count = 0
    
    # Process each ocean region
    for ocean in OCEANS:
        print(f"\n{'='*50}")
        print(f"Processing ocean region: {ocean}")
        print(f"{'='*50}")
        
        valid_rows, header, valid_count = process_ocean_data(ocean, INPUT_DIR1, INPUT_DIR2, OUTPUT_CSV_DIR)
        
        if valid_rows and header:
            all_valid_rows.extend(valid_rows)
            merged_header = header
            total_valid_count += valid_count
    
    # Generate plots if valid data exists
    if all_valid_rows:
        print(f"\n{'='*50}")
        print(f"Generating combined plot for {len(OCEANS)} ocean regions")
        print(f"Total valid rows for plotting: {total_valid_count}")
        print(f"{'='*50}")
        
        plot_combined_density_plots(all_valid_rows, merged_header, OUTPUT_PLOT_DIR)
    else:
        print("\nWarning: No valid data collected - skipping plot generation")
    
    print("\nAll ocean regions processed successfully!")