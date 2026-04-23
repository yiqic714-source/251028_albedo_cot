import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Basic Configuration
OCEANS = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']
DATA_ROOT = "/home/chenyiqi/251028_albedo_cot/processed_data/merged_msk_and_ret_csv"
COT_COL = "cot_mod08"
CER_COL = "cer_mod08"
OUT_DIR = "figs/"
os.makedirs(OUT_DIR, exist_ok=True)

# Plot Style Configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
colors = plt.cm.Set3(np.linspace(0, 1, len(OCEANS)))
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']  # Unique marker for each ocean

# -------------------------- Step 1: Preprocess all data (one CSV read only) --------------------------
valid_cer_data = []
ocean_data_dict = {}  # key: ocean name, value: valid CER array (COT≥3)

for ocean in OCEANS:
    file_path = os.path.join(DATA_ROOT, f"{ocean}.csv")
    if not os.path.exists(file_path):
        print(f"Warning: {ocean} file not found - {file_path}, skipped")
        continue
    df = pd.read_csv(file_path)
    if not all(col in df.columns for col in [COT_COL, CER_COL]):
        print(f"Warning: {ocean} missing {COT_COL}/{CER_COL} columns, skipped")
        continue
    # Data cleaning: COT≥3, CER≥0, remove NaNs/non-numerics
    df_clean = df[[COT_COL, CER_COL]].dropna()
    df_clean = df_clean[pd.to_numeric(df_clean[COT_COL], errors='coerce').notna()]
    df_clean = df_clean[pd.to_numeric(df_clean[CER_COL], errors='coerce').notna()]
    df_clean = df_clean[(df_clean[COT_COL] >= 3) & (df_clean[CER_COL] >= 0)]
    if len(df_clean) < 10:
        print(f"Warning: {ocean} insufficient valid data (<10), skipped")
        continue
    # Save valid CER data
    cer_vals = df_clean[CER_COL].values
    valid_cer_data.extend(cer_vals)
    ocean_data_dict[ocean] = cer_vals

# Terminate if no valid data
if not ocean_data_dict:
    print("Error: No valid ocean region data found, program exited")
    exit()

# Unified fixed bins for all oceans (ensure comparability)
global_cer_min, global_cer_max = np.min(valid_cer_data), np.max(valid_cer_data)
bins = np.linspace(global_cer_min, global_cer_max, 50)  # 50 bins (adjustable)
bin_width = bins[1] - bins[0]  # Calculate bin width for normalization

# -------------------------- Step 2: Plot normalized line-style CER histogram --------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 8), tight_layout=True)

for idx, (ocean, cer_vals) in enumerate(ocean_data_dict.items()):
    # Calculate raw histogram counts
    counts, bin_edges = np.histogram(cer_vals, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centers for smooth X-axis
    
    # 核心：数据量归一化 → 转换为概率密度（Probability Density）
    # 公式：density = 计数 / (总数据量 * 分箱宽度)，保证所有曲线下面积之和为1
    total_data = len(cer_vals)
    density = counts / (total_data * bin_width)

    # Plot normalized line (no bar, unique color+marker)
    ax.plot(
        bin_centers, density,
        color=colors[idx],
        marker=markers[idx],
        markersize=4,
        linewidth=2,
        alpha=0.8,
        label=ocean
    )

# -------------------------- Step 3: Plot Styling & Save --------------------------
ax.set_title('Normalized CER Histogram (Line Style) for Ocean Regions (COT ≥ 3)', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('CER (cer_mod08)', fontsize=14)
ax.set_ylabel('Probability Density (Normalized)', fontsize=14)  # Update Y-label for normalization
ax.grid(True, alpha=0.3, linestyle='-', color='gray')
ax.tick_params(axis='both', labelsize=12)
ax.legend(loc='best', fontsize=10, ncol=2, frameon=True, shadow=True)

# Save high-resolution figure
save_path = os.path.join(OUT_DIR, 'normalized_cer_histogram_line_singleplot_cot3plus.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

# -------------------------- Step 4: Print Detailed Statistics --------------------------
print(f"\nNormalized CER histogram saved to: {save_path}")
print("="*85)
print("Detailed Normalized CER Statistics (COT ≥ 3, Probability Density):")
print("="*85)
for ocean, cer_vals in ocean_data_dict.items():
    dens = np.histogram(cer_vals, bins=bins)[0] / (len(cer_vals) * bin_width)
    print(f"{ocean}:")
    print(f"  Valid Data: {len(cer_vals)}, Density Sum: {np.sum(dens):.4f} (≈1 = normalized)")
    print(f"  CER Mean: {np.mean(cer_vals):.4f}, Std: {np.std(cer_vals):.4f}")
    print(f"  CER Min: {np.min(cer_vals):.4f}, Max: {np.max(cer_vals):.4f}")
    print("-"*60)
print(f"\nTotal valid ocean regions plotted: {len(ocean_data_dict)}")