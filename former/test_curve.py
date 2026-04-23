import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.stats import linregress
import os

# Configuration parameters
cot_range = np.exp(np.linspace(np.log(3), 3, 15))  # COT value range for calculation
sza_list = [55, 35, 15]                            # Solar Zenith Angles (degrees)
season_dict = {'MAM': [3,4,5], 'JJA': [6,7,8], 'SON': [9,10,11], 'DJF': [12,1,2]}

# Plot style config: SZA->color (55:red,35:purple,15:orange); Method->linestyle (sbdart:solid, quadrature:dashed)
colors = ["#50A2E9", '#FF8C00', '#9370DB']  # '#E63946'Red
plot_style = {}
for idx, sza in enumerate(sza_list):
    plot_style[f'analytical_{sza}'] = {'color': colors[idx], 'linestyle': '--', 'lw': 2.5, 'label': f'Quadrature, SZA={sza}°'}
    plot_style[f'sbdart_{sza}'] = {'color': colors[idx], 'linestyle': '-', 'lw': 2.5, 'label': f'SBDART-dcp, SZA={sza}°'}

# Coordinate transformation functions
def cot_to_x(cot):
    """Convert COT to logarithmic value for x-axis"""
    return np.log(cot)

def albedo_to_y(albedo):
    """Convert albedo to log-odds value for y-axis (clip to avoid inf)"""
    albedo = np.clip(albedo, 1e-6, 1 - 1e-6)
    return np.log(albedo / (1 - albedo))

# Core function: Convert COT to albedo with specified method
def cot_to_albedo(cot, method, sza=None, season='DJF', ocean='NAO'):
    if method == 'sbdart':
        albedo = np.full(cot.shape, np.nan)
        file_path = f'/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table/cot_sza_to_albedo_lookup_table_dcp/cot_sza_to_albedo_lookup_table_{ocean}_{season}.csv'
        df = pd.read_csv(file_path, index_col=0)
        
        sz_grid = np.array(df.index, dtype=float)
        tval_grid = np.array(df.columns, dtype=float)
        albedo_grid = df.values
        sz_mesh, tval_mesh = np.meshgrid(sz_grid, tval_grid, indexing='ij')
        points = np.column_stack([sz_mesh.ravel(), tval_mesh.ravel()])
        values = albedo_grid.ravel()
        valid_mask = ~np.isnan(values)
        
        sza_arr = np.full_like(cot, sza, dtype=float)
        target_points = np.column_stack([sza_arr, np.atleast_1d(cot)])
        albedo = griddata(points[valid_mask], values[valid_mask], target_points, method='linear', fill_value=np.nan)
        return albedo
    
    elif method == 'l74':
        cot = np.asarray(cot)
        g = 0.85
        b = np.sqrt(3)/2*(1-g)
        return b * cot / (1 + b * cot)
    
    elif method == 'quadrature':
        cot = np.asarray(cot)
        g = 0.85
        b = np.sqrt(3)/2*(1-g)
        miu = np.cos(np.radians(sza))
        return (b * cot + (1/2 - np.sqrt(3)/2*miu) * (1 - np.exp(-cot/miu))) / (1 + b * cot)
    
    elif method == 'm80':
        cot = np.asarray(cot)
        g = 0.85
        miu = np.cos(np.radians(sza))
        return ((1-g) * cot + (2/3 - miu) * (1 - np.exp(-cot/miu))) / (4/3 + (1-g) * cot)

    else:
        print("Supported methods: ['sbdart', 'l74', 'quadrature', 'm80']")
        return np.nan

# Calculate albedo for all method-SZA combinations
def calculate_albedo_for_plot():
    albedo_results = {}
    for sza in sza_list:
        albedo_results[f'analytical_{sza}'] = cot_to_albedo(cot_range, method='quadrature', sza=sza)
        albedo_results[f'sbdart_{sza}'] = cot_to_albedo(cot_range, method='sbdart', sza=sza)
    return albedo_results

# Plot 1x2 subplots: (a) raw curves (COT vs Ac) | (b) fitted lines (lnCOT vs log-odds Ac)
def plot_double_subplots(albedo_results):
    save_path = 'figs/albedo_vs_cot_3sza_1×2_fitted.png'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.7), dpi=150)
    x_trans = cot_to_x(cot_range)  # Transformed x for subplot (b)
    fit_artists = []               # Store legend artists for custom order in subplot (b)

    # -------------------------- Subplot (a): Raw Coordinates (COT vs Ac) --------------------------
    x_raw = cot_range
    for key in plot_style.keys():
        ax1.plot(x_raw, albedo_results[key], **plot_style[key])
    # Format subplot (a)
    ax1.set_xlabel('COT', fontsize=14, fontweight='medium')
    ax1.set_ylabel(r'$A_{\mathrm{c}}$', fontsize=14, fontweight='medium')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3, color='gray', lw=1)
    ax1.set_xlim(x_raw.min() - 0.1, x_raw.max() + 0.1)
    ax1.set_ylim(bottom=0)
    ax1.set_title(r'$\mathbf{(a)}$', fontsize=16, pad=10, loc='left')
    # Custom legend order for (a): SBDART first, then Quadrature
    sbdart_labels = [f'SBDART-dcp, SZA={sza}°' for sza in sza_list]
    quad_labels = [f'Quadrature, SZA={sza}°' for sza in sza_list]
    custom_order_a = sbdart_labels + quad_labels
    handles_a, labels_a = ax1.get_legend_handles_labels()
    ordered_handles_a = [handles_a[labels_a.index(lab)] for lab in custom_order_a]
    ax1.legend(ordered_handles_a, custom_order_a, loc='lower right', fontsize=9, framealpha=0.9)

    # -------------------------- Subplot (b): Transformed Coordinates (lnCOT vs log-odds Ac) --------------------------
    # Pre-define order: SBDART (55/35/15) -> Quadrature (55/35/15)
    plot_order = [f'sbdart_{sza}' for sza in sza_list] + [f'analytical_{sza}' for sza in sza_list]
    
    for key in plot_order:
        # Get transformed y values and filter NaN
        y_vals = albedo_to_y(albedo_results[key])
        valid_idx = ~np.isnan(y_vals)
        x_fit = x_trans[valid_idx]
        y_fit = y_vals[valid_idx]
        
        # Linear regression fit
        fit_result = linregress(x_fit, y_fit)
        slope, intercept = fit_result.slope, fit_result.intercept

        # Generate formatted fit equation: y=slope.2fx{sign}intercept.1f
        slope_ann = round(slope, 2)
        intercept_ann = round(intercept, 1)
        sign = '+' if intercept_ann >= 0 else ''
        fit_eq = f'y={slope_ann:.2f}x{sign}{intercept_ann:.1f}'

        # Plot fitted line with empty label (no auto-legend)
        style_fit = plot_style[key].copy()
        style_fit['label'] = ''
        y_fitted = slope * x_trans + intercept
        ax2.plot(x_trans, y_fitted, **style_fit)

        # Create empty plot for legend and store for custom order
        style_legend = plot_style[key].copy()
        style_legend['label'] = fit_eq
        artist = ax2.plot([], [], **style_legend)[0]
        fit_artists.append( (artist, fit_eq) )

    # Format subplot (b)
    ax2.set_xlabel(r'$\ln(\text{COT})$', fontsize=14, fontweight='medium')
    ax2.set_ylabel(r'$\ln\left[A_{\mathrm{c}}/(1-A_{\mathrm{c}})\right]$', fontsize=14, fontweight='medium')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3, color='gray', lw=1)
    ax2.set_xlim(x_trans.min() - 0.1, x_trans.max() + 0.1)
    ax2.set_title(r'$\mathbf{(b)}$', fontsize=16, pad=10, loc='left')
    # Create custom legend for (b) with SBDART first
    handles_b = [art for art, eq in fit_artists]
    labels_b = [eq for art, eq in fit_artists]
    ax2.legend(handles_b, labels_b, loc='lower right', fontsize=8, framealpha=0.9)

    # Final layout and save
    plt.tight_layout()
    os.makedirs('figs', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to: {os.path.abspath(save_path)}")

# Main execution
if __name__ == "__main__":
    albedo_data = calculate_albedo_for_plot()
    plot_double_subplots(albedo_data)