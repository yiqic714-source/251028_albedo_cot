import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Paths and basic settings
# =========================
KLNB_OCEAN_FILE = '/home/chenyiqi/251028_albedo_cot/processed_data/k_lnb_by_seasons_oceans.csv'
OUTPUT_PNG = 'figs/k_lnb_unc_3panels.png'

OCEANS = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']
SEASONS = ['MAM', 'JJA', 'SON', 'DJF']

HEATMAP_CMAP = plt.cm.GnBu
LNB_CMAP = plt.cm.pink_r

SIZE_PARAMS = {
    'tick': 12,
    'title': 17,
    'cbar_tick': 10.5,
    'cbar_label': 14,
}


def format_panel_tag(panel_idx, icon_style):
    if icon_style == 'science':
        letter = chr(ord('A') + panel_idx)
        return rf'$\mathbf{{{letter}}}$'

    letter = chr(ord('a') + panel_idx)
    return rf'$\mathbf{{({letter})}}$'

os.makedirs('figs', exist_ok=True)


def load_uncor_from_klnb(file_path, var_type='SlopeUnc', method='ret'):
    """
    Load seasonal uncertainty values by ocean basins and seasons.
    """
    df = pd.read_csv(file_path)

    if 'Ocean' not in df.columns:
        raise ValueError("Input CSV must contain 'Ocean' column.")

    df = df[df['Ocean'] != 'Global'].copy()
    df['Ocean'] = pd.Categorical(df['Ocean'], categories=OCEANS, ordered=True)
    df = df.sort_values('Ocean')

    out = pd.DataFrame()
    out['ocean'] = df['Ocean'].astype(str)

    for season in SEASONS:
        col = f'{season}_{var_type}_{method}'
        out[season] = pd.to_numeric(df[col], errors='coerce') if col in df.columns else np.nan

    return out


def plot_heatmap(ax, df, title, cmap):
    """Plot heatmap: oceans (y), seasons (x)"""
    heatmap_data = df[SEASONS].values.astype(float)
    oceans = df['ocean'].tolist()
    heatmap_data = np.where(np.isinf(heatmap_data), np.nan, heatmap_data)

    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto')

    ax.set_xticks(np.arange(len(SEASONS)))
    ax.set_yticks(np.arange(len(oceans)))
    ax.set_xticklabels(SEASONS, fontsize=SIZE_PARAMS['tick'])
    ax.set_yticklabels(oceans, fontsize=SIZE_PARAMS['tick'])

    for i in range(len(oceans)):
        for j in range(len(SEASONS)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='k', fontsize=10, fontweight='bold')

    ax.set_title(title, fontsize=SIZE_PARAMS['title'], loc='left', pad=8)
    return im

# =========================
# Print global mean only
# =========================
def print_global_means(file_path):
    methods = ['ret', 'msk', 'cp', 'dcp']
    var_types = ['SlopeUnc', 'InterceptUnc']
    
    print("==== Uncertainty Global Means ====")
    for var in var_types:
        for m in methods:
            try:
                df = load_uncor_from_klnb(file_path, var_type=var, method=m)
                mean_val = np.nanmean(df[SEASONS].values)
                print(f"{var:12s} {m:4s}: {mean_val:.4f}")
            except:
                print(f"{var:12s} {m:4s}: NaN")
    print("===================================\n")


def create_uncertainty_plot(icon_style='nature'):
    if icon_style not in ('nature', 'science'):
        raise ValueError("icon_style must be 'nature' or 'science'.")

    df_slopeunc_ret = load_uncor_from_klnb(KLNB_OCEAN_FILE, var_type='SlopeUnc', method='ret')
    df_interceptunc_ret = load_uncor_from_klnb(KLNB_OCEAN_FILE, var_type='InterceptUnc', method='ret')
    df_slopeunc_msk = load_uncor_from_klnb(KLNB_OCEAN_FILE, var_type='SlopeUnc', method='msk')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    im1 = plot_heatmap(axes[0], df_slopeunc_ret,
                       f'{format_panel_tag(0, icon_style)}   $k_{{\mathrm{{ret}}}}$ uncertainty', cmap=HEATMAP_CMAP)
    im1.set_clim(0.0, 0.3)

    im2 = plot_heatmap(axes[1], df_interceptunc_ret,
                       f'{format_panel_tag(1, icon_style)}   ln$b_{{\mathrm{{ret}}}}$ uncertainty', cmap=LNB_CMAP)
    im2.set_clim(0.0, 0.45)

    im3 = plot_heatmap(axes[2], df_slopeunc_msk,
                       f'{format_panel_tag(2, icon_style)}   $k_{{\mathrm{{msk}}}}$ uncertainty', cmap=HEATMAP_CMAP)
    im3.set_clim(0.0, 0.3)

    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.1, pad=0.04)
    cbar1.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])

    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.1, pad=0.04)
    cbar2.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])

    cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.1, pad=0.04)
    cbar3.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved to: {OUTPUT_PNG}')


if __name__ == '__main__':
    # Choose panel tag style here: 'nature' -> (a)(b)(c), 'science' -> A B C.
    icon_style = 'science'

    print_global_means(KLNB_OCEAN_FILE)
    create_uncertainty_plot(icon_style=icon_style)