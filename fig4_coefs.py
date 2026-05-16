import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_fitting import oceans, format_panel_tag

# =========================
# Paths and basic settings
# =========================
BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
COEF_CSV = os.path.join(BASE_PATH, 'processed_data', 'sensitivity_albedo_vs_cot_ratio.csv')
FIG_SAVE_PATH = os.path.join(BASE_PATH, 'figs', 'fig4_coefs.png')
os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)

OCEANS = oceans  # 8 ocean regions
SEASONS = ['MAM', 'JJA', 'SON', 'DJF']

HEATMAP_CMAP = plt.cm.GnBu
B_CMAP = plt.cm.pink_r
UNC_CMAP = plt.cm.Reds

SIZE_PARAMS = {
    'small_tick': 9.5,
    'title': 16,
    'xylabel': 14,
    'cbar_tick': 9.5,
    'cbar_label': 12,
    'legend': 13,
    'text_label': 22,
}


def load_coef_data():
    """Load the merged coefficient CSV."""
    df = pd.read_csv(COEF_CSV)
    return df


def get_data_matrix(df, method, var_type, value_name):
    """
    Extract a (8 oceans x 4 seasons) matrix for a given variable.
    """
    mdf = df[(df['Method'] == method)].copy()
    matrix = np.full((len(OCEANS), len(SEASONS)), np.nan)
    for i, ocean in enumerate(OCEANS):
        for j, season in enumerate(SEASONS):
            mask = (mdf['Ocean'] == ocean) & (mdf['Season'] == season)
            if mask.any():
                matrix[i, j] = mdf.loc[mask, value_name].values[0]
    return matrix


def plot_single_heatmap(ax, data, x_labels, y_labels, cmap, vmin=None, vmax=None, text_label=''):
    """
    Plot a single heatmap on the given axes.
    """
    masked_data = np.ma.masked_invalid(data)
    im = ax.imshow(masked_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=SIZE_PARAMS['small_tick'])
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=SIZE_PARAMS['small_tick'])
    
    # Remove tick marks (keep labels)
    ax.tick_params(length=0)
    
    # Add text label centered on the heatmap
    if text_label:
        n_y, n_x = data.shape
        ax.text(n_x / 2 - 0.5, n_y / 2 - 0.5, text_label,
                fontsize=SIZE_PARAMS['text_label'], fontweight='bold',
                color='k', ha='center', va='center', fontname='DejaVu Sans')
    
    return im


def main():
    df = load_coef_data()
    
    # Extract data matrices for coefficients
    ret_k_1030 = get_data_matrix(df, 'ret', '1030', 'Slope_1030')
    ret_lnb_1030 = get_data_matrix(df, 'ret', '1030', 'Intercept_1030')
    ret_b_1030 = np.exp(ret_lnb_1030)
    
    msk_k_1030 = get_data_matrix(df, 'msk', '1030', 'Slope_1030')
    msk_lnb_1030 = get_data_matrix(df, 'msk', '1030', 'Intercept_1030')
    msk_b_1030 = np.exp(msk_lnb_1030)
    
    ret_k_day = get_data_matrix(df, 'ret', 'day', 'Slope_Daytime')
    ret_lnb_day = get_data_matrix(df, 'ret', 'day', 'Intercept_Daytime')
    ret_b_day = np.exp(ret_lnb_day)
    
    msk_k_day = get_data_matrix(df, 'msk', 'day', 'Slope_Daytime')
    msk_lnb_day = get_data_matrix(df, 'msk', 'day', 'Intercept_Daytime')
    msk_b_day = np.exp(msk_lnb_day)
    
    # Extract uncertainty matrices
    ret_k_1030_unc = get_data_matrix(df, 'ret', '1030', 'Slope_1030_Unc')
    ret_lnb_1030_unc = get_data_matrix(df, 'ret', '1030', 'Intercept_1030_Unc')
    ret_b_1030_unc = ret_b_1030 * ret_lnb_1030_unc  # propagate lnb unc to b
    
    msk_k_1030_unc = get_data_matrix(df, 'msk', '1030', 'Slope_1030_Unc')
    msk_lnb_1030_unc = get_data_matrix(df, 'msk', '1030', 'Intercept_1030_Unc')
    msk_b_1030_unc = msk_b_1030 * msk_lnb_1030_unc
    
    ret_k_day_unc = get_data_matrix(df, 'ret', 'day', 'Slope_Daytime_Unc')
    ret_lnb_day_unc = get_data_matrix(df, 'ret', 'day', 'Intercept_Daytime_Unc')
    ret_b_day_unc = ret_b_day * ret_lnb_day_unc
    
    msk_k_day_unc = get_data_matrix(df, 'msk', 'day', 'Slope_Daytime_Unc')
    msk_lnb_day_unc = get_data_matrix(df, 'msk', 'day', 'Intercept_Daytime_Unc')
    msk_b_day_unc = msk_b_day * msk_lnb_day_unc
    
    # Determine global vmin/vmax for shared colorbars
    # k values
    all_k = np.concatenate([ret_k_1030.ravel(), msk_k_1030.ravel(),
                            ret_k_day.ravel(), msk_k_day.ravel()])
    all_k = all_k[np.isfinite(all_k)]
    k_vmin = np.floor(np.min(all_k) * 10) / 10 if len(all_k) > 0 else 0
    k_vmax = np.ceil(np.max(all_k) * 10) / 10 if len(all_k) > 0 else 1
    
    # b values
    all_b = np.concatenate([ret_b_1030.ravel(), ret_b_day.ravel(),
                            msk_b_1030.ravel(), msk_b_day.ravel()])
    all_b = all_b[np.isfinite(all_b)]
    b_vmin = np.floor(np.min(all_b) * 10) / 10 if len(all_b) > 0 else 0
    b_vmax = np.ceil(np.max(all_b) * 10) / 10 if len(all_b) > 0 else 1
    
    # k uncertainty
    all_k_unc = np.concatenate([ret_k_1030_unc.ravel(), msk_k_1030_unc.ravel(),
                                ret_k_day_unc.ravel(), msk_k_day_unc.ravel()])
    all_k_unc = all_k_unc[np.isfinite(all_k_unc)]
    k_unc_vmax = np.ceil(np.max(all_k_unc) * 20) / 20 if len(all_k_unc) > 0 else 0.1
    
    # b uncertainty
    all_b_unc = np.concatenate([ret_b_1030_unc.ravel(), ret_b_day_unc.ravel(),
                                msk_b_1030_unc.ravel(), msk_b_day_unc.ravel()])
    all_b_unc = all_b_unc[np.isfinite(all_b_unc)]
    b_unc_vmax = np.ceil(np.max(all_b_unc) * 20) / 20 if len(all_b_unc) > 0 else 0.1
    
    # =========================
    # Create figure layout
    # =========================
    # Two rows of 2x4 heatmaps:
    #   Row 0 (subplot a): coefficients (k_ret_1030, k_ret_day, k_msk_1030, k_msk_day,
    #                                    b_ret_1030, b_ret_day, b_msk_1030, b_msk_day)
    #   Row 1 (subplot b): uncertainties (same layout)
    
    fig = plt.figure(figsize=(14, 11), dpi=100)
    
    left_margin = 0.06
    right_margin = 0.02
    top_margin = 0.04
    bottom_margin = 0.06
    
    n_rows, n_cols = 2, 4
    heatmap_total_height = 0.42
    gap = 0.04
    
    # Colorbar settings
    cbar_width = 0.012
    
    # Individual heatmap dimensions
    hm_w = (1 - left_margin - right_margin - cbar_width - 0.01) / n_cols
    hm_h = (heatmap_total_height - gap) / 2
    
    # ---- Subplot (a): Coefficient heatmaps ----
    k_data_list = [
        (ret_k_1030, '$k_{\\mathrm{ret,1030}}$'),
        (ret_k_day,   '$k_{\\mathrm{ret,day}}$'),
        (msk_k_1030,  '$k_{\\mathrm{msk,1030}}$'),
        (msk_k_day,   '$k_{\\mathrm{msk,day}}$'),
    ]
    b_data_list = [
        (ret_b_1030, '$b_{\\mathrm{ret,1030}}$'),
        (ret_b_day,   '$b_{\\mathrm{ret,day}}$'),
        (msk_b_1030,  '$b_{\\mathrm{msk,1030}}$'),
        (msk_b_day,   '$b_{\\mathrm{msk,day}}$'),
    ]
    
    # Row 0 of subplot (a): k heatmaps
    im_k_list = []
    for col, (data, label) in enumerate(k_data_list):
        left = left_margin + col * hm_w
        bottom = bottom_margin + heatmap_total_height + gap + hm_h
        ax = fig.add_axes([left, bottom, hm_w, hm_h])
        im = plot_single_heatmap(ax, data, SEASONS, OCEANS,
                                 HEATMAP_CMAP, vmin=k_vmin, vmax=k_vmax,
                                 text_label=label)
        im_k_list.append(im)
        if col > 0:
            ax.set_yticklabels([])
    
    # Row 1 of subplot (a): b heatmaps
    im_b_list = []
    for col, (data, label) in enumerate(b_data_list):
        left = left_margin + col * hm_w
        bottom = bottom_margin + heatmap_total_height + gap
        ax = fig.add_axes([left, bottom, hm_w, hm_h])
        im = plot_single_heatmap(ax, data, SEASONS, OCEANS,
                                 B_CMAP, vmin=b_vmin, vmax=b_vmax,
                                 text_label=label)
        im_b_list.append(im)
        if col > 0:
            ax.set_yticklabels([])
    
    # Title for subplot (a)
    fig.text(left_margin, bottom_margin + heatmap_total_height + gap + hm_h * 2 + 0.01,
             f'{format_panel_tag(0, "nature")} Correction Coefficients',
             fontsize=SIZE_PARAMS['title'], ha='left', va='bottom')
    
    # Colorbars for subplot (a)
    cbar_right_x = 1 - right_margin - cbar_width
    
    cax_k = fig.add_axes([cbar_right_x, bottom_margin + heatmap_total_height + gap + hm_h + 0.01,
                          cbar_width, hm_h - 0.02])
    cbar_k = fig.colorbar(im_k_list[0], cax=cax_k, orientation='vertical')
    cbar_k.set_label('$k$', fontsize=SIZE_PARAMS['cbar_label'])
    cbar_k.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])
    
    cax_b = fig.add_axes([cbar_right_x, bottom_margin + heatmap_total_height + gap + 0.01,
                          cbar_width, hm_h - 0.02])
    cbar_b = fig.colorbar(im_b_list[0], cax=cax_b, orientation='vertical')
    cbar_b.set_label('$b$', fontsize=SIZE_PARAMS['cbar_label'])
    cbar_b.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])
    
    # ---- Subplot (b): Uncertainty heatmaps ----
    k_unc_data_list = [
        (ret_k_1030_unc, '$\\sigma(k_{\\mathrm{ret,1030}})$'),
        (ret_k_day_unc,   '$\\sigma(k_{\\mathrm{ret,day}})$'),
        (msk_k_1030_unc,  '$\\sigma(k_{\\mathrm{msk,1030}})$'),
        (msk_k_day_unc,   '$\\sigma(k_{\\mathrm{msk,day}})$'),
    ]
    b_unc_data_list = [
        (ret_b_1030_unc, '$\\sigma(b_{\\mathrm{ret,1030}})$'),
        (ret_b_day_unc,   '$\\sigma(b_{\\mathrm{ret,day}})$'),
        (msk_b_1030_unc,  '$\\sigma(b_{\\mathrm{msk,1030}})$'),
        (msk_b_day_unc,   '$\\sigma(b_{\\mathrm{msk,day}})$'),
    ]
    
    # Row 0 of subplot (b): k uncertainty heatmaps
    im_k_unc_list = []
    for col, (data, label) in enumerate(k_unc_data_list):
        left = left_margin + col * hm_w
        bottom = bottom_margin + hm_h
        ax = fig.add_axes([left, bottom, hm_w, hm_h])
        im = plot_single_heatmap(ax, data, SEASONS, OCEANS,
                                 UNC_CMAP, vmin=0, vmax=k_unc_vmax,
                                 text_label=label)
        im_k_unc_list.append(im)
        if col > 0:
            ax.set_yticklabels([])
    
    # Row 1 of subplot (b): b uncertainty heatmaps
    im_b_unc_list = []
    for col, (data, label) in enumerate(b_unc_data_list):
        left = left_margin + col * hm_w
        bottom = bottom_margin
        ax = fig.add_axes([left, bottom, hm_w, hm_h])
        im = plot_single_heatmap(ax, data, SEASONS, OCEANS,
                                 UNC_CMAP, vmin=0, vmax=b_unc_vmax,
                                 text_label=label)
        im_b_unc_list.append(im)
        if col > 0:
            ax.set_yticklabels([])
    
    # Title for subplot (b)
    fig.text(left_margin, bottom_margin + hm_h * 2 + 0.01,
             f'{format_panel_tag(1, "nature")} Coefficient Uncertainties',
             fontsize=SIZE_PARAMS['title'], ha='left', va='bottom')
    
    # Colorbars for subplot (b)
    cax_k_unc = fig.add_axes([cbar_right_x, bottom_margin + hm_h + 0.01,
                              cbar_width, hm_h - 0.02])
    cbar_k_unc = fig.colorbar(im_k_unc_list[0], cax=cax_k_unc, orientation='vertical')
    cbar_k_unc.set_label('$\\sigma(k)$', fontsize=SIZE_PARAMS['cbar_label'])
    cbar_k_unc.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])
    
    cax_b_unc = fig.add_axes([cbar_right_x, bottom_margin + 0.01,
                              cbar_width, hm_h - 0.02])
    cbar_b_unc = fig.colorbar(im_b_unc_list[0], cax=cax_b_unc, orientation='vertical')
    cbar_b_unc.set_label('$\\sigma(b)$', fontsize=SIZE_PARAMS['cbar_label'])
    cbar_b_unc.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])
    
    # =========================
    # Save figure
    # =========================
    plt.savefig(FIG_SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {FIG_SAVE_PATH}")
    plt.close(fig)


if __name__ == '__main__':
    main()
