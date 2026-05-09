import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_fitting import oceans, format_panel_tag

# =========================
# Paths and basic settings
# =========================
BASE_PATH = '/home/chenyiqi/251028_albedo_cot'
COEF_CSV = os.path.join(BASE_PATH, 'processed_data', 'sensitivity_albedo_vs_cot.csv')
FIG_SAVE_PATH = os.path.join(BASE_PATH, 'figs', 'fig4_plt_corr_coef.png')
os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)

OCEANS = oceans  # 8 ocean regions
SEASONS = ['MAM', 'JJA', 'SON', 'DJF']

HEATMAP_CMAP = plt.cm.GnBu
B_CMAP = plt.cm.pink_r
M_CMAP = plt.cm.YlOrRd

SIZE_PARAMS = {
    'small_tick': 8,
    'title': 14,
    'cbar_tick': 10,
    'cbar_label': 12,
    'legend': 10,
    'text_label': 22,
}

# =========================
# Data loading
# =========================

def load_coef_data():
    """Load the merged coefficient CSV."""
    df = pd.read_csv(COEF_CSV)
    return df


def get_data_matrix(df, method, var_type, value_name):
    """
    Extract a (8 oceans x 4 seasons) matrix for a given variable.
    
    Parameters
    ----------
    df : pd.DataFrame
    method : str, 'ret' or 'msk'
    var_type : str, '1030' or 'Daytime'
    value_name : str, column name in the CSV (e.g., 'Slope_1030', 'Intercept_Daytime')
    
    Returns
    -------
    np.ndarray, shape (8, 4) — oceans x seasons
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
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : np.ndarray, shape (n_y, n_x)
    x_labels : list of str, labels for x-axis (seasons)
    y_labels : list of str, labels for y-axis (oceans)
    cmap : colormap
    vmin, vmax : float
    text_label : str, text to display centered on the heatmap
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
                color='k', ha='center', va='center', fontname='Verdana')
    
    return im


# =========================
# Main plotting
# =========================

def main():
    df = load_coef_data()
    
    # Extract data matrices
    # ret 1030: slope = k, intercept = lnb -> b = exp(lnb)
    ret_k_1030 = get_data_matrix(df, 'ret', '1030', 'Slope_1030')
    ret_lnb_1030 = get_data_matrix(df, 'ret', '1030', 'Intercept_1030')
    ret_b_1030 = np.exp(ret_lnb_1030)  # convert lnb to b
    
    # msk 1030
    msk_k_1030 = get_data_matrix(df, 'msk', '1030', 'Slope_1030')
    msk_m_1030 = np.full_like(msk_k_1030, 1.0)  # m=1 for all
    
    # ret Daytime
    ret_k_day = get_data_matrix(df, 'ret', 'Daytime', 'Slope_Daytime')
    ret_lnb_day = get_data_matrix(df, 'ret', 'Daytime', 'Intercept_Daytime')
    ret_b_day = np.exp(ret_lnb_day)  # convert lnb to b
    
    # msk Daytime
    msk_k_day = get_data_matrix(df, 'msk', 'Daytime', 'Slope_Daytime')
    msk_m_day = get_data_matrix(df, 'msk', 'Daytime', 'Albedo_Ratio_Daytime_o_1030')
    
    # Determine global vmin/vmax for shared colorbars
    # k values (ret and msk share similar range)
    all_k = np.concatenate([ret_k_1030.ravel(), msk_k_1030.ravel(),
                            ret_k_day.ravel(), msk_k_day.ravel()])
    all_k = all_k[np.isfinite(all_k)]
    k_vmin = np.floor(np.min(all_k) * 10) / 10 if len(all_k) > 0 else 0
    k_vmax = np.ceil(np.max(all_k) * 10) / 10 if len(all_k) > 0 else 1
    
    # b values (ret only, now in linear space)
    all_b = np.concatenate([ret_b_1030.ravel(), ret_b_day.ravel()])
    all_b = all_b[np.isfinite(all_b)]
    b_vmin = np.floor(np.min(all_b) * 10) / 10 if len(all_b) > 0 else 0
    b_vmax = np.ceil(np.max(all_b) * 10) / 10 if len(all_b) > 0 else 1
    
    # m values (msk only)
    all_m = np.concatenate([msk_m_1030.ravel(), msk_m_day.ravel()])
    all_m = all_m[np.isfinite(all_m)]
    m_vmin = np.floor(np.min(all_m) * 10) / 10 if len(all_m) > 0 else 0.8
    m_vmax = np.ceil(np.max(all_m) * 10) / 10 if len(all_m) > 0 else 1.4
    
    # =========================
    # Create figure layout
    # =========================
    # 2x2 grid of subplot groups, each group has 2 heatmaps (k + b/m)
    # Each heatmap has its own colorbar directly below it
    
    fig = plt.figure(figsize=(8.5, 6), dpi=100)
    
    # Layout parameters
    left_margin = 0.06
    right_margin = 0.02
    top_margin = 0.04
    bottom_margin = 0.08
    h_space = 0.12
    w_space = 0.14
    inner_w_space = 0.0  # no space between k and b/m within a subplot
    
    # Space reserved for colorbars below each heatmap in bottom row
    cbar_height = 0.02
    cbar_gap_from_heatmap = 0.05
    
    # Subplot group dimensions
    group_width = (1 - left_margin - right_margin - w_space) / 2
    # Bottom row has extra space for colorbars
    group_height = (1 - top_margin - bottom_margin - h_space - cbar_height - cbar_gap_from_heatmap) / 2
    
    # Within each group: 2 heatmaps side by side, no gap
    inner_width = (group_width - inner_w_space) / 2
    
    def get_group_rect(row, col):
        """Get the bounding box for a subplot group (row, col)."""
        left = left_margin + col * (group_width + w_space)
        if row == 0:
            # Top row
            bottom = bottom_margin + cbar_height + cbar_gap_from_heatmap + group_height + h_space
        else:
            # Bottom row: above colorbar area
            bottom = bottom_margin + cbar_height + cbar_gap_from_heatmap
        return left, bottom, group_width, group_height
    
    def get_inner_rect(group_left, group_bottom, is_left):
        """Get the bounding box for left or right heatmap within a group."""
        if is_left:
            return (group_left, group_bottom, inner_width, group_height)
        else:
            return (group_left + inner_width + inner_w_space, group_bottom, inner_width, group_height)
    
    def get_cbar_rect(group_left, is_left):
        """Get the bounding box for a colorbar below a heatmap in the bottom row."""
        if is_left:
            cbar_left = group_left + 0.005
        else:
            cbar_left = group_left + inner_width + inner_w_space + 0.005
        return (cbar_left, bottom_margin, inner_width-0.01, cbar_height)
    
    # =========================
    # Plot subplots
    # =========================
    
    # Subplot (a): ret 1030
    gl_a, gb_a, gw_a, gh_a = get_group_rect(0, 0)
    
    # Left: k_ret 1030
    ax_a_k = fig.add_axes(get_inner_rect(gl_a, gb_a, True))
    im_a_k = plot_single_heatmap(ax_a_k, ret_k_1030, SEASONS, OCEANS,
                                  HEATMAP_CMAP, vmin=k_vmin, vmax=k_vmax,
                                  text_label='$k_{\\mathrm{ret}}$')
    ax_a_k.set_title(
        f'{format_panel_tag(0, "nature")} Retrieval-Domain Coef., 10:30',
        fontsize=SIZE_PARAMS['title'], pad=10, loc='left', x=-0.40
    )
    
    # Right: b_ret 1030
    ax_a_b = fig.add_axes(get_inner_rect(gl_a, gb_a, False))
    im_a_b = plot_single_heatmap(ax_a_b, ret_b_1030, SEASONS, OCEANS,
                                 B_CMAP, vmin=b_vmin, vmax=b_vmax,
                                 text_label='$l$')
    ax_a_b.set_yticklabels([])
    
    # Subplot (b): msk 1030
    gl_b, gb_b, gw_b, gh_b = get_group_rect(0, 1)
    
    ax_b_k = fig.add_axes(get_inner_rect(gl_b, gb_b, True))
    im_b_k = plot_single_heatmap(ax_b_k, msk_k_1030, SEASONS, OCEANS,
                                 HEATMAP_CMAP, vmin=k_vmin, vmax=k_vmax,
                                 text_label='$k_{\\mathrm{msk}}$')
    ax_b_k.set_title(
        f'{format_panel_tag(1, "nature")} Mask-Domain Coef., 10:30',
        fontsize=SIZE_PARAMS['title'], pad=10, loc='left', x=-0.40
    )
    
    ax_b_m = fig.add_axes(get_inner_rect(gl_b, gb_b, False))
    im_b_m = plot_single_heatmap(ax_b_m, msk_m_1030, SEASONS, OCEANS,
                                 M_CMAP, vmin=m_vmin, vmax=m_vmax,
                                 text_label='$m$')
    ax_b_m.set_yticklabels([])
    
    # Subplot (c): ret Daytime
    gl_c, gb_c, gw_c, gh_c = get_group_rect(1, 0)
    
    ax_c_k = fig.add_axes(get_inner_rect(gl_c, gb_c, True))
    im_c_k = plot_single_heatmap(ax_c_k, ret_k_day, SEASONS, OCEANS,
                                 HEATMAP_CMAP, vmin=k_vmin, vmax=k_vmax,
                                 text_label='$k_{\\mathrm{ret}}$')
    ax_c_k.set_title(
        f'{format_panel_tag(2, "nature")} Retrieval-Domain Coef., Daytime',
        fontsize=SIZE_PARAMS['title'], pad=10, loc='left', x=-0.40
    )
    
    ax_c_b = fig.add_axes(get_inner_rect(gl_c, gb_c, False))
    im_c_b = plot_single_heatmap(ax_c_b, ret_b_day, SEASONS, OCEANS,
                                 B_CMAP, vmin=b_vmin, vmax=b_vmax,
                                 text_label='$l$')
    ax_c_b.set_yticklabels([])
    
    # Subplot (d): msk Daytime
    gl_d, gb_d, gw_d, gh_d = get_group_rect(1, 1)
    
    ax_d_k = fig.add_axes(get_inner_rect(gl_d, gb_d, True))
    im_d_k = plot_single_heatmap(ax_d_k, msk_k_day, SEASONS, OCEANS,
                                 HEATMAP_CMAP, vmin=k_vmin, vmax=k_vmax,
                                 text_label='$k_{\\mathrm{msk}}$')
    ax_d_k.set_title(
        f'{format_panel_tag(3, "nature")} Mask-Domain Coef., Daytime',
        fontsize=SIZE_PARAMS['title'], pad=10, loc='left', x=-0.40
    )
    
    ax_d_m = fig.add_axes(get_inner_rect(gl_d, gb_d, False))
    im_d_m = plot_single_heatmap(ax_d_m, msk_m_day, SEASONS, OCEANS,
                                 M_CMAP, vmin=m_vmin, vmax=m_vmax,
                                 text_label='$m$')
    ax_d_m.set_yticklabels([])
    
    # =========================
    # Colorbars below each heatmap in bottom row (c) and (d)
    # =========================
    # 4 colorbars: k_ret (below c-left), b (below c-right),
    #              k_msk (below d-left), m (below d-right)
    
    cbar_specs = [
        (im_a_k, gl_c, True, '$k$'),
        (im_a_b, gl_c, False, '$l$'),
        (im_b_k, gl_d, True, '$k$'),
        (im_b_m, gl_d, False, '$m$'),
    ]
    
    for im, group_left, is_left, label in cbar_specs:
        cax = fig.add_axes(get_cbar_rect(group_left, is_left))
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label(label, fontsize=SIZE_PARAMS['cbar_label'])
        cbar.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])
    
    fig.savefig(FIG_SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {FIG_SAVE_PATH}")


if __name__ == "__main__":
    main()
