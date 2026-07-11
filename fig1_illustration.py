# -*- coding: utf-8 -*-
"""
Draw original panels (a) and (b), plus the former Fig. 2a as panel (c).

Panel (a): Domains of variables
Panel (b): Data to build relationships
Panel (c): COT-albedo relationships
"""

import os
import matplotlib.pyplot as plt

from utils_fitting import format_panel_tag
from fig2_fittings_global_and_reasons import prepare_global_5curves_data, draw_global_5curves_panel


if __name__ == "__main__":
    icon_style = 'nature'
    if icon_style not in ('nature', 'science'):
        raise ValueError("icon_style must be 'nature' or 'science'.")

    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(
        2, 2,
        hspace=0.24, wspace=0.26,
        left=0.08, right=0.78,
        bottom=0.08, top=0.92
    )

    ax1 = fig.add_subplot(gs[0, 0])  # (a) Domains of variables
    ax2 = fig.add_subplot(gs[0, 1])  # (b) Data used to build relationships
    ax3 = fig.add_axes([0.08, 0.08, 0.41, 0.36])  # (c) Former Fig. 2a, wider than one column

    # -------------------------
    # (a) Domains of variables
    # -------------------------
    ax1.text(
        -0.03, 1.01,
        format_panel_tag(0, icon_style),
        transform=ax1.transAxes,
        fontsize=11,
        va='bottom',
        ha='left'
    )
    ax1.set_title('Domains of variables', fontsize=8.5, loc='center', pad=4.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(True)

    # -------------------------
    # (b) Data to build relationships
    # -------------------------
    ax2.text(
        -0.03, 1.01,
        format_panel_tag(1, icon_style),
        transform=ax2.transAxes,
        fontsize=11,
        va='bottom',
        ha='left'
    )
    ax2.set_title('Data to build relationships', fontsize=8.5, loc='center', pad=4.5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(True)

    # -------------------------
    # (c) Former Fig. 2a
    # -------------------------
    panel_c_data = prepare_global_5curves_data(verbose=True, include_simulations=False)
    draw_global_5curves_panel(
        ax3,
        panel_c_data,
        icon_style=icon_style,
        tag_index=2,
        tag_fontsize=11,
        axis_label_fontsize=9,
        tick_labelsize=8,
        legend_fontsize=7.2,
        legend_anchor=(1.06, 0.5),
        include_simulations=False,
    )
    ax3.set_title('Relationships', fontsize=8.5, loc='center', pad=4.5)

    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/fig1_illustration.png', dpi=300, bbox_inches='tight')
    plt.show()
