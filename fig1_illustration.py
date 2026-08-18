# -*- coding: utf-8 -*-
"""
Draw Fig. 1 illustration.

Left panel: blank placeholder.
Right panel: COT-albedo relationships, with legend below the right panel.
"""

import os
import matplotlib.pyplot as plt

from utils_fitting import format_panel_tag

from fig2_fittings_global_and_reasons import (
    prepare_global_5curves_data,
    draw_global_5curves_panel,
)


if __name__ == "__main__":
    icon_style = 'nature'
    if icon_style not in ('nature', 'science'):
        raise ValueError("icon_style must be 'nature' or 'science'.")

    fig = plt.figure(figsize=(6.7, 3.8))
    gs = fig.add_gridspec(
        1, 2,
        wspace=0.3,
        left=0.08,
        right=0.96,
        bottom=0.30,
        top=0.90,
    )

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # -------------------------
    # Blank panel
    # -------------------------
    ax_left.text(
        -0.03, 1.01,
        format_panel_tag(0, icon_style),
        transform=ax_left.transAxes,
        fontsize=11,
        va='bottom',
        ha='left',
    )
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_title('Represented Regions', fontsize=8.5, loc='center', pad=4.5)
    for spine in ax_left.spines.values():
        spine.set_visible(True)

    # -------------------------
    # Relationships panel
    # -------------------------
    panel_data = prepare_global_5curves_data(verbose=True, include_simulations=False)
    draw_global_5curves_panel(
        ax_right,
        panel_data,
        icon_style=icon_style,
        tag_index=1,
        tag_fontsize=11,
        axis_label_fontsize=9,
        tick_labelsize=7.7,
        legend_fontsize=7,
        legend_anchor=(0.5, -0.41),
        include_simulations=False,
    )
    ax_right.set_title('Relationships', fontsize=8.5, loc='center', pad=4.5)

    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/fig1_illustration.png', dpi=300, bbox_inches='tight')
    plt.show()
