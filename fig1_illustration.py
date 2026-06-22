# -*- coding: utf-8 -*-
"""
Draw original panels (a) and (b) only.

Panel (a): Domains of variables
Panel (b): Data to build relationships
"""

import os
import matplotlib.pyplot as plt
from utils_fitting import format_panel_tag


if __name__ == "__main__":
    icon_style = 'nature'
    if icon_style not in ('nature', 'science'):
        raise ValueError("icon_style must be 'nature' or 'science'.")

    fig = plt.figure(figsize=(7, 3.8))
    gs = fig.add_gridspec(
        1, 2,
        wspace=0.25,
        left=0.08, right=0.96,
        bottom=0.14, top=0.86
    )

    ax1 = fig.add_subplot(gs[0, 0])  # (a) Domains of variables
    ax2 = fig.add_subplot(gs[0, 1])  # (b) Data used to build relationships

    # -------------------------
    # (a) Domains of variables
    # -------------------------
    ax1.text(
        -0.01, 1.01,
        format_panel_tag(0, icon_style),
        transform=ax1.transAxes,
        fontsize=13.5,
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
        -0.01, 1.01,
        format_panel_tag(1, icon_style),
        transform=ax2.transAxes,
        fontsize=13.5,
        va='bottom',
        ha='left'
    )
    ax2.set_title('Data to build relationships', fontsize=8.5, loc='center', pad=4.5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(True)

    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/fig1_illustration.png', dpi=300, bbox_inches='tight')
    plt.show()
