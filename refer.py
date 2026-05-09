def plot_heatmap(ax, df, title, cmap=HEATMAP_CMAP, vmin=None, vmax=None):
    """
    Plot heatmap with oceans (x-axis), seasons (y-axis).
    """
    for col in SEASONS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    heatmap_data = df[SEASONS].values.astype(np.float64)
    oceans = df['ocean'].tolist()

    heatmap_data = np.where(np.isinf(heatmap_data), np.nan, heatmap_data)
    heatmap_data_t = heatmap_data.T

    im = ax.imshow(heatmap_data_t, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(oceans)))
    ax.set_yticks(np.arange(len(SEASONS)))
    ax.set_xticklabels(oceans, fontsize=SIZE_PARAMS['large_tick'], rotation=90, ha='right')
    ax.set_yticklabels(SEASONS, fontsize=SIZE_PARAMS['large_tick'])

    for i in range(len(SEASONS)):
        for j in range(len(oceans)):
            val = heatmap_data_t[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f'{val:.2f}',
                    ha="center", va="center",
                    color='k', fontsize=9.5, fontweight='bold'
                )

    ax.set_title(title, fontsize=SIZE_PARAMS['title'], pad=7, loc='left')
    return im


def plot_main_ax(
    ax,
    seasonal_stats,
    weight_dict,
    is_k_plot=True,
    cmap=HEATMAP_CMAP,
    panel_idx=None,
    icon_style='nature',
    title_y=None,
):
    """
    Plot main axis with SZA range 20-75° and legend at top-left.
    """
    all_sza = []
    for ocean in OCEANS:
        for season in SEASONS:
            cos_sza, _, _ = get_lookup_data(ocean, season)
            if cos_sza is not None:
                all_sza.extend(np.degrees(np.arccos(cos_sza)))

    unique_sza = np.sort(np.unique(all_sza)) if all_sza else np.array([])
    n_y = len(unique_sza)
    n_ocean = len(OCEANS)
    n_season = len(SEASONS)
    n_x = n_ocean * n_season

    main_data = np.full((n_y, n_x), np.nan)
    x_ticks = []
    ocean_label_pos = []
    ocean_labels = []

    mean_sza_x, mean_sza_y = [], []
    weighted_sza_x, weighted_sza_y = [], []

    for o_idx, ocean in enumerate(OCEANS):
        x_start = o_idx * n_season
        x_end = x_start + n_season
        ocean_label_pos.append((x_start + x_end - 1) / 2)
        ocean_labels.append(ocean)

        for s_idx, season in enumerate(SEASONS):
            x_pos = x_start + s_idx
            x_ticks.append(season)

            mean_sza_deg = seasonal_stats[(ocean, season)]
            weighted_sza_deg = weight_dict.get((ocean, season), np.nan)

            cos_sza, slope_vals, intercept_vals = get_lookup_data(ocean, season)

            if cos_sza is not None:
                sza_vals = np.degrees(np.arccos(cos_sza))
                for y_idx, target_sza in enumerate(unique_sza):
                    closest_idx = np.argmin(np.abs(sza_vals - target_sza))
                    if np.isclose(sza_vals[closest_idx], target_sza):
                        main_data[y_idx, x_pos] = (
                            slope_vals[closest_idx] if is_k_plot else intercept_vals[closest_idx]
                        )

            if not np.isnan(mean_sza_deg):
                mean_sza_x.append(x_pos)
                mean_sza_y.append(mean_sza_deg)

            if not np.isnan(weighted_sza_deg):
                weighted_sza_x.append(x_pos)
                weighted_sza_y.append(weighted_sza_deg)

    if not np.all(np.isnan(main_data)):
        vmin = K_VMIN if is_k_plot else LNB_VMIN
        vmax = K_VMAX if is_k_plot else LNB_VMAX

        ax.imshow(
            main_data,
            aspect='auto',
            cmap=cmap,
            extent=[-0.5, n_x - 0.5, 20, 75],
            vmin=vmin,
            vmax=vmax
        )

        mean_mask = (np.array(mean_sza_y) >= 20) & (np.array(mean_sza_y) <= 75)
        weighted_mask = (np.array(weighted_sza_y) >= 20) & (np.array(weighted_sza_y) <= 75)

        ax.scatter(
            np.array(mean_sza_x)[mean_mask],
            np.array(mean_sza_y)[mean_mask],
            color='red', s=50, marker='o',
            label='10:30', zorder=5, edgecolors='black'
        )
        ax.scatter(
            np.array(weighted_sza_x)[weighted_mask],
            np.array(weighted_sza_y)[weighted_mask],
            color='blue', s=60, marker='^',
            label='Daytime', zorder=5, edgecolors='black'
        )

        vline_positions = [n_season * (i + 1) - 0.5 for i in range(n_ocean - 1)]
        for vline_pos in vline_positions:
            ax.axvline(x=vline_pos, color='lightgray', linestyle='-', linewidth=1, zorder=4)

        ax.set_ylabel(r'SZA ($^\circ$)', fontsize=SIZE_PARAMS['large_tick'])
        ax.set_xlim(-0.5, n_x - 0.5)
        ax.set_ylim(20, 75)
        ax.set_yticks(np.arange(20, 76, 10))
        ax.set_yticklabels([f'{int(x)}' for x in np.arange(20, 76, 10)], fontsize=SIZE_PARAMS['large_tick'])

        ax.set_xticks(range(n_x))
        ax.set_xticklabels(x_ticks, fontsize=SIZE_PARAMS['small_tick'], va='top', rotation=90, ha='right')

        y_offset = 76
        for pos, label in zip(ocean_label_pos, ocean_labels):
            ax.text(pos, y_offset, label, ha='center', va='bottom', fontsize=SIZE_PARAMS['large_tick'])

        ax.legend(loc='upper center', fontsize=SIZE_PARAMS['legend'], frameon=True)

        if panel_idx is None:
            panel_idx = 3 if is_k_plot else 4

        panel_tag = format_panel_tag(panel_idx, icon_style)

        if is_k_plot:
            ax.set_title(
                f'{panel_tag}   $k_{{\mathrm{{cp}}}}$ vs. SZA',
                loc='left', fontsize=SIZE_PARAMS['title'], pad=1, y=title_y
            )
        else:
            ax.set_title(
                f'{panel_tag}   ln$b_{{\mathrm{{cp}}}}$ vs. SZA',
                loc='left', fontsize=SIZE_PARAMS['title'], pad=1, y=title_y
            )


def create_main_plot(icon_style='nature'):
    """
    Create combined plot with k_ret (row 1), lnb_ret (row 2), k_msk (row 3).
    """
    if icon_style not in ('nature', 'science'):
        raise ValueError("icon_style must be 'nature' or 'science'.")

    seasonal_stats = calculate_seasonal_stats(OCEANS, BASE_DATA_DIR)
    weight_dict = load_weighted_angles(WEIGHTED_FILE)

    diff_k_data = pd.DataFrame(index=OCEANS, columns=SEASONS, dtype=float)   # for k_ret
    diff_k1_data = pd.DataFrame(index=OCEANS, columns=SEASONS, dtype=float)  # for k_msk
    diff_b_data = pd.DataFrame(index=OCEANS, columns=SEASONS, dtype=float)   # for lnb_ret

    for ocean in OCEANS:
        for season in SEASONS:
            mean_sza_deg = seasonal_stats[(ocean, season)]
            weighted_sza_deg = weight_dict.get((ocean, season), np.nan)

            cos_sza, slope_vals, intercept_vals = get_lookup_data(ocean, season)

            if not np.isnan(mean_sza_deg) and not np.isnan(weighted_sza_deg):
                cos_mean_sza = np.cos(np.radians(mean_sza_deg))
                cos_weighted_sza = np.cos(np.radians(weighted_sza_deg))

                k_mean = get_value_at_sza(cos_sza, cos_mean_sza, slope_vals)
                k_weighted = get_value_at_sza(cos_sza, cos_weighted_sza, slope_vals)
                b_mean = get_value_at_sza(cos_sza, cos_mean_sza, intercept_vals)
                b_weighted = get_value_at_sza(cos_sza, cos_weighted_sza, intercept_vals)

                if np.isfinite(k_weighted) and np.isfinite(k_mean):
                    diff_k_data.loc[ocean, season] = k_weighted - k_mean
                    diff_k1_data.loc[ocean, season] = k_weighted - k_mean

                if np.isfinite(b_weighted) and np.isfinite(b_mean):
                    diff_b_data.loc[ocean, season] = b_weighted - b_mean

    # Directly load uncorrected values from k_lnb_by_seasons_oceans.csv
    uncor_k2_df = load_uncor_from_klnb(KLNB_OCEAN_FILE, var_type='Slope', method='ret').round(4)
    uncor_k1_df = load_uncor_from_klnb(KLNB_OCEAN_FILE, var_type='Slope', method='msk').round(4)
    uncor_lnb2_df = load_uncor_from_klnb(KLNB_OCEAN_FILE, var_type='Intercept', method='ret').round(4)

    # Save uncorrected CSVs in the same format
    uncor_k2_df.to_csv(UNCOR_K2_CSV, index=False)
    uncor_k1_df.to_csv(UNCOR_K1_CSV, index=False)
    uncor_lnb2_df.to_csv(UNCOR_LNB2_CSV, index=False)

    # Apply SZA corrections
    szacorr_k2_df = combine_diff_with_uncor(diff_k_data, uncor_k2_df).round(4)
    szacorr_k1_df = combine_diff_with_uncor(diff_k1_data, uncor_k1_df).round(4)
    szacorr_lnb2_df = combine_diff_with_uncor(diff_b_data, uncor_lnb2_df).round(4)

    # Save corrected CSVs
    szacorr_k2_df.to_csv(SZACORR_K2_CSV, index=False)
    szacorr_k1_df.to_csv(SZACORR_K1_CSV, index=False)
    szacorr_lnb2_df.to_csv(SZACORR_LNB2_CSV, index=False)

    fig = plt.figure(figsize=(18, 17), dpi=100)

    # Row 1 (3 panels): 10:30
    ax_k_a = fig.add_axes([0.06, 0.70, 0.26, 0.22])
    im_k_a = plot_heatmap(
        ax_k_a, uncor_k2_df,
        f'{format_panel_tag(0, icon_style)}   $k_{{\mathrm{{ret}}}}$, 10:30',
        vmin=K_VMIN, vmax=K_VMAX
    )

    ax_lnb_a = fig.add_axes([0.38, 0.70, 0.26, 0.22])
    im_lnb_a = plot_heatmap(
        ax_lnb_a, uncor_lnb2_df,
        f'{format_panel_tag(1, icon_style)}   ln$b_{{\mathrm{{ret}}}}$, 10:30',
        cmap=LNB_CMAP, vmin=LNB_VMIN, vmax=LNB_VMAX
    )

    ax_k1_g = fig.add_axes([0.70, 0.70, 0.26, 0.22])
    im_k1_g = plot_heatmap(
        ax_k1_g, uncor_k1_df,
        f'{format_panel_tag(2, icon_style)}   $k_{{\mathrm{{msk}}}}$, 10:30',
        vmin=K_VMIN, vmax=K_VMAX
    )

    # Row 2 (2 panels): cp vs SZA, left aligned
    ax_k_b = fig.add_axes([0.06, 0.40, 0.26, 0.22])
    plot_main_ax(
        ax_k_b,
        seasonal_stats,
        weight_dict,
        is_k_plot=True,
        panel_idx=3,
        icon_style=icon_style,
        title_y=1.10
    )

    ax_lnb_b = fig.add_axes([0.38, 0.40, 0.26, 0.22])
    plot_main_ax(
        ax_lnb_b,
        seasonal_stats,
        weight_dict,
        is_k_plot=False,
        cmap=LNB_CMAP,
        panel_idx=4,
        icon_style=icon_style,
        title_y=1.10
    )

    # Row 3 (2 panels): Daytime Mean, left aligned
    ax_k_c = fig.add_axes([0.06, 0.12, 0.26, 0.22])
    im_k_c = plot_heatmap(
        ax_k_c, szacorr_k2_df,
        f'{format_panel_tag(5, icon_style)}   $k_{{\mathrm{{ret}}}}$, Daytime Mean',
        vmin=K_VMIN, vmax=K_VMAX
    )

    ax_lnb_c = fig.add_axes([0.38, 0.12, 0.26, 0.22])
    im_lnb_c = plot_heatmap(
        ax_lnb_c, szacorr_lnb2_df,
        f'{format_panel_tag(6, icon_style)}   ln$b_{{\mathrm{{ret}}}}$, Daytime Mean',
        cmap=LNB_CMAP, vmin=LNB_VMIN, vmax=LNB_VMAX
    )

    cbar_c_ax = fig.add_axes([0.70, 0.65, 0.26, 0.014])
    cbar_c = fig.colorbar(im_k1_g, cax=cbar_c_ax, orientation='horizontal')
    cbar_c.set_label('$k$', fontsize=SIZE_PARAMS['xylabel'])
    cbar_c.set_ticks(np.arange(0.25, 0.91, 0.1))
    cbar_c.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])

    cbar_f_ax = fig.add_axes([0.06, 0.07, 0.26, 0.014])
    cbar_f = fig.colorbar(im_k_c, cax=cbar_f_ax, orientation='horizontal')
    cbar_f.set_label('$k$', fontsize=SIZE_PARAMS['xylabel'])
    cbar_f.set_ticks(np.arange(0.25, 0.91, 0.1))
    cbar_f.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])

    cbar_g_ax = fig.add_axes([0.38, 0.07, 0.26, 0.014])
    cbar_g = fig.colorbar(im_lnb_c, cax=cbar_g_ax, orientation='horizontal')
    cbar_g.set_label('ln$b$', fontsize=SIZE_PARAMS['xylabel'])
    cbar_g.set_ticks(np.arange(-2.7, -0.59, 0.3))
    cbar_g.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')