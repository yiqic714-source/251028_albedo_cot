from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import Ac_cot_fitting_utils as acfu


INPUT_DIR = Path("/home/chenyiqi/251028_albedo_cot/processed_data/merged_msk_and_ret_csv")
OUTPUT_FIG_UNR = Path("/home/chenyiqi/251028_albedo_cot/figs/scatter_unr_vs_kret_minus_kmsk.png")
OUTPUT_FIG_COTDISP = Path("/home/chenyiqi/251028_albedo_cot/figs/scatter_cotdisp_vs_kcp_minus_kret.png")


def filter_like_preprocess(df):
	"""Apply the same screening logic as preprocess_ocean_data."""
	df = df.copy()

	df["albedo"] = (
		(df["sw_all"] - df["sw_clr"] * (1 - df["cf_liq_ceres"])) /
		df["cf_liq_ceres"] / df["solar_incoming"]
	)
	df["month"] = pd.to_datetime(df["time"], format="mixed").dt.month

	for season_name, months in acfu.season_dict.items():
		df.loc[df["month"].isin(months), "season"] = season_name

	mask = (
		(df["cot_mod08"] > 2.5) &
		(df["ret_cot_cer"] > 2.5) &
		(df["cf_liq_ceres"] > 0.1) &
		(df["ret_albedo"] > 0) & (df["ret_albedo"] < 1) &
		(df["albedo"] > 0) & (df["albedo"] < 1)
	)
	return df.loc[mask].dropna().reset_index(drop=True)


def compute_ocean_season_k(ocean_data):
	"""Return season-wise k for ret/cp/msk using fig2_s4 settings."""
	fig, ax = plt.subplots(figsize=(4, 3))
	try:
		line_handles = []
		line_labels = []

		ret_result = acfu.plot_weighted_fit_line(
			ocean_data["ret_cot"],
			ocean_data["ret_albedo_list"][0],
			ocean_data["sza"],
			ocean_data["season"],
			ocean_data["x2"],
			"blue",
			"ret",
			line_handles,
			line_labels,
			ax,
			linestyle="--",
			cot_std=0.1,
			albedo_std=0.13,
		)

		cp_result = acfu.plot_weighted_fit_line(
			ocean_data["ret_cot"],
			ocean_data["ret_albedo_list"][1],
			ocean_data["sza"],
			ocean_data["season"],
			ocean_data["x2"],
			"orange",
			"cp",
			line_handles,
			line_labels,
			ax,
			linestyle="-",
			cot_std=0.0,
			albedo_std=0.03,
		)

		msk_result = acfu.plot_weighted_fit_line(
			ocean_data["msk_cot"],
			ocean_data["msk_albedo"],
			ocean_data["sza"],
			ocean_data["season"],
			ocean_data["x2"],
			"magenta",
			"msk",
			line_handles,
			line_labels,
			ax,
			linestyle="-",
			cot_std=0.1,
			albedo_std=0.20,
		)

		k_ret_season = ret_result[4]
		k_cp_season = cp_result[4]
		k_msk_season = msk_result[4]
		return k_ret_season, k_cp_season, k_msk_season
	finally:
		plt.close(fig)


def build_points():
	"""Build 8*4 ocean-season points: mean unr_fra vs (k_ret-k_msk)."""
	processed_by_ocean, _ = acfu.preprocess_ocean_data()

	rows = []
	usecols = list(dict.fromkeys(acfu.columns + ["cf_ret_liq_mod08", "clr_fra"]))

	for ocean in acfu.oceans:
		ocean_data = processed_by_ocean.get(ocean)
		if ocean_data is None:
			continue

		csv_path = INPUT_DIR / f"{ocean}.csv"
		df_raw = pd.read_csv(csv_path, usecols=usecols)
		df_filtered = filter_like_preprocess(df_raw)

		if len(df_filtered) == 0:
			continue

		df_filtered["unr_fra"] = 1 - df_filtered["cf_ret_liq_mod08"] - df_filtered["clr_fra"]
		df_filtered["cot_disp"] = df_filtered["ret_cotstd_cer"] / df_filtered["ret_cot_cer"]
		unr_by_season = df_filtered.groupby("season", dropna=True)["unr_fra"].mean().to_dict()
		cot_disp_by_season = df_filtered.groupby("season", dropna=True)["cot_disp"].mean().to_dict()

		k_ret_season, k_cp_season, k_msk_season = compute_ocean_season_k(ocean_data)

		for season in acfu.season_dict.keys():
			k_ret = k_ret_season.get(season, np.nan)
			k_cp = k_cp_season.get(season, np.nan)
			k_msk = k_msk_season.get(season, np.nan)
			rows.append(
				{
					"ocean": ocean,
					"season": season,
					"unr_fra": unr_by_season.get(season, np.nan),
					"k_ret_minus_k_msk": k_ret - k_msk,
					"cot_disp": cot_disp_by_season.get(season, np.nan),
					"k_cp_minus_k_ret": k_cp - k_ret,
				}
			)

	return pd.DataFrame(rows)


def plot_scatter_with_fit(points_df, ax):
	valid = points_df[["unr_fra", "k_ret_minus_k_msk"]].dropna()
	if len(valid) < 2:
		raise RuntimeError("Not enough valid points to fit a regression line.")

	slope, intercept, r_value, p_value, _ = stats.linregress(
		valid["unr_fra"].values,
		valid["k_ret_minus_k_msk"].values,
	)

	season_markers = {
		"MAM": "o",
		"JJA": "s",
		"SON": "^",
		"DJF": "D",
	}
	ocean_colors = {
		ocean: plt.cm.tab10(i % 10) for i, ocean in enumerate(acfu.oceans)
	}

	for ocean in acfu.oceans:
		for season in acfu.season_dict.keys():
			sub = points_df[(points_df["ocean"] == ocean) & (points_df["season"] == season)]
			if len(sub) == 0:
				continue
			ax.scatter(
				sub["unr_fra"],
				sub["k_ret_minus_k_msk"],
				s=58,
				alpha=0.9,
				marker=season_markers.get(season, "o"),
				color=ocean_colors[ocean],
			)

	x_line = np.linspace(valid["unr_fra"].min(), valid["unr_fra"].max(), 200)
	y_line = slope * x_line + intercept
	ax.plot(
		x_line,
		y_line,
		color="black",
		lw=2,
		label="fit line",
	)

	season_labels = list(acfu.season_dict.keys())
	ocean_labels = list(acfu.oceans)
	season_handles = {
		s: plt.Line2D([0], [0], marker=season_markers[s], color="black", linestyle="", markersize=7)
		for s in season_labels
	}
	ocean_handles = {
		o: plt.Line2D([0], [0], marker="o", color=ocean_colors[o], linestyle="", markersize=7)
		for o in ocean_labels
	}

	# Legend fills entries column-wise when ncol=3, so order by columns explicitly.
	half = len(ocean_labels) // 2
	left_oceans = ocean_labels[:half]
	right_oceans = ocean_labels[half:]
	combined_handles = [
		*[season_handles[s] for s in season_labels],
		*[ocean_handles[o] for o in left_oceans],
		*[ocean_handles[o] for o in right_oceans],
	]
	combined_labels = [
		*season_labels,
		*left_oceans,
		*right_oceans,
	]

	ax.legend(
		handles=combined_handles,
		labels=combined_labels,
		ncol=3,
		loc="center left",
		bbox_to_anchor=(1.02, 0.5),
		borderaxespad=0.0,
		framealpha=0.5,
		title="Season | Ocean",
	)
	ax.text(
		0.60,
		0.10,
		f"R={r_value:.2f}, p={p_value:.3f}",
		transform=ax.transAxes,
		ha="left",
		va="bottom",
		fontsize=11,
		bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "black", "linewidth": 0.8},
	)

	ax.set_xlabel("Twilight Zone Fraction", fontsize=12)
	ax.set_ylabel(r"$k_{\mathrm{ret}} - k_{\mathrm{msk}}$", fontsize=14)
	ax.grid(True, linestyle="--", alpha=0.3)



def plot_cotdisp_vs_kcp_minus_kret(points_df, ax):
	valid = points_df[["cot_disp", "k_cp_minus_k_ret"]].dropna()
	if len(valid) < 2:
		raise RuntimeError("Not enough valid points to fit cot_disp vs (k_cp-k_ret).")

	slope, intercept, r_value, p_value, _ = stats.linregress(
		valid["cot_disp"].values,
		valid["k_cp_minus_k_ret"].values,
	)

	season_markers = {
		"MAM": "o",
		"JJA": "s",
		"SON": "^",
		"DJF": "D",
	}
	ocean_colors = {
		ocean: plt.cm.tab10(i % 10) for i, ocean in enumerate(acfu.oceans)
	}

	for ocean in acfu.oceans:
		for season in acfu.season_dict.keys():
			sub = points_df[(points_df["ocean"] == ocean) & (points_df["season"] == season)]
			if len(sub) == 0:
				continue
			ax.scatter(
				sub["cot_disp"],
				sub["k_cp_minus_k_ret"],
				s=58,
				alpha=0.9,
				marker=season_markers.get(season, "o"),
				color=ocean_colors[ocean],
			)

	x_line = np.linspace(valid["cot_disp"].min(), valid["cot_disp"].max(), 200)
	y_line = slope * x_line + intercept
	ax.plot(
		x_line,
		y_line,
		color="black",
		lw=2,
		label="fit line",
	)

	season_labels = list(acfu.season_dict.keys())
	ocean_labels = list(acfu.oceans)
	season_handles = {
		s: plt.Line2D([0], [0], marker=season_markers[s], color="black", linestyle="", markersize=7)
		for s in season_labels
	}
	ocean_handles = {
		o: plt.Line2D([0], [0], marker="o", color=ocean_colors[o], linestyle="", markersize=7)
		for o in ocean_labels
	}

	# Legend fills entries column-wise when ncol=3, so order by columns explicitly.
	half = len(ocean_labels) // 2
	left_oceans = ocean_labels[:half]
	right_oceans = ocean_labels[half:]
	combined_handles = [
		*[season_handles[s] for s in season_labels],
		*[ocean_handles[o] for o in left_oceans],
		*[ocean_handles[o] for o in right_oceans],
	]
	combined_labels = [
		*season_labels,
		*left_oceans,
		*right_oceans,
	]

	ax.legend(
		handles=combined_handles,
		labels=combined_labels,
		ncol=1,
		loc="center left",
		bbox_to_anchor=(1.02, 0.5),
		borderaxespad=0.0,
		framealpha=0.5,
		title="Season | Ocean",
	)

	ax.text(
		0.60,
		0.10,
		f"R={r_value:.2f}, p={p_value:.3f}",
		transform=ax.transAxes,
		ha="left",
		va="bottom",
		fontsize=11,
		bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "black", "linewidth": 0.8},
	)

	ax.set_xlabel("Relative Dispersion of COT", fontsize=12)
	ax.set_ylabel(r"$k_{\mathrm{cp}} - k_{\mathrm{ret}}$", fontsize=14)
	ax.grid(True, linestyle="--", alpha=0.3)



def main():
	points_df = build_points()
	OUTPUT_FIG_UNR.parent.mkdir(parents=True, exist_ok=True)

	fig1, ax1 = plt.subplots(figsize=(6, 4.8))
	plot_scatter_with_fit(points_df, ax1)
	fig1.savefig(OUTPUT_FIG_UNR, dpi=300, bbox_inches="tight")
	plt.close(fig1)
	print(f"Saved scatter figure to: {OUTPUT_FIG_UNR}")

	fig2, ax2 = plt.subplots(figsize=(6, 4.8))
	plot_cotdisp_vs_kcp_minus_kret(points_df, ax2)
	fig2.savefig(OUTPUT_FIG_COTDISP, dpi=300, bbox_inches="tight")
	plt.close(fig2)
	print(f"Saved scatter figure to: {OUTPUT_FIG_COTDISP}")


if __name__ == "__main__":
	main()
