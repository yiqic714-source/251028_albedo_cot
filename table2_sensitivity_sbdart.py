from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import Ac_cot_fitting_utils as acfu


CASES = ["cbh2", "cth2", "cth4", "cer7", "cer13"]
BASE_DIR = Path("/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table")
OUTPUT_CSV = Path("/home/chenyiqi/251028_albedo_cot/processed_data/sensitivity_sbdart_k_lnb_cp.csv")


def load_lookup_table_points(csv_path, cot, sza):
	"""Sample lookup table albedo at provided cot/sza points."""
	df_lookup = pd.read_csv(csv_path, index_col=0)

	sza_grid = np.asarray(df_lookup.index, dtype=float)
	cot_grid = np.asarray(df_lookup.columns, dtype=float)
	albedo_grid = np.asarray(df_lookup.values, dtype=float)

	sza_mesh, cot_mesh = np.meshgrid(sza_grid, cot_grid, indexing="ij")
	points = np.column_stack([sza_mesh.ravel(), cot_mesh.ravel()])
	values = albedo_grid.ravel()
	valid_lookup = np.isfinite(values)

	if cot.size == 0 or sza.size == 0 or not np.any(valid_lookup):
		return np.array([]), np.array([]), np.array([], dtype=bool)

	target_points = np.column_stack([sza, cot])
	albedo = griddata(points[valid_lookup], values[valid_lookup], target_points, method="linear", fill_value=np.nan)

	valid = np.isfinite(cot) & np.isfinite(albedo) & (cot > 0) & (albedo > 0) & (albedo < 1)
	return cot[valid], albedo[valid], valid


def collect_global_points_for_case(case_name):
	"""Collect global cot/albedo points for one case by merging all oceans and seasons."""
	case_dir = BASE_DIR / f"sensitivity_{case_name}"
	all_cot = []
	all_albedo = []
	all_sza = []
	all_season = []

	for ocean in acfu.oceans:
		ocean_csv = Path(
			f"/home/chenyiqi/251028_albedo_cot/processed_data/merged_msk_and_ret_csv/{ocean}.csv"
		)
		if not ocean_csv.exists():
			continue

		df = pd.read_csv(ocean_csv, usecols=acfu.columns)
		df["albedo"] = (
			(df["sw_all"] - df["sw_clr"] * (1 - df["cf_liq_ceres"])) /
			df["cf_liq_ceres"] / df["solar_incoming"]
		)
		df["month"] = pd.to_datetime(df["time"], format="mixed").dt.month

		for season_name, months in acfu.season_dict.items():
			df.loc[df["month"].isin(months), "season"] = season_name

		base_mask = (
			(df["cot_mod08"] > 2.5) &
			(df["ret_cot_cer"] > 2.5) &
			(df["cf_liq_ceres"] > 0.1) &
			(df["ret_albedo"] > 0) & (df["ret_albedo"] < 1) &
			(df["albedo"] > 0) & (df["albedo"] < 1)
		)
		df_filtered = df[base_mask].dropna().reset_index(drop=True)
		if df_filtered.empty:
			continue

		for season in acfu.season_dict.keys():
			season_points = df_filtered[df_filtered["season"] == season]
			if season_points.empty:
				continue

			lookup_csv = case_dir / f"cot_sza_to_albedo_lookup_table_{ocean}_{season}.csv"
			if not lookup_csv.exists():
				continue

			cot = season_points["ret_cot_cer"].to_numpy(dtype=float)
			sza = season_points["sza"].to_numpy(dtype=float)
			cot_valid, albedo_valid, valid_mask = load_lookup_table_points(lookup_csv, cot, sza)
			if cot_valid.size == 0:
				continue

			all_cot.append(cot_valid)
			all_albedo.append(albedo_valid)
			all_sza.append(sza[valid_mask])
			all_season.append(np.full(cot_valid.shape, season, dtype=object))

	if not all_cot:
		return np.array([]), np.array([]), np.array([]), np.array([])

	return (
		np.concatenate(all_cot),
		np.concatenate(all_albedo),
		np.concatenate(all_sza),
		np.concatenate(all_season),
	)


def collect_dcp_points_for_case(case_name):
	"""Collect DCP points using only TPO MAM with fixed sza=54.4."""
	case_dir = BASE_DIR / f"sensitivity_dcp_{case_name}"
	ocean_csv = Path("/home/chenyiqi/251028_albedo_cot/processed_data/merged_msk_and_ret_csv/TPO.csv")
	if not ocean_csv.exists():
		return np.array([]), np.array([])

	df = pd.read_csv(ocean_csv, usecols=acfu.columns)
	df["albedo"] = (
		(df["sw_all"] - df["sw_clr"] * (1 - df["cf_liq_ceres"])) /
		df["cf_liq_ceres"] / df["solar_incoming"]
	)
	df["month"] = pd.to_datetime(df["time"], format="mixed").dt.month
	df.loc[df["month"].isin(acfu.season_dict["MAM"]), "season"] = "MAM"

	mask = (
		(df["cot_mod08"] > 2.5) &
		(df["ret_cot_cer"] > 2.5) &
		(df["cf_liq_ceres"] > 0.1) &
		(df["ret_albedo"] > 0) & (df["ret_albedo"] < 1) &
		(df["albedo"] > 0) & (df["albedo"] < 1) &
		(df["season"] == "MAM")
	)
	df_filtered = df[mask].dropna().reset_index(drop=True)
	if df_filtered.empty:
		return np.array([]), np.array([])

	lookup_csv = case_dir / "cot_sza_to_albedo_lookup_table_TPO_MAM.csv"
	if not lookup_csv.exists():
		return np.array([]), np.array([]), np.array([]), np.array([])

	cot = df_filtered["ret_cot_cer"].to_numpy(dtype=float)
	sza = np.full(cot.shape, 54.4, dtype=float)
	cot_valid, albedo_valid, valid_mask = load_lookup_table_points(lookup_csv, cot, sza)
	if cot_valid.size == 0:
		return np.array([]), np.array([]), np.array([]), np.array([])

	season = np.full(cot_valid.shape, "MAM", dtype=object)
	return cot_valid, albedo_valid, sza[valid_mask], season


def fit_with_group_weighting(cot, albedo, sza, season, label, cot_std=0.0, albedo_std=0.03):
	"""Fit with season+SZA subgrouping and sample-count weighted annual mean."""
	if cot.size < 3:
		return np.nan, np.nan

	fig, ax = plt.subplots(figsize=(4, 3))
	try:
		line_handles = []
		line_labels = []
		x2 = acfu.cot_to_x(np.linspace(np.nanmin(cot), np.nanmax(cot), 200))
		result = acfu.plot_weighted_fit_line(
			cot,
			albedo,
			sza,
			season,
			x2,
			"black",
			label,
			line_handles,
			line_labels,
			ax,
			linestyle="-",
			n_sza_groups=2,
			cot_std=cot_std,
			albedo_std=albedo_std,
		)
		return result[0], result[1]
	finally:
		plt.close(fig)


def run_once(case_name):
	"""Compute one global fit row for a given lookup-table case."""
	cp_cot, cp_albedo, cp_sza, cp_season = collect_global_points_for_case(
		case_name)
	dcp_cot, dcp_albedo, dcp_sza, dcp_season = collect_dcp_points_for_case(
		case_name
	)

	k_cp, lnb_cp = fit_with_group_weighting(
		cp_cot, cp_albedo, cp_sza, cp_season, label="cp", cot_std=0.0, albedo_std=0.03
	)
	k_dcp, lnb_dcp = fit_with_group_weighting(
		dcp_cot, dcp_albedo, dcp_sza, dcp_season, label="dcp", cot_std=0.0, albedo_std=0.03
	)

	return {
		"pass_name": case_name,
		"k_cp": k_cp,
		"lnb_cp": lnb_cp,
		"k_dcp": k_dcp,
		"lnb_dcp": lnb_dcp,
		"data_count": int(cp_cot.size),
		"dcp_data_count": int(dcp_cot.size),
	}


def main():
	rows = []
	for case in CASES:
		rows.append(
			run_once(
				case_name=case
			)
		)

	output_df = pd.DataFrame(rows)
	OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
	output_df.to_csv(OUTPUT_CSV, index=False)
	print(f"Saved cp sensitivity fit results to: {OUTPUT_CSV}")


if __name__ == "__main__":
	main()
