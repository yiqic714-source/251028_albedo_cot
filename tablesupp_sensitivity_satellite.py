from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Ac_cot_fitting_utils as acfu


OUTPUT_CSV = Path("/home/chenyiqi/251028_albedo_cot/processed_data/sensitivity_satellite_k_lnb_ret_msk.csv")


def compute_annual_fit(data):
	"""Compute annual weighted fit for ret and msk using the same fig2 workflow."""
	fig, ax = plt.subplots(figsize=(4, 3))
	try:
		line_handles = []
		line_labels = []

		ret_result = acfu.plot_weighted_fit_line(
			data["ret_cot"],
			data["ret_albedo_list"][0],
			data["sza"],
			data["season"],
			data["x2"],
			"blue",
			"ret",
			line_handles,
			line_labels,
			ax,
			linestyle="--",
			cot_std=0.1,
			albedo_std=0.13,
		)

		msk_result = acfu.plot_weighted_fit_line(
			data["msk_cot"],
			data["msk_albedo"],
			data["sza"],
			data["season"],
			data["x2"],
			"magenta",
			"msk",
			line_handles,
			line_labels,
			ax,
			linestyle="-",
			cot_std=0.1,
			albedo_std=0.20,
		)

		k_ret, lnb_ret = ret_result[0], ret_result[1]
		k_msk, lnb_msk = msk_result[0], msk_result[1]

		return k_ret, lnb_ret, k_msk, lnb_msk
	finally:
		plt.close(fig)


def run_once(pass_name, min_cot_mod08=2.0, min_ret_cot_cer=2.0, min_cf_liq_ceres=None):
	_, global_processed_data = acfu.preprocess_ocean_data(
		min_cot_mod08=min_cot_mod08,
		min_ret_cot_cer=min_ret_cot_cer,
		min_cf_liq_ceres=min_cf_liq_ceres,
	)

	if global_processed_data is None or int(global_processed_data.get("data_count", 0)) <= 0:
		return {
			"pass_name": pass_name,
			"k_ret": np.nan,
			"lnb_ret": np.nan,
			"k_msk": np.nan,
			"lnb_msk": np.nan,
			"data_count": 0,
		}

	k_ret, lnb_ret, k_msk, lnb_msk = compute_annual_fit(global_processed_data)
	return {
		"pass_name": pass_name,
		"k_ret": k_ret,
		"lnb_ret": lnb_ret,
		"k_msk": k_msk,
		"lnb_msk": lnb_msk,
		"data_count": int(global_processed_data.get("data_count", 0)),
	}


def main():
	rows = []

	# Pass 1: stricter COT thresholds.
	rows.append(
		run_once(
			pass_name="cot_gt_4",
			min_cot_mod08=4.0,
			min_ret_cot_cer=4.0,
			min_cf_liq_ceres=None,
		)
	)

	# Pass 2: cloud fraction constraint.
	rows.append(
		run_once(
			pass_name="cf_liq_gt_0.25",
			min_cot_mod08=2.5,
			min_ret_cot_cer=2.5,
			min_cf_liq_ceres=0.25,
		)
	)

	out_df = pd.DataFrame(rows)
	OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
	out_df.to_csv(OUTPUT_CSV, index=False)
	print(f"Saved satellite sensitivity results to: {OUTPUT_CSV}")


if __name__ == "__main__":
	main()
