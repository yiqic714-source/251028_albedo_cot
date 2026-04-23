from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_DIR = Path("/home/chenyiqi/251028_albedo_cot/processed_data/merged_msk_and_ret_csv")
OUTPUT_FIG = Path("/home/chenyiqi/251028_albedo_cot/figs/COTret_mod_vs_cer_joint.png")


def load_all_points(input_dir: Path) -> pd.DataFrame:
	"""Load ret_cot_mod and ret_cot_cer from all CSV files under input_dir."""
	csv_files = sorted(input_dir.glob("*.csv"))
	if not csv_files:
		raise FileNotFoundError(f"No CSV files found under: {input_dir}")

	frames = []
	for csv_path in csv_files:
		df = pd.read_csv(csv_path, usecols=["ret_cot_mod", "ret_cot_cer"])
		df["source"] = csv_path.stem
		frames.append(df)

	all_df = pd.concat(frames, ignore_index=True)

	# Keep only physically meaningful, finite COT pairs.
	mask = (
		np.isfinite(all_df["ret_cot_mod"]) &
		np.isfinite(all_df["ret_cot_cer"]) &
		(all_df["ret_cot_mod"] > 0) &
		(all_df["ret_cot_cer"] > 0)
	)
	all_df = all_df.loc[mask].reset_index(drop=True)

	if all_df.empty:
		raise RuntimeError("No valid ret_cot_mod/ret_cot_cer pairs after filtering.")

	return all_df


def plot_joint_distribution(df: pd.DataFrame, output_fig: Path) -> None:
	"""Plot a joint distribution using 2D probability density."""
	x = df["ret_cot_mod"].to_numpy(dtype=float)
	y = df["ret_cot_cer"].to_numpy(dtype=float)

	# Use log-space bins because COT is strongly right-skewed.
	bins = np.logspace(np.log10(0.5), np.log10(150), 70)

	fig, ax_joint = plt.subplots(figsize=(6, 5), dpi=160)
	h = ax_joint.hist2d(x, y, bins=[bins, bins], cmap=plt.cm.Blues, density=True)
	cbar = fig.colorbar(h[3], ax=ax_joint, pad=0.01)
	cbar.set_label("Probability density", fontsize=11)

	xy_min = 0.5
	xy_max = 150
	ax_joint.plot([xy_min, xy_max], [xy_min, xy_max], color="black", lw=1.2, ls="--")

	ax_joint.set_xscale("log")
	ax_joint.set_yscale("log")
	ax_joint.set_xlim(xy_min, xy_max)
	ax_joint.set_ylim(xy_min, xy_max)
	ax_joint.set_xlabel(r"MODIS COT$_{\mathrm{ret}}$", fontsize=12)
	ax_joint.set_ylabel(r"CERES COT$_{\mathrm{ret}}$", fontsize=12)
	ax_joint.grid(True, ls="--", alpha=0.25)

	r = np.corrcoef(np.log10(x), np.log10(y))[0, 1]
	ax_joint.text(
		0.03,
		0.97,
		f"R = {r:.3f}",
		transform=ax_joint.transAxes,
		ha="left",
		va="top",
		fontsize=11,
		bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.8},
	)

	output_fig.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_fig, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	df = load_all_points(INPUT_DIR)
	plot_joint_distribution(df, OUTPUT_FIG)
	print(f"Saved figure: {OUTPUT_FIG}")
	print(f"Valid points: {len(df):,}")


if __name__ == "__main__":
	main()
