from __future__ import annotations

import datetime as dt
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DATA_ROOT = Path('/home/chenyiqi/260320_ship_emission/processed_data/ml_xy_data')
GAMM = 1.37e-5
RANDOM_STATE = 42
VAL_FRAC_2020 = 0.10
YEARS = (2019, 2020)
SEASON = 'DJF'  # One of: DJF, MAM, JJA, SON
SEASON_MONTHS = {
	'DJF': {12, 1, 2},
	'MAM': {3, 4, 5},
	'JJA': {6, 7, 8},
	'SON': {9, 10, 11},
}
# Keep rows with SOX_COL quantile in [q_low, q_high], e.g. (0.0, 0.1) or (0.9, 1.0).
QUANTILE_RANGE = (0.9, 1.0)#(0, 0.1)
SOX_COL = 'weighted_sox_diff'
PLOT_DIR = Path('/home/chenyiqi/260320_ship_emission/processed_data/ml_xy_data/training_figs')
DEVICE = 'cpu'
CPU_THREADS = 64
EPOCHS = 24
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 6
TRANSFORMER_D_MODEL = 32
TRANSFORMER_NHEAD = 4
TRANSFORMER_LAYERS = 1
TRANSFORMER_DROPOUT = 0.1


class TabularTransformerRegressor(nn.Module):
	def __init__(self, n_features: int, d_model: int = 64, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
		super().__init__()
		if d_model % nhead != 0:
			raise ValueError('d_model must be divisible by nhead.')
		self.n_features = n_features
		self.feature_proj = nn.Linear(1, d_model)
		self.feature_embed = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=d_model * 4,
			dropout=dropout,
			batch_first=True,
			activation='gelu',
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.head = nn.Sequential(
			nn.LayerNorm(d_model),
			nn.Linear(d_model, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		tokens = self.feature_proj(x.unsqueeze(-1))
		tokens = tokens + self.feature_embed.unsqueeze(0)
		encoded = self.encoder(tokens)
		pooled = encoded.mean(dim=1)
		out = self.head(pooled).squeeze(-1)
		return out


def _train_transformer_regressor(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_val: np.ndarray,
	y_val: np.ndarray,
	X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	torch.manual_seed(RANDOM_STATE)
	np.random.seed(RANDOM_STATE)

	model = TabularTransformerRegressor(
		n_features=X_train.shape[1],
		d_model=TRANSFORMER_D_MODEL,
		nhead=TRANSFORMER_NHEAD,
		num_layers=TRANSFORMER_LAYERS,
		dropout=TRANSFORMER_DROPOUT,
	).to(DEVICE)
	optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
	loss_fn = nn.MSELoss()

	train_ds = TensorDataset(
		torch.from_numpy(X_train.astype(np.float32)),
		torch.from_numpy(y_train.astype(np.float32)),
	)
	train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

	X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(DEVICE)
	y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(DEVICE)
	X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(DEVICE)

	best_state = None
	best_val_loss = float('inf')
	patience = 0

	for epoch in range(EPOCHS):
		epoch_start = time.perf_counter()
		model.train()
		train_loss_sum = 0.0
		train_batches = 0
		for xb, yb in train_loader:
			xb = xb.to(DEVICE)
			yb = yb.to(DEVICE)
			optimizer.zero_grad(set_to_none=True)
			pred = model(xb)
			loss = loss_fn(pred, yb)
			loss.backward()
			optimizer.step()
			train_loss_sum += float(loss.item())
			train_batches += 1

		model.eval()
		with torch.no_grad():
			val_pred = model(X_val_t)
			val_loss = float(loss_fn(val_pred, y_val_t).item())

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
			patience = 0
			is_best = True
		else:
			patience += 1
			is_best = False

		epoch_sec = time.perf_counter() - epoch_start
		mean_train_loss = train_loss_sum / max(train_batches, 1)
		best_tag = ' *best' if is_best else ''
		print(
			f'Epoch {epoch + 1:02d}/{EPOCHS} '
			f'train_loss={mean_train_loss:.6f} val_loss={val_loss:.6f} '
			f'time={epoch_sec:.2f}s patience={patience}/{EARLY_STOP_PATIENCE}{best_tag}'
		)

		if patience >= EARLY_STOP_PATIENCE:
			print('Early stopping triggered.')
			break

	if best_state is not None:
		model.load_state_dict(best_state)

	model.eval()
	with torch.no_grad():
		pred_val = model(X_val_t).cpu().numpy()
		pred_test = model(X_test_t).cpu().numpy()

	return pred_val, pred_test


def _build_nd(df: pd.DataFrame) -> pd.Series:
	return GAMM * np.power(df['cot_mod08'], 0.5) * np.power(df['cer_mod08'] * 1e-6, -2.5) * 1e-6



def _extract_date_from_name(path: Path) -> dt.date | None:
	# Expected tail in filename: *_YYYYMMDD1330.csv
	m = re.search(r'_(\d{8})\d{4}\.csv$', path.name)
	if not m:
		return None
	return dt.datetime.strptime(m.group(1), '%Y%m%d').date()


def _load_season_data(data_root: Path, season: str) -> pd.DataFrame:
	season_key = season.strip().upper()
	if season_key not in SEASON_MONTHS:
		raise ValueError(f'Unsupported SEASON: {season}. Use one of {list(SEASON_MONTHS.keys())}.')
	months = SEASON_MONTHS[season_key]

	paths = sorted(data_root.glob('*/soxdiff_met_and_cld_*.csv'))
	frames: list[pd.DataFrame] = []
	for path in paths:
		date_value = _extract_date_from_name(path)
		if date_value is None:
			continue
		if date_value.year not in YEARS or date_value.month not in months:
			continue
		df = pd.read_csv(path)
		df['source_year'] = date_value.year
		frames.append(df)

	if not frames:
		raise ValueError(f'No {season_key} CSV files found for 2019/2020.')

	return pd.concat(frames, ignore_index=True)


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
	exclude = {
		'weighted_sox_diff',
		'cf_ret_liq_mod08',
		'cot_mod08',
		'cer_mod08',
		'cwp_mod08',
		'nd',
		'cf_ret_combined_mod08',
		'aod_mod08',
		'source_year',
	}

	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	feature_cols = []
	for col in numeric_cols:
		if col in exclude:
			continue
		# Exclude *_0 style columns (not pressure-level columns like *_1000).
		if col.endswith('_0'):
			continue
		feature_cols.append(col)
	return feature_cols


def _train_one_target(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	test_df: pd.DataFrame,
	feature_cols: list[str],
	target_col: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
	train_mask = np.isfinite(train_df[target_col].to_numpy(dtype=float))
	val_mask = np.isfinite(val_df[target_col].to_numpy(dtype=float))
	test_mask = np.isfinite(test_df[target_col].to_numpy(dtype=float))

	work_train = train_df.loc[train_mask, feature_cols + [target_col]].copy()
	work_val = val_df.loc[val_mask, feature_cols + [target_col]].copy()
	work_test = test_df.loc[test_mask, feature_cols + [target_col]].copy()

	if work_train.empty or work_val.empty or work_test.empty:
		raise ValueError(f'Insufficient finite rows for target: {target_col}')
	if 't0_lat' not in work_test.columns or 't0_lon' not in work_test.columns:
		raise ValueError('Test data must contain t0_lat and t0_lon for global plotting.')

	X_train = work_train[feature_cols].to_numpy(dtype=float)
	y_train = work_train[target_col].to_numpy(dtype=float)
	X_val = work_val[feature_cols].to_numpy(dtype=float)
	y_val = work_val[target_col].to_numpy(dtype=float)
	X_test = work_test[feature_cols].to_numpy(dtype=float)
	y_test = work_test[target_col].to_numpy(dtype=float)

	imputer = SimpleImputer(strategy='median')
	scaler = StandardScaler()
	X_train = scaler.fit_transform(imputer.fit_transform(X_train))
	X_val = scaler.transform(imputer.transform(X_val))
	X_test = scaler.transform(imputer.transform(X_test))

	pred_val, pred_test = _train_transformer_regressor(X_train, y_train, X_val, y_val, X_test)
	test_plot_df = work_test[['t0_lat', 't0_lon', target_col]].copy()
	test_plot_df['test_pred'] = pred_test
	test_plot_df = (
		test_plot_df
		.groupby(['t0_lat', 't0_lon'], as_index=False)
		.agg(test_target_mean=(target_col, 'mean'), test_pred_mean=('test_pred', 'mean'))
	)

	metrics = {
		'target': target_col,
		'n_rows': int(len(work_train) + len(work_val) + len(work_test)),
		'n_features': int(len(feature_cols)),
		'r2': float(r2_score(y_val, pred_val)),
		'rmse': float(np.sqrt(mean_squared_error(y_val, pred_val))),
		'mae': float(mean_absolute_error(y_val, pred_val)),
		'val_target_mean': float(np.mean(y_val)),
		'val_pred_mean': float(np.mean(pred_val)),
		'test_target_mean': float(np.mean(y_test)),
		'test_pred_mean': float(np.mean(pred_test)),
	}
	return metrics, test_plot_df


def _plot_test_global_distribution(
	test_plot_df: pd.DataFrame,
	target_col: str,
	out_dir: Path,
	season: str,
	q_low: float,
	q_high: float,
) -> Path:
	out_dir.mkdir(parents=True, exist_ok=True)
	q_low_pct = int(round(q_low * 100))
	q_high_pct = int(round(q_high * 100))
	out_path = out_dir / f'test_global_{target_col}_{season.upper()}_q{q_low_pct:02d}-{q_high_pct:02d}.png'

	fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=220, constrained_layout=True)
	plot_pairs = [
		('test_target_mean', f'Test Mean of {target_col}'),
		('test_pred_mean', f'Test Mean Prediction of {target_col}'),
	]

	v1 = test_plot_df['test_target_mean'].to_numpy(dtype=float)
	v2 = test_plot_df['test_pred_mean'].to_numpy(dtype=float)
	vall = np.concatenate([v1[np.isfinite(v1)], v2[np.isfinite(v2)]])
	vmin = float(np.nanmin(vall)) if vall.size else 0.0
	vmax = float(np.nanmax(vall)) if vall.size else 1.0

	for ax, (col_name, title) in zip(axes, plot_pairs):
		sc = ax.scatter(
			test_plot_df['t0_lon'].to_numpy(dtype=float),
			test_plot_df['t0_lat'].to_numpy(dtype=float),
			c=test_plot_df[col_name].to_numpy(dtype=float),
			s=9,
			cmap='viridis',
			vmin=vmin,
			vmax=vmax,
			linewidths=0,
			alpha=0.9,
		)
		ax.set_title(title)
		ax.set_xlabel('Longitude')
		ax.set_ylabel('Latitude')
		ax.set_xlim(-180, 180)
		ax.set_ylim(-90, 90)
		ax.grid(True, linestyle='--', alpha=0.3)
		fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)

	fig.savefig(out_path, bbox_inches='tight')
	plt.close(fig)
	return out_path

def main() -> None:
	torch.set_num_threads(CPU_THREADS)
	# Keep inter-op lower to reduce thread scheduling overhead on CPU training.
	torch.set_num_interop_threads(1)
	main_start = time.perf_counter()

	df = _load_season_data(DATA_ROOT, SEASON)
	if SOX_COL not in df.columns:
		raise ValueError(f'Missing soxdiff column: {SOX_COL}')
	df = df[np.isfinite(df[SOX_COL].to_numpy(dtype=float))].copy()
	if df.empty:
		raise ValueError('No finite rows in soxdiff column.')

	q_low, q_high = QUANTILE_RANGE
	low_thr = float(df[SOX_COL].quantile(q_low))
	high_thr = float(df[SOX_COL].quantile(q_high))
	df_top = df[(df[SOX_COL] >= low_thr) & (df[SOX_COL] <= high_thr)].copy()

	if df_top.empty:
		raise ValueError('No rows selected in configured quantile range.')

	df_2019 = df_top[df_top['source_year'] == 2019].copy()
	df_2020 = df_top[df_top['source_year'] == 2020].copy()
	df_2019['nd'] = _build_nd(df_2019)
	df_2020['nd'] = _build_nd(df_2020)
	df_2019['nd'] = np.log(df_2019['nd'].to_numpy(dtype=float) + 1e-9)
	df_2020['nd'] = np.log(df_2020['nd'].to_numpy(dtype=float) + 1e-9)
	df_2019['cwp_mod08'] = np.log(df_2019['cwp_mod08'].to_numpy(dtype=float) + 1e-9)
	df_2020['cwp_mod08'] = np.log(df_2020['cwp_mod08'].to_numpy(dtype=float) + 1e-9)

	if df_2019.empty or df_2020.empty:
		raise ValueError('Top soxdiff rows do not contain both 2019 and 2020 data.')

	train_2020, val_2020 = train_test_split(
		df_2020,
		test_size=VAL_FRAC_2020,
		random_state=RANDOM_STATE,
	)

	train_df = train_2020.reset_index(drop=True)
	val_df = val_2020.reset_index(drop=True)
	test_df = df_2019.reset_index(drop=True)

	feature_cols = _select_feature_columns(pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True))
	targets = ['cf_ret_liq_mod08', 'nd', 'cwp_mod08']

	for target in targets:
		if target not in train_df.columns:
			raise ValueError(f'Missing target column in data: {target}')

	print(f'Selected period: 2019/2020 {SEASON.upper()}')
	print(f'Model: transformer (device={DEVICE})')
	print(f'Torch CPU threads: intra_op={torch.get_num_threads()}, inter_op={torch.get_num_interop_threads()}')
	print(f'Rows -> loaded finite soxdiff: {len(df)}, quantile[{q_low:.2f}, {q_high:.2f}]: {len(df_top)}')
	print(f'Rows -> train(2020): {len(train_df)}, val(2020): {len(val_df)}, test(2019): {len(test_df)}')
	print(f'Feature count: {len(feature_cols)}')
	print('Features: ' + ', '.join(feature_cols))
	print('Targets:', ', '.join(targets))
	print('Start training...')

	results = []
	for target in targets:
		target_start = time.perf_counter()
		metrics, test_plot_df = _train_one_target(train_df, val_df, test_df, feature_cols, target)
		plot_path = _plot_test_global_distribution(test_plot_df, target, PLOT_DIR, SEASON, q_low, q_high)
		target_sec = time.perf_counter() - target_start
		print(f'Saved test global plot ({target}): {plot_path}')
		print(f'Target {target} finished in {target_sec:.2f}s')
		results.append(metrics)
	result_df = pd.DataFrame(results)

	total_sec = time.perf_counter() - main_start
	print(f'Quantile range: [{q_low:.2f}, {q_high:.2f}], thresholds=[{low_thr:.6g}, {high_thr:.6g}]')
	print(result_df.to_string(index=False))
	print(f'Total runtime: {total_sec:.2f}s')


if __name__ == '__main__':
	main()
