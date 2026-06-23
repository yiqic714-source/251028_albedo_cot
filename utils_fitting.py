"""
utils_fig_common.py

Shared utility functions and constants for fig2_curves_by_ocean_and_aod.py
and fig3_bias_attribution.py.
"""

import os
import numpy as np
import pandas as pd
from scipy import odr, stats
from scipy.interpolate import griddata

np.random.seed(0)

# ============================================================
# Constants
# ============================================================

oceans = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']

season_dict = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}

cot_range = np.exp(np.linspace(np.log(2.5), np.log(62.5), 15))


# ============================================================
# Transformation functions
# ============================================================

def cot_to_x(cot):
    return np.log(cot)


def albedo_to_y(albedo):
    albedo = np.clip(albedo, 1e-6, 1 - 1e-6)
    return np.log(albedo / (1 - albedo))


def cot_k_b_to_albedo(cot, k, b):
    """Corrected albedo: Ac = b * cot^k / (1 + b * cot^k)"""
    return b * cot ** k / (1 + b * cot ** k)


# ============================================================
# Fitting helper functions
# ============================================================

def _linear_func(beta, x):
    return beta[0] * x + beta[1]


def _as_sigma_array(sigma, n):
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        return np.full(n, float(sigma), dtype=float)
    sigma = sigma.ravel()
    if sigma.size != n:
        raise ValueError("sigma must be scalar or have the same length as the input data.")
    return sigma.astype(float)


def _raw_to_fit_arrays(cot, albedo, cot_sigma, albedo_sigma):
    cot = np.asarray(cot, dtype=float).ravel()
    albedo = np.asarray(albedo, dtype=float).ravel()
    cot_sigma = np.asarray(cot_sigma, dtype=float).ravel()
    albedo_sigma = np.asarray(albedo_sigma, dtype=float).ravel()

    cot = np.clip(cot, 1e-6, None)
    albedo = np.clip(albedo, 1e-6, 1 - 1e-6)

    x = cot_to_x(cot)
    y = albedo_to_y(albedo)

    sx = cot_sigma / cot
    sy = albedo_sigma / (albedo * (1 - albedo))

    x_scale = np.nanmax(np.abs(x)) if np.any(np.isfinite(x)) else 1.0
    y_scale = np.nanmax(np.abs(y)) if np.any(np.isfinite(y)) else 1.0
    sx_floor = max(x_scale * 1e-12, 1e-12)
    sy_floor = max(y_scale * 1e-12, 1e-12)

    sx = np.where(np.isfinite(sx) & (sx > 0), sx, sx_floor)
    sy = np.where(np.isfinite(sy) & (sy > 0), sy, sy_floor)

    return x, y, sx, sy


def _fit_odr_once(x, y, sx, sy, beta0=None):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    sx = np.asarray(sx, dtype=float).ravel()
    sy = np.asarray(sy, dtype=float).ravel()

    if x.size < 2:
        raise ValueError("At least 2 points are required.")

    if beta0 is None:
        k0, b0 = np.polyfit(x, y, 1)
        beta0 = [k0, b0]

    data = odr.RealData(x, y, sx=sx, sy=sy)
    model = odr.Model(_linear_func)
    out = odr.ODR(data, model, beta0=beta0).run()
    return out.beta[0], out.beta[1]


def mc_fit(cot, albedo, cot_std=0.0, albedo_std=0.0, n_mc=300, bootstrap=True, random_seed=42):
    cot = np.asarray(cot, dtype=float).ravel()
    albedo = np.asarray(albedo, dtype=float).ravel()

    if cot.size != albedo.size:
        raise ValueError("cot and albedo must have the same length.")

    cot_sigma = _as_sigma_array(cot_std, cot.size)
    albedo_sigma = _as_sigma_array(albedo_std, albedo.size)

    mask = (
        np.isfinite(cot) & np.isfinite(albedo) &
        np.isfinite(cot_sigma) & np.isfinite(albedo_sigma) &
        (cot > 0) & (albedo > 0) & (albedo < 1)
    )

    cot = cot[mask]
    albedo = albedo[mask]
    cot_sigma = cot_sigma[mask]
    albedo_sigma = albedo_sigma[mask]

    if cot.size < 3:
        return np.nan, np.nan, np.nan, np.nan

    x, y, sx, sy = _raw_to_fit_arrays(cot, albedo, cot_sigma, albedo_sigma)

    try:
        k_best, b_best = _fit_odr_once(x, y, sx, sy)
    except Exception:
        try:
            lr = stats.linregress(x, y)
            k_best, b_best = lr.slope, lr.intercept
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_seed)
    n = cot.size
    base_idx = np.arange(n)

    count = 0
    mean_k = 0.0
    mean_b = 0.0
    M2_k = 0.0
    M2_b = 0.0

    for _ in range(n_mc):
        idx = rng.integers(0, n, size=n) if bootstrap else base_idx

        cot_i = cot[idx]
        albedo_i = albedo[idx]
        cot_sigma_i = cot_sigma[idx]
        albedo_sigma_i = albedo_sigma[idx]

        cot_pert = cot_i + rng.normal(0.0, cot_sigma_i)
        albedo_pert = albedo_i + rng.normal(0.0, albedo_sigma_i)

        cot_pert = np.clip(cot_pert, 1e-6, None)
        albedo_pert = np.clip(albedo_pert, 1e-6, 1 - 1e-6)

        if np.unique(cot_pert).size < 2:
            continue

        x_i, y_i, sx_i, sy_i = _raw_to_fit_arrays(
            cot_pert, albedo_pert, cot_sigma_i, albedo_sigma_i
        )

        try:
            ki, bi = _fit_odr_once(x_i, y_i, sx_i, sy_i, beta0=[k_best, b_best])
        except Exception:
            try:
                lr = stats.linregress(x_i, y_i)
                ki, bi = lr.slope, lr.intercept
            except Exception:
                continue

        count += 1
        dk = ki - mean_k
        mean_k += dk / count
        M2_k += dk * (ki - mean_k)

        db = bi - mean_b
        mean_b += db / count
        M2_b += db * (bi - mean_b)

    if count < 2:
        return k_best, b_best, np.nan, np.nan

    k_unc = np.sqrt(M2_k / (count - 1))
    b_unc = np.sqrt(M2_b / (count - 1))

    return k_best, b_best, k_unc, b_unc


# ============================================================
# SBDART lookup table interpolation
# ============================================================

def cot_to_albedo(cot, method, sza=None, table_folder='dcp', ocean=None, season=None):
    """
    Compute cloud albedo from COT using various methods.

    Parameters
    ----------
    cot : array-like
        Cloud optical thickness.
    method : str
        'sbdart', 'l74', 'quadrature', or 'eddington'.
    sza : float or array-like, optional
        Solar zenith angle in degrees. Required for 'sbdart', 'quadrature', 'eddington'.
    table_folder : str, optional
        Subfolder name under build_sbdart_lookup_table/ (e.g., 'dcp', 'cp').
    ocean : str, optional
        Ocean region name (e.g., 'TPO'). If provided with season, reads region-specific LUT.
    season : str, optional
        Season name (e.g., 'MAM'). If provided with ocean, reads region-specific LUT.

    Returns
    -------
    array-like
        Cloud albedo values.
    """
    base_path = '/home/chenyiqi/251028_albedo_cot'
    cot = np.asarray(cot, dtype=float)

    if method == 'sbdart':
        # print(table_folder)
        if 'dcp' in table_folder:
            file_name = 'cot_sza_to_albedo_lookup_table_TPO_MAM.csv'
        else:
            file_name = f'cot_sza_to_albedo_lookup_table_{ocean}_{season}.csv'

        file_path = (
            f'{base_path}/build_sbdart_lookup_table/'
            f'cot_sza_to_albedo_lookup_table_{table_folder}/'
            f'{file_name}'
        )

        if not os.path.exists(file_path):
            return np.full(cot.shape, np.nan)

        df = pd.read_csv(file_path, index_col=0)
        sza_grid = np.array(df.index, dtype=float)
        cot_grid = np.array(df.columns, dtype=float)
        albedo_grid = df.values

        sza_mesh, cot_mesh = np.meshgrid(sza_grid, cot_grid, indexing='ij')
        points = np.column_stack([sza_mesh.ravel(), cot_mesh.ravel()])
        values = albedo_grid.ravel()

        valid = np.isfinite(values)

        cot_arr = np.atleast_1d(cot)
        if np.ndim(sza) == 0:
            sza_arr = np.full_like(cot_arr, sza, dtype=float)
        else:
            sza_arr = np.asarray(sza, dtype=float)

        target = np.column_stack([sza_arr, cot_arr])

        albedo = griddata(
            points[valid],
            values[valid],
            target,
            method='linear',
            fill_value=np.nan
        )

        return albedo.reshape(cot_arr.shape)

    if method == 'l74':
        g = 0.85
        b = 1 - g
        return b * cot / (1 + b * cot)

    if method == 'quadrature':
        g = 0.85
        mu = np.cos(np.radians(sza))
        b = np.sqrt(3) / 2 * (1 - g)
        return (
            b * cot + (1 / 2 - np.sqrt(3) / 2 * mu) * (1 - np.exp(-cot / mu))
        ) / (1 + b * cot)

    if method == 'eddington':
        g = 0.85
        mu = np.cos(np.radians(sza))
        return (
            (1 - g) * cot + (2 / 3 - mu) * (1 - np.exp(-cot / mu))
        ) / (4 / 3 + (1 - g) * cot)

    raise ValueError(f'Unsupported method: {method}')


# ============================================================
# Panel tag formatting
# ============================================================

def format_panel_tag(panel_idx, icon_style):
    """Format panel tag: 'science' -> A, B, C...; 'nature' -> (a), (b), (c)..."""
    if icon_style == 'science':
        letter = chr(ord('A') + panel_idx)
        return rf'$\mathbf{{{letter}}}$'
    letter = chr(ord('a') + panel_idx)
    return rf'$\mathbf{{({letter})}}$'
