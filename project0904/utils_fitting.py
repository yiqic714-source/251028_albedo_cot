"""
utils_fig_common.py

Shared utility functions and constants for fig2_curves_by_ocean_and_aod.py
and fig3_bias_attribution.py.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

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
    albedo = np.asarray(albedo, dtype=float)
    albedo = np.clip(albedo, 1e-6, 1 - 1e-6)
    return np.log(albedo / (1 - albedo))


def cot_k_b_to_albedo(cot, k, b):
    """Corrected albedo: Ac = b * cot^k / (1 + b * cot^k)"""
    return b * cot ** k / (1 + b * cot ** k)


# ============================================================
# Fitting helper functions
# ============================================================

def _raw_to_fit_arrays(cot, albedo):
    cot = np.asarray(cot, dtype=float).ravel()
    albedo = np.asarray(albedo, dtype=float).ravel()

    cot = np.clip(cot, 1e-6, None)
    albedo = np.clip(albedo, 1e-6, 1 - 1e-6)

    x = cot_to_x(cot)
    y = albedo_to_y(albedo)
    return x, y


def _fit_lstsq_with_uncertainty(x, y):
    """Fit y = k*x + ln(b) and estimate Jacobian-based standard errors.

    All binned points receive equal weight. Parameter covariance is estimated
    as s^2 * (J.T @ J)^-1, where s^2 is the residual variance and J is the
    Jacobian (the design matrix for this linear model).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.size < 3:
        raise ValueError("At least 3 points are required.")

    design = np.column_stack([x, np.ones_like(x)])
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < design.shape[1]:
        raise np.linalg.LinAlgError("The design matrix is rank deficient.")

    residual = y - design @ beta
    degrees_of_freedom = x.size - design.shape[1]
    residual_variance = np.sum(residual ** 2) / degrees_of_freedom
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    parameter_std = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    return beta[0], beta[1], parameter_std[0], parameter_std[1]


def fit_cot_albedo(cot, albedo):
    """Fit the COT-albedo relation by equal-weight least squares.

    Parameters
    ----------
    cot, albedo : array-like
        Binned mean COT and cloud albedo.
    Returns
    -------
    k, ln_b, k_std, ln_b_std : float
        Best-fit slope and intercept and their one-sigma standard errors.
    """
    cot = np.asarray(cot, dtype=float).ravel()
    albedo = np.asarray(albedo, dtype=float).ravel()

    if cot.size != albedo.size:
        raise ValueError("cot and albedo must have the same length.")

    mask = (
        np.isfinite(cot) & np.isfinite(albedo) &
        (cot > 0) & (albedo > 0) & (albedo < 1)
    )

    cot = cot[mask]
    albedo = albedo[mask]

    if cot.size < 3:
        return np.nan, np.nan, np.nan, np.nan

    x, y = _raw_to_fit_arrays(cot, albedo)

    try:
        return _fit_lstsq_with_uncertainty(x, y)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# ============================================================
# SBDART lookup table interpolation
# ============================================================

# 3D SBDART LUT (sza x nre(=cloud effective radius) x cot) produced by
# project0904/SBDART_LUT/run_sbdart.py.
SBDART_LUT_DIR = '/home/chenyiqi/251028_albedo_cot/project0904/SBDART_LUT'
_SBDART_3D_LUT_CACHE = {}


def _sbdart_3d_interpolator(table_folder, ocean, season):
    """Cached (sza, nre, cot) interpolator for one 3D SBDART LUT netCDF file."""
    if table_folder.endswith('vis'):
        # vis folders only contain the representative TPO/MAM LUT
        ocean, season = 'TPO', 'MAM'
    base = f'{SBDART_LUT_DIR}/{table_folder}'
    path = f'{base}/albedo_cot_cer_sza_LUT_{ocean}_{season}.nc'
    if not os.path.exists(path):
        # dcp-style folders store one representative (TPO/MAM) LUT for everything
        fallback = f'{base}/albedo_cot_cer_sza_LUT_TPO_MAM.nc'
        if os.path.exists(fallback):
            path = fallback
        else:
            return None
    if path not in _SBDART_3D_LUT_CACHE:
        with xr.open_dataset(path) as ds:
            grid = ds['albedo'].values.astype(float)
            axes = (
                ds['sza'].values.astype(float),
                ds['nre'].values.astype(float),
                ds['cot'].values.astype(float),
            )
        _SBDART_3D_LUT_CACHE[path] = RegularGridInterpolator(
            axes, grid, method='linear', bounds_error=False, fill_value=np.nan
        )
    return _SBDART_3D_LUT_CACHE[path]


def cot_to_albedo(cot, method, miu=None, sza=None, cer=None, table_folder='dcp', ocean=None, season=None):
    """
    Compute cloud albedo from COT using various methods.

    Parameters
    ----------
    cot : array-like
        Cloud optical thickness.
    method : str
        'sbdart', 'analy'.
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
    cot = np.asarray(cot, dtype=float)

    if method == 'sbdart':
        # 3D lookup by (cot, sza, cer = cloud effective radius) from the
        # netCDF LUT produced by project0904/SBDART_LUT/run_sbdart.py:
        #   SBDART_LUT/<table_folder>/albedo_cot_cer_sza_LUT_{ocean}_{season}.nc
        cot_arr = np.atleast_1d(np.asarray(cot, dtype=float))
        interpolator = _sbdart_3d_interpolator(table_folder, ocean, season)
        if interpolator is None:
            return np.full(cot_arr.shape, np.nan)
        if np.ndim(sza) == 0:
            sza_arr = np.full_like(cot_arr, float(sza))
        else:
            sza_arr = np.asarray(sza, dtype=float).ravel()
        if np.ndim(cer) == 0:
            cer_arr = np.full_like(cot_arr, float(cer))
        else:
            cer_arr = np.asarray(cer, dtype=float).ravel()
        points = np.column_stack([sza_arr, cer_arr, cot_arr])
        return interpolator(points).reshape(cot_arr.shape)

    if method == 'analy':
        g = 0.85
        b = (1 - g) / 2 / miu
        return b * cot / (1 + b * cot)

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
    return rf'$\mathbf{{{letter}}}$'
