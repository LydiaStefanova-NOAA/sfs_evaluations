"""
Grid-point linear detrending utilities.
"""
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def detrend_dim(
    ds: xr.Dataset,
    dim: str = "init",
    deg: int = 1,
) -> xr.Dataset:
    """
    Remove polynomial trend along a specified dimension (default: 'init' years)
    using xarray polyfit and polyval.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset (typically forecast anomalies).
    dim : str
        Dimension along which to fit trend. Default is 'init'.
    deg : int
        Degree of polynomial. Default is 1 (linear detrending).

    Returns
    -------
    xr.Dataset
        Detrended dataset with identical variables and dimensions as input.
    """
    if dim not in ds.dims:
        logger.warning(f"Dimension '{dim}' not found in dataset. Skipping detrending.")
        return ds

    logger.info(f"Applying linear detrending (deg={deg}) along dimension '{dim}'...")

    # Create 0-indexed numeric coordinate for polynomial evaluation
    x = xr.DataArray(np.arange(ds.sizes[dim]), dims=[dim], coords={dim: ds[dim]})

    # Fit polynomial coefficients
    fit = ds.polyfit(dim=dim, deg=deg)

    # Rename coefficient variables to match original dataset variable names
    fit_rename = {
        v: v.replace("_polyfit_coefficients", "")
        for v in fit.data_vars
        if "_polyfit_coefficients" in v
    }
    if fit_rename:
        fit = fit.rename(fit_rename)

    # Reconstruct fitted trend line
    trend = xr.polyval(x, fit)

    # Subtract trend from original dataset
    ds_detrended = ds - trend

    return ds_detrended
