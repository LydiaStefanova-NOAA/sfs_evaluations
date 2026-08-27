"""
Anomaly Correlation Coefficient (ACC) Metric Functions
"""
import logging
import numpy as np
import xarray as xr
from preprocess.climatology import _linear_detrend

logger = logging.getLogger(__name__)


#def _linear_detrend(da: xr.DataArray, dim: str = "year") -> xr.DataArray:
#    """Subtract a linear trend along the specified dimension using xarray polyfit/polyval."""
#    # Fit 1st degree polynomial across time dimension
#    poly_coeffs = da.polyfit(dim=dim, deg=1)
#    trend = xr.polyval(da[dim], poly_coeffs.polyfit_coefficients)
#    return da - trend


def compute_acc(
    sfs_da: xr.DataArray,
    obs_da: xr.DataArray,
    sfs_clim: xr.DataArray,
    obs_clim: xr.DataArray,
    year_dim: str = "year",
    member_dim: str = "member",
    detrend: bool = True,
) -> xr.DataArray:
    """
    Compute cell-wise Anomaly Correlation Coefficient (ACC) across reforecast years.

    Parameters
    ----------
    sfs_da : xr.DataArray
        SFS reforecast DataArray.
    obs_da : xr.DataArray
        Observational DataArray matching SFS target years.
    sfs_clim : xr.DataArray
        30-year SFS climatology DataArray.
    obs_clim : xr.DataArray
        30-year Observational climatology DataArray.
    year_dim : str
        Name of the interannual time dimension (default: "year").
    member_dim : str
        Name of the ensemble member dimension (default: "member").
    detrend : bool
        If True (default), linearly detrend anomalies before computing correlation.

    Returns
    -------
    xr.DataArray
        2D spatial DataArray of ACC skill values (-1.0 to 1.0).
    """
    logger.info("Computing cell-wise Anomaly Correlation Coefficient (ACC)...")

    # 1. Standardize interannual dimension name ('init' -> 'year')
    if "init" in sfs_da.dims and year_dim not in sfs_da.dims:
        sfs_da = sfs_da.rename({"init": year_dim})
    if "init" in sfs_clim.dims and year_dim not in sfs_clim.dims:
        sfs_clim = sfs_clim.rename({"init": year_dim})

    # 2. Average ensemble members if present
    if member_dim in sfs_da.dims:
        sfs_ens = sfs_da.mean(dim=member_dim, skipna=True)
    else:
        sfs_ens = sfs_da

    # 3. Compute raw anomalies relative to 1991-2020 baselines
    sfs_anom = sfs_ens - sfs_clim
    obs_anom = obs_da - obs_clim

    # 4. Align spatial grid if slight offset exists
    if sfs_anom.shape != obs_anom.shape:
        obs_anom = obs_anom.interp(
            lat=sfs_anom.lat,
            lon=sfs_anom.lon,
            method="linear"
        )

    # 5. Intersect common years
    common_years = np.intersect1d(sfs_anom[year_dim].values, obs_anom[year_dim].values)
    sfs_anom = sfs_anom.sel({year_dim: common_years})
    obs_anom = obs_anom.sel({year_dim: common_years})

    # 6. Apply Linear Detrending (Optional)
    if detrend:
        logger.info(f"Applying linear detrending along '{year_dim}' dimension...")
        sfs_anom = _linear_detrend(sfs_anom, dim=year_dim)
        obs_anom = _linear_detrend(obs_anom, dim=year_dim)

    # 7. Compute Pearson correlation across detrended years
    acc = xr.corr(sfs_anom, obs_anom, dim=year_dim)
    acc.name = "ACC"

    logger.info("✅ ACC computation complete!")
    return acc
