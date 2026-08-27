"""
Compute climatology-relative, (optionally) detrended anomalies immediately
after fields are read in, so every downstream metric (ACC, SNR, variance
ratios) operates on one consistent anomaly definition instead of each
metric independently re-deriving climatology offsets and/or trends.
"""
import logging
import xarray as xr

logger = logging.getLogger(__name__)


def _linear_detrend(da: xr.DataArray, dim: str = "year") -> xr.DataArray:
    """Subtract a linear trend along the specified dimension using xarray polyfit/polyval."""
    poly_coeffs = da.polyfit(dim=dim, deg=1)
    trend = xr.polyval(da[dim], poly_coeffs.polyfit_coefficients)
    return da - trend


def compute_obs_anomaly(
    obs_da: xr.DataArray,
    obs_clim: xr.DataArray,
    year_dim: str = "year",
    detrend: bool = True,
) -> xr.DataArray:
    """
    Subtract climatology, then (optionally) linearly detrend, for an
    observational field.

    Call this once, right after the obs field and its climatology are read
    in and aligned to the SFS grid. Every downstream metric should consume
    the result rather than re-deriving anomalies itself.
    """
    obs_anom = obs_da - obs_clim
    if detrend:
        obs_anom = _linear_detrend(obs_anom, dim=year_dim)
    return obs_anom


def compute_sfs_anomaly(
    sfs_da: xr.DataArray,
    sfs_clim: xr.DataArray,
    year_dim: str = "year",
    member_dim: str = "member",
    detrend: bool = True,
) -> xr.DataArray:
    """
    Subtract climatology, then (optionally) linearly detrend, for an SFS
    ensemble field.

    Call this once, right after the SFS field and its climatology are read
    in. Every downstream metric should consume the result rather than
    re-deriving anomalies itself.

    If detrending, the trend is fit on the ensemble-mean anomaly and that
    SAME trend is subtracted from every member (rather than detrending each
    member independently), so genuine cross-member spread is preserved
    rather than partially detrended away.
    """
    if "init" in sfs_da.dims and year_dim not in sfs_da.dims:
        sfs_da = sfs_da.rename({"init": year_dim})
    if "init" in sfs_clim.dims and year_dim not in sfs_clim.dims:
        sfs_clim = sfs_clim.rename({"init": year_dim})

    sfs_anom = sfs_da - sfs_clim

    if detrend:
        if member_dim in sfs_anom.dims:
            ens_mean_anom = sfs_anom.mean(dim=member_dim, skipna=True)
        else:
            ens_mean_anom = sfs_anom
        poly_coeffs = ens_mean_anom.polyfit(dim=year_dim, deg=1)
        trend = xr.polyval(ens_mean_anom[year_dim], poly_coeffs.polyfit_coefficients)
        sfs_anom = sfs_anom - trend

    return sfs_anom
