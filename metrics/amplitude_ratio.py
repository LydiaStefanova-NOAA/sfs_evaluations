"""
ECMWF Anomaly Amplitude Ratio Metric Functions
"""
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

def compute_amplitude_ratio(
    sfs_anom: xr.DataArray,
    obs_anom: xr.DataArray,
    year_dim: str = "year",
    member_dim: str = "member",
    mode: str = "members",
    detrended: bool = True,  # Indicates whether input anomalies were detrended upstream
    eps: float = 1e-12,
) -> xr.DataArray:

    # Set degrees of freedom without re-running detrending logic
    ddof = 2 if detrended else 1

    std_obs = obs_anom.std(dim=year_dim, ddof=ddof, skipna=True)
    
    if mode == "members":
        pool_dims = [year_dim, member_dim] if member_dim in sfs_anom.dims else [year_dim]
        std_mod = sfs_anom.std(dim=pool_dims, ddof=ddof, skipna=True)
    else:
        ens_mean = sfs_anom.mean(dim=member_dim, skipna=True) if member_dim in sfs_anom.dims else sfs_anom
        std_mod = ens_mean.std(dim=year_dim, ddof=ddof, skipna=True)

    return std_mod / std_obs.where(std_obs > eps)
