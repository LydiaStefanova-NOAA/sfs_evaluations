"""
Persistent Disk Caching Engine for SFS and Observational Climatologies & Trends.
Files are permanently saved to /scratch3/.../cache/ as NetCDF files.
"""
import os
import logging
import pandas as pd
import numpy as np
import xarray as xr

from preprocess.climatology import compute_climatology
from preprocess.detrend import detrend_dim

logger = logging.getLogger(__name__)

# Default persistent disk directory on scratch
CACHE_DIR = "/scratch3/NCEPDEV/global/Lydia.B.Stefanova/project/SFSbeta/data/cache"


def _get_cache_path(filename: str, cache_dir: str = CACHE_DIR) -> str:
    """Ensure directory exists and return absolute file path."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, filename)


# =====================================================================
# 1. SFS Model Climatology & Ensemble-Mean Trend (Persistent Disk)
# =====================================================================

def get_or_compute_sfs_climatology(
    ds_sfs: xr.Dataset,
    domain: str,
    init_month: str,
    clim_years: tuple[int, int] = (1991, 2020),
    cache_dir: str = CACHE_DIR,
    force_recompute: bool = False,
) -> xr.Dataset:
    """
    Load SFS lead-dependent climatology from disk if available; 
    otherwise compute across (init x member), save permanently to disk, and return.
    """
    fname = f"sfs_clim_{clim_years[0]}-{clim_years[1]}_init{init_month}_{domain}.nc"
    cache_path = _get_cache_path(fname, cache_dir)

    if os.path.exists(cache_path) and not force_recompute:
        logger.info(f"⚡ Loading cached SFS Climatology from DISK: {cache_path}")
        return xr.open_dataset(cache_path)

    logger.info(f"⚙️ Computing SFS Climatology ({clim_years[0]}-{clim_years[1]})...")
    ds_clim = compute_climatology(ds_sfs, clim_years=clim_years)

    # Save permanently to disk
    ds_clim.to_netcdf(cache_path)
    logger.info(f"💾 Saved SFS Climatology to DISK: {cache_path}")
    return ds_clim


def get_or_compute_sfs_trend(
    ds_sfs: xr.Dataset,
    domain: str,
    init_month: str,
    cache_dir: str = CACHE_DIR,
    force_recompute: bool = False,
) -> xr.Dataset:
    """
    Compute linear trend fitted on the INTERANNUAL ENSEMBLE MEAN, 
    and save the trend line permanently to disk.
    """
    fname = f"sfs_ensmean_trend_init{init_month}_{domain}.nc"
    cache_path = _get_cache_path(fname, cache_dir)

    if os.path.exists(cache_path) and not force_recompute:
        logger.info(f"⚡ Loading cached SFS Ensemble-Mean Trend from DISK: {cache_path}")
        return xr.open_dataset(cache_path)

    logger.info("⚙️ Computing SFS Ensemble-Mean Linear Trend...")
    # 1. Compute Ensemble Mean across members first (preserving 'init' dimension)
    ens_mean = ds_sfs.mean(dim="member", skipna=True)

    # 2. Fit polyfit along 'init' years
    x = xr.DataArray(np.arange(ens_mean.sizes["init"]), dims=["init"], coords={"init": ens_mean["init"]})
    fit = ens_mean.polyfit(dim="init", deg=1)

    # Rename polyfit variables back to original names
    fit_rename = {v: v.replace("_polyfit_coefficients", "") for v in fit.data_vars if "_polyfit_coefficients" in v}
    fit = fit.rename(fit_rename)

    # 3. Calculate 3D trend field (init, lead, lat, lon)
    trend = xr.polyval(x, fit)

    # Save permanently to disk
    trend.to_netcdf(cache_path)
    logger.info(f"💾 Saved SFS Ensemble-Mean Trend to DISK: {cache_path}")
    return trend


# =====================================================================
# 2. Observational Climatology Aligned to SFS Leads
# =====================================================================

def get_or_compute_obs_climatology_for_sfs(
    ds_obs: xr.Dataset,
    obs_name: str,  # 'era5' or 'oras5'
    init_month: str,
    clim_years: tuple[int, int] = (1991, 2020),
    num_leads: int = 12,
    cache_dir: str = CACHE_DIR,
    force_recompute: bool = False,
) -> xr.Dataset:
    """
    Compute 12-month observational climatology (1991-2020), map it to the exact
    calendar target months corresponding to (init_month + lead), and cache to disk.
    
    Output has dimensions: (lead, lat, lon)
    """
    fname = f"{obs_name}_clim_{clim_years[0]}-{clim_years[1]}_mapped_init{init_month}.nc"
    cache_path = _get_cache_path(fname, cache_dir)

    if os.path.exists(cache_path) and not force_recompute:
        logger.info(f"⚡ Loading cached Lead-Aligned {obs_name.upper()} Climatology from DISK: {cache_path}")
        return xr.open_dataset(cache_path)

    logger.info(f"⚙️ Computing {obs_name.upper()} 12-month Climatology ({clim_years[0]}-{clim_years[1]})...")
    
    # 1. Subset baseline years
    ds_baseline = ds_obs.sel(time=slice(str(clim_years[0]), str(clim_years[1])))

    # 2. Compute 12-month climatology (grouped by calendar month 1..12)
    obs_monthly_clim = ds_baseline.groupby("time.month").mean(dim="time", skipna=True)

    # 3. Map leads (0..11) for this init_month to calendar target months (1..12)
    init_m = int(init_month)
    target_months = [((init_m - 1 + lead) % 12) + 1 for lead in range(num_leads)]

    # 4. Extract target months and reassign 'month' dimension to SFS 'lead'
    obs_lead_clim = obs_monthly_clim.sel(month=target_months)
    obs_lead_clim = obs_lead_clim.rename({"month": "lead"}).assign_coords(lead=list(range(num_leads)))

    # Save permanently to disk
    obs_lead_clim.to_netcdf(cache_path)
    logger.info(f"💾 Saved Lead-Aligned {obs_name.upper()} Climatology to DISK: {cache_path}")
    return obs_lead_clim
