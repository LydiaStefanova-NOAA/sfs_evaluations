"""
Lead-dependent climatology and anomaly calculation utilities.
"""
import logging
import xarray as xr

logger = logging.getLogger(__name__)


def compute_climatology(
    ds: xr.Dataset,
    clim_years: tuple[int, int] | None = (1991, 2020),
) -> xr.Dataset:
    """
    Compute lead-dependent climatology averaged across init years and members.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing 'init', 'lead', and 'member' dimensions.
    clim_years : tuple of int, optional
        Year range for baseline climatology (inclusive). Default is (1991, 2020).

    Returns
    -------
    xr.Dataset
        Climatology indexed by ('lead', 'lat', 'lon').
    """
    ds_clim = ds

    if clim_years and "init" in ds.coords:
        start_yr, end_yr = clim_years
        logger.info(f"Subsetting climatology baseline to years {start_yr}-{end_yr}...")
        ds_clim = ds.sel(init=slice(str(start_yr), str(end_yr)))

    # Average over initialization years and ensemble members
    dims_to_mean = [d for d in ["init", "member"] if d in ds_clim.dims]
    climatology = ds_clim.mean(dim=dims_to_mean, skipna=True)

    return climatology


def compute_anomalies(
    ds: xr.Dataset,
    clim_years: tuple[int, int] | None = (1991, 2020),
    climatology: xr.Dataset | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Compute lead-dependent forecast anomalies by subtracting baseline climatology.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        (anomalies_ds, climatology_ds)
    """
    if climatology is None:
        climatology = compute_climatology(ds, clim_years=clim_years)

    anomalies = ds - climatology
    return anomalies, climatology
