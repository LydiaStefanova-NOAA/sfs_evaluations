"""
Lead-dependent climatology and anomaly calculation utilities.
"""
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


#def _linear_detrend(
#    da: xr.DataArray, 
#    dim: str = "year",
#    clim_years: tuple[int, int] | None = (1991, 2020),
#) -> xr.DataArray:
#    """Subtract a linear trend along the specified dimension using xarray polyfit/polyval."""
#    # Fit 1st degree polynomial across time dimension
#    poly_coeffs = da.polyfit(dim=dim, deg=1)
#    trend = xr.polyval(da[dim], poly_coeffs.polyfit_coefficients)
#    return da - trend

def _linear_detrend(
    obj: xr.Dataset | xr.DataArray,
    dim: str = "year",
    clim_years: tuple[int, int] | None = (1991, 2020),
) -> xr.Dataset | xr.DataArray:
    """
    Subtract a linear trend along the specified dimension using xarray polyfit/polyval.

    Raises KeyError if the requested dimension is not found in the input object.
    """
    if dim not in obj.dims:
        raise KeyError(
            f"Dimension '{dim}' not found in input object. Available dimensions: {list(obj.dims)}"
        )

    obj_fit = obj

    if clim_years and dim in obj.coords:
        start_yr, end_yr = clim_years
        coord_vals = obj[dim].values

        # Handle integer vs datetime coordinate types correctly
        if np.issubdtype(coord_vals.dtype, np.integer):
            obj_fit = obj.sel({dim: slice(start_yr, end_yr)})
        else:
            obj_fit = obj.sel({dim: slice(str(start_yr), str(end_yr))})

    # Fit 1st degree polynomial across specified baseline window
    poly_coeffs = obj_fit.polyfit(dim=dim, deg=1)

    # Evaluate trend polynomial across the full dataset/array time dimension
    trend = xr.polyval(obj[dim], poly_coeffs.polyfit_coefficients)

    return obj - trend

def compute_climatology(
    obj: xr.Dataset | xr.DataArray,
    clim_years: tuple[int, int] | None = (1991, 2020),
) -> xr.Dataset | xr.DataArray:
    """
    Compute climatology averaged across time/init years and members.
    """
    obj_clim = obj

    # Detect active time dimension
    time_dim = next((d for d in ["year", "init", "time"] if d in obj.dims), None)

    if clim_years and time_dim and time_dim in obj.coords:
        start_yr, end_yr = clim_years
        coord_vals = obj[time_dim].values
        
        # Handle integer vs datetime coordinate types correctly
        if np.issubdtype(coord_vals.dtype, np.integer):
            obj_clim = obj.sel({time_dim: slice(start_yr, end_yr)})
        else:
            obj_clim = obj.sel({time_dim: slice(str(start_yr), str(end_yr))})

    # Average over time/init and ensemble members
    dims_to_mean = [d for d in ["year", "init", "time", "member", "number", "ens"] if d in obj_clim.dims]
    climatology = obj_clim.mean(dim=dims_to_mean, skipna=True) if dims_to_mean else obj_clim

    return climatology


def compute_anomalies(
    obj: xr.Dataset | xr.DataArray,
    clim_years: tuple[int, int] | None = (1991, 2020),
    climatology: xr.Dataset | xr.DataArray | None = None,
) -> tuple[xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray]:
    """
    Compute anomalies by subtracting baseline climatology.
    """
    if climatology is None:
        climatology = compute_climatology(obj, clim_years=clim_years)

    anomalies = obj - climatology
    return anomalies, climatology
