"""
Shared spatial + temporal alignment helpers for diagnostics.

Purpose:
- Keep coordinate conventions consistent (lon wrapping, monotonic sorting)
- Align observation fields to model grid explicitly
- Standardize year coordinates and overlapping evaluation windows
- Provide lightweight grid diagnostics for debugging map discontinuities
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def canonicalize_lonlat(da: xr.DataArray) -> xr.DataArray:
    """
    Canonicalize horizontal coordinates for robust interpolation/plotting.

    - lon -> [0, 360) if lon is 1D
    - sort lon ascending (1D)
    - sort lat ascending (1D)
    """
    out = da

    if "lon" in out.coords and out["lon"].ndim == 1:
        out = out.assign_coords(lon=(out["lon"] % 360))
        out = out.sortby("lon")

    if "lat" in out.coords and out["lat"].ndim == 1:
        out = out.sortby("lat")

    return out


def standardize_year_dim(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure temporal coordinate is named 'year' and numeric years.
    """
    out = da
    if "init" in out.dims and "year" not in out.dims:
        out = out.rename({"init": "year"})

    if "year" not in out.coords:
        raise ValueError(f"DataArray has no 'year' coordinate after standardization. dims={out.dims}")

    if np.issubdtype(out["year"].dtype, np.datetime64):
        out = out.assign_coords(year=out["year"].dt.year)

    return out


def align_obs_to_sfs_grid(obs_da: xr.DataArray, sfs_da: xr.DataArray, method: str = "linear") -> xr.DataArray:
    """
    Interpolate OBS field onto SFS lat/lon grid (strict collocation).
    """
    if not all(c in sfs_da.coords for c in ["lat", "lon"]):
        raise ValueError("SFS DataArray missing lat/lon coordinates for alignment.")
    if not all(c in obs_da.coords for c in ["lat", "lon"]):
        raise ValueError("OBS DataArray missing lat/lon coordinates for alignment.")

    obs_can = canonicalize_lonlat(obs_da)
    sfs_can = canonicalize_lonlat(sfs_da)

    return obs_can.interp(lat=sfs_can["lat"], lon=sfs_can["lon"], method=method)


def select_common_eval_years(
    sfs_da: xr.DataArray,
    obs_da: xr.DataArray,
    start_year: int,
    end_year: int,
) -> Tuple[xr.DataArray, xr.DataArray, list[int]]:
    """
    Intersect SFS and OBS years and keep only requested window.
    """
    common_years = np.intersect1d(sfs_da.year.values, obs_da.year.values)
    eval_years = [int(y) for y in common_years if int(start_year) <= int(y) <= int(end_year)]

    if len(eval_years) == 0:
        raise ValueError(f"No overlapping years found in range {start_year}-{end_year}.")

    sfs_sel = sfs_da.sel(year=eval_years)
    obs_sel = obs_da.sel(year=eval_years)
    return sfs_sel, obs_sel, eval_years


def grid_report(tag: str, da: xr.DataArray) -> None:
    """
    Optional debug log of grid geometry.
    """
    logger.info(f"[{tag}] dims={da.dims}, sizes={dict(da.sizes)}")

    if "lon" in da.coords and da["lon"].ndim == 1 and da.sizes.get("lon", 0) > 1:
        dlon = da["lon"].diff("lon").values
        logger.info(
            f"[{tag}] lon: min={float(da.lon.min()):.3f}, max={float(da.lon.max()):.3f}, "
            f"median_step={float(np.nanmedian(np.abs(dlon))):.3f}, monotonic_inc={bool(np.all(dlon > 0))}"
        )

    if "lat" in da.coords and da["lat"].ndim == 1 and da.sizes.get("lat", 0) > 1:
        dlat = da["lat"].diff("lat").values
        logger.info(
            f"[{tag}] lat: min={float(da.lat.min()):.3f}, max={float(da.lat.max()):.3f}, "
            f"median_step={float(np.nanmedian(np.abs(dlat))):.3f}, monotonic_inc={bool(np.all(dlat > 0))}"
        )


def nan_report(tag: str, da: xr.DataArray) -> None:
    """
    Optional debug log of NaN coverage and value range.
    """
    arr = da.values
    nan_frac = float(np.isnan(arr).mean())
    vmin = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else np.nan
    vmax = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else np.nan
    logger.info(f"[{tag}] nan_frac={nan_frac:.4f}, min={vmin:.6g}, max={vmax:.6g}")
