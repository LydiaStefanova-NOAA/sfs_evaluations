"""
Ingestion and coordinate standardization functions for ERA5 atmospheric reanalysis.
"""
import os
import logging
import xarray as xr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ERA5_DEFAULT_PATHS = {
    "local_scratch": "/scratch3/NCEPDEV/global/Lydia.B.Stefanova/project/SFSbeta/data/era5_monthly_1deg_1991-2022.zarr",
    "gcp_public": "gs://gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr",
}

# Standard variable alias mapping (Standard name -> ERA5 Zarr inventory names)
ERA5_VAR_MAP = {
    "TMP2m": ["2m_temperature", "tmp2m", "t2m"],
    "MSLP": ["mean_sea_level_pressure", "msl", "prmsl"],
    "PRATE": ["total_precipitation", "tp"],
    "U10m": ["10m_u_component_of_wind", "u10"],
    "V10m": ["10m_v_component_of_wind", "v10"],
    "Z500": ["z500", "geopotential_500"],
    "Z200": ["z200"],
    "Z700": ["z700"],
    "Z850": ["z850"],
    "T850": ["t850"],
    "T200": ["t200"],
    "U850": ["u850"],
    "V850": ["v850"],
    "U200": ["u200"],
    "V200": ["v200"],
}


def open_era5(path_or_url: str = None) -> xr.Dataset:
    """
    Open ERA5 dataset from local scratch disk or Google Cloud Storage.

    Parameters
    ----------
    path_or_url : str, optional
        Custom file path or GCS URI. If None, checks local scratch disk first,
        falling back to GCP public Zarr.
    """
    if path_or_url is None:
        local_path = ERA5_DEFAULT_PATHS["local_scratch"]
        if os.path.exists(local_path):
            target = local_path
        else:
            logger.info("Local ERA5 scratch store not found. Falling back to GCP public store.")
            target = ERA5_DEFAULT_PATHS["gcp_public"]
    else:
        target = path_or_url

    logger.info(f"Opening ERA5 store: {target}")

    if target.startswith("gs://"):
        return xr.open_zarr(target, storage_options={"token": "anon"})
    
    return xr.open_zarr(target)


def standardize_era5_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize ERA5 spatial and temporal coordinate names.
    """
    rename_map = {}

    for lat_name in ["latitude", "lat"]:
        if lat_name in ds.coords or lat_name in ds.variables:
            rename_map[lat_name] = "lat"
            break

    for lon_name in ["longitude", "lon"]:
        if lon_name in ds.coords or lon_name in ds.variables:
            rename_map[lon_name] = "lon"
            break

    for time_name in ["valid_time", "initial_time"]:
        if time_name in ds.coords or time_name in ds.variables:
            rename_map[time_name] = "time"
            break

    if rename_map:
        ds = ds.rename(rename_map)

    # Ensure latitude is sorted south-to-north
    if "lat" in ds.coords and ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")

    return ds


def get_era5_data(
    requested_vars: list[str] | str = None, 
    path_or_url: str = None,
    **kwargs,
) -> xr.Dataset:
    """
    High-level function to load, clean, and subset ERA5 atmospheric data.
    """
    if requested_vars is None and "var_name" in kwargs:
        requested_vars = kwargs["var_name"]

    if isinstance(requested_vars, str):
        requested_vars = [requested_vars]

    ds = open_era5(path_or_url=path_or_url)
    ds = standardize_era5_coords(ds)

    if requested_vars:
        matched_vars = []
        rename_map = {}

        for req in requested_vars:
            if req in ds.data_vars:
                matched_vars.append(req)
            else:
                candidates = ERA5_VAR_MAP.get(req, [req])
                found = False
                for cand in candidates:
                    if cand in ds.data_vars:
                        matched_vars.append(cand)
                        rename_map[cand] = req  # Standardize variable name to pipeline standard
                        found = True
                        break
                if not found:
                    logger.warning(f"Variable '{req}' not found in ERA5 store.")

        if matched_vars:
            ds = ds[matched_vars]
            if rename_map:
                ds = ds.rename(rename_map)

    return ds
