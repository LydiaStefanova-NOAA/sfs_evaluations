"""
Ingestion and coordinate standardization functions for ECMWF ORAS5 reanalysis.
"""
import os
import logging
import xarray as xr
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ORAS5_ENDPOINTS = {
    "consolidated": "https://arco.datastores.ecmwf.int/cadl-arco-geo-001/arco/reanalysis_oras5/consolidated/geoChunked.zarr",
    "operational": "https://arco.datastores.ecmwf.int/cadl-arco-geo-001/arco/reanalysis_oras5/operational/geoChunked.zarr",
}

# Standard variable alias mapping (Standard pipeline name -> ORAS5 inventory names)
ORAS5_VAR_MAP = {
    "SST": ["sosstsst", "sosstmet", "tos", "sst"],
    "SSS": ["sosaline", "sss", "sos"],
    "SSH": ["sossheig", "zos", "ssh"],
    "MLD_003": ["somxl030"],
    "MLD_010": ["somxl010"],
    "ocnheat": ["sohtc300"],
    "dt20c": ["so20chgt"],
    "taux": ["sozotaux"],
    "tauy": ["sometauy"],
    "aice_h": ["ileadfra"],
    "hi_h": ["iicethic"],
    "u_ice": ["iicevelu"],
    "v_ice": ["iicevelv"],
}


def _get_cds_api_key() -> str | None:
    """Detect CDS API key from environment variable or ~/.cdsapirc file."""
    if os.getenv("CDSAPI_KEY"):
        return os.getenv("CDSAPI_KEY")

    cdsapirc_path = os.path.expanduser("~/.cdsapirc")
    if os.path.exists(cdsapirc_path):
        try:
            with open(cdsapirc_path, "r") as f:
                for line in f:
                    if line.strip().startswith("key:"):
                        return line.split(":", 1)[1].strip().strip('"\'')
        except Exception as e:
            logger.warning(f"Failed reading ~/.cdsapirc: {e}")
    return None


def open_oras5(version: str = "consolidated", custom_url_or_path: str = None) -> xr.Dataset:
    """
    Open ECMWF ORAS5 Zarr store from remote ARCO endpoint or local store.
    """
    target = custom_url_or_path or ORAS5_ENDPOINTS.get(version, ORAS5_ENDPOINTS["consolidated"])
    logger.info(f"Opening ORAS5 store: {target}")

    if target.startswith(("http://", "https://")):
        cds_key = _get_cds_api_key()
        storage_opts = {"headers": {"Authorization": f"Bearer {cds_key}"}} if cds_key else {}
        return xr.open_zarr(target, storage_options=storage_opts)
    
    return xr.open_zarr(target)


def standardize_oras5_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize ORAS5 spatial/temporal coordinate names and align longitudes to 0..360.
    """
    rename_map = {}
    
    # Standardize coordinate names
    for lat_name in ["nav_lat", "latitude", "lat"]:
        if lat_name in ds.coords or lat_name in ds.variables:
            rename_map[lat_name] = "lat"
            break

    for lon_name in ["nav_lon", "longitude", "lon"]:
        if lon_name in ds.coords or lon_name in ds.variables:
            rename_map[lon_name] = "lon"
            break

    for time_name in ["time_counter", "valid_time"]:
        if time_name in ds.coords or time_name in ds.variables:
            rename_map[time_name] = "time"
            break

    if rename_map:
        ds = ds.rename(rename_map)

    # Convert longitudes to [0, 360) and sort monotonically
    if "lon" in ds.coords:
        # Wrap [-180, 180] -> [0, 360]
        lon_360 = np.where(ds["lon"].values < 0, ds["lon"].values + 360.0, ds["lon"].values)
        ds = ds.assign_coords(lon=lon_360)
        ds = ds.sortby("lon")

    # Ensure latitude is sorted south-to-north
    if "lat" in ds.coords and ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")

    return ds

def get_oras5_data(
    requested_vars: list[str] | str = None,
    version: str = "consolidated", 
    custom_url_or_path: str = None,
    **kwargs,
) -> xr.Dataset:
    """
    High-level function to load, clean, and subset ORAS5 reanalysis data.
    """
    if requested_vars is None and "var_name" in kwargs:
        requested_vars = kwargs["var_name"]

    if isinstance(requested_vars, str):
        requested_vars = [requested_vars]

    ds = open_oras5(version=version, custom_url_or_path=custom_url_or_path)
    ds = standardize_oras5_coords(ds)

    if requested_vars:
        matched_vars = []
        rename_vars = {}

        for req in requested_vars:
            if req in ds.data_vars:
                matched_vars.append(req)
            else:
                candidates = ORAS5_VAR_MAP.get(req, [req])
                found = False
                for cand in candidates:
                    if cand in ds.data_vars:
                        matched_vars.append(cand)
                        rename_vars[cand] = req  # Standardize name to requested pipeline name
                        found = True
                        break
                if not found:
                    logger.warning(f"Variable '{req}' not found in ORAS5 store.")

        if matched_vars:
            ds = ds[matched_vars]
            if rename_vars:
                ds = ds.rename(rename_vars)

    return ds
