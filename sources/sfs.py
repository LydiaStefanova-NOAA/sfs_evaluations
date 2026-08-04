"""
Ingestion and coordinate standardization functions for NOAA SFS reforecasts.
"""
import logging
import xarray as xr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SFS_ENDPOINTS = {
    "ocn": "s3://noaa-oar-sfsdev-pds/experiments/beta1/reforecast/{init_month}/ocn_monthly.zarr",
    "atm": "s3://noaa-oar-sfsdev-pds/experiments/beta1/reforecast/{init_month}/atm_monthly.zarr",
    "ice": "s3://noaa-oar-sfsdev-pds/experiments/beta1/reforecast/{init_month}/ice_monthly.zarr",
}

# Standard variable alias mapping (Standard name -> SFS Zarr inventory names)
SFS_VAR_MAP = {
    # Atmospheric Domain
    "TMP2m": ["tmp2m", "TMP2m", "tmpsfc"],
    "Z500": ["z500", "HGT500", "hgt500"],
    "HGT500": ["z500", "HGT500", "hgt500"],
    "Z200": ["z200", "HGT200"],
    "Z700": ["z700", "HGT700"],
    "Z850": ["z850", "HGT850"],
    "MSLP": ["prmsl", "PRMSL"],
    "PRATE": ["pratesfc", "apcpsfc", "precip"],
    "U10m": ["u10m", "u10"],
    "V10m": ["v10m", "v10"],
    "T850": ["t850"],
    "T200": ["t200"],
    "U850": ["u850"],
    "V850": ["v850"],
    "U200": ["u200"],
    "V200": ["v200"],
    # Ocean Domain
    "SST": ["SST", "sst"],
    "SSH": ["SSH", "ssh"],
    "SSS": ["so", "sss"],
    "MLD_003": ["MLD_003"],
    "MLD_0125": ["MLD_0125"],
    "dt20c": ["dt20c"],
    "ocnheat": ["ocnheat"],
    "taux": ["taux"],
    "tauy": ["tauy"],
    # Sea Ice Domain
    "aice_h": ["aice_h", "aice"],
    "hi_h": ["hi_h", "hithick"],
    "hs_h": ["hs_h"],
    "Tsfc_h": ["Tsfc_h"],
    "uvel_h": ["uvel_h"],
    "vvel_h": ["vvel_h"],
}


def open_sfs_zarr(
    init_month: str = "05",
    domain: str = "ocn",
    custom_endpoint: str = None,
) -> xr.Dataset:
    """Open SFS Zarr store from NOAA S3 bucket."""
    if custom_endpoint:
        endpoint = custom_endpoint
    else:
        month_str = f"{int(init_month):02d}"
        endpoint = SFS_ENDPOINTS.get(domain, SFS_ENDPOINTS["ocn"]).format(init_month=month_str)

    logger.info(f"Opening SFS S3 store: {endpoint}")
    return xr.open_zarr(endpoint, storage_options={"anon": True})


def standardize_sfs_coords(ds: xr.Dataset) -> xr.Dataset:
    """Standardize spatial dimension names across SFS domains to ('lat', 'lon')."""
    rename_map = {}
    if "latitude" in ds.coords or "latitude" in ds.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords or "longitude" in ds.dims:
        rename_map["longitude"] = "lon"

    return ds.rename(rename_map) if rename_map else ds


def get_sfs_data(
    init_month: str = "05",
    domain: str = "ocn",
    requested_vars: list[str] | str = None,
    custom_endpoint: str = None,
) -> xr.Dataset:
    """High-level function to load, standardize, and subset SFS reforecast data."""
    ds = open_sfs_zarr(init_month=init_month, domain=domain, custom_endpoint=custom_endpoint)
    ds = standardize_sfs_coords(ds)

    if isinstance(requested_vars, str):
        requested_vars = [requested_vars]

    if requested_vars:
        matched_vars = []
        rename_map = {}

        for req in requested_vars:
            if req in ds.data_vars:
                matched_vars.append(req)
            else:
                candidates = SFS_VAR_MAP.get(req, [req, req.lower()])
                found = False
                for cand in candidates:
                    if cand in ds.data_vars:
                        matched_vars.append(cand)
                        rename_map[cand] = req  # Standardize name to requested name
                        found = True
                        break
                if not found:
                    logger.warning(
                        f"Variable '{req}' not found in SFS '{domain}' store. Available variables: {list(ds.data_vars)}"
                    )

        if matched_vars:
            ds = ds[matched_vars]
            if rename_map:
                ds = ds.rename(rename_map)
    # Inside get_sfs_data() in sources/sfs.py:

    if "SSS" in requested_vars and "z_l" in ds.dims:
        logger.info("Variable 'SSS' requested: selecting top surface level (z_l=0)...")
        ds = ds.isel(z_l=0, drop=True)

    return ds
