"""
Unified preprocessing pipeline workflows for SFS evaluations.
"""
import logging
import xarray as xr
from preprocess.time_coords import add_valid_time
from preprocess.regrid import regrid_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_ice_dataset(
    ds: xr.Dataset,
    target_res: str = "0.25deg",
    regrid_method: str = "bilinear",
    ice_threshold: float = 0.15,
) -> xr.Dataset:
    """
    Preprocess Sea Ice datasets (coordinate standardization, time alignment, masking, regridding).
    """
    logger.info("--- Starting Sea Ice Preprocessing Pipeline ---")

    # Ensure TLON/TLAT map to standard lon/lat and drop ambiguous U-grid coordinates
    if "lon" not in ds.coords and "lat" not in ds.coords:
        if "TLON" in ds.coords or "TLON" in ds.data_vars:
            logger.info("Assigning CICE T-grid (TLON, TLAT) to standard (lon, lat)...")
            ds = ds.assign_coords(lon=ds["TLON"], lat=ds["TLAT"])

    # Drop U-grid coordinates if present to prevent xESMF cf_xarray ambiguity
    drop_u_coords = [c for c in ["ULON", "ULAT"] if c in ds.coords or c in ds.data_vars]
    if drop_u_coords:
        ds = ds.drop_vars(drop_u_coords)

    if "init" in ds.coords and "lead" in ds.coords and "valid_time" not in ds.coords:
        logger.info("Step 1/2: Attaching 2D valid_time coordinate matrix...")
        ds = add_valid_time(ds)

    logger.info(f"Step 2/2: Regridding sea ice dataset to '{target_res}' using {regrid_method}...")
    ds_out = regrid_dataset(ds, target_res=target_res, method=regrid_method)

    if "aice_h" in ds_out:
        logger.info(f"Applying WMO {ice_threshold*100}% ice concentration mask...")
        ice_mask = ds_out["aice_h"] >= ice_threshold
        for var in ds_out.data_vars:
            if var != "aice_h" and "lat" in ds_out[var].dims:
                ds_out[var] = ds_out[var].where(ice_mask)

    logger.info("--- Sea Ice Preprocessing Pipeline Complete ---")
    return ds_out

def preprocess_ocn_dataset(
    ds: xr.Dataset,
    target_res: str = "1.0deg",
    regrid_method: str = "bilinear",
) -> xr.Dataset:
    """Preprocess Ocean datasets (SST, SSS, SSH, ORAS5 verification, etc.)."""
    logger.info("--- Starting Ocean Preprocessing Pipeline ---")

    # Always standardize valid_time coordinate
    if "init" in ds.coords and "lead" in ds.coords:
        logger.info("Step 1/2: Attaching 2D valid_time coordinate matrix...")
        ds = add_valid_time(ds)

    logger.info(f"Step 2/2: Regridding ocean dataset to '{target_res}' using {regrid_method}...")
    ds_out = regrid_dataset(ds, target_res=target_res, method=regrid_method)

    logger.info("--- Ocean Preprocessing Pipeline Complete ---")
    return ds_out

def preprocess_atm_dataset(
    ds: xr.Dataset,
    target_res: str = "1.0deg",
    regrid_method: str = "bilinear",
) -> xr.Dataset:
    """
    Preprocess Atmospheric datasets (ERA5, SFS atmosphere).
    """
    logger.info("--- Starting Atmospheric Preprocessing Pipeline ---")

    if "init" in ds.coords and "lead" in ds.coords and "valid_time" not in ds.coords:
        logger.info("Step 1/2: Attaching 2D valid_time coordinate matrix...")
        ds = add_valid_time(ds)

    logger.info(f"Step 2/2: Regridding atmospheric dataset to '{target_res}' using {regrid_method}...")
    ds_out = regrid_dataset(ds, target_res=target_res, method=regrid_method)

    logger.info("--- Atmospheric Preprocessing Pipeline Complete ---")
    return ds_out
