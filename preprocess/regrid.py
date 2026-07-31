"""
Spatial regridding functions using xESMF for sfs_evaluations.
Decoupled from physical domain names to support flexible grid resolutions.
"""
import os
import logging
import numpy as np
import xarray as xr
import xesmf as xe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEIGHTS_DIR = "preprocess/weights"
_REGRIDDER_CACHE = {}


def _get_grid_label(ds: xr.Dataset) -> str:
    """
    Infer a clean, standardized grid resolution label from dataset dimensions.
    Returns strings like '0.25deg', '1.0deg', or 'tripolar_0.25deg'.
    """
    nlat = ds.sizes.get("lat", ds.sizes.get("y", ds.sizes.get("yh", ds.sizes.get("yq", None))))
    nlon = ds.sizes.get("lon", ds.sizes.get("x", ds.sizes.get("xh", ds.sizes.get("xq", None))))

    # Curvilinear / Tripolar MOM6 raw grids
    if nlat == 1080 and nlon == 1440:
        return "tripolar_0.25deg"
    elif nlat == 721 and nlon == 1440:
        return "0.25deg"
    elif nlat == 181 and nlon == 360:
        return "1.0deg"
    elif nlat == 361 and nlon == 720:
        return "0.5deg"
    
    # Fallback to explicit dimension shape if non-standard
    return f"{nlat}x{nlon}"


def get_target_grid(target_res: str) -> xr.Dataset:
    """
    Generate target latitude/longitude grid specification datasets.
    """
    res_key = str(target_res).lower().replace("deg", "").strip()

    if res_key in ["0.25", "1/4"]:
        lat = np.linspace(-90.0, 90.0, 721)
        lon = np.linspace(0.0, 359.75, 1440)
        d_lat, d_lon = 0.25, 0.25
    elif res_key in ["1.0", "1", "1deg"]:
        lat = np.linspace(-90.0, 90.0, 181)
        lon = np.linspace(0.0, 359.0, 360)
        d_lat, d_lon = 1.0, 1.0
    elif res_key in ["0.5", "1/2"]:
        lat = np.linspace(-90.0, 90.0, 361)
        lon = np.linspace(0.0, 359.5, 720)
        d_lat, d_lon = 0.5, 0.5
    else:
        raise ValueError(f"Unsupported target_res: '{target_res}'. Options: '1.0deg', '0.25deg', '0.5deg'.")

    ds_target = xr.Dataset(
        coords={
            "lat": (["lat"], lat, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": (["lon"], lon, {"units": "degrees_east", "long_name": "longitude"}),
        }
    )

    lat_b = np.concatenate([[-90.0], (lat[:-1] + lat[1:]) / 2.0, [90.0]])
    lon_b = np.concatenate([[lon[0] - d_lon / 2.0], (lon[:-1] + lon[1:]) / 2.0, [lon[-1] + d_lon / 2.0]])
    
    return ds_target.assign_coords(lat_b=(["lat_b"], lat_b), lon_b=(["lon_b"], lon_b))


def get_regridder(
    ds_in: xr.Dataset,
    target_res: str = "1.0deg",
    method: str = "bilinear",
    weights_dir: str = WEIGHTS_DIR,
    reuse_weights: bool = True,
) -> xe.Regridder:
    """
    Retrieve or create an xESMF Regridder object with symmetric naming conventions.
    """
    os.makedirs(weights_dir, exist_ok=True)
    ds_target = get_target_grid(target_res)

    # Clean resolution labels
    src_label = _get_grid_label(ds_in)
    tgt_label = str(target_res).lower().replace("deg", "") + "deg"

    # Symmetric, self-describing filename
    weight_filename = f"weights_{src_label}_to_{tgt_label}_{method}.nc"
    weight_path = os.path.join(weights_dir, weight_filename)

    cache_key = (src_label, tgt_label, method)

    if cache_key in _REGRIDDER_CACHE:
        return _REGRIDDER_CACHE[cache_key]

    should_reuse = reuse_weights and os.path.exists(weight_path)

    if should_reuse:
        logger.info(f"Loading cached xESMF weights: {weight_path}")
    else:
        logger.info(f"Computing xESMF weights ({method}): {src_label} -> {tgt_label} -> {weight_path}")

    regridder = xe.Regridder(
        ds_in,
        ds_target,
        method=method,
        filename=weight_path,
        reuse_weights=should_reuse,
        periodic=True,
    )

    _REGRIDDER_CACHE[cache_key] = regridder
    return regridder


def regrid_dataset(
    ds_in: xr.Dataset,
    target_res: str = "1.0deg",
    method: str = "bilinear",
    weights_dir: str = WEIGHTS_DIR,
    reuse_weights: bool = True,
) -> xr.Dataset:
    """
    Remap input dataset to a specified target resolution using xESMF.
    """
    regridder = get_regridder(
        ds_in,
        target_res=target_res,
        method=method,
        weights_dir=weights_dir,
        reuse_weights=reuse_weights
    )

    # Prevent chunk explosion by keeping spatial target dims in a single chunk
    output_chunks = {"lat": -1, "lon": -1}

    return regridder(ds_in, keep_attrs=True, output_chunks=output_chunks)
