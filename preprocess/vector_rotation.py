"""
Trigonometric vector rotation routines for tripolar ocean/ice grids.
"""
import os
import logging
import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Primary HPC path for MOM6/CICE static ocean grid info
DEFAULT_STATIC_GRID_PATH = "/scratch4/BMC/gsienkf/Philip.Pegion/replay_evaluation/ocn_data/ocn_grid_info.nc"


def load_static_grid_rotation(static_path: str = DEFAULT_STATIC_GRID_PATH) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Load exact analytical cosrot and sinrot from MOM6 static grid specification file.
    """
    if not os.path.exists(static_path):
        raise FileNotFoundError(f"Static grid file not found at path: {static_path}")

    logger.info(f"Loading analytical grid rotation parameters from: {static_path}")
    ds_static = xr.open_dataset(static_path)

    # 1. Search for pre-calculated cosine and sine variables
    cos_var = next((v for v in ["cosrot", "cos_rot", "COSROT"] if v in ds_static), None)
    sin_var = next((v for v in ["sinrot", "sin_rot", "SINROT"] if v in ds_static), None)

    if cos_var and sin_var:
        logger.info(f"Found analytical rotation arrays: '{cos_var}', '{sin_var}'")
        return ds_static[cos_var], ds_static[sin_var]

    # 2. Search for raw angle variable (angle_dx, anglet, angle, etc.)
    angle_var = next((v for v in ["angle_dx", "anglet", "angle", "rot_angle"] if v in ds_static), None)
    if angle_var:
        logger.info(f"Found analytical grid angle array: '{angle_var}'")
        angle = ds_static[angle_var]
        units = angle.attrs.get("units", "").lower()
        
        if "deg" in units or np.nanmax(np.abs(angle.values)) > 2 * np.pi:
            angle_rad = np.radians(angle)
        else:
            angle_rad = angle
            
        return np.cos(angle_rad), np.sin(angle_rad)

    raise KeyError(
        f"Could not locate rotation arrays ('cosrot'/'sinrot' or 'angle_dx') in {static_path}. "
        f"Available variables: {list(ds_static.data_vars.keys())}"
    )


def compute_grid_angle_fallback(lat: xr.DataArray, lon: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    FALLBACK ONLY: Approximate grid angle using central differences if static file is missing.
    """
    logger.warning(
        "⚠️ Using finite-difference fallback for tripolar rotation angle. "
        "For publication-quality metrics, supply the analytical static grid file."
    )
    lat_vals = lat.values
    lon_vals = lon.values

    dlat_dx = np.gradient(lat_vals, axis=-1)
    dlon_dx = np.gradient(lon_vals, axis=-1)
    dlon_dx = (dlon_dx + 180.0) % 360.0 - 180.0

    dx = np.radians(dlon_dx) * np.cos(np.radians(lat_vals))
    dy = np.radians(dlat_dx)

    angle_rad = np.arctan2(dy, dx)
    return xr.DataArray(np.cos(angle_rad), coords=lat.coords), xr.DataArray(np.sin(angle_rad), coords=lat.coords)


def rotate_tripolar_vectors(
    u_grid: xr.DataArray,
    v_grid: xr.DataArray,
    static_grid_path: str = DEFAULT_STATIC_GRID_PATH,
    cos_rot: xr.DataArray = None,
    sin_rot: xr.DataArray = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Rotate grid-aligned vector components (u_grid, v_grid) on a tripolar grid
    to true Earth-oriented components (u_east, v_north).
    """
    # 1. User provided cos_rot and sin_rot directly
    if cos_rot is not None and sin_rot is not None:
        pass
    # 2. Try loading exact analytical parameters from static file
    elif static_grid_path and os.path.exists(static_grid_path):
        try:
            cos_rot, sin_rot = load_static_grid_rotation(static_grid_path)
        except Exception as e:
            logger.warning(f"Could not load static file ({e}); falling back to coordinate derivation.")
            lat = u_grid.coords.get("lat") if "lat" in u_grid.coords else u_grid.coords.get("geolat")
            lon = u_grid.coords.get("lon") if "lon" in u_grid.coords else u_grid.coords.get("geolon")
            cos_rot, sin_rot = compute_grid_angle_fallback(lat, lon)
    # 3. Fallback to finite difference calculation
    else:
        lat = u_grid.coords.get("lat") if "lat" in u_grid.coords else u_grid.coords.get("geolat")
        lon = u_grid.coords.get("lon") if "lon" in u_grid.coords else u_grid.coords.get("geolon")
        cos_rot, sin_rot = compute_grid_angle_fallback(lat, lon)

    # -------------------------------------------------------------
    # Ensure static grid dimension names match the input DataArray's spatial dimensions
    # (e.g., mapping 'yq'/'xq' or 'yh'/'xh' to 'lat'/'lon') to perform element-wise
    # multiplication rather than outer product broadcasting

    spatial_dims = u_grid.dims[-2:]  # Grab the last two dimensions (e.g., 'lat', 'lon')

    if cos_rot.dims != spatial_dims:
        cos_rot = cos_rot.rename({cos_rot.dims[0]: spatial_dims[0], cos_rot.dims[1]: spatial_dims[1]})
        sin_rot = sin_rot.rename({sin_rot.dims[0]: spatial_dims[0], sin_rot.dims[1]: spatial_dims[1]})

    # Standard Cartesian rotation
    u_east = u_grid * cos_rot - v_grid * sin_rot
    v_north = u_grid * sin_rot + v_grid * cos_rot

    u_east.name = "u_east"
    u_east.attrs = u_grid.attrs.copy()
    u_east.attrs["long_name"] = "Eastward Sea Ice Velocity"

    v_north.name = "v_north"
    v_north.attrs = v_grid.attrs.copy()
    v_north.attrs["long_name"] = "Northward Sea Ice Velocity"

    return u_east, v_north
