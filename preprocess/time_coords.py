"""
Time coordinate utilities for SFS evaluations.
"""
import logging
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def _parse_init_to_timestamp(init_val) -> pd.Timestamp:
    """Robustly parse an init coordinate value into a pandas Timestamp."""
    # Handle scalar numpy arrays
    if isinstance(init_val, np.ndarray) and init_val.ndim == 0:
        init_val = init_val.item()

    # Handle string ISO dates or numpy datetime64
    if isinstance(init_val, (np.datetime64, str)):
        return pd.to_datetime(str(init_val)[:10])

    # Handle cftime objects from NetCDF/Zarr
    if hasattr(init_val, "strftime"):
        return pd.to_datetime(init_val.strftime("%Y-%m-%d"))

    # Fallback for pandas parsing
    try:
        ts = pd.to_datetime(init_val)
        # Handle numeric YYYYMMDD integers if present
        if ts.year == 1970 and isinstance(init_val, (int, float, np.integer, np.floating)):
            s = str(int(init_val))
            if len(s) == 8:
                return pd.to_datetime(s, format="%Y%m%d")
            elif len(s) == 6:
                return pd.to_datetime(s, format="%Y%m")
        return ts
    except Exception:
        logger.warning(f"Could not parse init date '{init_val}'. Defaulting to 1991-05-01.")
        return pd.Timestamp("1991-05-01")


def add_valid_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Attach a standardized 2D 'valid_time' coordinate matrix (init x lead) as datetime64[ns].
    """
    if "init" not in ds.coords or "lead" not in ds.coords:
        return ds

    init_vals = np.atleast_1d(ds["init"].values)
    lead_vals = np.atleast_1d(ds["lead"].values)

    valid_matrix = []
    for init_val in init_vals:
        base_ts = _parse_init_to_timestamp(init_val)
        row = [base_ts + pd.DateOffset(months=int(lead)) for lead in lead_vals]
        valid_matrix.append(row)

    valid_array = np.array(valid_matrix, dtype="datetime64[ns]")

    # Determine coordinate dimensions
    if len(init_vals) == 1 and len(lead_vals) == 1:
        valid_coords = ([], valid_array.squeeze())
    elif len(init_vals) == 1:
        valid_coords = (["lead"], valid_array.squeeze(axis=0))
    elif len(lead_vals) == 1:
        valid_coords = (["init"], valid_array.squeeze(axis=1))
    else:
        valid_coords = (["init", "lead"], valid_array)

    # Drop raw integer valid_time if already present
    if "valid_time" in ds.coords or "valid_time" in ds.variables:
        ds = ds.drop_vars("valid_time")

    return ds.assign_coords(valid_time=valid_coords)
