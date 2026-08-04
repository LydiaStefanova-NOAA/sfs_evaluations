# preprocess/engine.py (or inside preprocess/pipeline.py)
import logging
import os
import numpy as np
import xarray as xr

from sources.sfs import get_sfs_data
from sources.era5 import get_era5_data
from sources.oras5 import get_oras5_data

from preprocess.pipeline import (
    preprocess_atm_dataset,
    preprocess_ice_dataset,
    preprocess_ocn_dataset,
)
from preprocess.cache import get_or_compute_sfs_climatology
from preprocess.temporal import resolve_target_season_leads, get_season_name

logger = logging.getLogger(__name__)

PREPROCESS_MAP = {"atm": preprocess_atm_dataset, "ice": preprocess_ice_dataset, "ocn": preprocess_ocn_dataset}
OBS_DATA_MAP = {"atm": get_era5_data, "ice": get_oras5_data, "ocn": get_oras5_data}
OBS_NAME_MAP = {"atm": "ERA5", "ice": "ORAS5", "ocn": "ORAS5"}
DOMAIN_LABEL_MAP = {"atm": "Atmospheric", "ice": "SeaIce", "ocn": "Oceanic"}
DEFAULT_VARS = {"atm": "TMP2m", "ice": "aice_h", "ocn": "dt20c"}

DOMAIN_VARS = {
    "atm": ["TMP2m", "Z500", "HGT500", "Z200", "Z700", "Z850", "MSLP", "PRATE", "U10m", "V10m", "T850", "T200", "U850", "V850", "U200", "V200"],
    "ocn": ["SST", "SSH", "SSS", "MLD_003", "MLD_0125", "dt20c", "ocnheat", "taux", "tauy"],
    "ice": ["aice_h", "hi_h", "hs_h", "Tsfc_h", "uvel_h", "vvel_h"],
}

def prepare_seasonal_evaluation_data(
    component: str = "atm",
    var_name: str = None,
    target_months: list[int] = [6, 7, 8],
    init_month: int = 5,
    start_year: int = 1991,
    end_year: int = 2022,
):
    """
    Modular Data Engine: Loads SFS & Obs data, preprocesses, computes/updates climatology cache,
    aligns common evaluation years, and eagerly loads data into RAM.
    
    Returns
    -------
    tuple: (sfs_season_da, obs_season_da, sfs_clim_da, obs_clim_da, meta)
    """
    comp = component.lower()
    if comp not in PREPROCESS_MAP:
        raise ValueError(f"Invalid component '{component}'. Must be one of: {list(PREPROCESS_MAP.keys())}")

    if var_name is None:
        var_name = DEFAULT_VARS[comp]

    if var_name not in DOMAIN_VARS[comp]:
        raise ValueError(f"Variable '{var_name}' not supported for domain '{comp}'. Allowed: {DOMAIN_VARS[comp]}")

    season_str = get_season_name(target_months)
    resolved_inits = resolve_target_season_leads(target_months)
    init_info = next((item for item in resolved_inits if int(item[0]) == int(init_month)), None)

    if init_info is None:
        available_inits = [item[0] for item in resolved_inits]
        raise ValueError(f"Init month {init_month:02d} cannot target season {season_str}. Options: {available_inits}")

    _, leads, lead_label = init_info
    
    # 1. Ingest SFS & Climatology
    ds_sfs_raw = get_sfs_data(init_month=init_month, domain=comp, requested_vars=[var_name])
    ds_sfs = PREPROCESS_MAP[comp](ds_sfs_raw, target_res="1.0deg")
    sfs_clim = get_or_compute_sfs_climatology(ds_sfs, domain=comp, init_month=init_month)

    # Cache update logic if variable is missing
    if var_name not in sfs_clim:
        logger.info(f"Updating climatology cache with missing variable '{var_name}'...")
        cache_file = getattr(sfs_clim, "encoding", {}).get("source")
        sfs_clim_updated = sfs_clim.load().copy()
        sfs_clim.close()

        clim_dims = [d for d in ["year", "init", "time", "member", "number", "ens"] if d in ds_sfs[var_name].dims]
        var_clim = ds_sfs[var_name].mean(dim=clim_dims, skipna=True) if clim_dims else ds_sfs[var_name]
        sfs_clim_updated[var_name] = var_clim

        if cache_file and os.path.exists(cache_file):
            tmp_cache_file = cache_file + ".tmp"
            sfs_clim_updated.to_netcdf(tmp_cache_file)
            os.replace(tmp_cache_file, cache_file)
        sfs_clim = sfs_clim_updated

    sfs_season_da = ds_sfs[var_name].sel(lead=leads).mean(dim="lead", skipna=True)
    sfs_clim_da = sfs_clim[var_name].sel(lead=leads).mean(dim="lead", skipna=True)

    # 2. Ingest Observations
    ds_obs_raw = OBS_DATA_MAP[comp](requested_vars=[var_name])
    ds_obs_base = ds_obs_raw.sel(time=slice(str(start_year), str(end_year)))
    obs_season = ds_obs_base.where(ds_obs_base["time.month"].isin(target_months), drop=True)
    obs_season_da = obs_season[var_name].groupby("time.year").mean(dim="time", skipna=True)
    obs_clim_da = obs_season_da.mean(dim="year", skipna=True)

    # 3. Standardize dimensions & coerce years
    if "init" in sfs_season_da.dims and "year" not in sfs_season_da.dims:
        sfs_season_da = sfs_season_da.rename({"init": "year"})
    if "init" in sfs_clim_da.dims and "year" not in sfs_clim_da.dims:
        sfs_clim_da = sfs_clim_da.rename({"init": "year"})

    if np.issubdtype(sfs_season_da.year.dtype, np.datetime64):
        sfs_season_da["year"] = sfs_season_da.year.dt.year
    if np.issubdtype(obs_season_da.year.dtype, np.datetime64):
        obs_season_da["year"] = obs_season_da.year.dt.year

    common_years = np.intersect1d(sfs_season_da.year.values, obs_season_da.year.values)
    eval_years = [y for y in common_years if start_year <= y <= end_year]

    if len(eval_years) == 0:
        raise ValueError(f"No overlapping years found in range {start_year}-{end_year}.")

    sfs_season_da = sfs_season_da.sel(year=eval_years)
    obs_season_da = obs_season_da.sel(year=eval_years)

    # 4. Eager load into RAM
    sfs_season_da = sfs_season_da.load()
    obs_season_da = obs_season_da.load()
    sfs_clim_da = sfs_clim_da.load()
    obs_clim_da = obs_clim_da.load()

    meta = {
        "component": comp,
        "domain_label": DOMAIN_LABEL_MAP[comp],
        "var_name": var_name,
        "season_str": season_str,
        "lead_label": lead_label,
        "actual_start": eval_years[0],
        "actual_end": eval_years[-1],
        "eval_years": eval_years,
    }

    return sfs_season_da, obs_season_da, sfs_clim_da, obs_clim_da, meta
