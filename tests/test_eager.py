"""
Unified ACC Skill & SNR Potential Predictability Diagnostic Driver
Calculates interannual ACC against observations (ERA5 or ORAS5) and overlays SNR contours.
Supports Atmospheric (atm), Sea Ice (ice), and Ocean (ocn) domains with variable validation,
disk-cache updates, and eager memory loading for high-performance plotting.
"""
import argparse
import logging
import warnings
import numpy as np
import xarray as xr

# Source Loaders
from sources.sfs import get_sfs_data
from sources.era5 import get_era5_data
from sources.oras5 import get_oras5_data

# Preprocessing Pipelines
from preprocess.pipeline import (
    preprocess_atm_dataset,
    preprocess_ice_dataset,
    preprocess_ocn_dataset,
)
from preprocess.cache import get_or_compute_sfs_climatology
from preprocess.temporal import resolve_target_season_leads, get_season_name

# Metrics & Plotting
from metrics.acc import compute_acc
from metrics.snr import compute_snr
from viz.spatial import plot_acc_snr_overlay, plot_acc_vs_rpot

# Suppress minor runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Domain Variable Registry ---
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

DOMAIN_VARS = {
    "atm": [
        "TMP2m", "Z500", "HGT500", "Z200", "Z700", "Z850",
        "MSLP", "PRATE", "U10m", "V10m", "T850", "T200",
        "U850", "V850", "U200", "V200"
    ],
    "ocn": [
        "SST", "SSH", "SSS", "MLD_003", "MLD_0125",
        "dt20c", "ocnheat", "taux", "tauy"
    ],
    "ice": [
        "aice_h", "hi_h", "hs_h", "Tsfc_h", "uvel_h", "vvel_h"
    ],
}

# --- Component Dispatch Configurations ---
PREPROCESS_MAP = {
    "atm": preprocess_atm_dataset,
    "ice": preprocess_ice_dataset,
    "ocn": preprocess_ocn_dataset,
}

OBS_DATA_MAP = {
    "atm": get_era5_data,
    "ice": get_oras5_data,
    "ocn": get_oras5_data,
}

OBS_NAME_MAP = {
    "atm": "ERA5",
    "ice": "ORAS5",
    "ocn": "ORAS5",
}

DOMAIN_LABEL_MAP = {
    "atm": "Atmospheric",
    "ice": "SeaIce",
    "ocn": "Oceanic",
}

DEFAULT_VARS = {
    "atm": "TMP2m",
    "ice": "aice_h",
    "ocn": "dt20c",
}


def run_acc_snr_diagnostic(
    component: str = "atm",
    var_name: str = None,
    target_months: list[int] = [6, 7, 8],  # Default: JJA
    init_month: int = 5,                  # Default: May init
    start_year: int = 1991,
    end_year: int = 2022,
    detrend: bool = True,
    output_png: str = None,
    plot_rpot: bool = None,
):
    """
    Execute ACC skill and SNR predictability diagnostic for a target season and domain component.
    """
    comp = component.lower()
    if comp not in PREPROCESS_MAP:
        raise ValueError(f"Invalid component '{component}'. Must be one of: {list(PREPROCESS_MAP.keys())}")

    # Resolve default variable if not provided
    if var_name is None:
        var_name = DEFAULT_VARS[comp]

    # Validate variable against domain registry
    valid_domain_vars = DOMAIN_VARS[comp]
    if var_name not in valid_domain_vars:
        raise ValueError(
            f"Variable '{var_name}' is not supported for domain '{comp}'. "
            f"Allowed variables for '{comp}' are: {valid_domain_vars}"
        )

    # Side-by-side potential skill plotting flag default
    if plot_rpot is None:
        plot_rpot = (comp == "ocn")

    season_str = get_season_name(target_months)

    # 1. Resolve lead times for requested init_month and target season
    resolved_inits = resolve_target_season_leads(target_months)
    init_info = next(
        (item for item in resolved_inits if int(item[0]) == int(init_month)),
        None
    )

    if init_info is None:
        available_inits = [item[0] for item in resolved_inits]
        raise ValueError(
            f"Init month {init_month:02d} cannot target season {season_str} within valid leads (0-11). "
            f"Available init options for {season_str} are: {available_inits}"
        )

    _, leads, label = init_info
    domain_label = DOMAIN_LABEL_MAP[comp]
    obs_label = OBS_NAME_MAP[comp]

    logger.info(f"=== Starting {season_str} {domain_label} ACC + SNR Diagnostic ({var_name}) ===")
    logger.info(f"Targeting window: {label} | Evaluation Years: {start_year}-{end_year}")

    if output_png is None:
        detrend_str = "_detrended" if detrend else ""
        output_png = (
            f"figures/{comp}_{season_str.lower()}_{var_name.lower()}_acc_snr_"
            f"init{init_month:02d}_{start_year}-{end_year}{detrend_str}.png"
        )

    # 2. Ingest and Preprocess SFS Reforecast Data
    logger.info(f"Loading SFS {comp} store for Init {init_month:02d}...")
    ds_sfs_raw = get_sfs_data(init_month=init_month, domain=comp, requested_vars=[var_name])
    preprocess_fn = PREPROCESS_MAP[comp]
    ds_sfs = preprocess_fn(ds_sfs_raw, target_res="1.0deg")

    # Load cached domain climatology baseline
    sfs_clim = get_or_compute_sfs_climatology(ds_sfs, domain=comp, init_month=init_month)

    # Ensure requested variable exists in cached climatology; if missing, calculate and append to disk cache
    if var_name not in sfs_clim:
        logger.info(
            f"Variable '{var_name}' missing from cached climatology dataset ({comp}, init {init_month:02d}). "
            f"Computing climatology for '{var_name}' and updating disk cache..."
        )
        clim_dims = [d for d in ["year", "init", "time", "member", "number", "ens"] if d in ds_sfs[var_name].dims]
        var_clim = ds_sfs[var_name].mean(dim=clim_dims, skipna=True) if clim_dims else ds_sfs[var_name]

        cache_file = sfs_clim.encoding.get("source")
        sfs_clim = sfs_clim.load()
        sfs_clim[var_name] = var_clim

        if cache_file:
            sfs_clim.to_netcdf(cache_file, mode="w")
            logger.info(f"Successfully updated cache file on disk: {cache_file}")
        else:
            cache_file = f"cache/sfs_clim_{comp}_init{init_month:02d}.nc"
            sfs_clim.to_netcdf(cache_file, mode="w")
            logger.info(f"Successfully saved updated climatology to cache: {cache_file}")

    # Extract target season mean across resolved leads
    sfs_season_da = ds_sfs[var_name].sel(lead=leads).mean(dim="lead", skipna=True)
    sfs_clim_da = sfs_clim[var_name].sel(lead=leads).mean(dim="lead", skipna=True)

    # 3. Ingest and Process Observational Verification
    logger.info(f"Loading {obs_label} observational baseline ({start_year}-{end_year})...")
    get_obs_fn = OBS_DATA_MAP[comp]
    ds_obs_raw = get_obs_fn(requested_vars=[var_name])
    ds_obs_base = ds_obs_raw.sel(time=slice(str(start_year), str(end_year)))

    # Extract matching seasonal slice for observation per year
    obs_season = ds_obs_base.where(ds_obs_base["time.month"].isin(target_months), drop=True)
    obs_season_da = obs_season[var_name].groupby("time.year").mean(dim="time", skipna=True)
    obs_clim_da = obs_season_da.mean(dim="year", skipna=True)

    # 4. Standardize Dimension Names & Coerce Coordinates to Integer Years
    if "init" in sfs_season_da.dims and "year" not in sfs_season_da.dims:
        sfs_season_da = sfs_season_da.rename({"init": "year"})
    if "init" in sfs_clim_da.dims and "year" not in sfs_clim_da.dims:
        sfs_clim_da = sfs_clim_da.rename({"init": "year"})

    if np.issubdtype(sfs_season_da.year.dtype, np.datetime64):
        sfs_season_da["year"] = sfs_season_da.year.dt.year
    if np.issubdtype(obs_season_da.year.dtype, np.datetime64):
        obs_season_da["year"] = obs_season_da.year.dt.year

    # Intersect and filter evaluation window to requested [start_year, end_year]
    common_years = np.intersect1d(sfs_season_da.year.values, obs_season_da.year.values)
    eval_years = [y for y in common_years if start_year <= y <= end_year]

    if len(eval_years) == 0:
        raise ValueError(f"No overlapping years found in range {start_year}-{end_year}.")

    sfs_season_da = sfs_season_da.sel(year=eval_years)
    obs_season_da = obs_season_da.sel(year=eval_years)
    logger.info(f"Filtered evaluation window to {len(eval_years)} years ({eval_years[0]}-{eval_years[-1]}).")

    # EAGER LOAD STEP 1: Stream input data into RAM once
    logger.info("Fetching seasonal evaluation slices into RAM...")
    sfs_season_da = sfs_season_da.load()
    obs_season_da = obs_season_da.load()
    sfs_clim_da = sfs_clim_da.load()
    obs_clim_da = obs_clim_da.load()

    # 5. Compute Metrics
    logger.info("Calculating Signal-to-Noise Ratio (SNR)...")
    if comp == "ocn":
        snr_da = compute_snr(sfs_season_da, detrend=detrend, use_std_ratio=False)
    else:
        snr_da = compute_snr(sfs_season_da)

    logger.info(f"Calculating Anomaly Correlation Coefficient (ACC) [detrend={detrend}]...")
    acc_da = compute_acc(
        sfs_da=sfs_season_da,
        obs_da=obs_season_da,
        sfs_clim=sfs_clim_da,
        obs_clim=obs_clim_da,
        detrend=detrend,
    )

    # EAGER LOAD STEP 2: Explicitly evaluate metrics into memory
    logger.info("Evaluating metric maps in memory...")
    acc_da = acc_da.load()
    snr_da = snr_da.load()

    # 6. Render Overlay Plot
    logger.info("Rendering Panel A Overlay Plot...")
    title_overlay = f"SFS {var_name} {season_str} Skill & Predictability ({start_year}-{end_year}) | {label}"
    plot_acc_snr_overlay(
        acc_da=acc_da,
        snr_da=snr_da,
        title_str=title_overlay,
        output_png=output_png,
    )

    # 7. Render Side-by-Side ACC vs Potential Skill Plot (optional)
    if plot_rpot:
        logger.info("Rendering Side-by-Side ACC vs. Potential Skill Plot...")
        output_rpot_png = output_png.replace("_acc_snr_", "_acc_vs_rpot_")
        title_rpot = f"SFS {var_name} {season_str} Realized Skill (ACC) vs. Potential Skill ({start_year}-{end_year}) | {label}"
        plot_acc_vs_rpot(
            acc_da=acc_da,
            snr_da=snr_da,
            title_str=title_rpot,
            output_png=output_rpot_png,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified SFS ACC & SNR Diagnostic Driver")
    parser.add_argument("--component", "-c", type=str, default="atm", choices=["atm", "ice", "ocn"], help="Target component domain")
    parser.add_argument("--var", "-v", type=str, default=None, help="Target variable name (e.g. TMP2m, Z500, SST, dt20c, aice_h)")
    parser.add_argument("--init", "-i", type=int, default=5, help="Initialization month (1-12)")
    parser.add_argument("--start-year", type=int, default=1991, help="Start evaluation year")
    parser.add_argument("--end-year", type=int, default=2022, help="End evaluation year")
    parser.add_argument("--no-detrend", action="store_true", help="Disable linear detrending")
    parser.add_argument("--plot-rpot", action="store_true", help="Force generation of ACC vs. Potential Skill plot")

    args = parser.parse_args()

    run_acc_snr_diagnostic(
        component=args.component,
        var_name=args.var,
        init_month=args.init,
        start_year=args.start_year,
        end_year=args.end_year,
        detrend=not args.no_detrend,
        plot_rpot=True if args.plot_rpot else None,
    )
