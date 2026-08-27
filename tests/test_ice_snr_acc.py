"""
SeaIce ACC Skill & SNR Potential Predictability Diagnostic Driver
Calculates interannual ACC against ERA5 and overlays SNR contours for a target season.
Supports year-range filtering (e.g., 1991-2000 rapid prototyping vs 1991-2020 evaluation).
"""
import logging
import warnings
import numpy as np
import xarray as xr

from sources.sfs import get_sfs_data
from sources.oras5 import get_oras5_data
from preprocess.pipeline import preprocess_ice_dataset
from preprocess.cache import get_or_compute_sfs_climatology
from preprocess.temporal import resolve_target_season_leads, get_season_name
from metrics.acc import compute_acc
from metrics.snr import compute_snr
from viz.spatial import plot_acc_snr_overlay

# Suppress minor runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ice_acc_snr_diagnostic(
    target_months: list[int] = [6, 7, 8],  # Default: JJA (June-July-August)
    var_name: str = "SST",
    init_month: int = 5,                  # Default: May init (Leads 1, 2, 3 for JJA)
    start_year: int = 1991,               # Subsetting start year
    end_year: int = 2022,                 # Subsetting end year (e.g. 2000 or 2020)
    detrend: bool = True,                 # Linearly detrend anomalies before computing ACC
    output_png: str = None,
):
    """
    Execute SeaIceACC skill and SNR predictability diagnostic for a specific target season.
    """
    season_str = get_season_name(target_months)

    # 1. Resolve lead times for requested init_month and target season
    resolved_inits = resolve_target_season_leads(target_months)

    # Type-safe init lookup (handles '05' vs 5 string mismatches)
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
    logger.info(f"=== Starting {season_str} SeaIce ACC + SNR Diagnostic ({var_name}) ===")
    logger.info(f"Targeting window: {label} | Evaluation Years: {start_year}-{end_year}")

    if output_png is None:
        detrend_str = "_detrended" if detrend else ""
        output_png = (
            f"figures/{season_str.lower()}_{var_name.lower()}_acc_snr_"
            f"init{init_month:02d}_{start_year}-{end_year}{detrend_str}.png"
        )

    # 2. Ingest and Preprocess SFS Reforecast Data
    logger.info(f"Loading SFS ice store for Init {init_month:02d}...")
    ds_sfs_raw = get_sfs_data(init_month=init_month, domain="ice", requested_vars=[var_name])
    ds_sfs = preprocess_ice_dataset(ds_sfs_raw, target_res="1.0deg")

    # Load / Compute 1991-2020 SFS Climatology Baseline (from disk cache)
    sfs_clim = get_or_compute_sfs_climatology(ds_sfs, domain="ice", init_month=init_month)

    # Extract target season mean across resolved leads
    sfs_season_da = ds_sfs[var_name].sel(lead=leads).mean(dim="lead", skipna=True)
    sfs_clim_da = sfs_clim[var_name].sel(lead=leads).mean(dim="lead", skipna=True)

    # 3. Ingest and Process ERA5 Verification
    logger.info(f"Loading ERA5 observational baseline ({start_year}-{end_year})...")
    ds_oras5_raw = get_oras5_data(requested_vars=[var_name])
    ds_oras5_base = ds_oras5_raw.sel(time=slice(str(start_year), str(end_year)))

    # Extract matching seasonal slice for ERA5 per year
    oras5_season = ds_oras5_base.where(ds_oras5_base["time.month"].isin(target_months), drop=True)
    oras5_season_da = oras5_season[var_name].groupby("time.year").mean(dim="time", skipna=True)
    oras5_clim_da = oras5_season_da.mean(dim="year", skipna=True)

    # 4. Standardize Dimension Names & Coerce Coordinates to Integer Years
    if "init" in sfs_season_da.dims and "year" not in sfs_season_da.dims:
        sfs_season_da = sfs_season_da.rename({"init": "year"})
    if "init" in sfs_clim_da.dims and "year" not in sfs_clim_da.dims:
        sfs_clim_da = sfs_clim_da.rename({"init": "year"})

    if np.issubdtype(sfs_season_da.year.dtype, np.datetime64):
        sfs_season_da["year"] = sfs_season_da.year.dt.year
    if np.issubdtype(oras5_season_da.year.dtype, np.datetime64):
        oras5_season_da["year"] = oras5_season_da.year.dt.year

    # Intersect and filter evaluation window to requested [start_year, end_year]
    common_years = np.intersect1d(sfs_season_da.year.values, oras5_season_da.year.values)
    eval_years = [y for y in common_years if start_year <= y <= end_year]

    if len(eval_years) == 0:
        raise ValueError(f"No overlapping years found in range {start_year}-{end_year}.")

    sfs_season_da = sfs_season_da.sel(year=eval_years)
    oras5_season_da = oras5_season_da.sel(year=eval_years)
    logger.info(f"Filtered evaluation window to {len(eval_years)} years ({eval_years[0]}-{eval_years[-1]}).")

    # 5. Compute Metrics
    logger.info("Calculating Signal-to-Noise Ratio (SNR)...")
    snr_da = compute_snr(sfs_season_da)

    logger.info(f"Calculating Anomaly Correlation Coefficient (ACC) [detrend={detrend}]...")
    acc_da = compute_acc(
        sfs_da=sfs_season_da,
        obs_da=oras5_season_da,
        sfs_clim=sfs_clim_da,
        obs_clim=oras5_clim_da,
        detrend=detrend,
    )

    # 6. Render Combined Overlay Plot via viz/spatial.py
    title_str = f"SFS {var_name} {season_str} Skill & Predictability ({start_year}-{end_year}) | {label}"
    plot_acc_snr_overlay(
        acc_da=acc_da,
        snr_da=snr_da,
        title_str=title_str,
        output_png=output_png,
    )


if __name__ == "__main__":
    # Example: Run JJA TMP2m ACC + SNR diagnostic for May Init over 1991-2000
    run_ice_acc_snr_diagnostic(
        target_months=[6, 7, 8],  # JJA Target
        var_name="aice_h",
        init_month=5,              # May Init -> Leads 1, 2, 3
        start_year=1991,
        end_year=2022,
        detrend=True,
    )
