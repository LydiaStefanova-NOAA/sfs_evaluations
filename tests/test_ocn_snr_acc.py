"""
Oceanic ACC Skill & SNR Potential Predictability Diagnostic Driver
Calculates interannual ACC against ORAS5 and overlays SNR contours for a target season.
Generates both Panel A Overlay and ACC vs Potential Skill side-by-side diagnostics.
"""
import logging
import warnings
import numpy as np
import xarray as xr

from sources.sfs import get_sfs_data
from sources.oras5 import get_oras5_data
from preprocess.pipeline import preprocess_ocn_dataset
from preprocess.cache import get_or_compute_sfs_climatology
from preprocess.temporal import resolve_target_season_leads, get_season_name
from metrics.acc import compute_acc
from metrics.snr import compute_snr
from viz.spatial import plot_acc_snr_overlay, plot_acc_vs_rpot

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ocn_acc_snr_diagnostic(
    target_months: list[int] = [6, 7, 8],  # Default: JJA
    var_name: str = "dt20c",
    init_month: int = 5,                   # Default: May init (Leads 1, 2, 3)
    start_year: int = 1991,
    end_year: int = 2022,
    detrend: bool = True,
    output_png: str = None,
):
    """
    Execute Oceanic ACC skill and SNR predictability diagnostic for a specific target season.
    """
    season_str = get_season_name(target_months)

    # 1. Resolve lead times
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
    logger.info(f"=== Starting {season_str} Oceanic ACC + SNR Diagnostic ({var_name}) ===")
    logger.info(f"Targeting window: {label} | Evaluation Years: {start_year}-{end_year}")

    if output_png is None:
        detrend_str = "_detrended" if detrend else ""
        output_png = (
            f"figures/{season_str.lower()}_{var_name.lower()}_acc_snr_"
            f"init{init_month:02d}_{start_year}-{end_year}{detrend_str}.png"
        )

    # 2. Ingest SFS Reforecast
    logger.info(f"Loading SFS ocn store for Init {init_month:02d}...")
    ds_sfs_raw = get_sfs_data(init_month=init_month, domain="ocn", requested_vars=[var_name])
    ds_sfs = preprocess_ocn_dataset(ds_sfs_raw, target_res="1.0deg")

    sfs_clim = get_or_compute_sfs_climatology(ds_sfs, domain="ocn", init_month=init_month)

    sfs_season_da = ds_sfs[var_name].sel(lead=leads).mean(dim="lead", skipna=True)
    sfs_clim_da = sfs_clim[var_name].sel(lead=leads).mean(dim="lead", skipna=True)

    # 3. Ingest ORAS5 Verification
    logger.info(f"Loading ORAS5 observational baseline ({start_year}-{end_year})...")
    ds_oras5_raw = get_oras5_data(requested_vars=[var_name])
    ds_oras5_base = ds_oras5_raw.sel(time=slice(str(start_year), str(end_year)))

    oras5_season = ds_oras5_base.where(ds_oras5_base["time.month"].isin(target_months), drop=True)
    oras5_season_da = oras5_season[var_name].groupby("time.year").mean(dim="time", skipna=True)
    oras5_clim_da = oras5_season_da.mean(dim="year", skipna=True)

    # 4. Standardize Dimensions
    if "init" in sfs_season_da.dims and "year" not in sfs_season_da.dims:
        sfs_season_da = sfs_season_da.rename({"init": "year"})
    if "init" in sfs_clim_da.dims and "year" not in sfs_clim_da.dims:
        sfs_clim_da = sfs_clim_da.rename({"init": "year"})

    if np.issubdtype(sfs_season_da.year.dtype, np.datetime64):
        sfs_season_da["year"] = sfs_season_da.year.dt.year
    if np.issubdtype(oras5_season_da.year.dtype, np.datetime64):
        oras5_season_da["year"] = oras5_season_da.year.dt.year

    common_years = np.intersect1d(sfs_season_da.year.values, oras5_season_da.year.values)
    eval_years = [y for y in common_years if start_year <= y <= end_year]

    if len(eval_years) == 0:
        raise ValueError(f"No overlapping years found in range {start_year}-{end_year}.")

    sfs_season_da = sfs_season_da.sel(year=eval_years)
    oras5_season_da = oras5_season_da.sel(year=eval_years)
    logger.info(f"Filtered evaluation window to {len(eval_years)} years ({eval_years[0]}-{eval_years[-1]}).")

    # 5. Compute Metrics
    logger.info("Calculating Signal-to-Noise Ratio (SNR)...")
    snr_da = compute_snr(sfs_season_da, detrend=detrend, use_std_ratio=False)

    logger.info(f"Calculating Anomaly Correlation Coefficient (ACC) [detrend={detrend}]...")
    acc_da = compute_acc(
        sfs_da=sfs_season_da,
        obs_da=oras5_season_da,
        sfs_clim=sfs_clim_da,
        obs_clim=oras5_clim_da,
        detrend=detrend,
    )

    # 6. Render Overlay Plot
    title_overlay = f"SFS {var_name} {season_str} Skill & Predictability ({start_year}-{end_year}) | {label}"
    plot_acc_snr_overlay(
        acc_da=acc_da,
        snr_da=snr_da,
        title_str=title_overlay,
        output_png=output_png,
    )

    # 7. Render Side-by-Side ACC vs Potential Skill Plot
    output_rpot_png = output_png.replace("_acc_snr_", "_acc_vs_rpot_")
    title_rpot = f"SFS {var_name} {season_str} Realized Skill (ACC) vs. Potential Skill ({start_year}-{end_year}) | {label}"
    plot_acc_vs_rpot(
        acc_da=acc_da,
        snr_da=snr_da,
        title_str=title_rpot,
        output_png=output_rpot_png,
    )


if __name__ == "__main__":
    run_ocn_acc_snr_diagnostic(
        target_months=[6, 7, 8],  # JJA Target
        var_name="dt20c",
        init_month=5,            # May Init -> Leads 1, 2, 3
        start_year=1991,
        end_year=2022,
        detrend=True,
    )
