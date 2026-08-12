"""
Unified ACC Skill & SNR Potential Predictability Diagnostic Driver
Lead-first capable: allows explicit lead selection to avoid month/year ambiguity.
"""
import argparse
import logging
import warnings
import numpy as np
import xarray as xr

# Source Loaders & Pipelines
from sources.sfs import get_sfs_data
from sources.era5 import get_era5_data
from sources.oras5 import get_oras5_data
from preprocess.pipeline import (
    preprocess_atm_dataset,
    preprocess_ice_dataset,
    preprocess_ocn_dataset,
)
from preprocess.temporal import (
    resolve_target_season_leads,
    get_season_name,
    leads_to_target_months,
    seasonal_mean_obs_by_init_and_leads,
)

# Metrics & Viz
from metrics.acc import compute_acc
from metrics.snr import compute_snr, compute_potential_skill, compute_rpc
from viz.spatial import plot_acc_snr_overlay, plot_skill_predictability_trio

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Registries ---
DOMAIN_VARS = {
    "atm": ["TMP2m", "Z500", "HGT500", "Z200", "Z700", "Z850", "MSLP", "PRATE", "U10m", "V10m", "T850", "T200", "U850", "V850", "U200", "V200"],
    "ocn": ["SST", "SSH", "SSS", "MLD_003", "MLD_0125", "dt20c", "ocnheat", "taux", "tauy"],
    "ice": ["aice_h", "hi_h", "hs_h", "Tsfc_h", "uvel_h", "vvel_h"],
}
PREPROCESS_MAP = {"atm": preprocess_atm_dataset, "ice": preprocess_ice_dataset, "ocn": preprocess_ocn_dataset}
OBS_DATA_MAP = {"atm": get_era5_data, "ice": get_oras5_data, "ocn": get_oras5_data}
OBS_NAME_MAP = {"atm": "ERA5", "ice": "ORAS5", "ocn": "ORAS5"}
DOMAIN_LABEL_MAP = {"atm": "Atmospheric", "ice": "SeaIce", "ocn": "Oceanic"}
DEFAULT_VARS = {"atm": "TMP2m", "ice": "aice_h", "ocn": "dt20c"}


def run_acc_snr_diagnostic(
    component: str = "atm",
    var_name: str = None,
    target_months: list[int] = [6, 7, 8],   # legacy path
    init_month: int = 5,
    start_year: int = 1991,
    end_year: int = 2022,
    detrend: bool = True,
    output_png: str = None,
    plot_rpot: bool = True,
    leads_override: list[int] = None,       # preferred path
):
    comp = component.lower()
    if comp not in PREPROCESS_MAP:
        raise ValueError(f"Invalid component '{component}'. Must be one of: {list(PREPROCESS_MAP.keys())}")

    if var_name is None:
        var_name = DEFAULT_VARS[comp]

    if var_name not in DOMAIN_VARS[comp]:
        raise ValueError(f"Variable '{var_name}' not supported for domain '{comp}'. Allowed: {DOMAIN_VARS[comp]}")

    if plot_rpot is None:
        #plot_rpot = (comp == "ocn")
        plot_rpot = True

    # Resolve leads
    if leads_override is not None and len(leads_override) > 0:
        leads = [int(L) for L in leads_override]
        if any(L < 0 for L in leads):
            raise ValueError(f"Invalid negative lead in {leads}")
        target_months = leads_to_target_months(init_month, leads)
        season_str = get_season_name(target_months)
        label = f"Init {init_month:02d} | Leads {leads[0]}-{leads[-1]}" if len(leads) > 1 else f"Init {init_month:02d} | Lead {leads[0]}"
    else:
        season_str = get_season_name(target_months)
        resolved_inits = resolve_target_season_leads(target_months)
        init_info = next((item for item in resolved_inits if int(item[0]) == int(init_month)), None)
        if init_info is None:
            available_inits = [item[0] for item in resolved_inits]
            raise ValueError(f"Init month {init_month:02d} cannot target season {season_str}. Available: {available_inits}")
        _, leads, label = init_info

    domain_label = DOMAIN_LABEL_MAP[comp]
    logger.info(f"=== Starting {season_str} {domain_label} ACC + SNR Diagnostic ({var_name}) ===")

    # 1) Ingest SFS and compute seasonal lead-mean
    ds_sfs_raw = get_sfs_data(init_month=init_month, domain=comp, requested_vars=[var_name])
    ds_sfs = PREPROCESS_MAP[comp](ds_sfs_raw, target_res="1.0deg")

    sfs_season_da = ds_sfs[var_name].sel(lead=leads).mean(dim="lead", skipna=True)

    # 1b) Recompute SFS climatology from *current ds_sfs only* (avoid cache inconsistency)
    clim_dims = [d for d in ["year", "init", "time", "member", "number", "ens"] if d in ds_sfs[var_name].dims]
    sfs_clim_full = ds_sfs[var_name].mean(dim=clim_dims, skipna=True) if clim_dims else ds_sfs[var_name]

    if "lead" not in sfs_clim_full.dims:
        raise ValueError(f"SFS climatology for {var_name} has no 'lead' dimension; dims={sfs_clim_full.dims}")

    sfs_clim_da = sfs_clim_full.sel(lead=leads).mean(dim="lead", skipna=True)

    # 2) Ingest observations aligned by init-year + leads
    ds_obs_raw = OBS_DATA_MAP[comp](requested_vars=[var_name])
    obs_season_da = seasonal_mean_obs_by_init_and_leads(
        ds_obs_raw[var_name],
        init_month=init_month,
        leads=leads,
        start_year=start_year,
        end_year=end_year,
        time_dim="time",
        require_complete=True,
    )
    obs_clim_da = obs_season_da.mean(dim="year", skipna=True)

    # 3) Standardize year dimensions
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

    logger.info(
        f"SFS years: {int(np.min(sfs_season_da.year.values))}-{int(np.max(sfs_season_da.year.values))} "
        f"(n={sfs_season_da.year.size}) | "
        f"OBS years: {int(np.min(obs_season_da.year.values))}-{int(np.max(obs_season_da.year.values))} "
        f"(n={obs_season_da.year.size})"
    )
    logger.info(f"Common overlapping years in requested window: n={len(eval_years)}")

    if len(eval_years) == 0:
        raise ValueError(f"No overlapping years found in range {start_year}-{end_year}.")

    sfs_season_da = sfs_season_da.sel(year=eval_years)
    obs_season_da = obs_season_da.sel(year=eval_years)

    actual_start, actual_end = eval_years[0], eval_years[-1]
    logger.info(f"Evaluation window: {len(eval_years)} actual years ({actual_start}-{actual_end}).")

    if output_png is None:
        detrend_str = "_detrended" if detrend else ""
        output_png = (
            f"figures/{comp}_{season_str.lower()}_{var_name.lower()}_acc_snr_"
            f"init{init_month:02d}_{actual_start}-{actual_end}{detrend_str}.png"
        )

    # 4) Eager load
    sfs_season_da = sfs_season_da.load()
    obs_season_da = obs_season_da.load()
    sfs_clim_da = sfs_clim_da.load()
    obs_clim_da = obs_clim_da.load()

    # 5) Metrics
    logger.info("Calculating Signal-to-Noise Ratio (SNR)...")
    snr_da = compute_snr(sfs_season_da, detrend=detrend).load()

    logger.info(f"Calculating Anomaly Correlation Coefficient (ACC) [detrend={detrend}]...")
    acc_da = compute_acc(
        sfs_da=sfs_season_da,
        obs_da=obs_season_da,
        sfs_clim=sfs_clim_da,
        obs_clim=obs_clim_da,
        detrend=detrend,
    ).load()

    # 6) Overlay plot
    logger.info("Rendering Panel A Overlay Plot...")
    title_overlay = f"{component}: SFS {var_name} {season_str} Skill & Predictability ({actual_start}-{actual_end}) | {label}"
    plot_acc_snr_overlay(
        acc_da=acc_da,
        snr_da=snr_da,
        title_str=title_overlay,
        output_png=output_png,
    )

    # 7) Trio plot
    if plot_rpot:
        logger.info("Calculating Potential Skill (r_pot) & RPC...")
        rpot_da = compute_potential_skill(snr_da).load()
        rpc_da = compute_rpc(acc_da, rpot_da).load()

        logger.info("Rendering 3-Panel Diagnostic Stack (ACC, r_pot, RPC)...")
        output_trio_png = output_png.replace("_acc_snr_", "_skill_trio_")
        main_header = ""
        subtitle_str = f"{component}: SFS {var_name} {season_str} ({actual_start}–{actual_end}) | {label}"

        plot_skill_predictability_trio(
            acc_da=acc_da,
            rpot_da=rpot_da,
            rpc_da=rpc_da,
            main_title=main_header,
            subtitle_str=subtitle_str,
            output_png=output_trio_png,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified SFS ACC & SNR Diagnostic Driver")
    parser.add_argument("--component", "-c", type=str, default="atm", choices=["atm", "ice", "ocn"])
    parser.add_argument("--var", "-v", type=str, default=None)
    parser.add_argument("--init", "-i", type=int, default=5)
    parser.add_argument("--leads", "-L", type=int, nargs="+", default=None)  # preferred
    parser.add_argument("--target", "-t", type=int, nargs="+", default=[1, 2, 3])  # legacy
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument("--plot-rpot", action="store_true")

    args = parser.parse_args()

    run_acc_snr_diagnostic(
        component=args.component,
        var_name=args.var,
        init_month=args.init,
        target_months=args.target,
        start_year=args.start_year,
        end_year=args.end_year,
        detrend=not args.no_detrend,
        plot_rpot=True if args.plot_rpot else None,
        leads_override=args.leads,
    )
