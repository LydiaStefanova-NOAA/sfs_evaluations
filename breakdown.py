"""
Spatial Diagnostic Script: Unpack High SNR into Signal Variance vs. Noise Variance Ratios
Supports Atmospheric (atm), Oceanic (ocn), and Sea Ice (ice) variables.
"""
import argparse
import logging
import warnings
import numpy as np
import xarray as xr

# Source Loaders & Preprocessing Pipelines
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
from preprocess.spatial_alignment import (
    canonicalize_lonlat,
    standardize_year_dim,
    align_obs_to_sfs_grid,
    select_common_eval_years,
    grid_report,
    nan_report,
)
from preprocess.climatology import compute_climatology, _linear_detrend
from metrics.acc import compute_acc
from metrics.snr import compute_snr, compute_potential_skill, compute_rpc
from metrics.compute_variance_diagnostic import compute_variance_ratios
from viz.spatial import plot_variance_diagnostics_map
from viz.plot_regimes import plot_anomaly_time_series_regimes

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOMAIN_VARS = {
    "atm": ["TMP2m", "Z500", "HGT500", "Z200", "Z700", "Z850", "MSLP", "PRATE", "U10m", "V10m", "T850", "T200", "U850", "V850", "U200", "V200"],
    "ocn": ["SST", "SSH", "SSS", "MLD_003", "MLD_0125", "dt20c", "ocnheat", "taux", "tauy"],
    "ice": ["aice_h", "hi_h", "hs_h", "Tsfc_h", "uvel_h", "vvel_h"],
}
PREPROCESS_MAP = {"atm": preprocess_atm_dataset, "ice": preprocess_ice_dataset, "ocn": preprocess_ocn_dataset}
OBS_DATA_MAP = {"atm": get_era5_data, "ice": get_oras5_data, "ocn": get_oras5_data}
OBS_NAME_MAP = {"atm": "ERA5", "ice": "ORAS5", "ocn": "ORAS5"}
DOMAIN_LABEL_MAP = {"atm": "Atmospheric", "ice": "SeaIce", "ocn": "Oceanic"}
DEFAULT_VARS = {"atm": "TMP2m", "ice": "aice_h", "ocn": "SSH"}

clim_years = (1991, 2020)


def run_variance_diagnostic_cli(
    component: str = "ocn",
    var_name: str = None,
    target_months: list[int] = [6, 7, 8],
    init_month: int = 5,
    start_year: int = 1991,
    end_year: int = 2022,
    detrend: bool = True,
    output_png: str = None,
    leads_override: list[int] = None,
    varobs_eps: float = 1e-12,
    mse_eps: float = 1e-12,
    nvr_method: str = "mse",
    acc_eps: float = 1e-12,
    debug: bool = False,
):
    comp = component.lower()
    if comp not in PREPROCESS_MAP:
        raise ValueError(f"Invalid component '{component}'. Allowed: {list(PREPROCESS_MAP.keys())}")

    if var_name is None:
        var_name = DEFAULT_VARS[comp]

    if var_name not in DOMAIN_VARS[comp]:
        raise ValueError(f"Variable '{var_name}' not supported for domain '{comp}'. Allowed: {DOMAIN_VARS[comp]}")

    # Resolve leads
    if leads_override is not None and len(leads_override) > 0:
        leads = [int(L) for L in leads_override]
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
    obs_label = OBS_NAME_MAP[comp]
    logger.info(f"=== Starting {season_str} {domain_label} SNR Variance Breakdown Diagnostic ({var_name}) ===")
    logger.info(f"NVR method: {nvr_method}")

    # 1) SFS seasonal field
    ds_sfs_raw = get_sfs_data(init_month=init_month, domain=comp, requested_vars=[var_name])
    ds_sfs = PREPROCESS_MAP[comp](ds_sfs_raw)
    sfs_season_da = ds_sfs[var_name].sel(lead=leads).mean(dim="lead", skipna=True)

    # 2) OBS seasonal field
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

    # 3) Standardize + Canonicalize + Year alignment
    sfs_season_da = canonicalize_lonlat(standardize_year_dim(sfs_season_da))
    obs_season_da = canonicalize_lonlat(standardize_year_dim(obs_season_da))

    sfs_season_da, obs_season_da, eval_years = select_common_eval_years(
        sfs_season_da, obs_season_da, start_year=start_year, end_year=end_year
    )
    obs_season_da = align_obs_to_sfs_grid(obs_season_da, sfs_season_da, method="linear")

    # 4) Compute Climatology & Anomaly fields
    sfs_clim_da = compute_climatology(sfs_season_da, clim_years=clim_years).squeeze(drop=True).load()
    obs_clim_da = compute_climatology(obs_season_da, clim_years=clim_years).squeeze(drop=True).load()
    sfs_season_da = sfs_season_da.squeeze(drop=True).load()
    obs_season_da = obs_season_da.squeeze(drop=True).load()

    actual_start, actual_end = eval_years[0], eval_years[-1]
    logger.info(f"Evaluation window: {len(eval_years)} actual years ({actual_start}-{actual_end}).")

    if output_png is None:
        detrend_str = "_detrended" if detrend else ""
        method_str = "_nvraccvarobs" if nvr_method == "acc_varobs" else ""
        output_png = (
            f"figures/{comp}_{season_str.lower()}_{var_name.lower()}_variance_breakdown_"
            f"init{init_month:02d}_{actual_start}-{actual_end}{detrend_str}{method_str}.png"
        )
    # 5) Compute Anomalies & Metrics
    logger.info(f"Computing Variance Ratios (SVR & NVR) against {obs_label}...")

    # 5.1) Compute raw anomalies relative to climatology
    sfs_anom_da = sfs_season_da - sfs_clim_da
    obs_anom_da = obs_season_da - obs_clim_da

    # 5.2) Apply Ensemble-Mean Linear Detrending
    if detrend:
        logger.info("Applying ensemble-mean linear detrending...")
        ens_mean_anom = (
            sfs_anom_da.mean(dim="member", skipna=True)
            if "member" in sfs_anom_da.dims
            else sfs_anom_da
        )
        ens_mean_detrended = _linear_detrend(ens_mean_anom, dim="year")
        sfs_trend = ens_mean_anom - ens_mean_detrended

        # Subtract forced trend from all members (preserves internal spread)
        sfs_anom_da = sfs_anom_da - sfs_trend
        obs_anom_da = _linear_detrend(obs_anom_da, dim="year")

    # 5.3) Compute ACC using full fields and climatologies
    acc_da = compute_acc(
        sfs_da=sfs_season_da,
        obs_da=obs_season_da,
        sfs_clim=sfs_clim_da,
        obs_clim=obs_clim_da,
        detrend=detrend,
    )

    # 5.4) Primary Metrics Pipeline
    svr_da, nvr_da = compute_variance_ratios(
        sfs_anom=sfs_anom_da,
        obs_anom=obs_anom_da,
        acc_da=acc_da,
        varobs_eps=varobs_eps,
        mse_eps=mse_eps,
        nvr_method=nvr_method,
        acc_eps=acc_eps,
        detrend=detrend,  # Signals ddof=2 when data is detrended
    )

    snr_da = compute_snr(sfs_season_da, year_dim="year", member_dim="member", detrend=detrend)
    rpot_da = compute_potential_skill(snr_da)
    rpc_da = compute_rpc(acc_da, rpot_da)

    # 6) Plot Diagnostics Map
    title_str = (
        f"{component.upper()}: SFS {var_name} {season_str} Signal vs. Noise Variance "
        f"({actual_start}–{actual_end}) | {label}"
    )
    logger.info("Plotting variance diagnostics spatial map...")
    plot_variance_diagnostics_map(
        svr_da=svr_da,
        nvr_da=nvr_da,
        n_years=len(eval_years),
        n_members=int(sfs_season_da.member.size),
        detrend=detrend,
        title_str=title_str,
        output_png=output_png,
    )

    # 7) Plot 4-Regime Anomaly Time Series Subplots
    show_calibrated=True
    ts_suffix = "_regimes_ts_recalibrated.png" if show_calibrated else "_regimes_ts.png"
    ts_output_png = output_png.replace(".png", ts_suffix)
    logger.info(f"Plotting regime time series diagnostics to {ts_output_png}...")
    plot_anomaly_time_series_regimes(
        sfs_anom_da=canonicalize_lonlat(sfs_anom_da),
        obs_anom_da=canonicalize_lonlat(obs_anom_da),
        svr_da=svr_da,
        nvr_da=nvr_da,
        acc_da=acc_da,
        rpot_da=rpot_da,
        rpc_da=rpc_da,
        show_calibrated=show_calibrated,  # <--- ADDED
        output_png=ts_output_png,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deconstruct SNR into Signal Variance and Noise Variance Ratios")
    parser.add_argument("--component", "-c", type=str, default="ocn", choices=["atm", "ice", "ocn"])
    parser.add_argument("--var", "-v", type=str, default=None)
    parser.add_argument("--init", "-i", type=int, default=5)
    parser.add_argument("--leads", "-L", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--target", "-t", type=int, nargs="+", default=None)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument("--varobs-eps", type=float, default=1e-12)
    parser.add_argument("--mse-eps", type=float, default=1e-12)
    parser.add_argument("--nvr-method", type=str, default="mse", choices=["mse", "acc_varobs"])
    parser.add_argument("--acc-eps", type=float, default=1e-12)
    #parser.add_argument("--show-calibrated", action="store_true", help="Overlay recalibrated mean and spread in regime plots")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    run_variance_diagnostic_cli(
        component=args.component,
        var_name=args.var,
        init_month=args.init,
        target_months=args.target,
        start_year=args.start_year,
        end_year=args.end_year,
        detrend=not args.no_detrend,
        leads_override=args.leads,
        varobs_eps=args.varobs_eps,
        mse_eps=args.mse_eps,
        nvr_method=args.nvr_method,
        acc_eps=args.acc_eps,
    #    show_calibrated=args.show_calibrated,
        debug=args.debug,
    )
