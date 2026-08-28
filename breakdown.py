"""
Spatial Diagnostic Script: Unpack High SNR into Signal Variance vs. Noise Variance Ratios
Supports Atmospheric (atm), Oceanic (ocn), and Sea Ice (ice) variables.
"""
import argparse
import calendar
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
)
from preprocess.climatology import compute_climatology, _linear_detrend
from metrics.acc import compute_acc
from metrics.snr import compute_snr, compute_potential_skill, compute_rpc
from metrics.compute_variance_diagnostic import compute_variance_ratios
from viz.spatial import plot_variance_diagnostics_map, plot_skill_predictability_trio
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


def format_plot_title(
    component: str,
    var_name: str,
    init_val: int | str,
    lead_str: str | int,
    start_year: int | str,
    end_year: int | str,
    label: str,
) -> str:
    """Standardized title header generator: COMP: SFS VAR Init 05 Lead 1-3 (START-END) | LABEL"""
    init_num = None
    if isinstance(init_val, int):
        init_num = init_val
    else:
        clean_init = str(init_val).strip().lower().replace("init", "").strip()
        if clean_init.isdigit():
            init_num = int(clean_init)
        else:
            month_abbrs = [m.lower() for m in calendar.month_abbr]
            month_names = [m.lower() for m in calendar.month_name]
            if clean_init[:3] in month_abbrs:
                init_num = month_abbrs.index(clean_init[:3])
            elif clean_init in month_names:
                init_num = month_names.index(clean_init)

    init_fmt = f"Init {init_num:02d}" if init_num is not None else f"Init {init_val}"
    lead_fmt = str(lead_str) if str(lead_str).lower().startswith("lead") else f"Lead {lead_str}"

    return f"{component.upper()}: SFS {var_name} {init_fmt} {lead_fmt} ({start_year}-{end_year}) | {label}"


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
    nvr_method: str = "acc_varobs",
    acc_eps: float = 1e-12,
    show_calibrated: bool = True,
    debug: bool = False,
):
    comp = component.lower()
    if comp not in PREPROCESS_MAP:
        raise ValueError(f"Invalid component '{component}'. Allowed: {list(PREPROCESS_MAP.keys())}")

    if var_name is None:
        var_name = DEFAULT_VARS[comp]

    if var_name not in DOMAIN_VARS[comp]:
        raise ValueError(f"Variable '{var_name}' not supported for domain '{comp}'. Allowed: {DOMAIN_VARS[comp]}")

    # Resolve leads and lead string representation
    if leads_override is not None and len(leads_override) > 0:
        leads = [int(L) for L in leads_override]
        target_months = leads_to_target_months(init_month, leads)
        season_str = get_season_name(target_months)
        lead_str = f"{leads[0]}-{leads[-1]}" if len(leads) > 1 else f"{leads[0]}"
    else:
        season_str = get_season_name(target_months)
        resolved_inits = resolve_target_season_leads(target_months)
        init_info = next((item for item in resolved_inits if int(item[0]) == int(init_month)), None)
        if init_info is None:
            available_inits = [item[0] for item in resolved_inits]
            raise ValueError(f"Init month {init_month:02d} cannot target season {season_str}. Available: {available_inits}")
        _, leads, _ = init_info
        lead_str = f"{leads[0]}-{leads[-1]}" if len(leads) > 1 else f"{leads[0]}"

    domain_label = DOMAIN_LABEL_MAP[comp]
    obs_label = OBS_NAME_MAP[comp]
    logger.info(f"=== Starting {season_str} {domain_label} SNR Variance Breakdown Diagnostic ({var_name}) ===")
    logger.info(f"NVR method: {nvr_method}")

    # 1) Load SFS seasonal field
    ds_sfs_raw = get_sfs_data(init_month=init_month, domain=comp, requested_vars=[var_name])
    ds_sfs = PREPROCESS_MAP[comp](ds_sfs_raw)
    sfs_season_da = ds_sfs[var_name].sel(lead=leads).mean(dim="lead", skipna=True)

    # 2) Load OBS seasonal field
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

    # 3) Standardize, Canonicalize, Align coordinates
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

    # Master title generator closure
    def make_title(label: str) -> str:
        return format_plot_title(
            component=comp,
            var_name=var_name,
            init_val=init_month,
            lead_str=lead_str,
            start_year=actual_start,
            end_year=actual_end,
            label=label,
        )

    # 5) Compute Anomalies & Metrics
    logger.info(f"Computing Variance Ratios (SVR & NVR) against {obs_label}...")

    sfs_anom_da = sfs_season_da - sfs_clim_da
    obs_anom_da = obs_season_da - obs_clim_da

    if detrend:
        logger.info("Applying ensemble-mean linear detrending...")
        ens_mean_anom = sfs_anom_da.mean(dim="member", skipna=True) if "member" in sfs_anom_da.dims else sfs_anom_da
        ens_mean_detrended = _linear_detrend(ens_mean_anom, dim="year")
        sfs_trend = ens_mean_anom - ens_mean_detrended

        sfs_anom_da = sfs_anom_da - sfs_trend
        obs_anom_da = _linear_detrend(obs_anom_da, dim="year")

    acc_da = compute_acc(
        sfs_da=sfs_season_da,
        obs_da=obs_season_da,
        sfs_clim=sfs_clim_da,
        obs_clim=obs_clim_da,
        detrend=detrend,
    )

    svr_da, nvr_da = compute_variance_ratios(
        sfs_anom=sfs_anom_da,
        obs_anom=obs_anom_da,
        acc_da=acc_da,
        varobs_eps=varobs_eps,
        mse_eps=mse_eps,
        nvr_method=nvr_method,
        acc_eps=acc_eps,
        detrend=detrend,
    )

    snr_da = compute_snr(sfs_season_da, year_dim="year", member_dim="member", detrend=detrend)
    rpot_da = compute_potential_skill(snr_da)
    rpc_da = compute_rpc(acc_da, rpot_da)

    # 6) Plot Diagnostics Maps & Plots
    
    # Step 6a: Skill & Predictability Trio Map
    trio_png = output_png.replace(".png", "_trio.png")
    logger.info(f"Plotting Skill & Predictability Trio map to {trio_png}...")
    plot_skill_predictability_trio(
        acc_da=canonicalize_lonlat(acc_da),
        rpot_da=canonicalize_lonlat(rpot_da),
        rpc_da=canonicalize_lonlat(rpc_da),
        main_title=make_title("Skill & Predictability Trio (ACC & RPC)"),
        output_png=trio_png,
    )

    # Step 6b: Variance Diagnostics Map
    logger.info("Plotting variance diagnostics spatial map...")
    n_members_val = sfs_season_da.sizes.get("member", 30)
    plot_variance_diagnostics_map(
        svr_da=canonicalize_lonlat(svr_da),
        nvr_da=canonicalize_lonlat(nvr_da),
        n_years=len(eval_years),
        n_members=int(n_members_val),
        detrend=detrend,
        title_str=make_title("SNR Variance Breakdown (SVR & NVR)"),
        output_png=output_png,
    )

    # Step 7: 5-Regime Anomaly Time Series Subplots
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
        show_calibrated=show_calibrated,
        title=make_title("5-Regime Diagnostic Time Series Overlays"),
        output_png=ts_output_png,
    )

    # Step 8: Verification & Recalibration Suite (acc_varobs mode only)
    if nvr_method == "acc_varobs":
        from metrics.verification import compute_recalibration_metrics
        from viz.spatial_verification import plot_scaling_factors_map, plot_msess_map
        from metrics.ser import compute_ser_before_after
        from viz.spatial_ser import plot_ser_comparison_map
        #from metrics.reliability import compute_reliability_curve
        from metrics.reliability import compute_nino34_reliability_curves
        from viz.plot_reliability import plot_reliability_diagram

        logger.info("Step 8: Generating Section 5 verification & recalibration plots...")
        alpha_da, beta_da, pct_mse_reduction_da = compute_recalibration_metrics(
            sfs_anom=sfs_anom_da,
            obs_anom=obs_anom_da,
            svr_da=svr_da,
            nvr_da=nvr_da,
            acc_da=acc_da,
        )

        verif_prefix = output_png.replace(".png", "_verif")

        # 8a) Scaling Factors Map (alpha, beta)
        scaling_png = f"{verif_prefix}_scaling_factors.png"
        plot_scaling_factors_map(
            alpha_da=canonicalize_lonlat(alpha_da),
            beta_da=canonicalize_lonlat(beta_da),
            title=make_title("Calibration Scaling Factors (α & β)"),
            output_png=scaling_png,
        )

        # 8b) Skill Payoff Map (MSESS)
        msess_png = f"{verif_prefix}_msess.png"
        plot_msess_map(
            pct_mse_reduction_da=canonicalize_lonlat(pct_mse_reduction_da),
            title=make_title("Global Forecast Skill Payoff (MSESS)"),
            output_png=msess_png,
        )

        # 8c) SER Before/After Comparison Map
        ser_raw, ser_cal = compute_ser_before_after(
            sfs_anom=sfs_anom_da,
            obs_anom=obs_anom_da,
            alpha_da=alpha_da,
            beta_da=beta_da,
        )
        ser_png = f"{verif_prefix}_ser_comparison.png"
        plot_ser_comparison_map(
            ser_raw_da=canonicalize_lonlat(ser_raw),
            ser_cal_da=canonicalize_lonlat(ser_cal),
            title=make_title("Spread-to-Error Ratio (SER)"),
            output_png=ser_png,
        )

        # Step 8d in breakdown.py
        rel_png = f"{verif_prefix}_reliability_tercile.png"
        plot_reliability_diagram(
            rel_dict=compute_nino34_reliability_curves(
                sfs_anom=canonicalize_lonlat(sfs_anom_da),
                obs_anom=canonicalize_lonlat(obs_anom_da),
                alpha_da=canonicalize_lonlat(alpha_da),
                beta_da=canonicalize_lonlat(beta_da),
                acc_da=canonicalize_lonlat(acc_da),
                quantile=0.67,
                n_bins=5,
                min_acc_threshold=0.3,
                threshold_mode="model_relative",
            ),
            title=make_title("Upper-Tercile Event Reliability (>67th Percentile)"),
            output_png=rel_png,
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
    parser.add_argument("--nvr-method", type=str, default="acc_varobs", choices=["mse", "acc_varobs"])
    parser.add_argument("--acc-eps", type=float, default=1e-12)
    parser.add_argument("--no-show-calibrated", action="store_true", help="Disable recalibrated mean and spread overlay in regime plots")
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
        show_calibrated=not args.no_show_calibrated,
        debug=args.debug,
    )
