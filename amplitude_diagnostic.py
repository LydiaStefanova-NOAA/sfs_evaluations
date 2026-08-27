"""
Spatial Diagnostic Script: Compute & Plot ECMWF Anomaly Amplitude Ratio
Supports Atmospheric (atm), Oceanic (ocn), and Sea Ice (ice) variables.
"""
import argparse
import logging
import warnings
import xarray as xr

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
from metrics.amplitude_ratio import compute_amplitude_ratio
from viz.plot_amplitude_ratio import plot_amplitude_ratio_map
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


def run_amplitude_diagnostic_cli(
    component: str = "ocn",
    var_name: str = None,
    target_months: list[int] = [6, 7, 8],
    init_month: int = 5,
    start_year: int = 1991,
    end_year: int = 2022,
    detrend: bool = True,
    output_png: str = None,
    leads_override: list[int] = None,
    ar_mode: str = "members",
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
    logger.info(f"=== Starting {season_str} {domain_label} ECMWF Amplitude Ratio Diagnostic ({var_name}) ===")

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

    # 3) Standardize & Align
    sfs_season_da = canonicalize_lonlat(standardize_year_dim(sfs_season_da))
    obs_season_da = canonicalize_lonlat(standardize_year_dim(obs_season_da))

    sfs_season_da, obs_season_da, eval_years = select_common_eval_years(
        sfs_season_da, obs_season_da, start_year=start_year, end_year=end_year
    )
    obs_season_da = align_obs_to_sfs_grid(obs_season_da, sfs_season_da, method="linear")

    # 4) Compute Climatology & Anomalies
    sfs_clim_da = compute_climatology(sfs_season_da, clim_years=clim_years).squeeze(drop=True).load()
    obs_clim_da = compute_climatology(obs_season_da, clim_years=clim_years).squeeze(drop=True).load()
    sfs_season_da = sfs_season_da.squeeze(drop=True).load()
    obs_season_da = obs_season_da.squeeze(drop=True).load()

    actual_start, actual_end = eval_years[0], eval_years[-1]
    logger.info(f"Evaluation window: {len(eval_years)} actual years ({actual_start}-{actual_end}).")

    if output_png is None:
        detrend_str = "_detrended" if detrend else ""
        output_png = (
            f"figures/{comp}_{season_str.lower()}_{var_name.lower()}_amplitude_ratio_"
            f"init{init_month:02d}_{actual_start}-{actual_end}{detrend_str}.png"
        )

    # 5) Compute Anomalies & Metrics
    sfs_anom_da = sfs_season_da - sfs_clim_da
    obs_anom_da = obs_season_da - obs_clim_da

    if detrend:
        logger.info("Applying ensemble-mean linear detrending...")
        # A. Fit linear trend strictly to the forced ensemble mean signal
        ens_mean_anom = (
            sfs_anom_da.mean(dim="member", skipna=True)
            if "member" in sfs_anom_da.dims
            else sfs_anom_da
        )
        ens_mean_detrended = _linear_detrend(ens_mean_anom, dim="year")
        sfs_trend = ens_mean_anom - ens_mean_detrended

        # B. Subtract forced trend from all members (preserves internal chaos)
        sfs_anom_da = sfs_anom_da - sfs_trend
        obs_anom_da = _linear_detrend(obs_anom_da, dim="year")

    # Compute ECMWF Amplitude Ratio
    #ar_da = compute_amplitude_ratio(sfs_anom_da, obs_anom_da, mode=ar_mode)
    ar_da = compute_amplitude_ratio(
        sfs_anom=sfs_anom_da,
        obs_anom=obs_anom_da,
        mode=ar_mode,
        detrended=detrend,  # <-- Pass detrend flag to switch ddof (2 vs 1)
    )

    ar_da = canonicalize_lonlat(ar_da)

    # Optional: Secondary diagnostics (Uncomment if passing to regime plots or saving)
    # acc_da = compute_acc(sfs_da=sfs_season_da, obs_da=obs_season_da, sfs_clim=sfs_clim_da, obs_clim=obs_clim_da, detrend=detrend)
    # svr_da, nvr_da = compute_variance_ratios(sfs_anom=sfs_anom_da, obs_anom=obs_anom_da, acc_da=acc_da, detrend=False)
    # snr_da = compute_snr(sfs_season_da, year_dim="year", member_dim="member", detrend=detrend)
    # rpot_da = compute_potential_skill(snr_da)
    # rpc_da = compute_rpc(acc_da, rpot_da)
    # plot_anomaly_time_series_regimes(ar_da=ar_da, rpc_da=rpc_da, ...)

    # 6) Plot Spatial Amplitude Map
    title_str = (
        f"{component.upper()}: SFS {var_name} {season_str} ECMWF Anomaly Amplitude Ratio "
        f"({actual_start}–{actual_end}) | {label}"
    )
    logger.info("Plotting Anomaly Amplitude Ratio spatial map...")
    plot_amplitude_ratio_map(ar_da=ar_da, title_str=title_str, output_png=output_png)

    # 6) Plot Spatial Amplitude Map
    title_str = (
        f"{component.upper()}: SFS {var_name} {season_str} ECMWF Anomaly Amplitude Ratio "
        f"({actual_start}–{actual_end}) | {label}"
    )
    logger.info("Plotting Anomaly Amplitude Ratio spatial map...")
    plot_amplitude_ratio_map(ar_da=ar_da, title_str=title_str, output_png=output_png)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ECMWF Anomaly Amplitude Ratio")
    parser.add_argument("--component", "-c", type=str, default="ocn", choices=["atm", "ice", "ocn"])
    parser.add_argument("--var", "-v", type=str, default=None)
    parser.add_argument("--init", "-i", type=int, default=5)
    parser.add_argument("--leads", "-L", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--target", "-t", type=int, nargs="+", default=None)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument("--ar-mode", type=str, default="members", choices=["members", "ens_mean"])
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    run_amplitude_diagnostic_cli(
        component=args.component,
        var_name=args.var,
        init_month=args.init,
        target_months=args.target,
        start_year=args.start_year,
        end_year=args.end_year,
        detrend=not args.no_detrend,
        leads_override=args.leads,
        ar_mode=args.ar_mode,
        debug=args.debug,
    )
