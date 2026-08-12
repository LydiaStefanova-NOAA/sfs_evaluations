"""
Spatial Diagnostic Script: Unpack High SNR into Signal Variance vs. Noise Variance Ratios
Supports Atmospheric (atm), Oceanic (ocn), and Sea Ice (ice) variables.

Minimal-variant updates (no forced target_res):
- Keep preprocess defaults per domain
- Canonicalize lon/lat and sort coordinates
- Explicitly align OBS to SFS grid before metric computation
- Stabilize SVR/NVR denominators via epsilon masks
- Optional debug logging for grid and NaN diagnostics
"""
import argparse
import logging
import os
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
from metrics.snr import _linear_detrend
from viz.spatial import _prepare_cyclic_grid, LAND_GRAY

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
DEFAULT_VARS = {"atm": "TMP2m", "ice": "aice_h", "ocn": "SSH"}


def _canonicalize_lonlat(da: xr.DataArray) -> xr.DataArray:
    """Convert lon to [0,360) and sort lon/lat when 1D."""
    out = da
    if "lon" in out.coords and out["lon"].ndim == 1:
        out = out.assign_coords(lon=(out["lon"] % 360)).sortby("lon")
    if "lat" in out.coords and out["lat"].ndim == 1:
        out = out.sortby("lat")
    return out


def _align_obs_to_sfs_grid(obs_da: xr.DataArray, sfs_da: xr.DataArray) -> xr.DataArray:
    """Interpolate OBS onto SFS lat/lon grid for strict collocation."""
    if not all(c in sfs_da.coords for c in ["lat", "lon"]):
        raise ValueError("SFS DataArray missing lat/lon coordinates.")
    if not all(c in obs_da.coords for c in ["lat", "lon"]):
        raise ValueError("OBS DataArray missing lat/lon coordinates.")

    obs_can = _canonicalize_lonlat(obs_da)
    sfs_can = _canonicalize_lonlat(sfs_da)

    return obs_can.interp(lat=sfs_can["lat"], lon=sfs_can["lon"], method="linear")


def _grid_report(tag: str, da: xr.DataArray) -> None:
    logger.info(f"[{tag}] dims={da.dims}, sizes={dict(da.sizes)}")
    if "lon" in da.coords and da["lon"].ndim == 1 and da.sizes.get("lon", 0) > 1:
        dlon = da["lon"].diff("lon").values
        logger.info(
            f"[{tag}] lon min={float(da.lon.min()):.3f}, max={float(da.lon.max()):.3f}, "
            f"median_step={float(np.nanmedian(np.abs(dlon))):.3f}, monotonic_inc={bool(np.all(dlon > 0))}"
        )
    if "lat" in da.coords and da["lat"].ndim == 1 and da.sizes.get("lat", 0) > 1:
        dlat = da["lat"].diff("lat").values
        logger.info(
            f"[{tag}] lat min={float(da.lat.min()):.3f}, max={float(da.lat.max()):.3f}, "
            f"median_step={float(np.nanmedian(np.abs(dlat))):.3f}, monotonic_inc={bool(np.all(dlat > 0))}"
        )


def _nan_report(tag: str, da: xr.DataArray) -> None:
    arr = da.values
    logger.info(f"[{tag}] nan_frac={float(np.isnan(arr).mean()):.4f}")


def compute_variance_ratios(
    sfs_da: xr.DataArray,
    obs_da: xr.DataArray,
    year_dim: str = "year",
    member_dim: str = "member",
    detrend: bool = True,
    varobs_eps: float = 1e-12,
    mse_eps: float = 1e-12,
    debug: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes Signal Variance Ratio (SVR) and Noise-to-MSE Variance Ratio (NVR).
    """
    common_years = np.intersect1d(sfs_da[year_dim].values, obs_da[year_dim].values)
    sfs_da = sfs_da.sel({year_dim: common_years})
    obs_da = obs_da.sel({year_dim: common_years})

    if detrend:
        ens_mean_raw = sfs_da.mean(dim=member_dim, skipna=True)
        poly_coeffs = ens_mean_raw.polyfit(dim=year_dim, deg=1)
        trend = xr.polyval(ens_mean_raw[year_dim], poly_coeffs.polyfit_coefficients)

        sfs_proc = sfs_da - trend
        obs_proc = _linear_detrend(obs_da, dim=year_dim)
    else:
        sfs_proc = sfs_da
        obs_proc = obs_da

    ens_mean = sfs_proc.mean(dim=member_dim, skipna=True)

    var_signal_mod = ens_mean.var(dim=year_dim, ddof=1, skipna=True)
    var_obs = obs_proc.var(dim=year_dim, ddof=1, skipna=True)

    internal_var = sfs_proc.var(dim=member_dim, ddof=1, skipna=True)
    var_noise_mod = internal_var.mean(dim=year_dim, skipna=True)

    mse = ((ens_mean - obs_proc) ** 2).mean(dim=year_dim, skipna=True)

    # Epsilon-stabilized denominators
    svr_da = var_signal_mod / var_obs.where(var_obs > varobs_eps)
    svr_da.name = "svr"
    svr_da.attrs["long_name"] = "Ratio of model signal vs obs variance (Var_signal / Var_obs)"

    nvr_da = var_noise_mod / mse.where(mse > mse_eps)
    nvr_da.name = "nvr"
    nvr_da.attrs["long_name"] = "Ratio of model noise vs MSE (Var_noise / MSE)"

    if debug:
        _nan_report("var_obs", var_obs)
        _nan_report("mse", mse)
        _nan_report("svr", svr_da)
        _nan_report("nvr", nvr_da)

    return svr_da, nvr_da


def plot_variance_diagnostics_map(
    svr_da: xr.DataArray,
    nvr_da: xr.DataArray,
    title_str: str = "SNR Variance Breakdown Diagnostic",
    output_png: str = "figures/variance_breakdown.png",
):
    """
    Plots a 2-Panel Side-by-Side Spatial Map:
    (a) Signal Variance Ratio (Var_signal_mod / Var_obs)
    (b) Noise-to-MSE Variance Ratio (Var_noise_mod / MSE)
    """
    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(1, 2, wspace=0.12, top=0.88, bottom=0.15, left=0.04, right=0.96)

    proj = ccrs.Robinson(central_longitude=0)
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    ax2 = fig.add_subplot(gs[0, 1], projection=proj)

    for ax in (ax1, ax2):
        ax.set_facecolor(LAND_GRAY)
        ax.add_feature(cfeature.LAND, facecolor=LAND_GRAY, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":", zorder=5)
        ax.gridlines(draw_labels=False, linestyle=":", color="gray", alpha=0.3, zorder=6)

    # Panel (a)
    lon2d, lat2d, cyclic_svr, svr_vals = _prepare_cyclic_grid(svr_da)

    svr_bounds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    svr_colors = ['#f7fcf5', '#e0f3db', '#c7e9c0', '#a1d99b', '#fdae6b', '#f16913', '#d94801', '#8c2d04']
    cmap_svr = mcolors.ListedColormap(svr_colors)
    cmap_svr.set_bad(color=(0, 0, 0, 0))
    cmap_svr.set_over('#4a1403')
    norm_svr = mcolors.BoundaryNorm(svr_bounds, cmap_svr.N)

    im1 = ax1.pcolormesh(
        lon2d, lat2d, cyclic_svr, cmap=cmap_svr, norm=norm_svr, shading="auto",
        transform=ccrs.PlateCarree(), zorder=3
    )
    med_svr = float(np.nanmedian(svr_vals))
    ax1.set_title(
        rf"$\mathbf{{(a)\ Signal\ Variance\ Ratio}}\ (\sigma^2_{{\mathrm{{signal,mod}}}} / \sigma^2_{{\mathrm{{obs}}}}) \mid \mathbf{{Spatial\ Median:\ {med_svr:.2f}}}$"
        "\n"
        "[ ≤ 1.0: Physically Bounded  |  > 1.0: Signal Inflated ]",
        fontsize=9, pad=6
    )

    cbar_ax1 = fig.add_axes([0.08, 0.08, 0.38, 0.025])
    cbar1 = fig.colorbar(
        im1, cax=cbar_ax1, orientation="horizontal", ticks=svr_bounds, extend="max",
        label="Signal Variance Ratio"
    )
    cbar1.ax.tick_params(labelsize=8)

    # Panel (b)
    _, _, cyclic_nvr, nvr_vals = _prepare_cyclic_grid(nvr_da)

    nvr_bounds = [0.0, 0.1, 0.25, 0.5, 0.8, 1.2, 2.0, 5.0]
    nvr_colors = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c7e9c0', '#fdae6b', '#e6550d']
    cmap_nvr = mcolors.ListedColormap(nvr_colors)
    cmap_nvr.set_bad(color=(0, 0, 0, 0))
    cmap_nvr.set_over('#a50f15')
    norm_nvr = mcolors.BoundaryNorm(nvr_bounds, cmap_nvr.N)

    im2 = ax2.pcolormesh(
        lon2d, lat2d, cyclic_nvr, cmap=cmap_nvr, norm=norm_nvr, shading="auto",
        transform=ccrs.PlateCarree(), zorder=3
    )
    med_nvr = float(np.nanmedian(nvr_vals))
    ax2.set_title(
        rf"$\mathbf{{(b)\ Noise\text{{-}}to\text{{-}}MSE\ Variance\ Ratio}}\ (\sigma^2_{{\mathrm{{noise,mod}}}} / \mathrm{{MSE}}) \mid \mathbf{{Spatial\ Median:\ {med_nvr:.2f}}}$"
        "\n"
        "[ « 1.0: Severe Under-Dispersion  |  ≈ 1.0: Well-Calibrated Spread ]",
        fontsize=9, pad=6
    )

    cbar_ax2 = fig.add_axes([0.54, 0.08, 0.38, 0.025])
    cbar2 = fig.colorbar(
        im2, cax=cbar_ax2, orientation="horizontal", ticks=nvr_bounds, extend="max",
        label="Noise-to-MSE Variance Ratio"
    )
    cbar2.ax.tick_params(labelsize=8)

    fig.suptitle(title_str, fontsize=12, fontweight="bold", y=0.97)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Spatial variance breakdown plot saved to '{output_png}'!")


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
    debug: bool = False,
):
    comp = component.lower()
    if comp not in PREPROCESS_MAP:
        raise ValueError(f"Invalid component '{component}'. Must be one of: {list(PREPROCESS_MAP.keys())}")

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

    # 1) SFS (keep preprocess defaults; do not force target_res)
    ds_sfs_raw = get_sfs_data(init_month=init_month, domain=comp, requested_vars=[var_name])
    ds_sfs = PREPROCESS_MAP[comp](ds_sfs_raw)
    sfs_season_da = ds_sfs[var_name].sel(lead=leads).mean(dim="lead", skipna=True)

    # 2) OBS aligned by init-year + leads
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

    # 3) Standardize year dim
    if "init" in sfs_season_da.dims and "year" not in sfs_season_da.dims:
        sfs_season_da = sfs_season_da.rename({"init": "year"})

    if np.issubdtype(sfs_season_da.year.dtype, np.datetime64):
        sfs_season_da["year"] = sfs_season_da.year.dt.year
    if np.issubdtype(obs_season_da.year.dtype, np.datetime64):
        obs_season_da["year"] = obs_season_da.year.dt.year

    # Canonicalize coordinates before temporal intersection and alignment
    sfs_season_da = _canonicalize_lonlat(sfs_season_da)
    obs_season_da = _canonicalize_lonlat(obs_season_da)

    common_years = np.intersect1d(sfs_season_da.year.values, obs_season_da.year.values)
    eval_years = [y for y in common_years if start_year <= y <= end_year]
    if len(eval_years) == 0:
        raise ValueError(f"No overlapping years found in range {start_year}-{end_year}.")

    sfs_season_da = sfs_season_da.sel(year=eval_years).squeeze(drop=True)
    obs_season_da = obs_season_da.sel(year=eval_years).squeeze(drop=True)

    # Explicit collocation: obs -> sfs grid
    obs_season_da = _align_obs_to_sfs_grid(obs_season_da, sfs_season_da)

    # Eager load
    sfs_season_da = sfs_season_da.load()
    obs_season_da = obs_season_da.load()

    actual_start, actual_end = eval_years[0], eval_years[-1]
    logger.info(f"Evaluation window: {len(eval_years)} actual years ({actual_start}-{actual_end}).")
    logger.info(
        f"SFS years={sfs_season_da.year.values.min()}..{sfs_season_da.year.values.max()} | "
        f"OBS years={obs_season_da.year.values.min()}..{obs_season_da.year.values.max()}"
    )

    if debug:
        _grid_report("SFS seasonal", sfs_season_da)
        _grid_report("OBS seasonal (aligned)", obs_season_da)
        _nan_report("SFS seasonal", sfs_season_da)
        _nan_report("OBS seasonal (aligned)", obs_season_da)

    if output_png is None:
        detrend_str = "_detrended" if detrend else ""
        output_png = (
            f"figures/{comp}_{season_str.lower()}_{var_name.lower()}_variance_breakdown_"
            f"init{init_month:02d}_{actual_start}-{actual_end}{detrend_str}.png"
        )

    # 4) Metrics
    logger.info(f"Computing Variance Ratios (SVR & NVR) against {obs_label}...")
    svr_da, nvr_da = compute_variance_ratios(
        sfs_season_da,
        obs_season_da,
        detrend=detrend,
        varobs_eps=varobs_eps,
        mse_eps=mse_eps,
        debug=debug,
    )

    # Keep coordinate order stable for plotting
    svr_da = _canonicalize_lonlat(svr_da)
    nvr_da = _canonicalize_lonlat(nvr_da)

    # 5) Plot
    title_str = (
        f"{component}: SFS {var_name} {season_str} Signal vs. Noise Variance "
        f"({actual_start}–{actual_end}) | {label}"
    )
    plot_variance_diagnostics_map(
        svr_da=svr_da,
        nvr_da=nvr_da,
        title_str=title_str,
        output_png=output_png,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deconstruct SNR into Signal Variance and Noise Variance Ratios")
    parser.add_argument("--component", "-c", type=str, default="ocn", choices=["atm", "ice", "ocn"])
    parser.add_argument("--var", "-v", type=str, default=None)
    parser.add_argument("--init", "-i", type=int, default=5)
    parser.add_argument("--leads", "-L", type=int, nargs="+", default=None)
    parser.add_argument("--target", "-t", type=int, nargs="+", default=[1, 2, 3])  # legacy path
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--no-detrend", action="store_true")

    # minimal new flags
    parser.add_argument("--varobs-eps", type=float, default=1e-12)
    parser.add_argument("--mse-eps", type=float, default=1e-12)
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
        debug=args.debug,
    )
