"""
Spatial Trend Diagnostic Script: Model vs Observed Linear Trends
Supports Atmospheric (atm), Oceanic (ocn), and Sea Ice (ice) variables.

Design mirrors breakdown.py and test_unified_snr_acc.py:
- same registry style
- same lead/season resolution pattern
- same preprocessing + temporal alignment helpers
- same map styling conventions
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

# Units map for colorbar labels (extend as needed)
VAR_UNITS_MAP = {
    "TMP2m": "K",
    "Z500": "m",
    "HGT500": "m",
    "Z200": "m",
    "Z700": "m",
    "Z850": "m",
    "MSLP": "Pa",
    "PRATE": "mm/day",
    "U10m": "m/s",
    "V10m": "m/s",
    "T850": "K",
    "T200": "K",
    "U850": "m/s",
    "V850": "m/s",
    "U200": "m/s",
    "V200": "m/s",
    "SST": "°C",
    "SSH": "m",
    "SSS": "psu",
    "MLD_003": "m",
    "MLD_0125": "m",
    "dt20c": "m",
    "ocnheat": "J/m²",
    "taux": "N/m²",
    "tauy": "N/m²",
    "aice_h": "fraction",
    "hi_h": "m",
    "hs_h": "m",
    "Tsfc_h": "°C",
    "uvel_h": "m/s",
    "vvel_h": "m/s",
}


def _compute_linear_trend_per_year(da: xr.DataArray, year_dim: str = "year") -> xr.DataArray:
    """
    Compute linear trend slope per year at each gridpoint.
    If 'member' exists, trend is computed on ensemble mean.
    """
    src = da
    if "member" in src.dims:
        src = src.mean(dim="member", skipna=True)

    if year_dim not in src.dims:
        raise ValueError(f"'{year_dim}' dimension not present. dims={src.dims}")

    if np.issubdtype(src[year_dim].dtype, np.datetime64):
        src = src.assign_coords({year_dim: src[year_dim].dt.year})

    coeff = src.polyfit(dim=year_dim, deg=1, skipna=True).polyfit_coefficients
    slope = coeff.sel(degree=1, drop=True)
    slope.name = "trend_per_year"
    return slope


def _symmetric_trend_levels(
    model_trend: xr.DataArray,
    obs_trend: xr.DataArray,
    q: float = 0.98,
    n_levels: int = 17,
):
    """
    Build shared symmetric bounds for model/obs panels from pooled trend magnitudes.
    """
    m = np.abs(model_trend.values.ravel())
    o = np.abs(obs_trend.values.ravel())
    pooled = np.concatenate([m[np.isfinite(m)], o[np.isfinite(o)]])
    if pooled.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.nanquantile(pooled, q))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(pooled)) if np.nanmax(pooled) > 0 else 1.0

    bounds = np.linspace(-vmax, vmax, n_levels)
    return bounds


def _draw_manual_graticules(ax):
    """
    Draw robust dashed lat/lon lines manually (works reliably across backends/projections).
    """
    # Meridians every 60°
    for lon in np.arange(-180, 181, 60):
        lats = np.linspace(-89.5, 89.5, 360)
        lons = np.full_like(lats, lon, dtype=float)
        ax.plot(
            lons,
            lats,
            transform=ccrs.PlateCarree(),
            linestyle="--",
            linewidth=0.3,
            color="black",
            alpha=0.35,
            zorder=20,
        )

    # Parallels every 30° (avoid exact poles)
    for lat in np.arange(-60, 61, 30):
        lons = np.linspace(-180, 180, 720)
        lats = np.full_like(lons, lat, dtype=float)
        ax.plot(
            lons,
            lats,
            transform=ccrs.PlateCarree(),
            linestyle="--",
            linewidth=0.6,
            color="black",
            alpha=0.35,
            zorder=20,
        )


def plot_model_obs_trend_maps(
    model_trend_da: xr.DataArray,
    obs_trend_da: xr.DataArray,
    obs_name: str,
    var_name: str,
    units: str,
    title_str: str,
    output_png: str,
):
    """
    Plot side-by-side trends using identical color scheme/range for model and obs.
    """
    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(1, 2, wspace=0.12, top=0.88, bottom=0.15, left=0.04, right=0.96)

    proj = ccrs.Robinson(central_longitude=0)
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    ax2 = fig.add_subplot(gs[0, 1], projection=proj)

    for ax in (ax1, ax2):
        ax.set_facecolor(LAND_GRAY)
        ax.add_feature(cfeature.LAND, facecolor=LAND_GRAY, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="black", zorder=12)
        ax.add_feature(cfeature.BORDERS, linewidth=0.45, edgecolor="dimgray", linestyle=":", zorder=12)

    # Prepare cyclic grids separately for each panel field
    lon2d_mod, lat2d_mod, cyclic_mod, mod_vals = _prepare_cyclic_grid(model_trend_da)
    lon2d_obs, lat2d_obs, cyclic_obs, obs_vals = _prepare_cyclic_grid(obs_trend_da)

    bounds = _symmetric_trend_levels(model_trend_da, obs_trend_da, q=0.98, n_levels=17)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color=(0, 0, 0, 0))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im1 = ax1.pcolormesh(
        lon2d_mod, lat2d_mod, cyclic_mod, cmap=cmap, norm=norm, shading="auto",
        transform=ccrs.PlateCarree(), zorder=3
    )
    med_mod = float(np.nanmedian(mod_vals))
    ax1.set_title(
        rf"$\mathbf{{(a)\ SFS\ Trend}}\ \mid\ \mathbf{{Spatial\ Median:\ {med_mod:.3g}\ {units}/yr}}$",
        fontsize=10, pad=6
    )

    im2 = ax2.pcolormesh(
        lon2d_obs, lat2d_obs, cyclic_obs, cmap=cmap, norm=norm, shading="auto",
        transform=ccrs.PlateCarree(), zorder=3
    )
    med_obs = float(np.nanmedian(obs_vals))
    ax2.set_title(
        rf"$\mathbf{{(b)\ {obs_name}\ Trend}}\ \mid\ \mathbf{{Spatial\ Median:\ {med_obs:.3g}\ {units}/yr}}$",
        fontsize=10, pad=6
    )

    # Draw lat/lon dashed lines on top of color field
    _draw_manual_graticules(ax1)
    _draw_manual_graticules(ax2)

    cbar_ax = fig.add_axes([0.12, 0.08, 0.76, 0.028])
    cbar = fig.colorbar(
        im2, cax=cbar_ax, orientation="horizontal",
        ticks=np.linspace(bounds.min(), bounds.max(), 9)
    )
    cbar.set_label(f"Linear Trend ({units}/year)")
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(title_str, fontsize=12, fontweight="bold", y=0.97)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Model/obs trend plot saved to '{output_png}'")


def run_trend_diagnostic_cli(
    component: str = "ocn",
    var_name: str = None,
    target_months: list[int] = [6, 7, 8],
    init_month: int = 5,
    start_year: int = 1991,
    end_year: int = 2022,
    output_png: str = None,
    leads_override: list[int] = None,
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
    obs_label = OBS_NAME_MAP[comp]
    units = VAR_UNITS_MAP.get(var_name, "units")

    logger.info(f"=== Starting {season_str} {domain_label} Trend Diagnostic ({var_name}) ===")

    # 1) SFS
    ds_sfs_raw = get_sfs_data(init_month=init_month, domain=comp, requested_vars=[var_name])
    ds_sfs = PREPROCESS_MAP[comp](ds_sfs_raw)
    sfs_season_da = ds_sfs[var_name].sel(lead=leads).mean(dim="lead", skipna=True)

    # 2) OBS (aligned by init-year + leads)
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

    # 3) Standardize + canonicalize + overlap years
    sfs_season_da = standardize_year_dim(sfs_season_da)
    obs_season_da = standardize_year_dim(obs_season_da)

    sfs_season_da = canonicalize_lonlat(sfs_season_da)
    obs_season_da = canonicalize_lonlat(obs_season_da)

    sfs_season_da, obs_season_da, eval_years = select_common_eval_years(
        sfs_season_da, obs_season_da, start_year=start_year, end_year=end_year
    )

    # Explicit spatial collocation
    obs_season_da = align_obs_to_sfs_grid(obs_season_da, sfs_season_da, method="linear")

    # Load
    sfs_season_da = sfs_season_da.squeeze(drop=True).load()
    obs_season_da = obs_season_da.squeeze(drop=True).load()

    if debug:
        grid_report("SFS seasonal", sfs_season_da)
        grid_report("OBS seasonal aligned", obs_season_da)
        nan_report("SFS seasonal", sfs_season_da)
        nan_report("OBS seasonal aligned", obs_season_da)

    actual_start, actual_end = eval_years[0], eval_years[-1]
    logger.info(f"Evaluation window: {len(eval_years)} actual years ({actual_start}-{actual_end}).")

    # 4) Trends
    logger.info("Computing linear trends...")
    model_trend_da = _compute_linear_trend_per_year(sfs_season_da, year_dim="year")
    obs_trend_da = _compute_linear_trend_per_year(obs_season_da, year_dim="year")

    model_trend_da = canonicalize_lonlat(model_trend_da)
    obs_trend_da = canonicalize_lonlat(obs_trend_da)

    # 5) Plot
    if output_png is None:
        output_png = (
            f"figures/{comp}_{season_str.lower()}_{var_name.lower()}_trend_model_vs_obs_"
            f"init{init_month:02d}_{actual_start}-{actual_end}.png"
        )

    title_str = (
        f"SFS {var_name} {season_str} Linear Trend vs {obs_label} "
        f"({actual_start}–{actual_end}) | {label}"
    )

    plot_model_obs_trend_maps(
        model_trend_da=model_trend_da,
        obs_trend_da=obs_trend_da,
        obs_name=obs_label,
        var_name=var_name,
        units=units,
        title_str=title_str,
        output_png=output_png,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot model vs observed spatial linear trends")
    parser.add_argument("--component", "-c", type=str, default="ocn", choices=["atm", "ice", "ocn"])
    parser.add_argument("--var", "-v", type=str, default=None)
    parser.add_argument("--init", "-i", type=int, default=5)
    parser.add_argument("--leads", "-L", type=int, nargs="+", default=None)
    parser.add_argument("--target", "-t", type=int, nargs="+", default=[1, 2, 3])  # legacy path
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--output-png", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    run_trend_diagnostic_cli(
        component=args.component,
        var_name=args.var,
        init_month=args.init,
        target_months=args.target,
        start_year=args.start_year,
        end_year=args.end_year,
        output_png=args.output_png,
        leads_override=args.leads,
        debug=args.debug,
    )
