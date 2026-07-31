"""
Fixed Target Season Model Drift Diagnostic Driver
Imports target lead mapping from preprocess.temporal.
"""
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr

from sources.sfs import get_sfs_data
from sources.era5 import get_era5_data
from preprocess.pipeline import preprocess_atm_dataset
from preprocess.cache import get_or_compute_sfs_climatology
from preprocess.temporal import resolve_target_season_leads, get_season_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_bias_colormap():
    """Explicit 11-shade colormap with distinct colors for outer intervals and overflow caps."""
    bounds = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 5.0]
    colors = plt.colormaps["RdBu_r"](np.linspace(0.0, 1.0, 11))
    cmap = mcolors.ListedColormap(colors[1:-1])
    cmap.set_under(colors[0])   # Deep Navy Blue (<-5.0)
    cmap.set_over(colors[-1])   # Deep Maroon (>5.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def run_seasonal_drift_diagnostic(
    target_months: list[int] = [6, 7, 8],  # Default: JJA
    var_name: str = "TMP2m",
    selected_panels: int = 4,
    output_png: str = None,
):
    """Execute seasonal climatological drift evaluation across lead times."""
    season_str = get_season_name(target_months)
    if output_png is None:
        output_png = f"{season_str.lower()}_{var_name.lower()}_drift_across_leads.png"

    # Automatically resolve available (init, lead_list, label) tuples from preprocess.temporal
    init_leads_list = resolve_target_season_leads(target_months)

    if len(init_leads_list) > selected_panels:
        idxs = np.linspace(0, len(init_leads_list) - 1, selected_panels, dtype=int)
        init_leads_list = [init_leads_list[i] for i in idxs]

    logger.info(f"=== Starting {season_str} Seasonal Drift Diagnostic ({var_name}) ===")

    # 1. Load ERA5 Observational Baseline for Target Season
    ds_era5_raw = get_era5_data(requested_vars=[var_name])
    ds_era5_base = ds_era5_raw.sel(time=slice("1991", "2020"))
    era5_season = ds_era5_base.where(ds_era5_base["time.month"].isin(target_months), drop=True)
    era5_target_da = era5_season[var_name].mean(dim="time", skipna=True).squeeze()

    if var_name == "TMP2m" and era5_target_da.mean() > 200:
        era5_target_da = era5_target_da - 273.15

    # 2. Compute Drift Fields
    drift_fields = {}
    for init_m, leads, label in init_leads_list:
        logger.info(f"Processing {label}...")

        ds_sfs_raw = get_sfs_data(init_month=init_m, domain="atm", requested_vars=[var_name])
        ds_sfs = preprocess_atm_dataset(ds_sfs_raw, target_res="1.0deg")

        sfs_clim = get_or_compute_sfs_climatology(ds_sfs, domain="atm", init_month=init_m)
        sfs_season_da = sfs_clim[var_name].sel(lead=leads).mean(dim="lead", skipna=True).squeeze()

        if var_name == "TMP2m" and sfs_season_da.mean() > 200:
            sfs_season_da = sfs_season_da - 273.15

        drift_fields[label] = sfs_season_da - era5_target_da

    # 3. Plot Grid
    plot_drift_grid(drift_fields, season_str, var_name, output_png=output_png)


def plot_drift_grid(
    drift_fields: dict[str, xr.DataArray],
    season_str: str,
    var_name: str,
    output_png: str,
):
    """Plot grid of bias fields using pcolormesh."""
    cmap, norm, bounds = create_bias_colormap()
    labels = list(drift_fields.keys())

    fig, axes = plt.subplots(
        2, 2, figsize=(18, 11),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=0)}
    )
    axes_flat = axes.flatten()

    for idx, label in enumerate(labels):
        ax = axes_flat[idx]
        bias_da = drift_fields[label]

        bias_vals = bias_da.values
        lats = bias_da.lat.values
        lons = bias_da.lon.values

        lons_clean = np.linspace(lons[0], lons[-1], len(lons))
        cyclic_data, cyclic_lon = add_cyclic_point(bias_vals, coord=lons_clean)
        lon2d, lat2d = np.meshgrid(cyclic_lon, lats)

        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black", zorder=5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":", zorder=5)
        ax.gridlines(draw_labels=False, linestyle=":", color="gray", alpha=0.4)

        im = ax.pcolormesh(
            lon2d, lat2d, cyclic_data,
            cmap=cmap,
            norm=norm,
            shading="auto",
            transform=ccrs.PlateCarree(),
            zorder=1,
        )

        mean_bias = float(bias_da.mean().values)
        rmse = float(np.sqrt((bias_da ** 2).mean()).values)

        title_str = f"{var_name} Climatology Bias ({season_str} Target) | {label}\n[Mean Bias: {mean_bias:+.2f}°C | RMSE: {rmse:.2f}°C]"
        ax.set_title(title_str, fontsize=11, fontweight="bold", pad=8)

    cbar_ax = fig.add_axes([0.25, 0.05, 0.50, 0.02])
    fig.colorbar(
        im, cax=cbar_ax, orientation="horizontal",
        ticks=bounds, extend="both",
        label=f"{season_str} {var_name} Climatological Bias (°C) [SFS minus ERA5]"
    )

    plt.subplots_adjust(wspace=0.08, hspace=0.18, bottom=0.12)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Seasonal drift diagnostic saved to '{output_png}'!")


if __name__ == "__main__":
    run_seasonal_drift_diagnostic(
        target_months=[12, 1, 2],  # DJF
        var_name="TMP2m",
    )
