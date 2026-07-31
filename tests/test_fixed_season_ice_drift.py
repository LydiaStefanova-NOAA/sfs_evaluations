"""
Fixed Target Season Model Drift Diagnostic for Sea Ice Concentration (aice_h)
Integrates SFS Ice domain, ORAS5 reanalysis verification, preprocess.temporal,
and dynamic multi-panel grid visualization.
"""
import logging
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

from sources.sfs import get_sfs_data
from sources.oras5 import get_oras5_data
from preprocess.pipeline import preprocess_ice_dataset
from preprocess.cache import get_or_compute_sfs_climatology
from preprocess.temporal import resolve_target_season_leads, get_season_name

# Suppress harmless interpolation land-slice NaNs
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_ice_bias_colormap():
    """
    Explicit 11-shade colormap for Sea Ice Concentration Bias (%).
    Bounds: [-50%, -30%, -20%, -10%, -5%, +5%, +10%, +20%, +30%, +50%]
    """
    bounds = [-50.0, -30.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 30.0, 50.0]
    colors = plt.colormaps["RdBu_r"](np.linspace(0.0, 1.0, 11))
    cmap = mcolors.ListedColormap(colors[1:-1])
    cmap.set_under(colors[0])   # Deep Navy Blue (<-50%)
    cmap.set_over(colors[-1])   # Deep Maroon (>+50%)
    
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def run_ice_drift_diagnostic(
    target_months: list[int] = [6, 7, 8],  # Default: JJA (Summer Melt Window)
    var_name: str = "aice_h",
    selected_panels: int | None = None,   # None = plot ALL valid initialization months
    output_png: str = None,
):
    """Execute seasonal sea ice concentration climatological drift evaluation across lead times."""
    season_str = get_season_name(target_months)
    if output_png is None:
        suffix = "all_leads" if selected_panels is None else f"{selected_panels}_leads"
        output_png = f"{season_str.lower()}_ice_drift_{suffix}.png"

    # Automatically resolve all valid (init, lead_list, label) tuples from preprocess.temporal
    init_leads_list = resolve_target_season_leads(target_months)

    # Subsample only if selected_panels is explicitly requested
    if selected_panels is not None and len(init_leads_list) > selected_panels:
        idxs = np.linspace(0, len(init_leads_list) - 1, selected_panels, dtype=int)
        init_leads_list = [init_leads_list[i] for i in idxs]

    logger.info(f"=== Starting {season_str} Sea Ice Concentration Drift Diagnostic ({var_name}) ===")
    logger.info(f"Evaluating {len(init_leads_list)} initialization window(s): {[item[2] for item in init_leads_list]}")

    # 1. Load ORAS5 Observational Baseline (ileadfra -> aice_h)
    logger.info("Loading ORAS5 observational baseline (1991-2019)...")
    ds_oras5_raw = get_oras5_data(requested_vars=[var_name])
    ds_oras5_base = ds_oras5_raw.sel(time=slice("1991", "2019"))
    
    oras5_season = ds_oras5_base.where(ds_oras5_base["time.month"].isin(target_months), drop=True)
    oras5_target_da = oras5_season[var_name].mean(dim="time", skipna=True).squeeze()

    # Convert fraction (0..1) to percentage (0..100) if needed
    if float(oras5_target_da.max()) <= 1.0:
        oras5_target_da = oras5_target_da * 100.0

    # 2. Compute Drift Fields across Initializations
    drift_fields = {}
    for init_m, leads, label in init_leads_list:
        logger.info(f"Processing {label}...")

        # Ingest SFS raw ice store and run through preprocess_ice_dataset
        ds_sfs_raw = get_sfs_data(init_month=init_m, domain="ice", requested_vars=[var_name])
        ds_sfs = preprocess_ice_dataset(ds_sfs_raw, target_res="1.0deg")

        # Compute or load disk cached SFS climatology
        sfs_clim = get_or_compute_sfs_climatology(ds_sfs, domain="ice", init_month=init_m)
        sfs_season_da = sfs_clim[var_name].sel(lead=leads).mean(dim="lead", skipna=True).squeeze()

        # Convert fraction to percentage if needed
        if float(sfs_season_da.max()) <= 1.0:
            sfs_season_da = sfs_season_da * 100.0

        # Align ORAS5 grid to SFS 1.0deg target grid if dimensions differ
        if sfs_season_da.shape != oras5_target_da.shape:
            oras5_aligned = oras5_target_da.interp(
                lat=sfs_season_da.lat, 
                lon=sfs_season_da.lon, 
                method="linear"
            )
        else:
            oras5_aligned = oras5_target_da

        bias_da = sfs_season_da - oras5_aligned
        drift_fields[label] = bias_da

    # 3. Render Plot Grid
    plot_drift_grid(drift_fields, season_str, output_png=output_png)


def plot_drift_grid(
    drift_fields: dict[str, xr.DataArray],
    season_str: str,
    output_png: str,
):
    """Plot grid of ice concentration bias fields using dynamic subplot arrangements."""
    cmap, norm, bounds = create_ice_bias_colormap()
    labels = list(drift_fields.keys())
    n_panels = len(labels)

    # Dynamic grid arrangement
    if n_panels == 1:
        nrows, ncols = 1, 1
    elif n_panels == 2:
        nrows, ncols = 1, 2
    elif n_panels <= 4:
        nrows, ncols = 2, 2
    elif n_panels <= 6:
        nrows, ncols = 2, 3
    else:
        ncols = 3
        nrows = int(np.ceil(n_panels / ncols))

    fig_w = 6.0 * ncols
    fig_h = 4.0 * nrows + 1.2

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=0)},
        squeeze=False,
    )
    axes_flat = axes.flatten()

    im = None
    for idx, label in enumerate(labels):
        ax = axes_flat[idx]
        bias_da = drift_fields[label]
        bias_vals = bias_da.values

        # 1. Extract coordinates and unmask if needed
        lats = np.asarray(bias_da.lat.values, dtype=float)
        lons = np.asarray(bias_da.lon.values, dtype=float)

        if np.ma.is_masked(lats):
            lats = lats.filled(np.nan)
        if np.ma.is_masked(lons):
            lons = lons.filled(np.nan)

        if lons.ndim > 1:
            lons = lons[0, :]
        if lats.ndim > 1:
            lats = lats[:, 0]

        if not np.all(np.isfinite(lons)):
            lons = np.linspace(0.0, 360.0, bias_vals.shape[1], endpoint=False)
        if not np.all(np.isfinite(lats)):
            lats = np.linspace(-90.0, 90.0, bias_vals.shape[0])

        # 2. Manual cyclic longitude wrap
        dlon = (lons[-1] - lons[0]) / (len(lons) - 1)
        cyclic_lon = np.append(lons, lons[-1] + dlon)
        cyclic_data = np.concatenate([bias_vals, bias_vals[:, 0:1]], axis=1)

        lon2d, lat2d = np.meshgrid(cyclic_lon, lats)

        # 3. Add map features and pcolormesh raster
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

        # 4. Safe metric calculations (handling all-NaN slices cleanly)
        if np.all(np.isnan(bias_vals)):
            stats_str = "[Mean Bias: N/A | RMSE: N/A]"
        else:
            mean_bias = float(np.nanmean(bias_vals))
            rmse = float(np.sqrt(np.nanmean(bias_vals ** 2)))
            stats_str = f"[Mean Bias: {mean_bias:+.1f}% | RMSE: {rmse:.1f}%]"

        # Compact individual panel title to prevent horizontal collisions
        ax.set_title(f"{label}\n{stats_str}", fontsize=10, fontweight="bold", pad=6)

    # Hide unneeded subplots
    for idx in range(n_panels, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    # Overarching Figure Title
    fig.suptitle(
        f"Sea Ice Concentration Climatological Bias ({season_str} Target) [SFS minus ORAS5]",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Shared colorbar
    cbar_bottom = 0.05 if nrows > 2 else 0.07
    cbar_ax = fig.add_axes([0.25, cbar_bottom, 0.50, 0.02])
    fig.colorbar(
        im, cax=cbar_ax, orientation="horizontal",
        ticks=bounds, extend="both",
        label="Sea Ice Concentration Bias (%)"
    )

    # Increased horizontal (wspace) and vertical (hspace) padding
    plt.subplots_adjust(wspace=0.15, hspace=0.32, top=0.91, bottom=cbar_bottom + 0.06)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Sea Ice Concentration drift diagnostic saved to '{output_png}'!")


if __name__ == "__main__":
    # Example: Run full JJA (June-July-August) drift diagnostic showing all valid inits
    run_ice_drift_diagnostic(
        target_months=[7],  # DJF Target
    )
