"""
Fixed Target Calendar Month Model Drift Diagnostic
Uses pcolormesh for crisp grid-cell resolution and an explicit 11-shade color norm.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVAILABLE_INITS = [3, 4, 5, 6, 7, 8, 11]


def resolve_inits_and_leads(target_month: int, max_lead: int = 11) -> list[tuple[str, int]]:
    """Resolve available (init_month, lead) pairs for a target calendar month."""
    inits_and_leads = []
    for init in AVAILABLE_INITS:
        lead = (target_month - init) % 12
        if 0 <= lead <= max_lead:
            init_str = f"{init:02d}"
            inits_and_leads.append((init_str, lead))

    inits_and_leads.sort(key=lambda x: x[1])
    return inits_and_leads


def create_bias_colormap():
    """
    Explicit 11-shade colormap with distinct colors for outer intervals and overflow caps.
    """
    bounds = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 5.0]
    
    # Sample 11 distinct colors across RdBu_r
    colors = plt.colormaps["RdBu_r"](np.linspace(0.0, 1.0, 11))
    
    # Inner 9 colors mapped to the 9 bounded intervals
    cmap = mcolors.ListedColormap(colors[1:-1])
    
    # Underflow (<-5.0) and Overflow (>5.0) get the darkest extreme shades
    cmap.set_under(colors[0])   # Deepest Navy Blue
    cmap.set_over(colors[-1])   # Deepest Maroon
    
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def run_fixed_month_drift_diagnostic(
    target_month: int = 7,  # July
    var_name: str = "TMP2m",
    selected_panels: int = 4,
    output_png: str = "july_t2m_drift_across_leads.png",
):
    """Execute fixed-month climatological drift evaluation across lead times."""
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    target_str = month_names[target_month - 1]

    inits_and_leads = resolve_inits_and_leads(target_month)
    
    if len(inits_and_leads) > selected_panels:
        idxs = np.linspace(0, len(inits_and_leads) - 1, selected_panels, dtype=int)
        inits_and_leads = [inits_and_leads[i] for i in idxs]

    logger.info(f"=== Target Month: {target_str} ({var_name}) ===")
    logger.info(f"Evaluated (Init, Lead) pairs: {inits_and_leads}")

    # 1. Load ERA5 Observational Climatology
    logger.info("Loading ERA5 observational baseline...")
    ds_era5_raw = get_era5_data(requested_vars=[var_name])
    ds_era5_base = ds_era5_raw.sel(time=slice("1991", "2020"))
    era5_clim = ds_era5_base.groupby("time.month").mean(dim="time", skipna=True)
    era5_target_da = era5_clim[var_name].sel(month=target_month).squeeze()

    if var_name == "TMP2m" and era5_target_da.mean() > 200:
        era5_target_da = era5_target_da - 273.15

    # 2. Compute Drift Fields
    drift_fields = {}
    for init_m, lead in inits_and_leads:
        label = f"Init {init_m} ({month_names[int(init_m)-1]}) | Lead {lead}"
        logger.info(f"Processing {label}...")

        ds_sfs_raw = get_sfs_data(init_month=init_m, domain="atm", requested_vars=[var_name])
        ds_sfs = preprocess_atm_dataset(ds_sfs_raw, target_res="1.0deg")

        sfs_clim = get_or_compute_sfs_climatology(ds_sfs, domain="atm", init_month=init_m)
        sfs_lead_da = sfs_clim[var_name].sel(lead=lead).squeeze()

        if var_name == "TMP2m" and sfs_lead_da.mean() > 200:
            sfs_lead_da = sfs_lead_da - 273.15

        bias_da = sfs_lead_da - era5_target_da
        drift_fields[label] = bias_da

    # 3. Plot Spatial Drift Grid
    plot_drift_grid(drift_fields, target_str, var_name, output_png=output_png)


def plot_drift_grid(
    drift_fields: dict[str, xr.DataArray],
    target_str: str,
    var_name: str,
    output_png: str,
):
    """Plot 2x2 grid using pcolormesh for crisp grid-cell boundaries."""
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

        # Sanitize longitude vector to eliminate NetCDF precision floating-point errors
        lons_clean = np.linspace(lons[0], lons[-1], len(lons))

        cyclic_data, cyclic_lon = add_cyclic_point(bias_vals, coord=lons_clean)
        lon2d, lat2d = np.meshgrid(cyclic_lon, lats)

        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black", zorder=5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":", zorder=5)
        ax.gridlines(draw_labels=False, linestyle=":", color="gray", alpha=0.4)

        # pcolormesh shows exact model grid cells without smoothing
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

        title_str = f"{var_name} Climatology Bias ({target_str} Target) | {label}\n[Mean Bias: {mean_bias:+.2f}°C | RMSE: {rmse:.2f}°C]"
        ax.set_title(title_str, fontsize=11, fontweight="bold", pad=8)

    # Colorbar with explicit extend='both'
    cbar_ax = fig.add_axes([0.25, 0.05, 0.50, 0.02])
    fig.colorbar(
        im, cax=cbar_ax, orientation="horizontal",
        ticks=bounds, extend="both",
        label=f"{target_str} {var_name} Climatological Bias (°C) [SFS minus ERA5]"
    )

    plt.subplots_adjust(wspace=0.08, hspace=0.18, bottom=0.12)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Fixed-month drift diagnostic saved to '{output_png}'!")


if __name__ == "__main__":
    run_fixed_month_drift_diagnostic(
        target_month=7,  # July
        var_name="TMP2m",
        output_png="july_t2m_drift_across_leads.png",
    )
