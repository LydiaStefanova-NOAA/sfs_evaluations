"""
Atmospheric Signal-to-Noise Ratio (SNR) Diagnostic Driver
Styling aligned with Panel A NOAA Verification Diagnostics:
  - Robinson Map Projection
  - Unmasked Land (Full global data display)
  - Custom Panel A Discrete Colormap (Yellow -> Green -> Purple)
  - Multi-tier SNR Hatching (0.22-0.5, 0.5-1, 1-2, >2)
"""
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr

from sources.sfs import get_sfs_data
from preprocess.pipeline import preprocess_atm_dataset
from preprocess.climatology import compute_anomalies
from preprocess.detrend import detrend_dim
from preprocess.temporal import select_target_leads
from metrics.snr import compute_signal_to_noise_ratio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure crisp hatch lines globally
plt.rcParams['hatch.linewidth'] = 0.6
plt.rcParams['hatch.color'] = 'black'


def create_panel_a_colormap():
    """
    Create discrete colormap & norm matching Panel A (Yellow -> Green -> Purple).
    """
    bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hex_colors = [
        "#fffae1",  # 0.1 - 0.2: Cream Yellow
        "#fce892",  # 0.2 - 0.3: Light Yellow
        "#b2e09b",  # 0.3 - 0.4: Soft Mint Green
        "#52c366",  # 0.4 - 0.5: Mid Green
        "#009d3e",  # 0.5 - 0.6: Vibrant Green
        "#006b27",  # 0.6 - 0.7: Dark Green
        "#b5b7e2",  # 0.7 - 0.8: Lavender
        "#7672bd",  # 0.8 - 0.9: Purple
        "#4b4395",  # 0.9 - 1.0: Deep Royal Purple
    ]
    cmap = mcolors.ListedColormap(hex_colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def plot_snr_grid(
    results: dict[str, xr.Dataset],
    vars_to_plot: list[str],
    output_png: str = "atm_snr_may_init.png",
):
    """
    Generate a 2x2 grid plot formatted in the style of Panel A.
    """
    labels = list(results.keys())
    cmap, norm, bounds = create_panel_a_colormap()

    fig, axes = plt.subplots(
        len(vars_to_plot), len(labels),
        figsize=(18, 11),
        squeeze=False,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=0)}
    )

    for i, var in enumerate(vars_to_plot):
        for j, label in enumerate(labels):
            ax = axes[i, j]
            ds_snr = results[label]

            # Identify target variable
            main_vars = [v for v in ds_snr.data_vars if not v.endswith("_signal") and not v.endswith("_noise")]
            var_key = var if var in ds_snr.data_vars else main_vars[min(i, len(main_vars) - 1)]

            snr_da = ds_snr[var_key].squeeze()

            # Ensure cyclic point so Robinson projection renders without a longitude gap
            snr_vals, lons, lats = snr_da.values, snr_da.lon.values, snr_da.lat.values
            cyclic_data, cyclic_lon = add_cyclic_point(snr_vals, coord=lons)
            lon2d, lat2d = np.meshgrid(cyclic_lon, lats)

            # Map Base Setup (Unmasked Land, Coastlines on top)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black", zorder=5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":", zorder=5)
            ax.gridlines(draw_labels=False, linestyle=":", color="gray", alpha=0.4)

            # 1. Base Filled Contour Field (Color Coding)
            im = ax.contourf(
                lon2d, lat2d, cyclic_data,
                levels=bounds,
                cmap=cmap,
                norm=norm,
                extend="both",
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

            # 2. SNR Multi-Tier Hatching Overlay (Panel A specifications)
            snr_levels = [0.22, 0.5, 1.0, 2.0, 100.0]
            snr_hatches = ["..", "//", "xx", "...."]

            ax.contourf(
                lon2d, lat2d, cyclic_data,
                levels=snr_levels,
                hatches=snr_hatches,
                colors="none",  # Transparent fill so base colors show through
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

            # Metadata Titles
            mean_snr = float(snr_da.mean().values)
            max_snr = float(snr_da.max().values)

            title_str = f"{var_key} SNR | {label}\n[Mean SNR: {mean_snr:.2f} | Max SNR: {max_snr:.2f}]"
            ax.set_title(title_str, fontsize=11, fontweight="bold", pad=10)

    # Hatching Legend (Matching Panel A bottom legend)
    hatch_patches = [
        Patch(facecolor="none", edgecolor="black", hatch="..", label="SNR 0.22 - 0.5"),
        Patch(facecolor="none", edgecolor="black", hatch="//", label="SNR 0.5 - 1"),
        Patch(facecolor="none", edgecolor="black", hatch="xx", label="SNR 1 - 2"),
        Patch(facecolor="none", edgecolor="black", hatch="....", label="SNR > 2"),
    ]
    fig.legend(
        handles=hatch_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.10),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        fontsize=10,
    )

    # Shared Colorbar
    cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.02])
    fig.colorbar(
        im, cax=cbar_ax, orientation="horizontal",
        ticks=bounds, label="Signal-to-Noise Ratio (SNR)"
    )

    plt.subplots_adjust(wspace=0.08, hspace=0.18, bottom=0.16)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Atmospheric SNR diagnostic plot saved to '{output_png}'!")


def run_atm_snr_pipeline(
    init_month: str = "05",
    requested_vars: list[str] = ["TMP2m", "Z500"],
    clim_years: tuple[int, int] = (1991, 2020),
    detrend: bool = True,
    target_lead_sets: list[tuple[str, list[int], bool]] = None,
    output_png: str = "atm_snr_may_init.png",
):
    """Execute full atmospheric SNR diagnostic pipeline."""
    if target_lead_sets is None:
        target_lead_sets = [
            ("Lead 0 (May)", [0], False),
            ("Lead 1-3 (JJA)", [1, 2, 3], True),
        ]

    logger.info(f"=== Step 1: Loading Raw SFS Atmospheric Data (Init: {init_month}) ===")
    ds_sfs_raw = get_sfs_data(
        init_month=init_month,
        domain="atm",
        requested_vars=requested_vars,
    )

    logger.info("=== Step 2: Preprocessing Atmospheric Grid & Time Coordinates ===")
    ds_sfs = preprocess_atm_dataset(ds_sfs_raw, target_res="1.0deg")

    logger.info(f"=== Step 3: Computing Lead-Dependent Anomalies ({clim_years[0]}-{clim_years[1]} Baseline) ===")
    ds_anom, ds_clim = compute_anomalies(ds_sfs, clim_years=clim_years)

    if detrend:
        logger.info("=== Step 4: Applying Linear Detrending Along Init Years ===")
        ds_anom = detrend_dim(ds_anom, dim="init", deg=1)

    logger.info("=== Step 5 & 6: Calculating SNR Across Variables and Lead Targets ===")
    results = {}

    for label, leads, average_flag in target_lead_sets:
        ds_target = select_target_leads(ds_anom, leads=leads, average_seasonal=average_flag)
        ds_snr = compute_signal_to_noise_ratio(ds_target, year_dim="init", member_dim="member")
        results[label] = ds_snr

    logger.info("=== Step 7: Plotting Robinson Diagnostic Maps (Panel A Styling) ===")
    plot_snr_grid(results, requested_vars, output_png=output_png)

    return results


if __name__ == "__main__":
    run_atm_snr_pipeline(
        init_month="05",
        requested_vars=["TMP2m", "Z500"],
        detrend=True,
    )
