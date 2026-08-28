"""
Spatial Visualization Module for Spread-to-Error Ratio (SER) Calibration
"""
import os
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from viz.spatial import _prepare_cyclic_grid, _draw_manual_graticules, LAND_GRAY

logger = logging.getLogger(__name__)


def plot_ser_comparison_map(
    ser_raw_da: xr.DataArray,
    ser_cal_da: xr.DataArray,
    title: str = "",  # <--- ADD THIS PARAMETER
    output_png: str = "figures/ser_before_after.png",
):
    """
    Plots 2-Panel Spatial Comparison Map:
    Panel (a): Raw Ensemble SER (showing overconfidence/underconfidence)
    Panel (b): Calibrated Ensemble SER (proving convergence to ~1.0)
    """
    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(1, 2, wspace=0.12, top=0.88, bottom=0.15, left=0.04, right=0.96)

    proj = ccrs.Robinson(central_longitude=0)
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    ax2 = fig.add_subplot(gs[0, 1], projection=proj)

    for ax in (ax1, ax2):
        ax.set_facecolor(LAND_GRAY)
        ax.add_feature(cfeature.LAND, facecolor=LAND_GRAY, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#404040", zorder=5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":", zorder=5)

    # 5-Tier SER Palette anchored around 1.0
    ser_bounds = [0.0, 0.5, 0.8, 1.2, 1.6, 5.0]
    ser_colors = [
        '#3182bd',  # < 0.5: Severe Overconfidence (Dark Blue)
        '#9ecae1',  # 0.5 - 0.8: Overconfident (Light Blue)
        '#c7e9c0',  # 0.8 - 1.2: Well-Calibrated Spread (Mint Green)
        '#fdae6b',  # 1.2 - 1.6: Underconfident Spread (Orange)
        '#e6550d',  # > 1.6: Severe Underconfidence (Dark Red)
    ]
    ser_cmap = mcolors.ListedColormap(ser_colors)
    ser_cmap.set_bad(color=(0, 0, 0, 0))
    ser_norm = mcolors.BoundaryNorm(ser_bounds, ser_cmap.N)

    # --- Panel (a): Raw SER ---
    lon2d, lat2d, cyclic_raw, raw_vals = _prepare_cyclic_grid(ser_raw_da)
    ax1.pcolormesh(
        lon2d, lat2d, cyclic_raw, cmap=ser_cmap, norm=ser_norm,
        shading="auto", transform=ccrs.PlateCarree(), zorder=3
    )
    med_raw = float(np.nanmedian(raw_vals))
    ax1.set_title(
        rf"$\mathbf{{(a)\ Raw\ Ensemble\ Spread\text{{-}}to\text{{-}}Error\ Ratio\ (SER)}}\ \mid\ \mathbf{{Spatial\ Median:\ {med_raw:.2f}}}$"
        "\n[ < 1.0: Overconfident / Spread Deficit  |  > 1.0: Underconfident ]",
        fontsize=8.5, pad=6
    )

    # --- Panel (b): Calibrated SER ---
    _, _, cyclic_cal, cal_vals = _prepare_cyclic_grid(ser_cal_da)
    ax2.pcolormesh(
        lon2d, lat2d, cyclic_cal, cmap=ser_cmap, norm=ser_norm,
        shading="auto", transform=ccrs.PlateCarree(), zorder=3
    )
    med_cal = float(np.nanmedian(cal_vals))
    ax2.set_title(
        rf"$\mathbf{{(b)\ Calibrated\ Ensemble\ Spread\text{{-}}to\text{{-}}Error\ Ratio\ (SER)}}\ \mid\ \mathbf{{Spatial\ Median:\ {med_cal:.2f}}}$"
        "\n[ Convergence to 1.0 Guarantees Statistical Spread Reliability ]",
        fontsize=8.5, pad=6
    )

    _draw_manual_graticules(ax1)
    _draw_manual_graticules(ax2)

    # Legend Handles
    legend_elements = [
        mpatches.Patch(facecolor="#3182bd", edgecolor="black", label="< 0.5: Severe Overconfidence"),
        mpatches.Patch(facecolor="#9ecae1", edgecolor="black", label="0.5 - 0.8: Overconfident"),
        mpatches.Patch(facecolor="#c7e9c0", edgecolor="black", label="0.8 - 1.2: Well-Calibrated Spread (Ideal)"),
        mpatches.Patch(facecolor="#fdae6b", edgecolor="black", label="1.2 - 1.6: Underconfident"),
        mpatches.Patch(facecolor="#e6550d", edgecolor="black", label="> 1.6: Severe Underconfidence"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=5, frameon=False, fontsize=8, columnspacing=1.2
    )

# Use passed-in title if provided, else fall back to default
    main_title = title if title else "Spread Reliability Verification [Spread-to-Error Ratio (SER)]"
    fig.suptitle(main_title, fontsize=12, fontweight="bold", y=0.97)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✅ SER comparison map saved to '{output_png}'!")
