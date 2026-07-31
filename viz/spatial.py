"""
Spatial Plotting Module for SFS Evaluations
Contains reusable Cartopy/Matplotlib spatial plotters and overlays.
Matches NOAA SFSv1 Verification Diagnostics Panel A visual standard.
"""
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

logger = logging.getLogger(__name__)


def create_acc_sfs_colormap():
    """
    Creates the exact 9-level ACC colormap used in NOAA SFSv1 verification diagnostics:
    0.1-0.3: Yellows | 0.3-0.7: Greens | 0.7-1.0: Purples
    Values < 0.1 are left white/unfilled.
    """
    bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hex_colors = [
        "#FFF59D",  # 0.1 - 0.2: Pale Yellow
        "#FFE082",  # 0.2 - 0.3: Light Gold
        "#C8E6C9",  # 0.3 - 0.4: Soft Sage
        "#81C784",  # 0.4 - 0.5: Medium Light Green
        "#388E3C",  # 0.5 - 0.6: Vibrant Green
        "#1B5E20",  # 0.6 - 0.7: Dark Forest Green
        "#E1BEE7",  # 0.7 - 0.8: Soft Lavender
        "#AB47BC",  # 0.8 - 0.9: Medium Purple
        "#4A148C",  # 0.9 - 1.0: Deep Purple
    ]
    cmap = mcolors.ListedColormap(hex_colors)
    cmap.set_under("white")  # ACC < 0.1 rendered transparent/white
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def plot_acc_snr_overlay(
    acc_da: xr.DataArray,
    snr_da: xr.DataArray,
    title_str: str = "Panel A: Deterministic Skill (ACC) & SNR",
    snr_levels: list[float] | None = None,
    output_png: str = "figures/acc_snr_overlay.png",
):
    """
    Plot spatial ACC skill with 4-tier SNR hatch pattern overlays on a Robinson projection.
    No boundary contour lines drawn around hatch regions.
    """
    fig, ax = plt.subplots(
        figsize=(12, 7.5),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=0)}
    )

    # Ensure strictly 2D spatial inputs (lat, lon)
    acc_da = acc_da.squeeze()
    snr_da = snr_da.squeeze()

    acc_vals = acc_da.values
    snr_vals = snr_da.values

    # Unmask coordinates
    lats = np.asarray(acc_da.lat.values, dtype=float)
    lons = np.asarray(acc_da.lon.values, dtype=float)

    if np.ma.is_masked(lats):
        lats = lats.filled(np.nan)
    if np.ma.is_masked(lons):
        lons = lons.filled(np.nan)

    if lons.ndim > 1:
        lons = lons[0, :]
    if lats.ndim > 1:
        lats = lats[:, 0]

    # Manual cyclic longitude wrap
    dlon = (lons[-1] - lons[0]) / (len(lons) - 1)
    cyclic_lon = np.append(lons, lons[-1] + dlon)
    cyclic_acc = np.concatenate([acc_vals, acc_vals[:, 0:1]], axis=-1)
    cyclic_snr = np.concatenate([snr_vals, snr_vals[:, 0:1]], axis=-1)

    lon2d, lat2d = np.meshgrid(cyclic_lon, lats)

    # Base map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black", zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":", zorder=5)
    ax.gridlines(draw_labels=False, linestyle=":", color="gray", alpha=0.3)

    # Layer 1: ACC Raster Background (0.1 to 1.0)
    cmap, norm, bounds = create_acc_sfs_colormap()
    im = ax.pcolormesh(
        lon2d, lat2d, cyclic_acc,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    # Layer 2: 4-Tier SNR Hatch Pattern Overlays (WITHOUT boundary contour lines)
    hatch_levels = [0.22, 0.5, 1.0, 2.0, 1e5]
    snr_hatches = ["...", "///", "xxx", "***"]

    plt.rcParams["hatch.linewidth"] = 0.5
    plt.rcParams["hatch.color"] = "black"

    ax.contourf(
        lon2d, lat2d, cyclic_snr,
        levels=hatch_levels,
        hatches=snr_hatches,
        colors="none",
        edgecolors="none",  # <--- Removes all region boundary contour lines
        transform=ccrs.PlateCarree(),
        zorder=6,
    )

    # SNR Hatch Legend placed directly below the map
    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="..", label="SNR 0.22 - 0.5"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="/", label="SNR 0.5 - 1"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="xx", label="SNR 1 - 2"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="....", label="SNR > 2"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        fontsize=9,
        handletextpad=0.5,
        columnspacing=1.5,
    )

    # ACC Colorbar placed at the bottom
    cbar_ax = fig.add_axes([0.18, 0.08, 0.64, 0.025])
    cbar = fig.colorbar(
        im, cax=cbar_ax, orientation="horizontal",
        ticks=bounds, extend="neither",
        label="Anomaly Correlation Coefficient (ACC)"
    )
    cbar.ax.tick_params(labelsize=9)

    mean_acc = float(np.nanmean(acc_vals))
    mean_snr = float(np.nanmean(snr_vals))

    ax.set_title(
        f"{title_str}\n[Mean ACC: {mean_acc:.2f} | Mean SNR: {mean_snr:.2f}]",
        fontsize=12, fontweight="bold", pad=12
    )

    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ ACC + SNR overlay diagnostic saved to '{output_png}'!")
