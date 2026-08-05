"""
Spatial Plotting Module for SFS Evaluations
Contains reusable Cartopy/Matplotlib spatial plotters and overlays.
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

# Standard soft gray for land and masked/undefined ocean background
LAND_GRAY = "#eaeaea"


def create_acc_sfs_colormap():
    """Exact 9-level ACC colormap used in NOAA SFSv1 verification diagnostics."""
    bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hex_colors = [
        '#ffffcc', '#ffeda0', '#c7e9c0', '#a1d99b', '#74c476',
        '#31a354', '#dadaeb', '#bcbddc', '#756bb1'
    ]
    cmap = mcolors.ListedColormap(hex_colors)
    cmap.set_under("white")  # Values < 0.1 rendered white/unfilled
    cmap.set_bad(color=(0, 0, 0, 0))  # Fully transparent NaNs for Cartopy cyclic wrapping
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def create_rpc_colormap():
    """5-tier discrete RPC colormap highlighting model calibration & Signal-to-Noise Paradox."""
    bounds = [0.0, 0.5, 0.8, 1.2, 1.6, 5.0]
    hex_colors = [
        '#3182bd',  # < 0.5: Highly Overconfident (Dark Blue)
        '#9ecae1',  # 0.5 - 0.8: Overconfident (Light Blue)
        '#c7e9c0',  # 0.8 - 1.2: Well-Calibrated (Mint Green)
        '#fdae6b',  # 1.2 - 1.6: Underconfident / S/N Paradox (Orange)
        '#e6550d',  # > 1.6: Severe S/N Paradox (Dark Red)
    ]
    cmap = mcolors.ListedColormap(hex_colors)
    cmap.set_under('#3182bd')  # Negative RPC (< 0.0, wrong-sign skill) mapped to Highly Overconfident
    cmap.set_bad(color=(0, 0, 0, 0))  # Fully transparent NaNs for Cartopy cyclic wrapping
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def _prepare_cyclic_grid(da: xr.DataArray):
    """Squeezes dimensions and applies cyclic longitude wrapping for Cartopy."""
    da = da.squeeze()
    vals = da.values
    lats = np.asarray(da.lat.values, dtype=float)
    lons = np.asarray(da.lon.values, dtype=float)

    if lons.ndim > 1:
        lons = lons[0, :]
    if lats.ndim > 1:
        lats = lats[:, 0]

    dlon = (lons[-1] - lons[0]) / (len(lons) - 1)
    cyclic_lon = np.append(lons, lons[-1] + dlon)
    cyclic_vals = np.concatenate([vals, vals[:, 0:1]], axis=-1)
    lon2d, lat2d = np.meshgrid(cyclic_lon, lats)
    
    return lon2d, lat2d, cyclic_vals, vals


def plot_acc_snr_overlay(
    acc_da: xr.DataArray,
    snr_da: xr.DataArray,
    title_str: str = "ACC & signal-to-noise ratio (SNR)",
    output_png: str = "figures/acc_snr_overlay.png",
):
    """Plot spatial ACC skill with 4-tier SNR hatch pattern overlays."""
    fig, ax = plt.subplots(
        figsize=(12, 7.5),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=0)}
    )
    ax.set_facecolor(LAND_GRAY)

    lon2d, lat2d, cyclic_acc, acc_vals = _prepare_cyclic_grid(acc_da)
    _, _, cyclic_snr, snr_vals = _prepare_cyclic_grid(snr_da)

    ax.add_feature(cfeature.LAND, facecolor=LAND_GRAY, zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black", zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":", zorder=5)
    ax.gridlines(draw_labels=False, linestyle=":", color="gray", alpha=0.3, zorder=6)

    cmap, norm, bounds = create_acc_sfs_colormap()
    im = ax.pcolormesh(
        lon2d, lat2d, cyclic_acc, cmap=cmap, norm=norm, shading="auto",
        transform=ccrs.PlateCarree(), zorder=3
    )

    hatch_levels = [0.22, 0.5, 1.0, 2.0, 1e5]
    snr_hatches = ["...", "///", "xxx", "***"]
    plt.rcParams["hatch.linewidth"] = 0.5
    plt.rcParams["hatch.color"] = "black"

    ax.contourf(
        lon2d, lat2d, cyclic_snr, levels=hatch_levels, hatches=snr_hatches,
        colors="none", edgecolors="none", transform=ccrs.PlateCarree(), zorder=7
    )

    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="...", label="SNR 0.22 - 0.5"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="SNR 0.5 - 1"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="xxx", label="SNR 1 - 2"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="***", label="SNR > 2"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=4, frameon=False, fontsize=9, handletextpad=0.5, columnspacing=1.5
    )

    cbar_ax = fig.add_axes([0.18, 0.08, 0.64, 0.025])
    cbar = fig.colorbar(
        im, cax=cbar_ax, orientation="horizontal", ticks=bounds, extend="neither",
        label="Correlation Coefficient (0.1 to 1.0)"
    )
    cbar.ax.tick_params(labelsize=9)

    snr_median = float(np.nanmedian(snr_vals))
    acc_median = float(np.nanmedian(acc_vals))
    full_title = f"{title_str}\n[Median ACC: {acc_median:.2f} | Median SNR: {snr_median:.2f}]"
    ax.set_title(full_title, fontsize=11, fontweight="bold", pad=12)

    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ ACC + SNR overlay diagnostic saved to '{output_png}'!")


def plot_skill_predictability_trio(
    acc_da: xr.DataArray,
    rpot_da: xr.DataArray,
    rpc_da: xr.DataArray,
    main_title: str = "ACC (real vs model-world) & their ratio (RPC)",
    subtitle_str: str = "",
    output_png: str = "figures/skill_trio_3panel.png",
):
    """
    Plots a 3-Panel Diagnostic Layout:
    Row 1: (a) Real-World ACC (r_mo) | (b) Model-World ACC (r_mw)
    Row 2: (c) Predictability Ratio (RPC) [Centered]
    """
    fig = plt.figure(figsize=(15, 11))
    
    gs = fig.add_gridspec(
        2, 4, 
        hspace=0.32, wspace=0.15, 
        top=0.91, bottom=0.08, left=0.04, right=0.96
    )

    proj = ccrs.Robinson(central_longitude=0)
    ax1 = fig.add_subplot(gs[0, :2], projection=proj)
    ax2 = fig.add_subplot(gs[0, 2:], projection=proj)
    ax3 = fig.add_subplot(gs[1, 1:3], projection=proj)

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(LAND_GRAY)  # Axis background matches light gray mask color
        ax.add_feature(cfeature.LAND, facecolor=LAND_GRAY, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":", zorder=5)
        ax.gridlines(draw_labels=False, linestyle=":", color="gray", alpha=0.3, zorder=6)

    lon2d, lat2d, cyclic_acc, acc_vals = _prepare_cyclic_grid(acc_da)
    _, _, cyclic_rpot, rpot_vals = _prepare_cyclic_grid(rpot_da)
    _, _, cyclic_rpc, rpc_vals = _prepare_cyclic_grid(rpc_da)

    cmap_acc, norm_acc, bounds_acc = create_acc_sfs_colormap()

    # --- Panel (a): Real-World ACC (r_mo) ---
    im1 = ax1.pcolormesh(
        lon2d, lat2d, cyclic_acc, cmap=cmap_acc, norm=norm_acc, shading="auto",
        transform=ccrs.PlateCarree(), zorder=3
    )
    median_acc = float(np.nanmedian(acc_vals))
    ax1.set_title(
        rf"$\mathbf{{(a)\ Real\text{{-}}World\ ACC}}\ (r_{{\mathrm{{mo}}}}) \mid \mathbf{{Spatial\ Median:\ {median_acc:.2f}}}$",
        fontsize=10, pad=6
    )

    # --- Panel (b): Model-World ACC (r_mw) ---
    ax2.pcolormesh(
        lon2d, lat2d, cyclic_rpot, cmap=cmap_acc, norm=norm_acc, shading="auto",
        transform=ccrs.PlateCarree(), zorder=3
    )
    median_rpot = float(np.nanmedian(rpot_vals))
    ax2.set_title(
        rf"$\mathbf{{(b)\ Model\text{{-}}World\ ACC}}\ (r_{{\mathrm{{mw}}}} = \sqrt{{\frac{{\mathrm{{SNR}}}}{{1 + \mathrm{{SNR}}}}}}) \mid \mathbf{{Spatial\ Median:\ {median_rpot:.2f}}}$",
        fontsize=10, pad=6
    )

    # Shared Colorbar under Row 1
    fig.canvas.draw()
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    
    cbar_x0 = pos1.x0 + 0.15 * (pos2.x1 - pos1.x0)
    cbar_width = 0.70 * (pos2.x1 - pos1.x0)
    cbar_y0 = pos1.y0 - 0.045
    
    cbar_ax1 = fig.add_axes([cbar_x0, cbar_y0, cbar_width, 0.015])
    cbar1 = fig.colorbar(
        im1, cax=cbar_ax1, orientation="horizontal", ticks=bounds_acc, extend="neither",
        label="Correlation Coefficient (0.1 to 1.0)"
    )
    cbar1.ax.tick_params(labelsize=8)

    # --- Panel (c): Predictability Ratio (RPC) ---
    cmap_rpc, norm_rpc, _ = create_rpc_colormap()
    ax3.pcolormesh(
        lon2d, lat2d, cyclic_rpc, cmap=cmap_rpc, norm=norm_rpc, shading="auto",
        transform=ccrs.PlateCarree(), zorder=3
    )
    median_rpc = float(np.nanmedian(rpc_vals))
    ax3.set_title(
        rf"$\mathbf{{(c)\ Predictability\ Ratio}}\ (\mathrm{{RPC}} = \frac{{\text{{Real-World ACC}}}}{{\text{{Model-World ACC}}}}) \mid \mathbf{{Spatial\ Median:\ {median_rpc:.2f}}}$",
        fontsize=10, pad=6
    )

    # Column-First Legend Handle Order for Matplotlib ncol=3:
    # Column 0: Dark Blue (< 0.5) top, Light Blue (0.5 - 0.8) bottom
    # Column 1: Mint Green (0.8 - 1.2) top, Light Gray (Masked) bottom
    # Column 2: Dark Red (> 1.6) top, Orange (1.2 - 1.6) bottom
    rpc_legend_elements = [
        # --- Column 0 (Left Side: Overconfident) ---
        mpatches.Patch(facecolor="#3182bd", edgecolor="black", label="< 0.5: Highly Overconfident"),        # Row 0, Col 0
        mpatches.Patch(facecolor="#9ecae1", edgecolor="black", label="0.5 - 0.8: Overconfident"),           # Row 1, Col 0

        # --- Column 1 (Middle: Well-Calibrated / Masked) ---
        mpatches.Patch(facecolor="#c7e9c0", edgecolor="black", label="0.8 - 1.2: Well-Calibrated"),          # Row 0, Col 1
        mpatches.Patch(facecolor=LAND_GRAY, edgecolor="black", label="Masked/No Signal (ACC<0.3)"),                 # Row 1, Col 1

        # --- Column 2 (Right Side: Underconfident) ---
        mpatches.Patch(facecolor="#e6550d", edgecolor="black", label="> 1.6: Highly Underconfident (Strong S/N Paradox"),  # Row 0, Col 2 (Dark Red)
        mpatches.Patch(facecolor="#fdae6b", edgecolor="black", label="1.2 - 1.6: Underconfident (S/N Paradox)"), # Row 1, Col 2 (Orange)
    ]

    ax3.legend(
        handles=rpc_legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.05),
        ncol=3, frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.2
    )

    # Master Title & Metadata Subtitle
    if subtitle_str:
        fig.suptitle(f"{main_title}\n{subtitle_str}", fontsize=12, fontweight="bold", y=0.97)
    else:
        fig.suptitle(main_title, fontsize=12, fontweight="bold", y=0.97)

    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ 3-Panel Skill, Predictability & RPC plot saved to '{output_png}'!")
