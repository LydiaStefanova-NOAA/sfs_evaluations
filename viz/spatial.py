"""
Spatial Plotting Module for SFS Evaluations
Contains reusable Cartopy/Matplotlib spatial plotters and overlays.
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

logger = logging.getLogger(__name__)

# Standard soft gray for land and masked/undefined ocean background
LAND_GRAY = "#eaeaea"

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

import os
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

logger = logging.getLogger(__name__)
LAND_GRAY = "#f0f0f0"  # Softer land background

def plot_variance_diagnostics_map(
    svr_da: xr.DataArray,
    nvr_da: xr.DataArray,
    n_years: int,
    n_members: int,
    detrend: bool = True,
    title_str: str = "SNR Variance Breakdown Diagnostic",
    output_png: str = "figures/variance_breakdown.png",
):
    from metrics.confidence import compute_dynamic_ci_bounds

    # Dynamically compute bounds
    _, _, svr_crit, (nvr_low, nvr_high) = compute_dynamic_ci_bounds(
        n_years=n_years, n_members=n_members, detrend=detrend
    )

    # 5-bin dynamic bounds anchored around 1.0
    low_outer = round(nvr_low * 0.5, 2)
    high_outer = round(nvr_high * 1.5, 2)
    five_bin_bounds = [0.0, low_outer, nvr_low, nvr_high, high_outer, 5.0]

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
        ax.gridlines(draw_labels=False, linestyle=":", color="gray", alpha=0.3, zorder=6)

    nvr_method = nvr_da.attrs.get("nvr_method", "mse")

    # =========================================================================
    # --- PANEL (a): SIGNAL VARIANCE RATIO (SVR) SETUP ---
    # =========================================================================
    # Muted palette: Steel Blue (Deficit) -> Off-White (1.0) -> Warm Rust (Inflation)
    svr_colors_5 = ['#4575b4', '#abd9e9', '#f7f7f7', '#fdae61', '#d73027']

    if nvr_method == "acc_varobs":
        svr_math = r"\sigma^2_{\mathrm{signal,mod}} / (\mathrm{ACC}^2 \cdot \sigma^2_{\mathrm{obs}})"
        svr_subtitle = (
            f"[ Sev. Deficit: <{low_outer:.2f} | Deficit: {low_outer:.2f}–{nvr_low:.2f} | "
            f"Calibrated: {nvr_low:.2f}–{nvr_high:.2f} | Inflated: {nvr_high:.2f}–{high_outer:.2f} | Sev. Infl.: >{high_outer:.2f} ]"
        )
        svr_bounds = five_bin_bounds
        svr_colors = svr_colors_5
        svr_cbar_label = "Signal / (ACC² · Var_obs)"
    else:
        svr_math = r"\sigma^2_{\mathrm{signal,mod}} / \sigma^2_{\mathrm{obs}}"
        svr_subtitle = f"[ ≤ 1.0: Bounded  |  1.0–{svr_crit:.2f}: Moderate Inflation  |  > {svr_crit:.2f}: Significant Inflation (95% CI, N={n_years}) ]"
        svr_bounds = [0.0, 0.25, 0.50, 0.75, 1.00, svr_crit, 3.00, 5.00]
        svr_colors = ['#313695', '#4575b4', '#74add1', '#f7f7f7', '#fdae61', '#f46d43', '#a50026']
        svr_cbar_label = "Signal / Var_obs"

    lon2d, lat2d, cyclic_svr, svr_vals = _prepare_cyclic_grid(svr_da)
    cmap_svr = mcolors.ListedColormap(svr_colors)
    cmap_svr.set_bad(color=(0, 0, 0, 0))
    cmap_svr.set_over('#7a0177')
    norm_svr = mcolors.BoundaryNorm(svr_bounds, cmap_svr.N)

    im1 = ax1.pcolormesh(lon2d, lat2d, cyclic_svr, cmap=cmap_svr, norm=norm_svr, shading="auto", transform=ccrs.PlateCarree(), zorder=3)
    med_svr = float(np.nanmedian(svr_vals))

    ax1.set_title(
        rf"$\mathbf{{(a)\ Signal\ Variance\ Ratio}}\ ({svr_math}) \mid \mathbf{{Spatial\ Median:\ {med_svr:.2f}}}$"
        f"\n{svr_subtitle}",
        fontsize=8.0, pad=6
    )

    cbar_ax1 = fig.add_axes([0.08, 0.08, 0.38, 0.025])
    cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation="horizontal", ticks=svr_bounds, extend="max", label=svr_cbar_label)
    cbar1.ax.tick_params(labelsize=8)

    # =========================================================================
    # --- PANEL (b): NOISE VARIANCE RATIO (NVR) SETUP ---
    # =========================================================================
    # Muted palette: Soft Teal (Under) -> Off-White (1.0) -> Soft Plum/Berry (Over)
    nvr_colors_5 = ['#01665e', '#80cdc1', '#f7f7f7', '#df65b0', '#980043']

    if nvr_method == "acc_varobs":
        nvr_math = r"\sigma^2_{\mathrm{noise,mod}} / ((1-\mathrm{ACC}^2)\sigma^2_{\mathrm{obs}})"
        nvr_cbar_label = "Noise / ((1 - ACC²) · Var_obs)"
    else:
        nvr_math = r"\sigma^2_{\mathrm{noise,mod}} / \mathrm{MSE}"
        nvr_cbar_label = "Noise / MSE"

    nvr_subtitle = (
        f"[ Sev. Under: <{low_outer:.2f} | Under: {low_outer:.2f}–{nvr_low:.2f} | "
        f"Calibrated: {nvr_low:.2f}–{nvr_high:.2f} | Over: {nvr_high:.2f}–{high_outer:.2f} | Sev. Over: >{high_outer:.2f} ]"
    )
    nvr_bounds = five_bin_bounds

    _, _, cyclic_nvr, nvr_vals = _prepare_cyclic_grid(nvr_da)
    cmap_nvr = mcolors.ListedColormap(nvr_colors_5)
    cmap_nvr.set_bad(color=(0, 0, 0, 0))
    cmap_nvr.set_over('#4c0519')
    norm_nvr = mcolors.BoundaryNorm(nvr_bounds, cmap_nvr.N)

    im2 = ax2.pcolormesh(lon2d, lat2d, cyclic_nvr, cmap=cmap_nvr, norm=norm_nvr, shading="auto", transform=ccrs.PlateCarree(), zorder=3)
    med_nvr = float(np.nanmedian(nvr_vals))

    ax2.set_title(
        rf"$\mathbf{{(b)\ Noise\ Variance\ Ratio}}\ ({nvr_math}) \mid \mathbf{{Spatial\ Median:\ {med_nvr:.2f}}}$"
        f"\n{nvr_subtitle}",
        fontsize=8.0, pad=6
    )

    cbar_ax2 = fig.add_axes([0.54, 0.08, 0.38, 0.025])
    cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation="horizontal", ticks=nvr_bounds, extend="max", label=nvr_cbar_label)
    cbar2.ax.tick_params(labelsize=8)

    # --- Final Layout & Save ---
    _draw_manual_graticules(ax1)
    _draw_manual_graticules(ax2)

    fig.suptitle(title_str, fontsize=12, fontweight="bold", y=0.97)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
def create_acc_sfs_colormap_orig():
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

def create_acc_sfs_colormap():
    """Exact 9-level ACC colormap used in NOAA SFSv1 verification diagnostics."""
    bounds = [ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hex_colors = [
        '#ffffcc', '#ffeda0',  '#a1d99b', 
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
    """Squeeze + normalize coord names + add cyclic lon for Cartopy."""
    import numpy as np
    import xarray as xr

    da = da.squeeze()

    # Normalize common coordinate aliases
    rename_map = {}
    if "lat" not in da.coords:
        for c in ["latitude", "Latitude", "nav_lat", "y"]:
            if c in da.coords:
                rename_map[c] = "lat"
                break
    if "lon" not in da.coords:
        for c in ["longitude", "Longitude", "nav_lon", "x"]:
            if c in da.coords:
                rename_map[c] = "lon"
                break
    if rename_map:
        da = da.rename(rename_map)

    # Also handle cases where dim names (not coords) are aliases
    dim_rename = {}
    if "lat" not in da.dims:
        for d in ["latitude", "Latitude", "y"]:
            if d in da.dims:
                dim_rename[d] = "lat"
                break
    if "lon" not in da.dims:
        for d in ["longitude", "Longitude", "x"]:
            if d in da.dims:
                dim_rename[d] = "lon"
                break
    if dim_rename:
        da = da.rename(dim_rename)

    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"_prepare_cyclic_grid needs lat/lon dims; got dims={da.dims}, coords={list(da.coords)}")

    da = da.transpose("lat", "lon")

    lats = np.asarray(da["lat"].values, dtype=float)
    lons = np.asarray(da["lon"].values, dtype=float)
    vals = da.values

    if lons.ndim > 1:
        lons = lons[0, :]
    if lats.ndim > 1:
        lats = lats[:, 0]

    dlon = (lons[-1] - lons[0]) / (len(lons) - 1)
    cyclic_lon = np.append(lons, lons[-1] + dlon)
    cyclic_vals = np.concatenate([vals, vals[:, 0:1]], axis=1)

    lon2d, lat2d = np.meshgrid(cyclic_lon, lats)

    if cyclic_vals.shape != lon2d.shape:
        raise ValueError(
            f"Cyclic shape mismatch: cyclic_vals={cyclic_vals.shape}, lon2d={lon2d.shape}, lat2d={lat2d.shape}"
        )

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
    hatch_levels = [ 0.1, 0.33, 0.96, 4.26, 1e5 ]
    snr_hatches = ["...", "///", "xxx", "***"]
    plt.rcParams["hatch.linewidth"] = 0.5
    plt.rcParams["hatch.color"] = "black"

    ax.contourf(
        lon2d, lat2d, cyclic_snr, levels=hatch_levels, hatches=snr_hatches,
        colors="none", edgecolors="none", transform=ccrs.PlateCarree(), zorder=7
    )

    legend_elements = [
        #mpatches.Patch(facecolor="white", edgecolor="black", hatch="...", label="SNR 0.22 - 0.5"),
        #mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="SNR 0.5 - 1"),
        #mpatches.Patch(facecolor="white", edgecolor="black", hatch="xxx", label="SNR 1 - 2"),
        #mpatches.Patch(facecolor="white", edgecolor="black", hatch="***", label="SNR > 2"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="...", label="Rmod 0.3 - 0.5"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Rmod 0.5 - 0.7"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="xxx", label="Rmod 0.7 - 0.9"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="***", label="Rmod > 0.9"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=4, frameon=False, fontsize=9, handletextpad=0.5, columnspacing=1.5
    )

    cbar_ax = fig.add_axes([0.18, 0.08, 0.64, 0.025])
    cbar = fig.colorbar(
        im, cax=cbar_ax, orientation="horizontal", ticks=bounds, extend="neither",
        label="Correlation Coefficient"
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

    def _draw_manual_graticules(ax):
        # Meridians
        for lon in np.arange(-180, 181, 60):
            lats = np.linspace(-89.5, 89.5, 360)
            lons = np.full_like(lats, lon, dtype=float)
            ax.plot(
                lons, lats,
                transform=ccrs.PlateCarree(),
                linestyle="--",
                linewidth=0.6,
                color="black",
                alpha=0.35,
                zorder=20,
            )
        # Parallels
        for lat in np.arange(-60, 61, 30):
            lons = np.linspace(-180, 180, 720)
            lats = np.full_like(lons, lat, dtype=float)
            ax.plot(
                lons, lats,
                transform=ccrs.PlateCarree(),
                linestyle="--",
                linewidth=0.6,
                color="black",
                alpha=0.35,
                zorder=20,
            )

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(LAND_GRAY)
        ax.add_feature(cfeature.LAND, facecolor=LAND_GRAY, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=12)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":", zorder=12)

    # Prepare cyclic grids separately for each field
    lon2d_acc, lat2d_acc, cyclic_acc, acc_vals = _prepare_cyclic_grid(acc_da)
    lon2d_rpot, lat2d_rpot, cyclic_rpot, rpot_vals = _prepare_cyclic_grid(rpot_da)
    lon2d_rpc, lat2d_rpc, cyclic_rpc, rpc_vals = _prepare_cyclic_grid(rpc_da)

    cmap_acc, norm_acc, bounds_acc = create_acc_sfs_colormap()

    # --- Panel (a): Real-World ACC (r_mo) ---
    im1 = ax1.pcolormesh(
        lon2d_acc, lat2d_acc, cyclic_acc, cmap=cmap_acc, norm=norm_acc, shading="auto",
        transform=ccrs.PlateCarree(), zorder=3
    )
    median_acc = float(np.nanmedian(acc_vals))
    ax1.set_title(
        rf"$\mathbf{{(a)\ Real\text{{-}}World\ ACC}}\ (r_{{\mathrm{{mo}}}}) \mid \mathbf{{Spatial\ Median:\ {median_acc:.2f}}}$",
        fontsize=10, pad=6
    )

    # --- Panel (b): Model-World ACC (r_mw) ---
    ax2.pcolormesh(
        lon2d_rpot, lat2d_rpot, cyclic_rpot, cmap=cmap_acc, norm=norm_acc, shading="auto",
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
        lon2d_rpc, lat2d_rpc, cyclic_rpc, cmap=cmap_rpc, norm=norm_rpc, shading="auto",
        transform=ccrs.PlateCarree(), zorder=3
    )
    median_rpc = float(np.nanmedian(rpc_vals))
    ax3.set_title(
        rf"$\mathbf{{(c)\ Predictability\ Ratio}}\ (\mathrm{{RPC}} = \frac{{\text{{Real-World ACC}}}}{{\text{{Model-World ACC}}}}) \mid \mathbf{{Spatial\ Median:\ {median_rpc:.2f}}}$",
        fontsize=10, pad=6
    )

    # Manual graticules LAST (ensures visible over color fill)
    _draw_manual_graticules(ax1)
    _draw_manual_graticules(ax2)
    _draw_manual_graticules(ax3)

    # Column-First Legend Handle Order for Matplotlib ncol=3
    rpc_legend_elements = [
        mpatches.Patch(facecolor="#3182bd", edgecolor="black", label="< 0.5: Highly Overconfident"),
        mpatches.Patch(facecolor="#9ecae1", edgecolor="black", label="0.5 - 0.8: Overconfident"),
        mpatches.Patch(facecolor="#c7e9c0", edgecolor="black", label="0.8 - 1.2: Well-Calibrated"),
        mpatches.Patch(facecolor=LAND_GRAY, edgecolor="black", label="Masked/No Signal"),
        mpatches.Patch(facecolor="#e6550d", edgecolor="black", label="> 1.6: Highly Underconfident (Strong S/N Paradox)"),
        mpatches.Patch(facecolor="#fdae6b", edgecolor="black", label="1.2 - 1.6: Underconfident (S/N Paradox)"),
    ]

    ax3.legend(
        handles=rpc_legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.05),
        ncol=3, frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.2
    )

    if subtitle_str:
        fig.suptitle(f"{main_title}\n{subtitle_str}", fontsize=12, fontweight="bold", y=0.97)
    else:
        fig.suptitle(main_title, fontsize=12, fontweight="bold", y=0.97)

    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ 3-Panel Skill, Predictability & RPC plot saved to '{output_png}'!")
