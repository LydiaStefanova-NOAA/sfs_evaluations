"""
Spatial Plotting Module for Recalibration Verification
"""
import os
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from viz.spatial import _prepare_cyclic_grid, _draw_manual_graticules, LAND_GRAY

logger = logging.getLogger(__name__)


def plot_scaling_factors_map(
    alpha_da: xr.DataArray,
    beta_da: xr.DataArray,
    title: str = "",
    output_png: str = "figures/scaling_factors.png",
):
    """Developer Diagnostic Map: 2-Panel Action Scaling Factors (alpha & beta)."""
    proj = ccrs.Robinson(central_longitude=0)
    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(1, 2, wspace=0.12, top=0.86, bottom=0.15, left=0.04, right=0.96)

    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    ax2 = fig.add_subplot(gs[0, 1], projection=proj)

    for ax in (ax1, ax2):
        ax.set_facecolor(LAND_GRAY)
        ax.add_feature(cfeature.LAND, facecolor=LAND_GRAY, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#404040", zorder=5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":", zorder=5)

    bounds_scale = [0.0, 0.33, 0.67, 0.90, 1.11, 1.50, 3.00, 5.00]
    colors_scale = ['#2166ac', '#4393c3', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d', '#b2182b']
    cmap_scale = mcolors.ListedColormap(colors_scale)
    cmap_scale.set_bad(color=(0, 0, 0, 0))
    cmap_scale.set_over('#67001f')
    norm_scale = mcolors.BoundaryNorm(bounds_scale, cmap_scale.N)

    # Panel (a): Alpha
    lon2d_a, lat2d_a, cyclic_alpha, alpha_vals = _prepare_cyclic_grid(alpha_da)
    im1 = ax1.pcolormesh(lon2d_a, lat2d_a, cyclic_alpha, cmap=cmap_scale, norm=norm_scale, shading="auto", transform=ccrs.PlateCarree(), zorder=3)
    med_alpha = float(np.nanmedian(alpha_vals))
    ax1.set_title(rf"$\mathbf{{(a)\ Signal\ Scaling\ (\alpha)}}\ \mid\ \mathbf{{Median:\ {med_alpha:.2f}}}$" "\n[ < 1: Dampen Mean  |  ~1: Untouched  |  > 1: Amplify Mean ]", fontsize=8.0, pad=6)
    cbar_ax1 = fig.add_axes([0.08, 0.08, 0.38, 0.025])
    fig.colorbar(im1, cax=cbar_ax1, orientation="horizontal", ticks=bounds_scale, extend="max", label=r"Signal Scaling Factor ($\alpha$)")

    # Panel (b): Beta
    lon2d_b, lat2d_b, cyclic_beta, beta_vals = _prepare_cyclic_grid(beta_da)
    im2 = ax2.pcolormesh(lon2d_b, lat2d_b, cyclic_beta, cmap=cmap_scale, norm=norm_scale, shading="auto", transform=ccrs.PlateCarree(), zorder=3)
    med_beta = float(np.nanmedian(beta_vals))
    ax2.set_title(rf"$\mathbf{{(b)\ Noise\ Spread\ Scaling\ (\beta)}}\ \mid\ \mathbf{{Median:\ {med_beta:.2f}}}$" "\n[ < 1: Compress Spread  |  ~1: Untouched  |  > 1: Inflate Spread ]", fontsize=8.0, pad=6)
    cbar_ax2 = fig.add_axes([0.54, 0.08, 0.38, 0.025])
    fig.colorbar(im2, cax=cbar_ax2, orientation="horizontal", ticks=bounds_scale, extend="max", label=r"Spread Scaling Factor ($\beta$)")

    _draw_manual_graticules(ax1)
    _draw_manual_graticules(ax2)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_msess_map(
    pct_mse_reduction_da: xr.DataArray,
    title: str = "",
    output_png: str = "figures/msess.png",
):
    """User Payoff Map: Global MSE Skill Score (MSESS)."""
    proj = ccrs.Robinson(central_longitude=0)
    fig = plt.figure(figsize=(10, 6.2))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_facecolor(LAND_GRAY)
    ax.add_feature(cfeature.LAND, facecolor=LAND_GRAY, zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#404040", zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":", zorder=5)

    mse_bounds = [-75.0, -50.0, -25.0, -10.0, -2.5, 2.5, 10.0, 25.0, 50.0, 75.0]
    mse_colors = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#f7f7f7', '#a6d96a', '#66bd63', '#1a9850', '#004529']
    mse_cmap = mcolors.ListedColormap(mse_colors)
    mse_cmap.set_bad(color=(0, 0, 0, 0))
    mse_norm = mcolors.BoundaryNorm(mse_bounds, mse_cmap.N)

    lon2d, lat2d, cyclic_mse, mse_vals = _prepare_cyclic_grid(pct_mse_reduction_da)
    im = ax.pcolormesh(lon2d, lat2d, cyclic_mse, cmap=mse_cmap, norm=mse_norm, shading="auto", transform=ccrs.PlateCarree(), zorder=3)

    min_val, med_val, max_val = float(np.nanmin(mse_vals)), float(np.nanmedian(mse_vals)), float(np.nanmax(mse_vals))
    sign_med = "+" if med_val >= 0 else ""
    sign_min = "+" if min_val >= 0 else ""
    sign_max = "+" if max_val >= 0 else ""

    ax.set_title(
        rf"$\mathbf{{MSESS_{{cal\ vs\ raw}}}}\ \mid\ \mathbf{{Range:\ {sign_min}{min_val:.1f}\%\ to\ {sign_max}{max_val:.1f}\%}}\ \mid\ \mathbf{{Median:\ {sign_med}{med_val:.1f}\%}}$"
        "\n[ Green = Error Reduced | White = Neutral (±2.5%) | Red = Error Increased ]",
        fontsize=8.5, pad=6
    )

    cbar_ax = fig.add_axes([0.15, 0.08, 0.70, 0.028])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=mse_bounds, extend="both", label=r"MSESS vs Raw Model (%)  $\left[\frac{\mathrm{MSE}_{\mathrm{raw}} - \mathrm{MSE}_{\mathrm{cal}}}{\mathrm{MSE}_{\mathrm{raw}}} \times 100\%\right]$")

    _draw_manual_graticules(ax)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
