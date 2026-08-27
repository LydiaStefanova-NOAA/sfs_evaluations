"""
Spatial Map Visualization for Anomaly Amplitude Ratio
Matches viz/spatial.py styling, projection, and cyclic grid handling.
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from viz.spatial import _draw_manual_graticules, _prepare_cyclic_grid, LAND_GRAY
from metrics.confidence import compute_dynamic_ci_bounds


def plot_amplitude_ratio_map(
    ar_da: xr.DataArray,
    n_years: int | None = None,
    n_members: int | None = None,
    detrend: bool = True,
    title_str: str = "ECMWF Anomaly Amplitude Ratio Diagnostic",
    output_png: str = "figures/amplitude_ratio.png",
):
    """
    Plots spatial map of ECMWF Anomaly Amplitude Ratio using Robinson projection,
    cyclic grid alignment, and F-test confidence bounds.
    """
    # 1. Prepare cyclic grid and spatial statistics
    lon2d, lat2d, cyclic_ar, ar_vals = _prepare_cyclic_grid(ar_da)
    med_ar = float(np.nanmedian(ar_vals))
    mode_str = ar_da.attrs.get("mode", "members")

    # 2. Derive AR confidence bounds directly from compute_dynamic_ci_bounds
    if n_years is not None and n_members is not None:
        _, _, _, (nvr_low, nvr_high) = compute_dynamic_ci_bounds(
            n_years=n_years, n_members=n_members, detrend=detrend
        )
        # AR = sqrt(Variance Ratio)
        ar_low = round(np.sqrt(nvr_low), 2)
        ar_high = round(np.sqrt(nvr_high), 2)

        ar_bounds = sorted(
            list(
                set(
                    [
                        0.0,
                        round(ar_low / 2, 2),
                        ar_low,
                        ar_high,
                        round(ar_high * 1.5, 2),
                        round(ar_high * 2.2, 2),
                    ]
                )
            )
        )
        subtitle_desc = f"[ < {ar_low:.2f}: Suppressed  |  {ar_low:.2f}–{ar_high:.2f}: Calibrated (95% CI)  |  > {ar_high:.2f}: Inflated ]"
    else:
        ar_bounds = [0.0, 0.5, 0.75, 0.9, 1.1, 1.35, 1.7, 2.2]
        subtitle_desc = "[ < 0.9: Suppressed Amplitude  |  0.9–1.1: Calibrated  |  > 1.1: Inflated Amplitude ]"

    # Diverging colors: Blues for underestimated (<ar_low), Reds for overestimated (>ar_high)
    ar_colors = [
        '#08519c', '#3182bd', '#9ecae1',  # Underestimated
        '#f7f7f7',                       # Calibrated (95% CI)
        '#fc9272', '#de2d26', '#a50f15'   # Overestimated
    ]

    cmap_ar = mcolors.ListedColormap(ar_colors[: len(ar_bounds) - 1])
    cmap_ar.set_bad(color=(0, 0, 0, 0))  # Transparent NaNs
    cmap_ar.set_over('#67000d')           # Extreme inflation
    norm_ar = mcolors.BoundaryNorm(ar_bounds, cmap_ar.N)

    # 3. Figure & Axis Setup (Robinson projection)
    fig = plt.figure(figsize=(10, 6.5))
    gs = fig.add_gridspec(1, 1, top=0.88, bottom=0.15, left=0.05, right=0.95)

    proj = ccrs.Robinson(central_longitude=0)
    ax = fig.add_subplot(gs[0, 0], projection=proj)

    ax.set_facecolor(LAND_GRAY)
    ax.add_feature(cfeature.LAND, facecolor=LAND_GRAY, zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":", zorder=5)
    ax.gridlines(draw_labels=False, linestyle=":", color="gray", alpha=0.3, zorder=6)

    # 4. Render Mesh
    im = ax.pcolormesh(
        lon2d,
        lat2d,
        cyclic_ar,
        cmap=cmap_ar,
        norm=norm_ar,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    # 5. Math Subtitle & Metadata
    subtitle_math = rf"$\mathbf{{Anomaly\ Amplitude\ Ratio}}\ (\sigma_{{\mathrm{{mod,{mode_str}}}}} / \sigma_{{\mathrm{{obs}}}}) \mid \mathbf{{Spatial\ Median:\ {med_ar:.2f}}}$"
    ax.set_title(f"{subtitle_math}\n{subtitle_desc}", fontsize=9, pad=8)

    # 6. Colorbar
    cbar_ax = fig.add_axes([0.25, 0.08, 0.50, 0.025])
    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="horizontal",
        ticks=ar_bounds,
        extend="max",
        label=r"Anomaly Amplitude Ratio ($\sigma_{\mathrm{mod}} / \sigma_{\mathrm{obs}}$)",
    )
    cbar.ax.tick_params(labelsize=8)

    # 7. Add Manual Graticules
    _draw_manual_graticules(ax)

    fig.suptitle(title_str, fontsize=12, fontweight="bold", y=0.97)

    # Save
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
