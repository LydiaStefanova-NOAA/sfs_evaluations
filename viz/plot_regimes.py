import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def select_regime_gridpoints(
    svr_da: xr.DataArray,
    nvr_da: xr.DataArray,
    acc_da: xr.DataArray,
    rpot_da: xr.DataArray,
    skill_threshold: float = 0.6,
) -> dict:
    """
    Selects 5 regime points targeting physical thresholds relative to 1.0 within
    the skilled domain (ACC > skill_threshold OR r_pot > skill_threshold).
    If a condition is unpopulated, returns None for that regime.
    """
    valid_mask = (
        ~np.isnan(svr_da.values)
        & ~np.isnan(nvr_da.values)
        & ~np.isnan(acc_da.values)
        & ~np.isnan(rpot_da.values)
    )

    skill_mask = (acc_da.values > skill_threshold) | (rpot_da.values > skill_threshold)
    target_mask = valid_mask & skill_mask

    svr_vals = svr_da.values
    nvr_vals = nvr_da.values
    lats = svr_da.lat.values
    lons = svr_da.lon.values

    regimes = {
        "Excessive Signal, Excessive Noise": {
            "target": (2.0, 2.0),
            "cond": (svr_vals > 1.0) & (nvr_vals > 1.0),
        },
        "Excessive Signal, Deficient Noise": {
            "target": (2.0, 0.2),
            "cond": (svr_vals > 1.0) & (nvr_vals < 1.0),
        },
        "Deficient Signal, Excessive Noise": {
            "target": (0.2, 2.0),
            "cond": (svr_vals < 1.0) & (nvr_vals > 1.0),
        },
        "Deficient Signal, Deficient Noise": {
            "target": (0.2, 0.2),
            "cond": (svr_vals < 1.0) & (nvr_vals < 1.0),
        },
    }

    selected_coords = {}

    for name, spec in regimes.items():
        target_s, target_n = spec["target"]
        quadrant_mask = target_mask & spec["cond"]

        if not np.any(quadrant_mask):
            logger.warning(f"No grid points met physical condition for '{name}'. Setting plot to blank.")
            selected_coords[name] = None
        else:
            dist = np.sqrt((svr_vals - target_s) ** 2 + (nvr_vals - target_n) ** 2)
            dist[~quadrant_mask] = np.inf

            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            selected_coords[name] = {
                "lat": float(lats[min_idx[0]]),
                "lon": float(lons[min_idx[1]]),
            }

    # Ideal Regime (Closest to SVR = 1.0 and NVR = 1.0)
    if not np.any(target_mask):
        logger.warning("No grid points met skill threshold for Ideal regime. Setting plot to blank.")
        selected_coords["Ideal Signal & Noise (SVR=1, NVR=1)"] = None
    else:
        dist_ideal = np.sqrt((svr_vals - 1.0) ** 2 + (nvr_vals - 1.0) ** 2)
        dist_ideal[~target_mask] = np.inf
        idx_ideal = np.unravel_index(np.argmin(dist_ideal), dist_ideal.shape)
        selected_coords["Ideal Signal & Noise (SVR=1, NVR=1)"] = {
            "lat": float(lats[idx_ideal[0]]),
            "lon": float(lons[idx_ideal[1]]),
        }

    return selected_coords


def plot_anomaly_time_series_regimes(
    sfs_anom_da: xr.DataArray,
    obs_anom_da: xr.DataArray,
    svr_da: xr.DataArray,
    nvr_da: xr.DataArray,
    acc_da: xr.DataArray,
    rpot_da: xr.DataArray,
    rpc_da: xr.DataArray,
    skill_threshold: float = 0.6,
    output_png: str = "figures/anomaly_timeseries_regimes.png",
):
    """
    Plots normalized anomaly time series at 5 physical regime points in a 3x2 grid layout.
    The 4 extreme regimes occupy rows 1-2, and the 'Ideal' regime sits on a line by itself (row 3).
    Unmet conditions render as a blank subplot labeled 'Condition Not Met'.
    """
    coords = select_regime_gridpoints(
        svr_da=svr_da,
        nvr_da=nvr_da,
        acc_da=acc_da,
        rpot_da=rpot_da,
        skill_threshold=skill_threshold,
    )

    years = sfs_anom_da.year.values

    # First Pass: Extract time series & compute global Y-axis bounds for valid points
    regime_data = []
    global_ymin = np.inf
    global_ymax = -np.inf

    for regime_name, point in coords.items():
        if point is None:
            regime_data.append({"name": regime_name, "valid": False})
            continue

        lat, lon = point["lat"], point["lon"]

        sfs_pt = sfs_anom_da.sel(lat=lat, lon=lon, method="nearest")
        obs_pt = obs_anom_da.sel(lat=lat, lon=lon, method="nearest")

        svr_val = float(svr_da.sel(lat=lat, lon=lon, method="nearest").values)
        nvr_val = float(nvr_da.sel(lat=lat, lon=lon, method="nearest").values)
        acc_val = float(acc_da.sel(lat=lat, lon=lon, method="nearest").values)
        rpot_val = float(rpot_da.sel(lat=lat, lon=lon, method="nearest").values)
        rpc_val = float(rpc_da.sel(lat=lat, lon=lon, method="nearest").values)

        obs_std = float(obs_pt.std(dim="year", ddof=1).values)
        norm_factor = obs_std if obs_std > 1e-12 else 1.0

        obs_norm = obs_pt.values / norm_factor
        sfs_norm = sfs_pt.values / norm_factor
        ens_mean_norm = sfs_pt.mean(dim="member").values / norm_factor
        noise_std_t = np.std(sfs_norm, axis=1, ddof=1)

        ymin_pt = min(np.min(sfs_norm), np.min(obs_norm), np.min(ens_mean_norm - noise_std_t))
        ymax_pt = max(np.max(sfs_norm), np.max(obs_norm), np.max(ens_mean_norm + noise_std_t))

        global_ymin = min(global_ymin, ymin_pt)
        global_ymax = max(global_ymax, ymax_pt)

        regime_data.append(
            {
                "name": regime_name,
                "valid": True,
                "lat": lat,
                "lon": lon,
                "obs_norm": obs_norm,
                "sfs_norm": sfs_norm,
                "ens_mean_norm": ens_mean_norm,
                "noise_std_t": noise_std_t,
                "svr_val": svr_val,
                "nvr_val": nvr_val,
                "acc_val": acc_val,
                "rpot_val": rpot_val,
                "rpc_val": rpc_val,
            }
        )

    # Set default bounds if no condition was met across any regime
    if np.isinf(global_ymin) or np.isinf(global_ymax):
        ylim_bounds = (-3.0, 3.0)
    else:
        y_padding = (global_ymax - global_ymin) * 0.05
        ylim_bounds = (global_ymin - y_padding, global_ymax + y_padding)

    # Second Pass: Plotting on 3x2 Grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, data in enumerate(regime_data):
        ax = axes_flat[i]
        ax.set_title(data["name"], fontsize=11, fontweight="bold")
        ax.set_ylabel(r"Normalized Anomaly ($\sigma_{\mathrm{obs}}$)")
        ax.set_ylim(ylim_bounds)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", zorder=0)

        if not data["valid"]:
            ax.text(
                0.5, 0.5, "Condition Not Met", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, fontweight="bold", color="dimgray"
            )
            continue

        lat, lon = data["lat"], data["lon"]
        sfs_norm = data["sfs_norm"]
        ens_mean_norm = data["ens_mean_norm"]
        noise_std_t = data["noise_std_t"]
        obs_norm = data["obs_norm"]

        # 1. Faint individual ensemble members
        for m in range(sfs_norm.shape[1]):
            ax.plot(
                years, sfs_norm[:, m], color="lightgray", alpha=0.35, linewidth=0.6, zorder=1,
                label="Ensemble Members" if m == 0 else ""
            )

        # 2. Shaded \pm 1\sigma_{noise} Spread Band
        ax.fill_between(
            years,
            ens_mean_norm - noise_std_t,
            ens_mean_norm + noise_std_t,
            color="crimson",
            alpha=0.20,
            zorder=2,
            label=r"$\pm 1\sigma_{\mathrm{noise}}$ Spread",
        )

        # 3. Ensemble Mean & Observations
        ax.plot(years, ens_mean_norm, color="crimson", linewidth=2.2, zorder=3, label="Ensemble Mean")
        ax.plot(years, obs_norm, color="black", linewidth=2.0, linestyle="--", zorder=4, label="Observed")

        metrics_text = (
            f"Lat: {lat:.2f}°, Lon: {lon:.2f}°\n"
            f"SVR = {data['svr_val']:.2f}\n"
            f"NVR = {data['nvr_val']:.2f}\n"
            f"ACC = {data['acc_val']:.2f}\n"
            rf"$r_{{\mathrm{{pot}}}}$ = {data['rpot_val']:.2f}" "\n"
            f"RPC = {data['rpc_val']:.2f}"
        )
        ax.text(
            0.02, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray")
        )

        if i == 0:
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # Hide unused 6th panel (Row 3, Column 2) so Ideal sits alone on Row 3
    axes_flat[5].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
