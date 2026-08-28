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
    high_quantile: float = 85.0,
    low_quantile: float = 15.0,
) -> dict:
    """
    Selects 5 regime points targeting dynamic extremes (percentiles) relative to 1.0 
    within the skilled domain (ACC > skill_threshold OR r_pot > skill_threshold).
    Uses scale-invariant log-distance so SVR magnitude does not dominate NVR.
    """
    valid_mask = (
        ~np.isnan(svr_da.values) & (svr_da.values > 1e-6)
        & ~np.isnan(nvr_da.values) & (nvr_da.values > 1e-6)
        & ~np.isnan(acc_da.values)
        & ~np.isnan(rpot_da.values)
    )

    skill_mask = (acc_da.values > skill_threshold) | (rpot_da.values > skill_threshold)
    target_mask = valid_mask & skill_mask

    svr_vals = svr_da.values
    nvr_vals = nvr_da.values
    lats = svr_da.lat.values
    lons = svr_da.lon.values

    # Quadrant Definitions
    quadrant_specs = [
        ("Inflated Signal, Over-dispersed Noise", svr_vals > 1.0, nvr_vals > 1.0, high_quantile, high_quantile),
        ("Inflated Signal, Under-dispersed Noise", svr_vals > 1.0, nvr_vals < 1.0, high_quantile, low_quantile),
        ("Deficient Signal, Over-dispersed Noise", svr_vals < 1.0, nvr_vals > 1.0, low_quantile, high_quantile),
        ("Deficient Signal, Under-dispersed Noise", svr_vals < 1.0, nvr_vals < 1.0, low_quantile, low_quantile),
    ]

    selected_coords = {}

    for name, cond_s, cond_n, q_s, q_n in quadrant_specs:
        quadrant_mask = target_mask & cond_s & cond_n

        if not np.any(quadrant_mask):
            logger.warning(f"No grid points met physical condition for '{name}'. Setting plot to blank.")
            selected_coords[name] = None
        else:
            # Dynamically compute target based on quadrant percentiles
            target_s = float(np.percentile(svr_vals[quadrant_mask], q_s))
            target_n = float(np.percentile(nvr_vals[quadrant_mask], q_n))

            # Scale-invariant log-distance to dynamic targets
            dist = np.sqrt(
                (np.log(np.maximum(svr_vals, 1e-6) / target_s)) ** 2
                + (np.log(np.maximum(nvr_vals, 1e-6) / target_n)) ** 2
            )
            dist[~quadrant_mask] = np.inf

            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            selected_coords[name] = {
                "lat": float(lats[min_idx[0]]),
                "lon": float(lons[min_idx[1]]),
            }

    # Calibrated Ideal Regime (Closest to SVR = 1.0 and NVR = 1.0)
    ideal_key = "Calibrated Signal & Noise (SVR=1, NVR=1)"
    if not np.any(target_mask):
        logger.warning("No grid points met skill threshold for Ideal regime. Setting plot to blank.")
        selected_coords[ideal_key] = None
    else:
        dist_ideal = np.sqrt(
            (np.log(np.maximum(svr_vals, 1e-6) / 1.0)) ** 2
            + (np.log(np.maximum(nvr_vals, 1e-6) / 1.0)) ** 2
        )
        dist_ideal[~target_mask] = np.inf
        idx_ideal = np.unravel_index(np.argmin(dist_ideal), dist_ideal.shape)
        selected_coords[ideal_key] = {
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
    show_calibrated: bool = False,
    title: str = "",  # <--- ADD THIS PARAMETER
    output_png: str = "figures/anomaly_timeseries_regimes.png",
):
    """
    Plots normalized anomaly time series at 5 physical regime points in a 3x2 grid layout.
    
    Parameters:
    -----------
    show_calibrated : bool
        If True, overlays the recalibrated ensemble mean (alpha * fbar) and noise spread (beta * sigma)
        in blue alongside the raw forecast and observations.
    """
    nvr_method = nvr_da.attrs.get("nvr_method", "mse")

    coords = select_regime_gridpoints(
        svr_da=svr_da,
        nvr_da=nvr_da,
        acc_da=acc_da,
        rpot_da=rpot_da,
        skill_threshold=skill_threshold,
    )

    years = sfs_anom_da.year.values

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

        # Recalibration factors
        alpha = 1.0 / np.sqrt(max(svr_val, 1e-4))
        beta = 1.0 / np.sqrt(max(nvr_val, 1e-4))
        ens_mean_cal = alpha * ens_mean_norm
        noise_std_cal = beta * noise_std_t

        # Y-bound accounting for raw vs. calibrated
        ymin_pt = min(np.min(sfs_norm), np.min(obs_norm), np.min(ens_mean_norm - noise_std_t))
        ymax_pt = max(np.max(sfs_norm), np.max(obs_norm), np.max(ens_mean_norm + noise_std_t))

        if show_calibrated:
            ymin_pt = min(ymin_pt, np.min(ens_mean_cal - noise_std_cal))
            ymax_pt = max(ymax_pt, np.max(ens_mean_cal + noise_std_cal))

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
                "ens_mean_cal": ens_mean_cal,
                "noise_std_cal": noise_std_cal,
                "alpha": alpha,
                "beta": beta,
                "svr_val": svr_val,
                "nvr_val": nvr_val,
                "acc_val": acc_val,
                "rpot_val": rpot_val,
                "rpc_val": rpc_val,
            }
        )

    # Padding so top text box never collides with data
    if np.isinf(global_ymin) or np.isinf(global_ymax):
        ylim_bounds = (-3.5, 3.5)
    else:
        y_padding = (global_ymax - global_ymin) * 0.18
        ylim_bounds = (global_ymin - y_padding, global_ymax + y_padding)

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

        if show_calibrated:
            # 1. Raw Forecast (Subdued Crimson)
            ax.plot(
                years, ens_mean_norm, color="crimson", linestyle="--", linewidth=1.5,
                alpha=0.6, zorder=2, label="Raw Mean"
            )
            ax.fill_between(
                years, ens_mean_norm - noise_std_t, ens_mean_norm + noise_std_t,
                color="crimson", alpha=0.10, zorder=1, label=r"Raw $\pm 1\sigma$ Spread"
            )

            # 2. Recalibrated Forecast (Prominent Blue)
            ens_mean_cal = data["ens_mean_cal"]
            noise_std_cal = data["noise_std_cal"]
            alpha = data["alpha"]
            beta = data["beta"]

            ax.plot(
                years, ens_mean_cal, color="#1e40af", linewidth=2.2, zorder=4,
                label=rf"Cal. Mean ($\alpha={alpha:.2f}$)"
            )
            ax.fill_between(
                years, ens_mean_cal - noise_std_cal, ens_mean_cal + noise_std_cal,
                color="#3b82f6", alpha=0.25, zorder=3, label=rf"Cal. Spread ($\beta={beta:.2f}$)"
            )
        else:
            # Standard View: Raw Members, Spread, and Mean
            for m in range(sfs_norm.shape[1]):
                ax.plot(
                    years, sfs_norm[:, m], color="lightgray", alpha=0.35, linewidth=0.6, zorder=1,
                    label="Ensemble Members" if m == 0 else ""
                )
            ax.fill_between(
                years, ens_mean_norm - noise_std_t, ens_mean_norm + noise_std_t,
                color="crimson", alpha=0.20, zorder=2, label=r"$\pm 1\sigma_{\mathrm{noise}}$ Spread"
            )
            ax.plot(years, ens_mean_norm, color="crimson", linewidth=2.2, zorder=3, label="Ensemble Mean")

        # 3. Observed Time Series (Black Dashed)
        ax.plot(years, obs_norm, color="black", linewidth=2.0, linestyle="--", zorder=5, label="Observed")

        # Metric Text Box
        if nvr_method == "acc_varobs":
            svr_lbl = f"SVR (vs ACC²·Var_obs) = {data['svr_val']:.2f}"
            nvr_lbl = f"NVR (vs (1-ACC²)·Var_obs) = {data['nvr_val']:.2f}"
        else:
            svr_lbl = f"SVR (vs Var_obs) = {data['svr_val']:.2f}"
            nvr_lbl = f"NVR (vs MSE) = {data['nvr_val']:.2f}"

        cal_info = f"\nα = {data['alpha']:.2f} | β = {data['beta']:.2f}" if show_calibrated else ""

        metrics_text = (
            f"Lat: {lat:.2f}°, Lon: {lon:.2f}°\n"
            f"{svr_lbl}\n"
            f"{nvr_lbl}\n"
            f"ACC = {data['acc_val']:.2f}\n"
            rf"$r_{{\mathrm{{pot}}}}$ = {data['rpot_val']:.2f}" "\n"
            f"RPC = {data['rpc_val']:.2f}"
            f"{cal_info}"
        )
        ax.text(
            0.02, 0.96, metrics_text, transform=ax.transAxes,
            fontsize=8.0, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="gray")
        )

        if i == 0:
            ax.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    # Re-enable bottom X-axis labels on panel 4 (Row 2, Right)
    axes[1, 1].tick_params(labelbottom=True)
    axes_flat[5].set_visible(False)

    method_title_str = "ACC² · Var_obs Baseline" if nvr_method == "acc_varobs" else "MSE Baseline"
    cal_title_str = " [Raw vs Recalibrated Overlay]" if show_calibrated else ""
    main_title = title if title else f"Regime Time Series Diagnostics [{method_title_str}]{cal_title_str}"
    fig.suptitle(main_title, fontsize=12, fontweight="bold", y=0.995)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
