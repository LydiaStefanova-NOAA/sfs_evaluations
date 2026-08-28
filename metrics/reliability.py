"""
Targeted Reliability Metric Calculation Module

Computes forecast probability vs. observed relative frequency for upper-tercile events
(e.g., >67th percentile) across specific spatial domains (Niño 3.4 & High-Skill regions).

Features:
- Auto-canonicalization of coordinates (lat: -90..90, lon: -180..180).
- Protection against xarray quantile coordinate alignment mismatches.
- Decoupled validity filtering (prevents calibration NaNs from wiping out raw forecasts).
- Support for model-relative quantile thresholding to preserve rank order.
"""
import logging
import numpy as np
import xarray as xr
from preprocess.spatial_alignment import canonicalize_lonlat

logger = logging.getLogger(__name__)


def _build_nino34_mask(da: xr.DataArray) -> xr.DataArray:
    """
    Creates a boolean spatial mask for the Niño 3.4 region (5°S–5°N, 170°W–120°W).
    Universally normalizes longitudes to [-180, 180] space.
    """
    lat_name = next((c for c in ["lat", "latitude"] if c in da.coords or c in da.dims), None)
    lon_name = next((c for c in ["lon", "longitude"] if c in da.coords or c in da.dims), None)

    if lat_name is None or lon_name is None:
        raise ValueError(f"Dataset missing latitude/longitude coordinates. Found coords: {list(da.coords)}")

    lat_da = da[lat_name]
    lon_da = da[lon_name]

    # Universal longitude normalization to [-180, 180]
    lon_norm = xr.where(lon_da > 180, lon_da - 360, lon_da)

    lat_mask = (lat_da >= -5.0) & (lat_da <= 5.0)
    lon_mask = (lon_norm >= -170.0) & (lon_norm <= -120.0)

    mask = lat_mask & lon_mask
    n_points = int(mask.sum())
    logger.info(f"Niño 3.4 spatial mask selected {n_points} grid cell(s).")
    return mask


def _compute_single_region_reliability(
    sfs_anom: xr.DataArray,
    obs_anom: xr.DataArray,
    alpha_da: xr.DataArray,
    beta_da: xr.DataArray,
    mask_da: xr.DataArray = None,
    quantile: float = 0.67,
    n_bins: int = 5,
    threshold_mode: str = "model_relative",
    year_dim: str = "year",
    member_dim: str = "member",
) -> dict:
    """
    Core engine to compute reliability curve metrics over a spatial domain.
    """
    n_members = sfs_anom.sizes[member_dim]

    # 1. Observed Event Threshold & Binary Indicator
    obs_thresh = obs_anom.quantile(quantile, dim=year_dim, skipna=True).drop_vars("quantile", errors="ignore")
    obs_binary = obs_anom > obs_thresh

    # 2. Construct Calibrated Ensemble Members with NaN protection
    alpha_clean = alpha_da.fillna(0.0)
    beta_clean = beta_da.fillna(1.0)

    ens_mean_raw = sfs_anom.mean(dim=member_dim, skipna=True)
    ens_pert_raw = sfs_anom - ens_mean_raw
    sfs_cal = (alpha_clean * ens_mean_raw) + (beta_clean * ens_pert_raw)

    # 3. Compute Forecast Probabilities (Stripping quantile coords to prevent xarray comparison NaNs)
    if threshold_mode == "model_relative":
        raw_thresh = sfs_anom.quantile(quantile, dim=[year_dim, member_dim], skipna=True).drop_vars("quantile", errors="ignore")
        cal_thresh = sfs_cal.quantile(quantile, dim=[year_dim, member_dim], skipna=True).drop_vars("quantile", errors="ignore")

        p_raw = (sfs_anom > raw_thresh).astype(float).mean(dim=member_dim, skipna=True)
        p_cal = (sfs_cal > cal_thresh).astype(float).mean(dim=member_dim, skipna=True)
    elif threshold_mode == "obs_absolute":
        p_raw = (sfs_anom > obs_thresh).astype(float).mean(dim=member_dim, skipna=True)
        p_cal = (sfs_cal > obs_thresh).astype(float).mean(dim=member_dim, skipna=True)
    else:
        raise ValueError(f"Invalid threshold_mode '{threshold_mode}'. Allowed: 'model_relative', 'obs_absolute'")

    # 4. Apply Spatial Mask if provided
    if mask_da is not None:
        obs_binary = obs_binary.where(mask_da)
        p_raw = p_raw.where(mask_da)
        p_cal = p_cal.where(mask_da)

    # Flatten arrays
    obs_flat = obs_binary.values.flatten()
    raw_flat = p_raw.values.flatten()
    cal_flat = p_cal.values.flatten()

    # 5. Decoupled Validity Filtering (Prevents calibration NaNs from wiping out the raw model line)
    valid_raw = ~np.isnan(obs_flat) & ~np.isnan(raw_flat)
    valid_cal = ~np.isnan(obs_flat) & ~np.isnan(cal_flat)

    obs_raw_vals, raw_vals = obs_flat[valid_raw], raw_flat[valid_raw]
    obs_cal_vals, cal_vals = obs_flat[valid_cal], cal_flat[valid_cal]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    prob_pred_raw, prob_obs_raw, counts_raw = [], [], []
    prob_pred_cal, prob_obs_cal, counts_cal = [], [], []

    # 6. Probability Binning Loop
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]

        # Raw model binning
        if len(raw_vals) > 0:
            idx_r = (raw_vals >= low) & (raw_vals < high if i < n_bins - 1 else raw_vals <= high)
            if np.sum(idx_r) > 0:
                prob_pred_raw.append(np.mean(raw_vals[idx_r]))
                prob_obs_raw.append(np.mean(obs_raw_vals[idx_r]))
                counts_raw.append(np.sum(idx_r))
            else:
                prob_pred_raw.append(bin_centers[i])
                prob_obs_raw.append(np.nan)
                counts_raw.append(0)
        else:
            prob_pred_raw.append(bin_centers[i])
            prob_obs_raw.append(np.nan)
            counts_raw.append(0)

        # Calibrated model binning
        if len(cal_vals) > 0:
            idx_c = (cal_vals >= low) & (cal_vals < high if i < n_bins - 1 else cal_vals <= high)
            if np.sum(idx_c) > 0:
                prob_pred_cal.append(np.mean(cal_vals[idx_c]))
                prob_obs_cal.append(np.mean(obs_cal_vals[idx_c]))
                counts_cal.append(np.sum(idx_c))
            else:
                prob_pred_cal.append(bin_centers[i])
                prob_obs_cal.append(np.nan)
                counts_cal.append(0)
        else:
            prob_pred_cal.append(bin_centers[i])
            prob_obs_cal.append(np.nan)
            counts_cal.append(0)

    return {
        "bin_centers": bin_centers,
        "prob_pred_raw": np.array(prob_pred_raw),
        "prob_obs_raw": np.array(prob_obs_raw),
        "counts_raw": np.array(counts_raw),
        "prob_pred_cal": np.array(prob_pred_cal),
        "prob_obs_cal": np.array(prob_obs_cal),
        "counts_cal": np.array(counts_cal),
        "valid_count": max(len(raw_vals), len(cal_vals)),
    }


def compute_nino34_reliability_curves(
    sfs_anom: xr.DataArray,
    obs_anom: xr.DataArray,
    alpha_da: xr.DataArray,
    beta_da: xr.DataArray,
    acc_da: xr.DataArray = None,
    quantile: float = 0.67,
    n_bins: int = 5,
    min_acc_threshold: float = 0.3,
    threshold_mode: str = "model_relative",
) -> dict:
    """
    Computes reliability curves for a 2-panel comparison:
    - Panel (a): Niño 3.4 Region (5°S–5°N, 170°W–120°W)
    - Panel (b): High-Skill Domain (ACC >= min_acc_threshold) or Global Baseline
    """
    logger.info(f"Computing Targeted Reliability Curves (q={quantile:.2f}, mode='{threshold_mode}')...")

    # Force canonical coordinates across all DataArrays to eliminate indexing mismatches
    sfs_anom = canonicalize_lonlat(sfs_anom)
    obs_anom = canonicalize_lonlat(obs_anom)
    alpha_da = canonicalize_lonlat(alpha_da)
    beta_da = canonicalize_lonlat(beta_da)
    if acc_da is not None:
        acc_da = canonicalize_lonlat(acc_da)

    # 1. Panel (a): Niño 3.4 Mask
    nino34_mask = _build_nino34_mask(sfs_anom)
    nino34_rel = _compute_single_region_reliability(
        sfs_anom=sfs_anom,
        obs_anom=obs_anom,
        alpha_da=alpha_da,
        beta_da=beta_da,
        mask_da=nino34_mask,
        quantile=quantile,
        n_bins=n_bins,
        threshold_mode=threshold_mode,
    )

    # 2. Panel (b): High-Skill Domain (ACC >= min_acc_threshold)
    if acc_da is not None:
        skill_mask = acc_da >= min_acc_threshold
        second_label = f"High-Skill Domain (ACC ≥ {min_acc_threshold:.1f})"
    else:
        skill_mask = None
        second_label = "Global Domain"

    second_rel = _compute_single_region_reliability(
        sfs_anom=sfs_anom,
        obs_anom=obs_anom,
        alpha_da=alpha_da,
        beta_da=beta_da,
        mask_da=skill_mask,
        quantile=quantile,
        n_bins=n_bins,
        threshold_mode=threshold_mode,
    )

    logger.info("✅ Targeted reliability metrics computation complete!")
    return {
        "nino34": nino34_rel,
        "second_domain": second_rel,
        "second_label": second_label,
    }


def compute_reliability_curve(
    sfs_anom: xr.DataArray,
    obs_anom: xr.DataArray,
    alpha_da: xr.DataArray,
    beta_da: xr.DataArray,
    quantile: float = 0.67,
    n_bins: int = 5,
    threshold_mode: str = "model_relative",
) -> dict:
    """
    Backward-compatible single-domain (global) reliability curve entry point.
    """
    return _compute_single_region_reliability(
        sfs_anom=sfs_anom,
        obs_anom=obs_anom,
        alpha_da=alpha_da,
        beta_da=beta_da,
        mask_da=None,
        quantile=quantile,
        n_bins=n_bins,
        threshold_mode=threshold_mode,
    )
