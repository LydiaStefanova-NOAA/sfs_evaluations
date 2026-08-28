"""
Reliability Metric Calculation Module

Computes forecast probability vs. observed relative frequency for extreme events
(e.g., upper-tercile events >67th percentile).

Supports:
- Model-relative quantile thresholding (preserves rank order & tail probabilities)
  vs. absolute observation thresholding.
- Targeted domain evaluation (Niño 3.4 region: 5°S–5°N, 170°W–120°W).
- Skill-filtered domain evaluation (filtering grid points where ACC >= threshold).
"""
import logging
import numpy as np
import xarray as xr

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

    # Normalize longitudes to [-180, 180] regardless of input convention (0..360 vs -180..180)
    lon_norm = xr.where(lon_da > 180, lon_da - 360, lon_da)

    lat_mask = (lat_da >= -5.0) & (lat_da <= 5.0)
    lon_mask = (lon_norm >= -170.0) & (lon_norm <= -120.0)

    mask = lat_mask & lon_mask
    n_points = int(mask.sum())
    logger.info(f"Niño 3.4 spatial mask selected {n_points} grid cell(s).")
    
    if n_points == 0:
        logger.warning("⚠️ Warning: Niño 3.4 mask selected 0 grid cells! Check lat/lon range of input dataset.")

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
    Core engine to compute reliability curve metrics over a masked spatial domain.

    Parameters
    ----------
    sfs_anom : xr.DataArray
        Raw model anomaly ensemble (year, member, lat, lon).
    obs_anom : xr.DataArray
        Observational anomaly field (year, lat, lon).
    alpha_da : xr.DataArray
        Signal scaling factor (alpha).
    beta_da : xr.DataArray
        Spread scaling factor (beta).
    mask_da : xr.DataArray, optional
        Boolean spatial mask. If provided, metrics are restricted to True grid cells.
    quantile : float, default=0.67
        Event threshold quantile (0.67 for upper tercile).
    n_bins : int, default=5
        Number of probability binning intervals between 0.0 and 1.0.
    threshold_mode : {"model_relative", "obs_absolute"}, default="model_relative"
        - "model_relative": Evaluates exceedance relative to each dataset's OWN quantile threshold.
        - "obs_absolute": Evaluates exceedance directly against Q(Obs) in physical units.
    """
    n_members = sfs_anom.sizes[member_dim]

    # 1. Observed Event Threshold & Binary Indicator
    obs_thresh = obs_anom.quantile(quantile, dim=year_dim)
    obs_binary = obs_anom > obs_thresh

    # 2. Construct Calibrated Ensemble Members
    ens_mean_raw = sfs_anom.mean(dim=member_dim, skipna=True)
    ens_pert_raw = sfs_anom - ens_mean_raw
    sfs_cal = (alpha_da * ens_mean_raw) + (beta_da * ens_pert_raw)

    # 3. Compute Forecast Probabilities Based on Selected Threshold Mode
    if threshold_mode == "model_relative":
        # Compute threshold relative to each ensemble's own distribution across years and members
        raw_thresh = sfs_anom.quantile(quantile, dim=[year_dim, member_dim])
        cal_thresh = sfs_cal.quantile(quantile, dim=[year_dim, member_dim])

        p_raw = (sfs_anom > raw_thresh).sum(dim=member_dim) / n_members
        p_cal = (sfs_cal > cal_thresh).sum(dim=member_dim) / n_members
    elif threshold_mode == "obs_absolute":
        # Direct physical threshold comparison
        p_raw = (sfs_anom > obs_thresh).sum(dim=member_dim) / n_members
        p_cal = (sfs_cal > obs_thresh).sum(dim=member_dim) / n_members
    else:
        raise ValueError(f"Invalid threshold_mode '{threshold_mode}'. Allowed: 'model_relative', 'obs_absolute'")

    # 4. Apply Spatial Mask if provided
    if mask_da is not None:
        obs_binary = obs_binary.where(mask_da)
        p_raw = p_raw.where(mask_da)
        p_cal = p_cal.where(mask_da)

    # Flatten arrays and drop NaNs
    obs_vals = obs_binary.values.flatten()
    raw_vals = p_raw.values.flatten()
    cal_vals = p_cal.values.flatten()

    valid = ~np.isnan(obs_vals) & ~np.isnan(raw_vals) & ~np.isnan(cal_vals)
    obs_vals = obs_vals[valid]
    raw_vals = raw_vals[valid]
    cal_vals = cal_vals[valid]

    # 5. Probability Binning into Reliability Histogram
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    prob_pred_raw, prob_obs_raw, counts_raw = [], [], []
    prob_pred_cal, prob_obs_cal, counts_cal = [], [], []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]

        # Raw binning
        idx_r = (raw_vals >= low) & (raw_vals < high if i < n_bins - 1 else raw_vals <= high)
        if np.sum(idx_r) > 0:
            prob_pred_raw.append(np.mean(raw_vals[idx_r]))
            prob_obs_raw.append(np.mean(obs_vals[idx_r]))
            counts_raw.append(np.sum(idx_r))
        else:
            prob_pred_raw.append(bin_centers[i])
            prob_obs_raw.append(np.nan)
            counts_raw.append(0)

        # Calibrated binning
        idx_c = (cal_vals >= low) & (cal_vals < high if i < n_bins - 1 else cal_vals <= high)
        if np.sum(idx_c) > 0:
            prob_pred_cal.append(np.mean(cal_vals[idx_c]))
            prob_obs_cal.append(np.mean(obs_vals[idx_c]))
            counts_cal.append(np.sum(idx_c))
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
    logger.info(
        f"Computing Targeted Reliability Curves (q={quantile:.2f}, mode='{threshold_mode}')..."
    )

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

    # 2. Panel (b): High Skill Mask (ACC >= 0.3) if ACC is supplied, else Global
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
