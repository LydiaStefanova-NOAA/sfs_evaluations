"""
Reliability Metric Calculation Module (Model-Relative Quantile Support)
"""
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def compute_reliability_curve(
    sfs_anom: xr.DataArray,
    obs_anom: xr.DataArray,
    alpha_da: xr.DataArray,
    beta_da: xr.DataArray,
    mask_da: xr.DataArray = None,
    quantile: float = 0.67,
    n_bins: int = 5,
    threshold_mode: str = "model_relative",  # "model_relative" vs "obs_absolute"
    year_dim: str = "year",
    member_dim: str = "member",
) -> dict:
    """
    Computes reliability curve data for upper-tercile events.

    Parameters
    ----------
    threshold_mode : str
        - "obs_absolute": Evaluates SFS exceedance against Q_0.67(Obs) in physical units.
        - "model_relative": Evaluates SFS exceedance against Q_0.67 of its OWN distribution.
    """
    # 1. Compute Observed Event Threshold
    obs_thresh = obs_anom.quantile(quantile, dim=year_dim)
    obs_binary = obs_anom > obs_thresh

    # 2. Construct Calibrated Ensemble
    ens_mean_raw = sfs_anom.mean(dim=member_dim, skipna=True)
    ens_pert_raw = sfs_anom - ens_mean_raw
    sfs_cal = (alpha_da * ens_mean_raw) + (beta_da * ens_pert_raw)

    n_members = sfs_anom.sizes[member_dim]

    # 3. Compute Event Thresholds Based on Selected Mode
    if threshold_mode == "model_relative":
        # Calculate quantiles relative to each ensemble's own distribution
        raw_thresh = sfs_anom.quantile(quantile, dim=[year_dim, member_dim])
        cal_thresh = sfs_cal.quantile(quantile, dim=[year_dim, member_dim])

        p_raw = (sfs_anom > raw_thresh).sum(dim=member_dim) / n_members
        p_cal = (sfs_cal > cal_thresh).sum(dim=member_dim) / n_members
    else:
        # Physical units (Obs threshold applied directly)
        p_raw = (sfs_anom > obs_thresh).sum(dim=member_dim) / n_members
        p_cal = (sfs_cal > obs_thresh).sum(dim=member_dim) / n_members

    # Apply spatial mask if provided
    if mask_da is not None:
        obs_binary = obs_binary.where(mask_da)
        p_raw = p_raw.where(mask_da)
        p_cal = p_cal.where(mask_da)

    # Flatten arrays
    obs_vals = obs_binary.values.flatten()
    raw_vals = p_raw.values.flatten()
    cal_vals = p_cal.values.flatten()

    valid = ~np.isnan(obs_vals) & ~np.isnan(raw_vals) & ~np.isnan(cal_vals)
    obs_vals, raw_vals, cal_vals = obs_vals[valid], raw_vals[valid], cal_vals[valid]

    # 4. Bin Probabilities into Reliability Histogram
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    prob_pred_raw, prob_obs_raw, counts_raw = [], [], []
    prob_pred_cal, prob_obs_cal, counts_cal = [], [], []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]

        idx_r = (raw_vals >= low) & (raw_vals < high if i < n_bins - 1 else raw_vals <= high)
        if np.sum(idx_r) > 0:
            prob_pred_raw.append(np.mean(raw_vals[idx_r]))
            prob_obs_raw.append(np.mean(obs_vals[idx_r]))
            counts_raw.append(np.sum(idx_r))
        else:
            prob_pred_raw.append(bin_centers[i])
            prob_obs_raw.append(np.nan)
            counts_raw.append(0)

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
