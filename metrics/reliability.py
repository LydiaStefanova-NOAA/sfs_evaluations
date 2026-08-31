"""
Targeted Reliability Metric Calculation Module (Brier Score Decomposition & Mean Bin Alignment)
"""
import logging
import numpy as np
import xarray as xr
from preprocess.spatial_alignment import canonicalize_lonlat

logger = logging.getLogger(__name__)


def _clean_da(da: xr.DataArray) -> xr.DataArray:
    if da is None:
        return None
    drop_coords = [c for c in da.coords if c not in da.dims and c not in ["year", "lat", "lon", "latitude", "longitude"]]
    return da.drop_vars(drop_coords, errors="ignore") if drop_coords else da


def _transpose_spatial(da: xr.DataArray, target_dims: list = None) -> xr.DataArray:
    if target_dims is None:
        target_dims = ["year", "lat", "lon"]
    existing = [d for d in target_dims if d in da.dims] + [d for d in da.dims if d not in target_dims]
    return da.transpose(*existing)


def _build_nino34_mask(da: xr.DataArray) -> xr.DataArray:
    lat_name = next((c for c in ["lat", "latitude"] if c in da.coords or c in da.dims), None)
    lon_name = next((c for c in ["lon", "longitude"] if c in da.coords or c in da.dims), None)

    if lat_name is None or lon_name is None:
        raise ValueError("Dataset missing latitude/longitude coordinates.")

    lat_da, lon_da = da[lat_name], da[lon_name]
    lon_norm = xr.where(lon_da > 180, lon_da - 360, lon_da)

    mask = (lat_da >= -5.0) & (lat_da <= 5.0) & (lon_norm >= -170.0) & (lon_norm <= -120.0)
    drop_c = [c for c in mask.coords if c not in mask.dims]
    if drop_c:
        mask = mask.drop_vars(drop_c, errors="ignore")

    logger.info(f"Niño 3.4 spatial mask selected {int(mask.sum())} grid cell(s).")
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
    sfs_anom, obs_anom = _clean_da(sfs_anom), _clean_da(obs_anom)
    alpha_da, beta_da = _clean_da(alpha_da), _clean_da(beta_da)

    n_members = sfs_anom.sizes[member_dim]

    obs_thresh = _clean_da(obs_anom.quantile(quantile, dim=year_dim, skipna=True))
    obs_binary = _transpose_spatial(obs_anom > obs_thresh, [year_dim, "lat", "lon"])

    alpha_clean, beta_clean = alpha_da.fillna(0.0), beta_da.fillna(1.0)
    ens_mean_raw = sfs_anom.mean(dim=member_dim, skipna=True)
    ens_pert_raw = sfs_anom - ens_mean_raw
    sfs_cal = (alpha_clean * ens_mean_raw) + (beta_clean * ens_pert_raw)

    if threshold_mode == "model_relative":
        raw_thresh = _clean_da(sfs_anom.quantile(quantile, dim=[year_dim, member_dim], skipna=True))
        cal_thresh = _clean_da(sfs_cal.quantile(quantile, dim=[year_dim, member_dim], skipna=True))
        p_raw = (sfs_anom > raw_thresh).astype(float).mean(dim=member_dim, skipna=True)
        p_cal = (sfs_cal > cal_thresh).astype(float).mean(dim=member_dim, skipna=True)
    else:
        p_raw = (sfs_anom > obs_thresh).astype(float).mean(dim=member_dim, skipna=True)
        p_cal = (sfs_cal > obs_thresh).astype(float).mean(dim=member_dim, skipna=True)

    p_raw = _transpose_spatial(p_raw, [year_dim, "lat", "lon"])
    p_cal = _transpose_spatial(p_cal, [year_dim, "lat", "lon"])

    if mask_da is not None:
        mask_2d = _transpose_spatial(mask_da, ["lat", "lon"]).values
        obs_binary, p_raw, p_cal = obs_binary.where(mask_2d), p_raw.where(mask_2d), p_cal.where(mask_2d)

    obs_flat, raw_flat, cal_flat = obs_binary.values.flatten(), p_raw.values.flatten(), p_cal.values.flatten()

    valid_raw = ~np.isnan(obs_flat) & ~np.isnan(raw_flat)
    valid_cal = ~np.isnan(obs_flat) & ~np.isnan(cal_flat)

    obs_raw_vals, raw_vals = obs_flat[valid_raw], raw_flat[valid_raw]
    obs_cal_vals, cal_vals = obs_flat[valid_cal], cal_flat[valid_cal]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    prob_pred_raw, prob_obs_raw, counts_raw = [], [], []
    prob_pred_cal, prob_obs_cal, counts_cal = [], [], []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]

        # Raw binning (y_bar_k, o_bar_k)
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

        # Calibrated binning (y_bar_k, o_bar_k)
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

    # --- Brier Score Decomposition (BSS, REL, RES) ---
    base_rate = float(np.mean(obs_raw_vals)) if len(obs_raw_vals) > 0 else (1.0 - quantile)
    unc = base_rate * (1.0 - base_rate)

    counts_raw_arr, prob_pred_raw_arr, prob_obs_raw_arr = np.array(counts_raw), np.array(prob_pred_raw), np.array(prob_obs_raw)
    counts_cal_arr, prob_pred_cal_arr, prob_obs_cal_arr = np.array(counts_cal), np.array(prob_pred_cal), np.array(prob_obs_cal)

    valid_r = (counts_raw_arr > 0) & (~np.isnan(prob_obs_raw_arr))
    if np.sum(counts_raw_arr[valid_r]) > 0:
        n_tot_r = np.sum(counts_raw_arr[valid_r])
        rel_raw = float(np.sum(counts_raw_arr[valid_r] * (prob_pred_raw_arr[valid_r] - prob_obs_raw_arr[valid_r]) ** 2) / n_tot_r)
        res_raw = float(np.sum(counts_raw_arr[valid_r] * (prob_obs_raw_arr[valid_r] - base_rate) ** 2) / n_tot_r)
        bss_raw = float((res_raw - rel_raw) / unc) if unc > 0 else 0.0
    else:
        rel_raw, res_raw, bss_raw = np.nan, np.nan, np.nan

    valid_c = (counts_cal_arr > 0) & (~np.isnan(prob_obs_cal_arr))
    if np.sum(counts_cal_arr[valid_c]) > 0:
        n_tot_c = np.sum(counts_cal_arr[valid_c])
        rel_cal = float(np.sum(counts_cal_arr[valid_c] * (prob_pred_cal_arr[valid_c] - prob_obs_cal_arr[valid_c]) ** 2) / n_tot_c)
        res_cal = float(np.sum(counts_cal_arr[valid_c] * (prob_obs_cal_arr[valid_c] - base_rate) ** 2) / n_tot_c)
        bss_cal = float((res_cal - rel_cal) / unc) if unc > 0 else 0.0
    else:
        rel_cal, res_cal, bss_cal = np.nan, np.nan, np.nan

    return {
        "bin_centers": bin_centers,
        "prob_pred_raw": prob_pred_raw_arr,
        "prob_obs_raw": prob_obs_raw_arr,
        "counts_raw": counts_raw_arr,
        "prob_pred_cal": prob_pred_cal_arr,
        "prob_obs_cal": prob_obs_cal_arr,
        "counts_cal": counts_cal_arr,
        "base_rate": base_rate,
        "rel_raw": rel_raw,
        "res_raw": res_raw,
        "bss_raw": bss_raw,
        "rel_cal": rel_cal,
        "res_cal": res_cal,
        "bss_cal": bss_cal,
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
    sfs_anom, obs_anom = canonicalize_lonlat(sfs_anom), canonicalize_lonlat(obs_anom)
    alpha_da, beta_da = canonicalize_lonlat(alpha_da), canonicalize_lonlat(beta_da)
    if acc_da is not None:
        acc_da = canonicalize_lonlat(acc_da)

    nino34_mask = _build_nino34_mask(sfs_anom)
    nino34_rel = _compute_single_region_reliability(
        sfs_anom, obs_anom, alpha_da, beta_da, mask_da=nino34_mask, quantile=quantile, n_bins=n_bins, threshold_mode=threshold_mode
    )

    skill_mask = (acc_da >= min_acc_threshold) if acc_da is not None else None
    second_label = f"High-Skill Domain (ACC ≥ {min_acc_threshold:.1f})" if acc_da is not None else "Global Domain"

    second_rel = _compute_single_region_reliability(
        sfs_anom, obs_anom, alpha_da, beta_da, mask_da=skill_mask, quantile=quantile, n_bins=n_bins, threshold_mode=threshold_mode
    )

    return {"nino34": nino34_rel, "second_domain": second_rel, "second_label": second_label}
