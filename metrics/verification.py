"""
Post-Processing & Recalibration Verification Metrics
Computes scaling factors (alpha, beta) and MSESS with skill-gated least-squares dampening.
"""
import logging
import numpy as np
import xarray as xr
from metrics.snr import compute_model_variances

logger = logging.getLogger(__name__)


def compute_recalibration_metrics(
    sfs_anom: xr.DataArray,
    obs_anom: xr.DataArray,
    svr_da: xr.DataArray,
    nvr_da: xr.DataArray,
    acc_da: xr.DataArray,
    max_alpha: float = 3.0,
    max_beta: float = 5.0,
    year_dim: str = "year",
    member_dim: str = "member",
    eps: float = 1e-12,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Computes linear scaling factors (alpha, beta) and percentage MSE reduction (MSESS).
    Uses skill-gated least-squares regression dampening to prevent error explosion in unskillful regions.
    """
    logger.info("Computing Section 5 verification metrics (alpha, beta, MSESS)...")

    # 1. Unbiased Model Signal Std and Observed Std
    unbiased_sig_var, _ = compute_model_variances(
        sfs_anom, year_dim=year_dim, member_dim=member_dim, detrend=True
    )
    std_signal_mod = np.sqrt(unbiased_sig_var)
    std_obs = obs_anom.std(dim=year_dim, ddof=2, skipna=True)

    # 2. Skill-Gated Alpha: alpha = max(ACC, 0) * std_obs / std_signal_mod
    # If ACC <= 0, alpha = 0 (predicts climatology, avoiding catastrophic error explosion)
    acc_clipped = acc_da.clip(min=0.0)
    denom_alpha = std_signal_mod.where(std_signal_mod > eps)
    alpha_raw = (acc_clipped * std_obs) / denom_alpha

    # Cap alpha at max_alpha (e.g. 3.0) to prevent amplifying noise floors
    alpha_da = alpha_raw.clip(max=max_alpha).fillna(0.0)

    # 3. Noise Spread Scaling Beta (1/sqrt(NVR))
    nvr_guarded = xr.where(nvr_da > eps, nvr_da, np.nan)
    beta_da = (1.0 / np.sqrt(nvr_guarded)).clip(max=max_beta).fillna(1.0)

    alpha_da.name = "alpha"
    alpha_da.attrs["long_name"] = "Signal Mean Scaling Factor (alpha)"

    beta_da.name = "beta"
    beta_da.attrs["long_name"] = "Member Spread Scaling Factor (beta)"

    # 4. Raw & Calibrated Ensemble Means
    ens_mean_raw = sfs_anom.mean(dim=member_dim, skipna=True) if member_dim in sfs_anom.dims else sfs_anom
    ens_mean_cal = alpha_da * ens_mean_raw

    # 5. Compute Mean Squared Error (MSE) raw vs calibrated
    mse_raw = ((ens_mean_raw - obs_anom) ** 2).mean(dim=year_dim, skipna=True)
    mse_cal = ((ens_mean_cal - obs_anom) ** 2).mean(dim=year_dim, skipna=True)

    # 6. MSESS (%) = ((MSE_raw - MSE_cal) / MSE_raw) * 100
    mse_raw_guarded = xr.where(mse_raw > eps, mse_raw, np.nan)
    pct_mse_reduction_da = ((mse_raw - mse_cal) / mse_raw_guarded) * 100.0

    pct_mse_reduction_da.name = "pct_mse_reduction"
    pct_mse_reduction_da.attrs["long_name"] = "MSE Skill Score MSESS (%)"

    logger.info("✅ Verification metrics computation complete!")
    return alpha_da, beta_da, pct_mse_reduction_da
