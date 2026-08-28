"""
Spread-to-Error Ratio (SER) Metric Functions
"""
import logging
import numpy as np
import xarray as xr
from metrics.snr import compute_model_variances

logger = logging.getLogger(__name__)


def compute_ser_before_after(
    sfs_anom: xr.DataArray,
    obs_anom: xr.DataArray,
    alpha_da: xr.DataArray,
    beta_da: xr.DataArray,
    year_dim: str = "year",
    member_dim: str = "member",
    eps: float = 1e-12,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes Spread-to-Error Ratio (SER) before and after post-processing calibration.

    Returns:
    --------
    tuple[xr.DataArray, xr.DataArray]
        (ser_raw, ser_cal)
    """
    logger.info("Computing spatial Spread-to-Error Ratio (SER) before & after calibration...")

    # 1. Raw Unbiased Noise Standard Deviation
    _, var_noise_mod = compute_model_variances(
        sfs_anom, year_dim=year_dim, member_dim=member_dim, detrend=True
    )
    std_noise_raw = np.sqrt(var_noise_mod)
    std_noise_cal = beta_da * std_noise_raw

    # 2. Raw vs Calibrated Ensemble Means
    ens_mean_raw = sfs_anom.mean(dim=member_dim, skipna=True) if member_dim in sfs_anom.dims else sfs_anom
    ens_mean_cal = alpha_da * ens_mean_raw

    # 3. Raw vs Calibrated RMSE
    rmse_raw = np.sqrt(((ens_mean_raw - obs_anom) ** 2).mean(dim=year_dim, skipna=True))
    rmse_cal = np.sqrt(((ens_mean_cal - obs_anom) ** 2).mean(dim=year_dim, skipna=True))

    # 4. Compute SER (Spread / RMSE)
    ser_raw = std_noise_raw / rmse_raw.where(rmse_raw > eps)
    ser_cal = std_noise_cal / rmse_cal.where(rmse_cal > eps)

    ser_raw.name = "ser_raw"
    ser_raw.attrs["long_name"] = "Raw Spread-to-Error Ratio (SER)"

    ser_cal.name = "ser_cal"
    ser_cal.attrs["long_name"] = "Calibrated Spread-to-Error Ratio (SER)"

    logger.info("✅ SER metrics computation complete!")
    return ser_raw, ser_cal
