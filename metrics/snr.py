"""
Signal-to-Noise Ratio (SNR) Metric Functions
"""
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _linear_detrend(da: xr.DataArray, dim: str = "year") -> xr.DataArray:
    """Subtract a linear trend along the specified dimension using xarray polyfit/polyval."""
    poly_coeffs = da.polyfit(dim=dim, deg=1)
    trend = xr.polyval(da[dim], poly_coeffs.polyfit_coefficients)
    return da - trend


def compute_snr(
    sfs_da: xr.DataArray,
    year_dim: str = "year",
    member_dim: str = "member",
    detrend: bool = True,
) -> xr.DataArray:
    """
    Compute Unbiased Signal-to-Noise Ratio (SNR) as a Variance Ratio across members and reforecast years.
    Lightweight Dask-friendly masking to prevent socket timeouts.
    """
    logger.info("Computing Signal-to-Noise Ratio (SNR) [Variance Ratio]...")

    if "init" in sfs_da.dims and year_dim not in sfs_da.dims:
        sfs_da = sfs_da.rename({"init": year_dim})

    M = sfs_da.sizes[member_dim]

    # 1. Ensemble Mean
    ens_mean = sfs_da.mean(dim=member_dim, skipna=True)

    # 2. Detrending
    if detrend:
        # Calculate trend ONLY on ensemble mean
        poly_coeffs = ens_mean.polyfit(dim=year_dim, deg=1)
        trend = xr.polyval(ens_mean[year_dim], poly_coeffs.polyfit_coefficients)

        # Subtract the EXACT SAME trend from both ensemble mean and all members
        ens_mean_proc = ens_mean - trend
        sfs_da_proc = sfs_da - trend
    else:
        ens_mean_proc = ens_mean
        sfs_da_proc = sfs_da

    # 3. Variances
    total_ens_var = ens_mean_proc.var(dim=year_dim, ddof=1, skipna=True)
    internal_var = sfs_da_proc.var(dim=member_dim, ddof=1, skipna=True)
    noise_var = internal_var.mean(dim=year_dim, skipna=True)

    # 4. Unbiased Signal Variance (1/M noise correction)
    #unbiased_sig_var = (total_ens_var - (noise_var / M)).clip(min=0.0)
    unbiased_sig_var = (total_ens_var - (noise_var / M))
    #raw_sig_var = total_ens_var - (noise_var / M)
    #pct_clipped = (raw_sig_var < 0).mean().values * 100
    #print(f"Percentage of spatial grid points clipped to 0: {pct_clipped:.1f}%")

    # 5. Lightweight Masking (Standard Xarray .where on noise_var threshold)
    snr_var = unbiased_sig_var / noise_var.where(noise_var > 0)
    snr_var.name = "SNR_var"
    snr_var.attrs["long_name"] = "Signal-to-Noise Variance Ratio"

    logger.info("✅ SNR computation complete!")
    return snr_var


def compute_potential_skill(snr_da: xr.DataArray) -> xr.DataArray:
    """
    Compute theoretical potential correlation skill (r_pot) from Signal-to-Noise Ratio (SNR).

    Formula:
    r_pot = sqrt(SNR / (1 + SNR))
    """
    # Ensure non-negative SNR values for square root stability
    snr_clean = xr.where(snr_da < 0, 0.0, snr_da).fillna(0.0)
    rpot = np.sqrt(snr_clean / (1.0 + snr_clean))
    rpot.name = "r_pot"
    rpot.attrs["long_name"] = "Potential Correlation Skill"

    # Mask out original NaN locations (e.g. land points or missing data)
    return rpot.where(snr_da.notnull())

def compute_rpc(acc_da: xr.DataArray, rpot_da: xr.DataArray) -> xr.DataArray:
    """
    Compute Ratio of Predictable Components (RPC):
    
    RPC = ACC / r_pot
    """
    # Guard against division by zero or extremely low potential skill
    rpot_guarded = rpot_da.where(rpot_da > 0.05)
    rpc = acc_da / rpot_guarded
    #rpc = rpc.where(acc_da >= 0.3) # mask out places where the RPC is irrelevant (ACC insignificant)
    # Trying out masking areas that have low ACC in both model and real-world
    rpc = rpc.where((acc_da >= 0.3)|(rpot_guarded >=0.3)) # mask out places where the RPC is irrelevant (ACC insignificant)

    
    rpc.name = "rpc"
    rpc.attrs["long_name"] = "Ratio of Predictable Components (RPC)"
    return rpc.where(acc_da.notnull() & rpot_da.notnull())

def diagnose_snr_components(
    sfs_da: xr.DataArray,
    obs_da: xr.DataArray,
    time_dim: str = "year",
    member_dim: str = "member"
):
    """
    Diagnoses whether high SNR stems from inflated signal or undersized noise.
    """
    # 1. Model Signal & Noise Standard Deviations
    ens_mean = sfs_da.mean(dim=member_dim, skipna=True)
    std_signal_mod = ens_mean.std(dim=time_dim, ddof=1, skipna=True)

    internal_var = (sfs_da - ens_mean).var(dim=member_dim, ddof=1, skipna=True)
    std_noise_mod = np.sqrt(internal_var.mean(dim=time_dim, skipna=True))

    # 2. Observed Total Standard Deviation
    std_obs = obs_da.std(dim=time_dim, ddof=1, skipna=True)

    # 3. Forecast RMSE
    rmse = np.sqrt(((ens_mean - obs_da) ** 2).mean(dim=time_dim, skipna=True))

    # Ratios
    signal_ratio = (std_signal_mod / std_obs).median().item()
    ser = (std_noise_mod / rmse).median().item()

    print(f"--- Diagnostic Summary ---")
    print(f"Signal Ratio  (Model Signal / Obs Total Std): {signal_ratio:.2f}  [> 1.0 => Signal inflated]")
    print(f"Spread/Error  (Model Noise / Forecast RMSE): {ser:.2f}  [< 1.0 => Noise too small]")
