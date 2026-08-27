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
    return (da - trend).assign_attrs(da.attrs)


def compute_model_variances(
    sfs_da: xr.DataArray,
    year_dim: str = "year",
    member_dim: str = "member",
    detrend: bool = True,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes unbiased signal variance and noise variance for an ensemble field.
    Returns: (unbiased_sig_var, noise_var)
    """
    M = sfs_da.sizes[member_dim]
    ens_mean = sfs_da.mean(dim=member_dim, skipna=True)

    if detrend:
        ens_mean_proc = _linear_detrend(ens_mean, dim=year_dim)
        sfs_da_proc = sfs_da - (ens_mean - ens_mean_proc)
        ddof = 2
    else:
        ens_mean_proc = ens_mean
        sfs_da_proc = sfs_da
        ddof = 1

    total_ens_var = ens_mean_proc.var(dim=year_dim, ddof=ddof, skipna=True)
    internal_var = sfs_da_proc.var(dim=member_dim, ddof=1, skipna=True)
    noise_var = internal_var.mean(dim=year_dim, skipna=True)

    # Unbiased Signal Variance (subtracting 1/M finite-ensemble noise contamination)
    unbiased_sig_var = (total_ens_var - (noise_var / M)).clip(min=0)

    return unbiased_sig_var, noise_var


def compute_snr(
    sfs_da: xr.DataArray,
    year_dim: str = "year",
    member_dim: str = "member",
    detrend: bool = True,
) -> xr.DataArray:
    """Computes Unbiased Signal-to-Noise Ratio using compute_model_variances."""
    unbiased_sig_var, noise_var = compute_model_variances(
        sfs_da, year_dim=year_dim, member_dim=member_dim, detrend=detrend
    )

    snr_var = unbiased_sig_var / noise_var.where(noise_var > 0)
    snr_var.name = "SNR_var"
    snr_var.attrs["long_name"] = "Signal-to-Noise Variance Ratio"
    return snr_var


def compute_potential_skill(snr_da: xr.DataArray) -> xr.DataArray:
    """
    Compute theoretical potential correlation skill (r_pot) from Signal-to-Noise Ratio (SNR).
    Formula: r_pot = sqrt(SNR_var / (1 + SNR_var))
    """
    snr_clean = xr.where(snr_da < 0, 0.0, snr_da).fillna(0.0)
    rpot = np.sqrt(snr_clean / (1.0 + snr_clean))
    rpot.name = "r_pot"
    rpot.attrs["long_name"] = "Potential Correlation Skill"
    return rpot.where(snr_da.notnull())


def compute_rpc(acc_da: xr.DataArray, rpot_da: xr.DataArray) -> xr.DataArray:
    """Compute Ratio of Predictable Components (RPC = ACC / r_pot)."""
    rpot_guarded = rpot_da.where(rpot_da > 0.05)
    rpc = acc_da / rpot_guarded
    rpc.name = "rpc"
    rpc.attrs["long_name"] = "Ratio of Predictable Components (RPC)"
    return rpc.where(acc_da.notnull() & rpot_da.notnull())


def diagnose_snr_components(
    sfs_da: xr.DataArray,
    obs_da: xr.DataArray,
    acc_da: xr.DataArray = None,
    time_dim: str = "year",
    member_dim: str = "member",
    detrend: bool = True,
):
    """Diagnoses whether overconfidence/high SNR stems from inflated signal or undersized noise."""
    # 1. Unbiased Model Signal & Noise Standard Deviations
    unbiased_sig_var, noise_var = compute_model_variances(
        sfs_da, year_dim=time_dim, member_dim=member_dim, detrend=detrend
    )
    std_signal_mod = np.sqrt(unbiased_sig_var)
    std_noise_mod = np.sqrt(noise_var)

    # 2. Observed Variances & Detrending
    if detrend:
        obs_proc = _linear_detrend(obs_da, dim=time_dim)
        ens_mean = sfs_da.mean(dim=member_dim, skipna=True)
        ens_mean_proc = _linear_detrend(ens_mean, dim=time_dim)
        ddof = 2
    else:
        obs_proc = obs_da
        ens_mean_proc = sfs_da.mean(dim=member_dim, skipna=True)
        ddof = 1

    std_obs = obs_proc.std(dim=time_dim, ddof=ddof, skipna=True)
    rmse = np.sqrt(((ens_mean_proc - obs_proc) ** 2).mean(dim=time_dim, skipna=True))

    # Diagnostics
    signal_total_ratio = (std_signal_mod / std_obs).median().item()
    ser = (std_noise_mod / rmse).median().item()

    print("--- Diagnostic Summary ---")
    print(f"Signal/Total-Obs Ratio : {signal_total_ratio:.2f}  [Model Signal vs Total Obs Std]")
    
    if acc_da is not None:
        std_signal_obs = (acc_da.clip(min=0) * std_obs)
        signal_pred_ratio = (std_signal_mod / std_signal_obs.where(std_signal_obs > 0)).median().item()
        print(f"Signal/Pred-Obs Ratio  : {signal_pred_ratio:.2f}  [> 1.0 => Signal is physically inflated]")

    print(f"Spread/Error Ratio (SER): {ser:.2f}  [< 1.0 => Noise is suppressed / under-dispersive]")
