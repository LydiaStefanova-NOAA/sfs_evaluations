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
    use_std_ratio: bool = False,
) -> xr.DataArray:
    """
    Compute Unbiased Signal-to-Noise Ratio (SNR) across members and reforecast years.

    Parameters
    ----------
    sfs_da : xr.DataArray
        SFS reforecast DataArray with member and year/init dimensions.
    year_dim : str
        Name of interannual time dimension (default: "year").
    member_dim : str
        Name of ensemble member dimension (default: "member").
    detrend : bool
        If True (default), linearly detrend before computing variances.
    use_std_ratio : bool
        If True (default), returns ratio of Std Devs (sqrt(SNR)) for Panel A hatch overlays.
        If False, returns raw variance ratio needed for Panel B RPC calculations.
    """
    logger.info("Computing Signal-to-Noise Ratio (SNR)...")

    # 1. Standardize interannual dimension name ('init' -> 'year')
    if "init" in sfs_da.dims and year_dim not in sfs_da.dims:
        sfs_da = sfs_da.rename({"init": year_dim})

    M = sfs_da.sizes[member_dim]

    # 2. Compute Ensemble Mean
    ens_mean = sfs_da.mean(dim=member_dim, skipna=True)

    # 3. Apply Detrending (Both to ensemble mean and full member array)
    if detrend:
        logger.info(f"Applying linear detrending along '{year_dim}' dimension for SNR...")
        ens_mean_proc = _linear_detrend(ens_mean, dim=year_dim)
        sfs_da_proc = _linear_detrend(sfs_da, dim=year_dim)
    else:
        ens_mean_proc = ens_mean
        sfs_da_proc = sfs_da

    # 4. Total Ensemble Mean Variance
    total_ens_var = ens_mean_proc.var(dim=year_dim, ddof=1, skipna=True)

    # 5. Average Internal Noise Variance across Ensemble Members
    internal_var = sfs_da_proc.var(dim=member_dim, ddof=1, skipna=True)
    noise_var = internal_var.mean(dim=year_dim, skipna=True)

    # 6. Unbiased Signal Variance (Subtract residual noise fraction 1/M)
    unbiased_sig_var = (total_ens_var - (noise_var / M)).clip(min=0.0)

    # 7. Calculate SNR (Variance Ratio)
    snr_var = unbiased_sig_var / noise_var

    # 8. Output as Std Dev ratio (Panel A) or Variance ratio (Panel B / RPC)
    if use_std_ratio:
        snr = np.sqrt(snr_var)
        snr.name = "SNR_std"
    else:
        snr = snr_var
        snr.name = "SNR_var"

    logger.info("✅ SNR computation complete!")
    return snr
