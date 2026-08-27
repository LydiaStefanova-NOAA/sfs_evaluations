# metrics/compute_variance_diagnostic.py

import xarray as xr
from metrics.acc import compute_acc
from metrics.snr import compute_model_variances


def compute_variance_ratios(
    sfs_anom: xr.DataArray,
    obs_anom: xr.DataArray,
    acc_da: xr.DataArray | None = None,
    year_dim: str = "year",
    member_dim: str = "member",
    detrend: bool = True,  # True if data was detrended upstream
    varobs_eps: float = 1e-12,
    mse_eps: float = 1e-12,
    nvr_method: str = "mse",
    acc_eps: float = 1e-12,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes SVR and NVR by reusing core variance helpers from metrics/snr.py.
    """
    ddof = 2 if detrend else 1

    # 1. Reuse SNR helper for unbiased model signal and noise variances
    var_signal_mod, var_noise_mod = compute_model_variances(
        sfs_anom, year_dim=year_dim, member_dim=member_dim, detrend=detrend
    )

    # 2. Observational Variance with proper ddof
    var_obs = obs_anom.var(dim=year_dim, ddof=ddof, skipna=True)

    # 3. Extract Ensemble Mean safely
    ens_mean = (
        sfs_anom.mean(dim=member_dim, skipna=True)
        if member_dim in sfs_anom.dims
        else sfs_anom
    )

    # 4. Compute Ratios based on chosen method
    if nvr_method == "mse":
        svr_da = var_signal_mod / var_obs.where(var_obs > varobs_eps)
        svr_da.name = "svr"
        svr_da.attrs["long_name"] = "Signal Variance Ratio (Var_signal / Var_obs)"

        mse = ((ens_mean - obs_anom) ** 2).mean(dim=year_dim, skipna=True)
        nvr_da = var_noise_mod / mse.where(mse > mse_eps)
        nvr_da.attrs["long_name"] = "Noise Variance Ratio (Var_noise / MSE)"

    elif nvr_method == "acc_varobs":
        if acc_da is None:
            acc_da = compute_acc(ens_mean, obs_anom, detrend=detrend)

        acc2 = (acc_da ** 2).where((acc_da ** 2) > acc_eps)
        denom_svr = (acc2 * var_obs).where((acc2 * var_obs) > varobs_eps)
        svr_da = var_signal_mod / denom_svr
        svr_da.name = "svr"
        svr_da.attrs["long_name"] = "Signal Variance Ratio (Var_signal / (ACC^2 * Var_obs))"

        one_minus_acc2 = (1.0 - (acc_da ** 2)).where((1.0 - (acc_da ** 2)) > acc_eps)
        denom_nvr = (one_minus_acc2 * var_obs).where((one_minus_acc2 * var_obs) > varobs_eps)
        nvr_da = var_noise_mod / denom_nvr
        nvr_da.attrs["long_name"] = "Noise Variance Ratio (Var_noise / ((1-ACC^2)*Var_obs))"

    else:
        raise ValueError("nvr_method must be 'mse' or 'acc_varobs'")

    nvr_da.name = "nvr"
    nvr_da.attrs["nvr_method"] = nvr_method

    return svr_da, nvr_da
