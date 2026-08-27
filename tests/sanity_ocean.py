"""
Sanity check script for Ocean SNR / dt20c member spread.
"""
import logging
import numpy as np
import xarray as xr

from sources.sfs import get_sfs_data
from preprocess.temporal import select_target_leads

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading SFS ocean dataset for sanity check...")

    var_name = "dt20c"

    # 1. Load SFS ocean dataset for Init 05 (May)
    ds_sfs = get_sfs_data(
        init_month="05",
        domain="ocn",
        requested_vars=[var_name],
    )
    da_sfs = ds_sfs[var_name]

    # 2. Select JJA leads (Leads 1, 2, 3 for May init) and average seasonally
    logger.info("Selecting target leads [1, 2, 3] and averaging seasonal target...")
    sfs_season_da = select_target_leads(da_sfs, leads=[1, 2, 3], average_seasonal=True)

    # 3. Detect Time/Year Dimension
    time_dim = None
    for candidate in ["year", "init", "time"]:
        if candidate in sfs_season_da.dims:
            time_dim = candidate
            break

    if time_dim is None:
        raise ValueError(f"Could not find time dimension in {sfs_season_da.dims}")

    logger.info(f"Detected time dimension: '{time_dim}' (shape: {sfs_season_da.sizes[time_dim]})")

    # Rename to 'year' for metric consistency if named 'init' or 'time'
    if time_dim != "year":
        sfs_season_da = sfs_season_da.rename({time_dim: "year"})
        time_dim = "year"

    # Handle datetime64 vs integer indexing for 1991-2020 filter
    year_vals = sfs_season_da[time_dim].values
    if np.issubdtype(year_vals.dtype, np.datetime64):
        # Extract integer year from datetime objects
        int_years = sfs_season_da[time_dim].dt.year.values
        mask = (int_years >= 1991) & (int_years <= 2020)
        sfs_season_da = sfs_season_da.isel({time_dim: mask})
    elif np.issubdtype(year_vals.dtype, np.integer):
        eval_years = [y for y in range(1991, 2021) if y in year_vals]
        sfs_season_da = sfs_season_da.sel({time_dim: eval_years})

    logger.info(f"Active evaluation window: {sfs_season_da.sizes[time_dim]} years")

    # Force eager load of subset into memory
    sfs_season_da = sfs_season_da.load()

    # 4. Select Tropical Pacific Sample Point (0N, 140W / 220E)
    max_lon = float(sfs_season_da.lon.max())
    target_lon = 220.0 if max_lon > 180 else -140.0
    sample_pt = sfs_season_da.sel(lat=0, lon=target_lon, method="nearest")

    print("\n" + "=" * 65)
    print("1. CHECKING ENSEMBLE MEMBER DIVERSITY (First Evaluation Year)")
    print("=" * 65)
    sample_yr = sample_pt.isel(year=0)
    member_vals = sample_yr.values

    # Print out formatted year value
    year_label = str(sample_yr.year.values)
    if len(year_label) >= 4:
        year_label = year_label[:4]

    print(f"Variable: {var_name}")
    print(f"Year: {year_label}")
    print(f"Point: Lat {float(sample_pt.lat):.2f}, Lon {float(sample_pt.lon):.2f}")
    print("Ensemble Members (m):")
    print(member_vals)
    print(f"Member Spread (Std Dev across members): {np.std(member_vals, ddof=1):.6f} m")

    print("\n" + "=" * 65)
    print("2. CHECKING VARIANCE MAGNITUDES ACROSS ALL YEARS")
    print("=" * 65)

    ens_mean = sample_pt.mean(dim="member")
    signal_var = float(ens_mean.var(dim="year", ddof=1))

    internal_var = sample_pt.var(dim="member", ddof=1)
    noise_var = float(internal_var.mean(dim="year"))

    print(f"Interannual Signal Variance  (Var of Ens Mean) : {signal_var:.6f} m²")
    print(f"Internal Member Noise Variance (Mean Member Var): {noise_var:.6f} m²")

    M = sample_pt.sizes["member"]
    unbiased_sig_var = max(0.0, signal_var - (noise_var / M))

    if noise_var > 0:
        snr_var = unbiased_sig_var / noise_var
        snr_std = np.sqrt(snr_var)
        print(f"Unbiased Signal Variance (1/M corrected)       : {unbiased_sig_var:.6f} m²")
        print(f"Point SNR (Variance Ratio)                      : {snr_var:.2f}")
        print(f"Point SNR (Std Dev Ratio)                       : {snr_std:.2f}")
    else:
        print("❌ WARNING: Noise Variance is ZERO (All ensemble members are identical!)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
