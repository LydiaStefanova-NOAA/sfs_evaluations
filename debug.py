"""
Standalone Debugger: Compare Dimensions & Shapes between SSH (working) and SSS (failing)
"""
import sys
import numpy as np
import xarray as xr

from sources.sfs import get_sfs_data
from sources.oras5 import get_oras5_data
from preprocess.pipeline import preprocess_ocn_dataset


def inspect_variable_pipeline(var_name: str):
    print(f"\n{'='*25} INSPECTING: {var_name} {'='*25}")
    
    # --- Step 1: Raw SFS Ingestion ---
    try:
        ds_sfs_raw = get_sfs_data(init_month=5, domain="ocn", requested_vars=[var_name])
        da_sfs_raw = ds_sfs_raw[var_name]
        print(f"1. Raw SFS [{var_name}] Dims      : {da_sfs_raw.dims}")
        print(f"   Raw SFS [{var_name}] Shape     : {da_sfs_raw.shape}")
    except Exception as e:
        print(f"1. Raw SFS [{var_name}] FAILED     : {e}")
        return

    # --- Step 2: Preprocessed SFS ---
    try:
        ds_sfs_proc = preprocess_ocn_dataset(ds_sfs_raw)
        da_sfs_proc = ds_sfs_proc[var_name]
        print(f"2. Preproc SFS [{var_name}] Dims    : {da_sfs_proc.dims}")
        print(f"   Preproc SFS [{var_name}] Shape   : {da_sfs_proc.shape}")
    except Exception as e:
        print(f"2. Preproc SFS [{var_name}] FAILED : {e}")
        return

    # --- Step 3: Raw ORAS5 Ingestion ---
    try:
        ds_obs_raw = get_oras5_data(requested_vars=[var_name])
        da_obs_raw = ds_obs_raw[var_name]
        print(f"3. Raw ORAS5 [{var_name}] Dims    : {da_obs_raw.dims}")
        print(f"   Raw ORAS5 [{var_name}] Shape   : {da_obs_raw.shape}")
    except Exception as e:
        print(f"3. Raw ORAS5 [{var_name}] FAILED   : {e}")
        return

    # --- Step 4: Seasonal Reduction ---
    sfs_season = da_sfs_proc.sel(lead=[1, 2, 3]).mean(dim="lead", skipna=True)
    
    obs_slice = da_obs_raw.sel(time=slice("1991", "2022"))
    obs_jja = obs_slice.where(obs_slice["time.month"].isin([6, 7, 8]), drop=True)
    obs_season = obs_jja.groupby("time.year").mean(dim="time", skipna=True)
    
    print(f"4. Seasonal SFS [{var_name}] Dims   : {sfs_season.dims}")
    print(f"4. Seasonal OBS [{var_name}] Dims   : {obs_season.dims}")

    # --- Step 5: Ensemble Mean & Variance Ratio Calculation ---
    member_dim = "member" if "member" in sfs_season.dims else "number"
    if member_dim not in sfs_season.dims:
        print(f"   ⚠️ WARNING: '{member_dim}' not found in SFS dims! Available: {sfs_season.dims}")
    
    ens_mean = sfs_season.mean(dim=member_dim, skipna=True)
    print(f"5. SFS Ens Mean Dims         : {ens_mean.dims}")

    year_dim_sfs = "year" if "year" in ens_mean.dims else "init"
    var_signal = ens_mean.var(dim=year_dim_sfs, ddof=1, skipna=True)
    var_obs = obs_season.var(dim="year", ddof=1, skipna=True)

    svr = var_signal / var_obs
    print(f"6. Final SVR Output Dims     : {svr.dims}")
    print(f"   Final SVR Output Shape    : {svr.shape}")

    if len(svr.dims) > 2:
        extra_dims = [d for d in svr.dims if d not in ["lat", "latitude", "lon", "longitude"]]
        print(f"\n💥 BINGO! Extra non-spatial dimension(s) detected in SVR: {extra_dims}")
    else:
        print(f"\n✅ Clean 2D array ready for plotting.")


if __name__ == "__main__":
    inspect_variable_pipeline("SSH")  # Control (Working)
    inspect_variable_pipeline("SSS")  # Test (Failing)
