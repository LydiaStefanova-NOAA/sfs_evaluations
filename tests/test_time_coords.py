"""
Test script for preprocess/time_coords.py
"""
import numpy as np
import pandas as pd
import xarray as xr
from preprocess.time_coords import add_valid_time, slice_obs_for_forecast
from sources.sfs import get_sfs_data


def test_synthetic_time_alignment():
    print("1. Testing time alignment logic with synthetic datasets...")

    # Create mock forecast dates (May 1st for 3 reforecast years: 2020, 2021, 2022)
    inits = pd.to_datetime(["2020-05-01", "2021-05-01", "2022-05-01"])
    leads = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 0 to 9 lead months

    ds_fcst = xr.Dataset(
        data_vars={"z500": (("init", "lead"), np.random.rand(3, 10))},
        coords={"init": inits, "lead": leads},
    )

    # 1. Test add_valid_time
    ds_fcst = add_valid_time(ds_fcst)
    assert "valid_time" in ds_fcst.coords, "valid_time was not added to dataset coordinates"
    
    # Verify exact math: May 2020 + lead 3 months = August 2020
    expected_sample_date = pd.Timestamp("2020-08-01")
    actual_sample_date = pd.Timestamp(ds_fcst.valid_time.values[0, 3])
    assert actual_sample_date == expected_sample_date, (
        f"Expected {expected_sample_date}, got {actual_sample_date}"
    )

    print("  ✓ 2D valid_time coordinate array generated correctly.")
    print(f"  ✓ Init 2020-05-01 + Lead 3 months -> Valid Time: {actual_sample_date.strftime('%Y-%m-%d')}")

    # 2. Test slice_obs_for_forecast
    # Create mock observation dataset covering 2018 through 2024 monthly
    obs_times = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
    ds_obs = xr.Dataset(
        data_vars={"z500": ("time", np.random.rand(len(obs_times)))},
        coords={"time": obs_times},
    )

    ds_obs_sliced = slice_obs_for_forecast(ds_fcst, ds_obs)

    min_valid = pd.Timestamp(ds_fcst.valid_time.values.min())
    max_valid = pd.Timestamp(ds_fcst.valid_time.values.max())

    sliced_min = pd.Timestamp(ds_obs_sliced.time.values[0])
    sliced_max = pd.Timestamp(ds_obs_sliced.time.values[-1])

    assert sliced_min == min_valid, f"Obs sliced min ({sliced_min}) does not match fcst min ({min_valid})"
    assert sliced_max == max_valid, f"Obs sliced max ({sliced_max}) does not match fcst max ({max_valid})"

    print("  ✓ Observational dataset sliced cleanly to match forecast bounds.")
    print(f"  ✓ Obs sliced window: {sliced_min.strftime('%Y-%m-%d')} to {sliced_max.strftime('%Y-%m-%d')}")
    print("✅ Synthetic test PASSED!\n")


def test_live_sfs_time_alignment():
    print("2. Testing time alignment against live SFS AWS S3 metadata...")
    try:
        # Load month 05 atmospheric data
        ds_sfs = get_sfs_data(init_month="05", domain="atm", requested_vars=["z500"])

        print(f"  - Raw SFS dimensions: {dict(ds_sfs.sizes)}")
        print(f"  - Coordinates found: {list(ds_sfs.coords.keys())}")

        # Add valid time coordinates to real SFS store
        ds_sfs = add_valid_time(ds_sfs)

        print("  ✓ Successfully added 'valid_time' to real SFS store!")
        print(f"  - Overall time range: {ds_sfs.valid_time.values.min()} to {ds_sfs.valid_time.values.max()}")
        print("✅ Live SFS test PASSED!\n")

    except Exception as e:
        print(f"❌ Live SFS time alignment test failed: {e}\n")


if __name__ == "__main__":
    test_synthetic_time_alignment()
    test_live_sfs_time_alignment()
