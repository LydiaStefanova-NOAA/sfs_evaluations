"""
Sanity check script for ORAS5 and ERA5 data ingestion.
"""
from sources.oras5 import get_oras5_data
from sources.era5 import get_era5_data


def test_oras5():
    print("Testing ORAS5 Ingestion...")
    # Request common ocean/ice variables (e.g., Sea Ice Fraction)
    ds = get_oras5_data(version="consolidated", requested_vars=["ileadfra", "sosstsst"])
    print("✅ ORAS5 Streamed Successfully!")
    print(f" - Dimensions: {dict(ds.sizes)}")
    print(f" - Coordinates: {list(ds.coords.keys())}")
    print(f" - Loaded Vars: {list(ds.data_vars.keys())}\n")


def test_era5():
    print("Testing ERA5 Ingestion...")
    # Request geopotential / 2m temperature
    ds = get_era5_data(requested_vars=["z500", "t2m"])
    print("✅ ERA5 Loaded Successfully!")
    print(f" - Dimensions: {dict(ds.sizes)}")
    print(f" - Coordinates: {list(ds.coords.keys())}")
    print(f" - Loaded Vars: {list(ds.data_vars.keys())}\n")


if __name__ == "__main__":
    try:
        test_oras5()
    except Exception as e:
        print(f"❌ ORAS5 test failed: {e}\n")

    try:
        test_era5()
    except Exception as e:
        print(f"❌ ERA5 test failed: {e}\n")
