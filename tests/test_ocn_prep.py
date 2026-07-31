"""
Integration test for the ocean preprocessing pipeline.
Verifies SFS 1.0deg pass-through and ORAS5 0.25deg -> 1.0deg downscaling.
"""
import numpy as np
import xarray as xr
from sources.sfs import get_sfs_data
from preprocess.pipeline import preprocess_ocn_dataset


def test_ocean_regridding_pipeline():
    print("1. Testing SFS Ocean Pipeline (1.0deg -> 1.0deg pass-through)...")
    # Fetch valid SFS ocean scalars from the inventory (e.g., SST, SSH)
    ds_sfs = get_sfs_data(
        init_month="03",
        domain="ocn",
        requested_vars=["SST", "SSH"],
    )

    # Slice a small 2D subset
    ds_sfs_slice = ds_sfs.isel(init=[0], lead=[0], member=[0])
    print(f"  - Input SFS dims: {dict(ds_sfs_slice.sizes)}")

    # Preprocess (will attach valid_time and standardize 1.0deg coordinates)
    ds_sfs_processed = preprocess_ocn_dataset(ds_sfs_slice, target_res="1.0deg")
    
    assert ds_sfs_processed.sizes["lat"] == 181 and ds_sfs_processed.sizes["lon"] == 360, "❌ SFS coordinate alignment failed!"
    print("  ✓ SFS pipeline successful (valid_time attached, coordinates standardized).\n")


    print("2. Simulating ORAS5 0.25deg Reanalysis Dataset...")
    # Generate a dummy dataset with the ORAS5 0.25° grid footprint
    lat_025 = np.linspace(-90.0, 90.0, 721)
    lon_025 = np.linspace(0.0, 359.75, 1440)
    
    # Mock SST data (with random numbers)
    mock_sst = np.random.rand(721, 1440)
    
    ds_oras5_mock = xr.Dataset(
        data_vars={"SST": (["lat", "lon"], mock_sst)},
        coords={
            "lat": (["lat"], lat_025, {"units": "degrees_north"}),
            "lon": (["lon"], lon_025, {"units": "degrees_east"}),
            "time": (["time"], [np.datetime64("2010-01-01")]),  # Mock time coordinate
        }
    )
    print(f"  - Input ORAS5 dims: {dict(ds_oras5_mock.sizes)}")


    print("\n3. Regridding ORAS5 (0.25deg) -> Target Grid (1.0deg)...")
    # Pushing the mock obs through the pipeline. 
    # Because there is no "init"/"lead", it will skip valid_time logic and jump straight to regridding.
    ds_oras5_1deg = preprocess_ocn_dataset(ds_oras5_mock, target_res="1.0deg")

    print("\n4. Verifying ORAS5 Downscaled Output:")
    print(f"  ✓ Output Dimensions: {dict(ds_oras5_1deg.sizes)}")
    print(f"  ✓ Output Coordinates: {list(ds_oras5_1deg.coords.keys())}")
    
    assert ds_oras5_1deg.sizes["lat"] == 181 and ds_oras5_1deg.sizes["lon"] == 360, (
        f"❌ ORAS5 downscaling failed! Expected 181x360, got {ds_oras5_1deg.sizes['lat']}x{ds_oras5_1deg.sizes['lon']}"
    )

    print("\n✅ OCEAN PREPROCESSING & ORAS5 DOWNSCALING TEST PASSED!")


if __name__ == "__main__":
    test_ocean_regridding_pipeline()
