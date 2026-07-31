"""
End-to-end integration test for the sea ice preprocessing pipeline.
"""
from sources.sfs import get_sfs_data
from preprocess.pipeline import preprocess_ice_dataset


def test_full_ice_preprocessing_pipeline():
    print("1. Ingesting native SFS Sea Ice dataset...")
    # Request sea ice concentration and velocities
    ds_raw = get_sfs_data(
        init_month="05",
        domain="ice",
        requested_vars=["aice_h", "uvel_h", "vvel_h"],
    )

    # Slice 1 time/member to keep test fast
    ds_slice = ds_raw.isel(init=slice(0, 2), lead=slice(0, 3), member=0)
    print(f"  - Input Raw Dataset dims: {dict(ds_slice.sizes)}")
    print(f"  - Input Data Variables: {list(ds_slice.data_vars.keys())}")

    print("\n2. Executing preprocess_ice_dataset() pipeline...")
    ds_processed = preprocess_ice_dataset(ds_slice)

    print("\n3. Verifying Processed Output:")
    print(f"  ✓ Output Dimensions: {dict(ds_processed.sizes)}")
    print(f"  ✓ Output Coordinates: {list(ds_processed.coords.keys())}")
    print(f"  ✓ Output Data Variables: {list(ds_processed.data_vars.keys())}")

    # Assertions to confirm pipeline transformed data properly
    assert "valid_time" in ds_processed.coords, "❌ Pipeline failed to add valid_time coordinate!"
    assert ds_processed.sizes["lat"] == 721 and ds_processed.sizes["lon"] == 1440, "❌ Spatial grid is not 0.25°!"
    assert "u_east" in ds_processed.data_vars and "v_north" in ds_processed.data_vars, "❌ Vector rotation was not performed!"
    assert "uvel_h" not in ds_processed.data_vars, "❌ Unrotated uvel_h was not dropped!"

    print("\n✅ FULL SEA ICE PREPROCESSING PIPELINE TEST PASSED!")


if __name__ == "__main__":
    test_full_ice_preprocessing_pipeline()
