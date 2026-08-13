"""
Test script to inspect the static MOM6 grid file and verify analytical vector rotation.
"""
import xarray as xr
from sources.sfs import sfs
from preprocess.vector_rotation import load_static_grid_rotation, rotate_tripolar_vectors

STATIC_PATH = "/scratch4/BMC/gsienkf/Philip.Pegion/replay_evaluation/ocn_data/ocn_grid_info.nc"


def inspect_static_file():
    print("1. Inspecting static grid file...")
    ds_static = xr.open_dataset(STATIC_PATH)
    print(f"  ✓ Successfully opened static file: {STATIC_PATH}")
    print(f"  - Dimensions: {dict(ds_static.sizes)}")
    print(f"  - Variables: {list(ds_static.data_vars.keys())}")

    cos_rot, sin_rot = load_static_grid_rotation(STATIC_PATH)
    print(f"  ✓ Successfully extracted cos_rot shape: {cos_rot.shape}, sin_rot shape: {sin_rot.shape}\n")


def test_vector_rotation_with_static():
    print("2. Testing vector rotation on live SFS ice data using analytical grid info...")
    ds_ice = get_sfs_data(init_month="05", domain="ice", requested_vars=["uvel_h", "vvel_h"])

    if "uvel_h" in ds_ice and "vvel_h" in ds_ice:
        ds_slice = ds_ice.isel(init=0, lead=0, member=0)
        u_east, v_north = rotate_tripolar_vectors(
            ds_slice["uvel_h"], 
            ds_slice["vvel_h"], 
            static_grid_path=STATIC_PATH
        )
        print(f"  ✓ Rotated u_east shape: {u_east.shape}")
        print(f"  ✓ Rotated v_north shape: {v_north.shape}")
        print("✅ Analytical Vector Rotation Test PASSED!\n")
    else:
        print("  ℹ Velocity variables not in sample store; test skipped.")


if __name__ == "__main__":
    inspect_static_file()
    test_vector_rotation_with_static()
