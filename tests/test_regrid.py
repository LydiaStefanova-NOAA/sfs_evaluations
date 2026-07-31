"""
Sanity check script for preprocess/regrid.py
"""
from sources.sfs import get_sfs_data
from preprocess.regrid import get_target_grid, regrid_dataset


def test_target_grids():
    print("1. Testing target grid creation...")
    ds_ice_target = get_target_grid("ice")
    ds_atm_target = get_target_grid("atm")

    print(f"  ✓ Sea Ice Target Grid (0.25°): {dict(ds_ice_target.sizes)}")
    print(f"  ✓ Atm/Ocn Target Grid (1.0°):  {dict(ds_atm_target.sizes)}")
    assert ds_ice_target.sizes["lat"] == 721
    assert ds_atm_target.sizes["lat"] == 181
    print("✅ Target grids generated correctly!\n")


def test_live_regrid():
    print("2. Testing live regridding on SFS atmospheric data...")
    try:
        # Load month 05 atmospheric dataset
        ds_sfs = get_sfs_data(init_month="05", domain="atm", requested_vars=["z500"])
        
        # Take a single 2D spatial slice (init 0, lead 0) to keep test fast
        ds_slice = ds_sfs.isel(init=0, lead=0)
        print(f"  - Source grid dims: {dict(ds_slice.sizes)}")

        # Regrid to 1.0-degree target grid
        ds_regridded = regrid_dataset(ds_slice, domain="atm", method="bilinear")

        print(f"  - Regridded grid dims: {dict(ds_regridded.sizes)}")
        print(f"  - Output coords: {list(ds_regridded.coords.keys())}")
        
        assert ds_regridded.sizes["lat"] == 181
        assert ds_regridded.sizes["lon"] == 360
        print("✅ Live regridding test PASSED!\n")

    except Exception as e:
        print(f"❌ Live regridding failed: {e}\n")


if __name__ == "__main__":
    test_target_grids()
    test_live_regrid()
