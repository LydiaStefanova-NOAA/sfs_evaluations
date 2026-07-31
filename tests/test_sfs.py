"""
Sanity check script to test sources/sfs.py
"""
from sources.sfs import get_sfs_data

def main():
    print("Testing SFS functional ingestion...")

    # Load month 05 atmospheric dataset
    ds_atm = get_sfs_data(init_month="05", domain="atm", requested_vars=["z500", "t2m"])

    print("\n✅ Streamed SFS Metadata successfully!")
    print(f"Dimensions: {dict(ds_atm.sizes)}")
    print(f"Coordinates: {list(ds_atm.coords.keys())}")
    print(f"Variables loaded: {list(ds_atm.data_vars.keys())}")

    # Inspect spatial coordinate shapes
    if "lat" in ds_atm.coords and "lon" in ds_atm.coords:
        print(f"Latitude shape: {ds_atm.lat.shape}")
        print(f"Longitude shape: {ds_atm.lon.shape}")

if __name__ == "__main__":
    main()
