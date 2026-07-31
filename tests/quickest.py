import xarray as xr
from sources.oras5 import get_oras5_data

# Load 1 year of ORAS5 SST
ds = get_oras5_data(requested_vars=["SST"])
ds_1992 = ds.sel(time="1992")

# Extract 12 months for a single North Atlantic point (45N, 330E)
point_sst = ds_1992["SST"].sel(lat=45.0, lon=330.0, method="nearest")

for t, val in zip(ds_1992["time"].values, point_sst.values):
    month_str = str(t)[:7]
    print(f"{month_str}: {float(val):.2f} °C")
