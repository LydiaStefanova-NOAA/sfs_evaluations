"""
Visual comparison and evaluation test:
  1. Native SFS SST (1.0°)
  2. Native ORAS5 SST (0.25°)
  3. Coarsened ORAS5 SST (1.0°)
  4. SFS minus ORAS5 Difference Field (1.0°)

Annotated with initialization, lead time, valid date, and global bias statistics.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

from sources.sfs import get_sfs_data
from sources.oras5 import get_oras5_data
from preprocess.pipeline import preprocess_ocn_dataset


def format_time_info(da: xr.DataArray) -> str:
    """Extract and format initialization, lead time, and valid date metadata."""
    info_parts = []

    # Check for initialization time
    if "init" in da.coords:
        init_val = da.coords["init"].values
        init_str = str(init_val)[:10] if isinstance(init_val, (str, np.datetime64, np.ndarray)) else str(init_val)
        info_parts.append(f"Init: {init_str}")

    # Check for lead time
    if "lead" in da.coords:
        lead_val = da.coords["lead"].values
        info_parts.append(f"Lead: {lead_val}")

    # Check for valid time (inspecting valid_time, time_counter, or time)
    time_coord_names = ["valid_time", "time_counter", "time"]
    time_key = next((k for k in time_coord_names if k in da.coords), None)

    if time_key:
        t_val = da.coords[time_key].values
        if isinstance(t_val, np.ndarray) and t_val.ndim > 0:
            t_val = t_val.item() if t_val.size == 1 else t_val[0]

        try:
            t_str = np.datetime_as_string(t_val, unit="M")
        except (TypeError, ValueError):
            t_str = str(t_val)[:7]
            
        info_parts.append(f"Valid: {t_str}")

    return " | ".join(info_parts) if info_parts else "Time Info: N/A"


def main():
    print("1. Fetching native 1.0° SFS SST...")
    ds_sfs_raw = get_sfs_data(
        init_month="11",
        domain="ocn",
        requested_vars=["SST"],
    )
    # Slice a single ensemble member, initialization year, and lead month
    ds_sfs_slice = ds_sfs_raw.isel(init=[0], lead=[10], member=[0])
    ds_sfs_processed = preprocess_ocn_dataset(ds_sfs_slice, target_res="1.0deg")
    sfs_sst = ds_sfs_processed["SST"].squeeze()

    print("\n2. Loading native 0.25° ORAS5 SST via sources/oras5.py...")
    ds_oras5_raw = get_oras5_data(requested_vars=["SST"])

    # Extract target valid time string (e.g., '1991-03') from SFS to select matching ORAS5 slice
    if "valid_time" in sfs_sst.coords:
        v_val = sfs_sst.coords["valid_time"].values
        if isinstance(v_val, np.ndarray) and v_val.ndim > 0:
            v_val = v_val.item()
        target_valid_time = np.datetime_as_string(v_val, unit="M")
    else:
        target_valid_time = "1991-03"

    print(f"  - Matching ORAS5 to SFS valid time target: {target_valid_time}")
    if "time" in ds_oras5_raw.coords and ds_oras5_raw.sizes["time"] > 1:
        ds_oras5_raw = ds_oras5_raw.sel(time=target_valid_time, method="nearest")

    oras5_raw_sst = ds_oras5_raw["SST"].squeeze()

    print("\n3. Coarsening ORAS5 SST to model grid (0.25° -> 1.0°)...")
    ds_oras5_regrid = preprocess_ocn_dataset(ds_oras5_raw, target_res="1.0deg")
    oras5_regrid_sst = ds_oras5_regrid["SST"].squeeze()

    print("\n4. Computing Difference Field (SFS - ORAS5)...")
    diff_sst = sfs_sst - oras5_regrid_sst
    
    # Calculate summary bias statistics
    mean_bias = float(diff_sst.mean().values)
    rmse = float(np.sqrt((diff_sst ** 2).mean()).values)

    # Format time metadata strings
    sfs_time_str = format_time_info(sfs_sst)
    oras5_time_str = format_time_info(oras5_raw_sst)

    print("\n5. Generating 2x2 comparison map with difference panel...")
    fig, axes = plt.subplots(
        2, 2, figsize=(18, 11), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    def setup_map_ax(ax, title):
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.gridlines(draw_labels=True, linestyle=":", color="gray", alpha=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    # Panel 1: Native SFS (1.0°)
    setup_map_ax(
        axes[0, 0], 
        f"1. SFS Model SST (Native 1.0°)\n[{sfs_time_str}]"
    )
    im_abs = axes[0, 0].pcolormesh(
        sfs_sst.lon, sfs_sst.lat, sfs_sst.values,
        transform=ccrs.PlateCarree(), cmap="plasma", vmin=-2.0, vmax=32.0
    )

    # Panel 2: ORAS5 Native (0.25°)
    setup_map_ax(
        axes[0, 1], 
        f"2. ORAS5 Reanalysis SST (Fine 0.25°)\n[{oras5_time_str}]"
    )
    axes[0, 1].pcolormesh(
        oras5_raw_sst.lon, oras5_raw_sst.lat, oras5_raw_sst.values,
        transform=ccrs.PlateCarree(), cmap="plasma", vmin=-2.0, vmax=32.0
    )

    # Panel 3: ORAS5 Coarsened (1.0°)
    setup_map_ax(
        axes[1, 0], 
        f"3. ORAS5 Reanalysis SST (Coarsened 1.0°)\n[{oras5_time_str}]"
    )
    axes[1, 0].pcolormesh(
        oras5_regrid_sst.lon, oras5_regrid_sst.lat, oras5_regrid_sst.values,
        transform=ccrs.PlateCarree(), cmap="plasma", vmin=-2.0, vmax=32.0
    )

    # Panel 4: Difference Field (SFS - ORAS5)
    setup_map_ax(
        axes[1, 1], 
        f"4. Difference: SFS minus ORAS5 (1.0° Grid)\n[Mean Bias: {mean_bias:+.2f}°C | RMSE: {rmse:.2f}°C]"
    )
    im_diff = axes[1, 1].pcolormesh(
        diff_sst.lon, diff_sst.lat, diff_sst.values,
        transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-5.0, vmax=5.0
    )

    # Shared colorbar for absolute fields
    cbar_abs_ax = fig.add_axes([0.15, 0.05, 0.32, 0.02])
    fig.colorbar(im_abs, cax=cbar_abs_ax, orientation="horizontal", label="Absolute SST (°C)")

    # Colorbar for difference field
    cbar_diff_ax = fig.add_axes([0.55, 0.05, 0.32, 0.02])
    fig.colorbar(im_diff, cax=cbar_diff_ax, orientation="horizontal", label="SST Difference / Bias (°C)")

    plt.subplots_adjust(wspace=0.15, hspace=0.25, bottom=0.12)

    out_png = "oras5_regrid_comparison.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Plot successfully saved to '{out_png}'!")
    print(f"  - SFS Metadata:      {sfs_time_str}")
    print(f"  - ORAS5 Metadata:    {oras5_time_str}")
    print(f"  - Global Mean Bias:  {mean_bias:+.3f} °C")
    print(f"  - Global RMSE:       {rmse:.3f} °C")


if __name__ == "__main__":
    main()
