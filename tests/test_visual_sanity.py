"""
Publication-quality visual sanity check generator for sea ice preprocessing.
Uses Cartopy polar stereographic projections for true geographic verification.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

from sources.sfs import get_sfs_data
from preprocess.pipeline import preprocess_ice_dataset


def generate_sanity_images():
    print("1. Fetching raw SFS sea ice data...")
    ds_raw = get_sfs_data(
        init_month="05",
        domain="ice",
        requested_vars=["aice_h", "uvel_h", "vvel_h"]
    )

    # Slice to a single 2D field
    ds_slice = ds_raw.isel(init=[0], lead=[0], member=[0])

    print("2. Running preprocessing pipeline...")
    ds_processed = preprocess_ice_dataset(ds_slice)

    # Squeeze out length-1 dimensions
    ds_2d = ds_processed.squeeze()

    print("3. Generating Cartopy visual sanity maps...")
    fig = plt.figure(figsize=(16, 14))

    # ---------------------------------------------------------
    # Panel 1: Global Concentration (PlateCarree with Land)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax1.gridlines(draw_labels=True, linestyle=":", color="gray", alpha=0.5)

    ice_conc = ds_2d["aice_h"].values
    
    # Use standard Blues colormap starting above 0 so open ocean is uncolored
    im1 = ax1.pcolormesh(
        ds_2d.lon, ds_2d.lat, ice_conc,
        transform=ccrs.PlateCarree(),
        cmap="Blues", vmin=0.01, vmax=1.0
    )
    ax1.set_title("1. Global Sea Ice Concentration\n[Check: Ice ONLY at Poles]", fontsize=12)
    fig.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.08, shrink=0.7, label="Concentration (0-1)")

    # ---------------------------------------------------------
    # Polar Projection Setup for Panels 2, 3, and 4
    # ---------------------------------------------------------
    polar_proj = ccrs.NorthPolarStereo()

    def setup_arctic_ax(ax, title):
        ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.gridlines(draw_labels=False, linestyle=":", color="gray")
        ax.set_title(title, fontsize=12)

    # Panel 2: Arctic Concentration
    ax2 = fig.add_subplot(2, 2, 2, projection=polar_proj)
    setup_arctic_ax(ax2, "2. Arctic Sea Ice Concentration\n[Polar Stereographic View]")
    im2 = ax2.pcolormesh(
        ds_2d.lon, ds_2d.lat, ice_conc,
        transform=ccrs.PlateCarree(),
        cmap="Blues", vmin=0.01, vmax=1.0
    )
    fig.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.7, label="Concentration")

    # Panel 3: Eastward Velocity (u_east)
    ax3 = fig.add_subplot(2, 2, 3, projection=polar_proj)
    setup_arctic_ax(ax3, "3. Eastward Ice Velocity (u_east)\n[Rotated & Regridded]")
    im3 = ax3.pcolormesh(
        ds_2d.lon, ds_2d.lat, ds_2d["u_east"].values,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=-0.15, vmax=0.15
    )
    fig.colorbar(im3, ax=ax3, orientation="horizontal", pad=0.05, shrink=0.7, label="m/s")

    # Panel 4: Northward Velocity (v_north)
    ax4 = fig.add_subplot(2, 2, 4, projection=polar_proj)
    setup_arctic_ax(ax4, "4. Northward Ice Velocity (v_north)\n[Rotated & Regridded]")
    im4 = ax4.pcolormesh(
        ds_2d.lon, ds_2d.lat, ds_2d["v_north"].values,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=-0.15, vmax=0.15
    )
    fig.colorbar(im4, ax=ax4, orientation="horizontal", pad=0.05, shrink=0.7, label="m/s")

    plt.tight_layout()
    out_file = "pipeline_visual_sanity.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Clean polar diagnostic map saved to '{out_file}'!")

    # Plot raw 2D array subset without map projections
    plt.figure(figsize=(6, 6))
    plt.pcolormesh(ds_2d["u_east"].sel(lat=slice(75, 80), lon=slice(10, 30)).values, cmap="RdBu_r")
    plt.colorbar(label="m/s")
    plt.title("Raw Grid Tiles (No Projection, No Interpolation)")
    plt.savefig("raw_pixel_tiles.png")

    print(f"\n✅ raw grid tile image saved to raw_pixel_tiles.png!")


if __name__ == "__main__":
    generate_sanity_images()

