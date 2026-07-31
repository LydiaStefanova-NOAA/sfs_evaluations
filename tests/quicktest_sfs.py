"""
Quick test script to verify SFS dataset ingestion, valid_time calculation,
and lead-to-calendar month mapping.
"""
import numpy as np
from sources.sfs import get_sfs_data
from preprocess.pipeline import preprocess_ocn_dataset


def main():
    INIT_MONTH = "11"  # May initializations

    print(f"Loading SFS dataset for Init Month: {INIT_MONTH}...")
    ds_sfs_raw = get_sfs_data(
        init_month=INIT_MONTH,
        domain="ocn",
        requested_vars=["SST"]
    )

    # Select first initialization year and ensemble member for testing
    ds_sfs_slice = ds_sfs_raw.isel(init=[0], member=[0])
    ds_sfs = preprocess_ocn_dataset(ds_sfs_slice, target_res="1.0deg")

    # Extract initialization date string
    init_val = ds_sfs["init"].values[0]
    init_date = str(init_val)[:10] if isinstance(init_val, (str, np.datetime64)) else str(init_val)

    print(f"\nInitialization Date: {init_date}")
    print("=" * 45)

    for lead_idx in range(ds_sfs.sizes["lead"]):
        slice_lead = ds_sfs.isel(lead=lead_idx)
        lead_val = int(slice_lead["lead"].values)

        # Extract scalar valid_time value
        v_val = slice_lead["valid_time"].values
        if isinstance(v_val, np.ndarray) and v_val.ndim > 0:
            v_val = v_val.item()

        # Format datetime value as YYYY-MM
        try:
            v_str = np.datetime_as_string(v_val, unit="M")
        except (TypeError, ValueError):
            v_str = str(v_val)[:7]

        print(f"  Lead {lead_val:2d}  -->  Valid Target Month: {v_str}")


if __name__ == "__main__":
    main()
