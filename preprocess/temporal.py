"""
Temporal selection, lead time mapping, and seasonal aggregation utilities.
"""
import logging
import xarray as xr

logger = logging.getLogger(__name__)

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEFAULT_AVAILABLE_INITS = [3, 4, 5, 6, 7, 8, 11]


def resolve_target_season_leads(
    target_months: list[int] | int,
    available_inits: list[int] = DEFAULT_AVAILABLE_INITS,
    max_lead: int = 11,
) -> list[tuple[str, list[int], str]]:
    """
    Calculate valid (init_month_str, lead_indices, formatted_label) tuples for
    user-specified target calendar month(s).

    Parameters
    ----------
    target_months : list[int] or int
        Target calendar month(s) (1..12), e.g., [6, 7, 8] for JJA or 7 for July.
    available_inits : list[int]
        List of available initialization months in the dataset.
    max_lead : int
        Maximum allowed lead time index (default: 11).

    Returns
    -------
    list[tuple[str, list[int], str]]
        List of (init_str, leads_list, label) tuples sorted by lead time ascending.
    """
    if isinstance(target_months, int):
        target_months = [target_months]

    resolved = []
    for init in available_inits:
        # Calculate lead for the first month in the target sequence
        lead_0 = (target_months[0] - init) % 12
        
        # Build contiguous lead indices for full target window
        leads = [lead_0 + i for i in range(len(target_months))]

        # Keep initialization only if full target window fits within max_lead
        if max(leads) <= max_lead:
            init_str = f"{init:02d}"
            init_name = MONTH_ABBR[init - 1]
            lead_str = f"Lead {leads[0]}" if len(leads) == 1 else f"Leads {leads[0]}-{leads[-1]}"
            label = f"Init {init_str} ({init_name}) | {lead_str}"
            resolved.append((init_str, leads, label))

    # Sort ascending by initial lead time
    resolved.sort(key=lambda x: x[1][0])
    return resolved


def get_season_name(target_months: list[int] | int) -> str:
    """Generate clean seasonal label (e.g., [6, 7, 8] -> 'JJA', [12, 1, 2] -> 'DJF')."""
    if isinstance(target_months, int):
        return MONTH_ABBR[target_months - 1]

    season_map = {
        (12, 1, 2): "DJF", (3, 4, 5): "MAM",
        (6, 7, 8): "JJA", (9, 10, 11): "SON"
    }
    return season_map.get(tuple(target_months), "".join([MONTH_ABBR[m - 1][0] for m in target_months]))


def select_target_leads(
    ds: xr.Dataset,
    leads: list[int] | int = 0,
    average_seasonal: bool = False,
) -> xr.Dataset:
    """
    Select specific forecast lead(s) and optionally average across them.
    """
    if "lead" not in ds.dims:
        raise ValueError("Dimension 'lead' not found in dataset.")

    if isinstance(leads, int):
        leads = [leads]

    ds_subset = ds.sel(lead=leads)

    if average_seasonal and len(leads) > 1:
        logger.info(f"Averaging across target leads {leads} (Seasonal Target)...")
        ds_subset = ds_subset.mean(dim="lead", keep_attrs=True)

    return ds_subset
