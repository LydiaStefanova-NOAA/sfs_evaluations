"""
Temporal selection, lead time mapping, and seasonal aggregation utilities.
"""
import logging
import xarray as xr
import numpy as np

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

def seasonal_mean_by_target_months(
    da: xr.DataArray,
    target_months: list[int],
    time_dim: str = "time",
    require_complete: bool = True,
) -> xr.DataArray:
    """
    Compute seasonal means for arbitrary target months with correct season-year handling.

    - For cross-year seasons (e.g., [12,1,2]), December is assigned to the following year.
    - Returns DataArray with dimension 'year' (plus non-time dims).
    - If require_complete=True, drops years missing one or more target months.
    """
    months = [int(m) for m in target_months]
    if len(months) == 0:
        raise ValueError("target_months must not be empty.")
    if any((m < 1 or m > 12) for m in months):
        raise ValueError(f"Invalid month(s) in target_months: {target_months}")

    months_unique = sorted(set(months))
    base = da.where(da[time_dim].dt.month.isin(months_unique), drop=True)

    # Cross-year if December plus any earlier month
    crosses_year = (12 in months_unique) and any(m < 12 for m in months_unique)

    season_year = base[time_dim].dt.year.astype(int)
    if crosses_year:
        season_year = season_year + (base[time_dim].dt.month == 12)

    season_year = season_year.rename("year")

    seasonal = base.groupby(season_year).mean(dim=time_dim, skipna=True)

    if require_complete:
        # Count unique contributing months per year; keep only full seasons
        month_da = xr.DataArray(
            base[time_dim].dt.month.values,
            coords={time_dim: base[time_dim]},
            dims=[time_dim],
        )
        month_counts = month_da.groupby(season_year).map(
            lambda x: xr.DataArray(len(np.unique(x.values)))
        )
        valid_years = month_counts["year"].where(month_counts >= len(months_unique), drop=True)
        seasonal = seasonal.sel(year=valid_years.values)

    return seasonal
