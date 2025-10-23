"""Utility functions for the SWOT toolkit."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
from pyproj import CRS
import xarray as xr
from pandas import Timestamp
from tqdm.auto import tqdm


def find_closest_items(
    ref_time: datetime | Timestamp | str,
    items_df: pd.DataFrame,
    max_days: int = 5,
) -> pd.DataFrame:
    """Find the closest Sentinel-2 images to a reference time."""
    # raise an error if "datetime" is not in the dataframe
    if "datetime" not in items_df.columns:
        msg = "The dataframe must contain a 'datetime' column."
        raise ValueError(msg)

    # get the delta time for all s2 images and filter those within max_days
    items_df["delta"] = items_df["datetime"] - pd.to_datetime(ref_time)
    delta = items_df[items_df["delta"].abs() < timedelta(days=max_days)]

    # now, order by the closest
    delta = delta.iloc[delta["delta"].abs().argsort()]

    # return the available images
    return items_df.loc[delta.index]


def project_root() -> Path:
    """Find the project root directory by looking for common project files.

    Returns
    -------
    Path
        Path to the project root directory

    Raises
    ------
    FileNotFoundError
        If project root cannot be determined

    """
    # Start from the current file's directory and navigate upwards
    current_file = Path(__file__).resolve()
    current_path = current_file.parent

    # Try to find the project root by looking for common project files
    while current_path != current_path.parent:
        if (current_path / "setup.py").exists() or (current_path / "pyproject.toml").exists():
            return current_path
        current_path = current_path.parent

    msg = "Could not determine project root directory"
    raise FileNotFoundError(msg)


def _find_temporal_matches(
    ref_time: datetime | Timestamp | str,
    target_df: pd.DataFrame,
    target_date_col: str,
    max_time_delta: timedelta | int = 10,
) -> pd.DataFrame:
    """Find target observations within the specified time window."""
    # Convert max_time_delta to timedelta if it's an integer (days)
    if isinstance(max_time_delta, int):
        max_time_delta = timedelta(days=max_time_delta)

    # Calculate time differences
    target_df_copy = target_df.copy()
    target_df_copy["delta"] = target_df_copy[target_date_col] - pd.to_datetime(ref_time)

    # Filter observations within time window
    return target_df_copy[target_df_copy["delta"].abs() < max_time_delta]


def _apply_matching_strategy(
    target_matches: pd.DataFrame,
    strategy: str,
    quality_col: str | None,
    *,
    quality_ascending: bool,
) -> pd.DataFrame:
    """Apply the specified matching strategy to select the best match."""
    if strategy == "closest":
        # Sort by absolute time difference
        sorted_matches = target_matches.iloc[target_matches["delta"].abs().argsort()]
        return sorted_matches.head(1)

    if strategy == "best_quality":
        if quality_col is None:
            msg = "quality_col must be specified when using 'best_quality' strategy"
            raise ValueError(msg)
        # Sort by quality metric
        sorted_matches = target_matches.sort_values(quality_col, ascending=quality_ascending)
        return sorted_matches.head(1)

    if strategy == "balanced":
        if quality_col is None:
            msg = "quality_col must be specified when using 'balanced' strategy"
            raise ValueError(msg)

        # Normalize time and quality scores (0-1 scale)
        delta_abs = cast("pd.TimedeltaIndex", target_matches["delta"].abs())
        time_scores = 1 - (delta_abs / delta_abs.max())

        quality_values = cast("pd.Series[float]", target_matches[quality_col])
        if not quality_ascending:
            # If higher quality values are better, normalize as-is
            quality_min, quality_max = quality_values.min(), quality_values.max()
            quality_scores = cast(
                "pd.Series[float]",
                (quality_values - quality_min) / (quality_max - quality_min),
            )
        else:
            # If lower quality values are better, invert the normalization
            quality_min, quality_max = quality_values.min(), quality_values.max()
            quality_scores = cast(
                "pd.Series[float]",
                1 - ((quality_values - quality_min) / (quality_max - quality_min)),
            )

        # Calculate balanced score (equal weighting)
        balanced_scores = 0.5 * time_scores + 0.5 * quality_scores

        # Select best balanced match
        best_idx = cast("pd.Series[int]", balanced_scores.idxmax())  # type: ignore[]
        return target_matches.loc[[best_idx]]  # type: ignore[]

    msg = f"Unknown strategy: {strategy}. Must be 'closest', 'best_quality', or 'balanced'"
    raise ValueError(msg)


def match_datasets_by_time(  # noqa: PLR0913
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    ref_date_col: str = "datetime",
    target_date_col: str = "datetime",
    max_time_delta: timedelta | int = 10,
    quality_col: str | None = None,
    quality_ascending: bool = False,
    strategy: Literal["closest", "best_quality", "balanced"] = "closest",
) -> pd.DataFrame:
    """Match two datasets by finding the closest temporal matches with optional quality filtering.

    This function matches observations from a reference dataset with observations from a target
    dataset based on temporal proximity. Optionally, it can also consider data quality metrics
    when selecting the best matches within the time window.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference dataset containing observations to be matched. Each row represents
        an observation that needs to find a corresponding match in the target dataset.
    target_df : pd.DataFrame
        Target dataset containing potential matches. The function will search this
        dataset to find the best temporal (and optionally quality) matches.
    ref_date_col : str, optional
        Column name containing datetime information in the reference dataset.
        Default is "datetime".
    target_date_col : str, optional
        Column name containing datetime information in the target dataset.
        Default is "datetime".
    max_time_delta : timedelta | int, optional
        Maximum temporal window for matching. If int, interpreted as days.
        Only target observations within this window will be considered.
        Default is 10 days.
    quality_col : str | None, optional
        Column name in target dataset containing quality metrics (e.g., "valid_pxls",
        "cloud_coverage"). If provided, quality will be considered in matching.
        Default is None (no quality filtering).
    quality_ascending : bool, optional
        Sort order for quality metric. False means higher values = better quality
        (e.g., for valid pixel percentage). True means lower values = better quality
        (e.g., for cloud coverage percentage). Default is False.
    strategy : {"closest", "best_quality", "balanced"}, optional
        Matching strategy:
        - "closest": Select temporally closest match within time window
        - "best_quality": Select best quality match within time window
        - "balanced": Weighted combination of temporal and quality factors
        Default is "closest".

    Returns
    -------
    pd.DataFrame
        Joined DataFrame containing reference observations with their matched target
        observations. Target columns are suffixed with "_target". Includes a "delta"
        column showing the time difference between reference and target observations.
        Multi-indexed by original reference index and time delta for easy filtering.

    Raises
    ------
    ValueError
        If required date columns are missing from either DataFrame.

    Notes
    -----
    - Both date columns should contain timezone-naive datetime objects for proper matching
    - For large datasets, this function may be time-intensive as it processes each
      reference observation individually
    - The "balanced" strategy uses equal weighting between normalized time and quality scores

    """
    # Validate required columns
    if ref_date_col not in reference_df.columns:
        msg = f"Reference DataFrame must contain '{ref_date_col}' column."
        raise ValueError(msg)
    if target_date_col not in target_df.columns:
        msg = f"Target DataFrame must contain '{target_date_col}' column."
        raise ValueError(msg)
    if quality_col and quality_col not in target_df.columns:
        msg = (
            f"Target DataFrame must contain '{quality_col}' column when quality_col is specified."
        )
        raise ValueError(msg)

    # Convert max_time_delta to timedelta if it's an integer (days)
    if isinstance(max_time_delta, int):
        max_time_delta = timedelta(days=max_time_delta)

    # Initialize DataFrame to accumulate matched results
    matched_results = pd.DataFrame()

    # Process each reference observation
    for row in tqdm(reference_df.itertuples(), total=len(reference_df), desc="Matching datasets"):
        ref_time = getattr(row, ref_date_col)

        # Find target observations within time window
        target_matches = _find_temporal_matches(
            ref_time=ref_time,
            target_df=target_df,
            target_date_col=target_date_col,
            max_time_delta=max_time_delta,
        )

        if target_matches.empty:
            continue

        # Apply matching strategy
        best_match = _apply_matching_strategy(
            target_matches=target_matches,
            strategy=strategy,
            quality_col=quality_col,
            quality_ascending=quality_ascending,
        )

        # Prepare matched result for joining
        best_match.index.name = "target_id"
        best_match["ref_index"] = row.Index
        best_match = best_match.reset_index()
        best_match = best_match.set_index("ref_index", drop=True)

        # Accumulate results
        matched_results = pd.concat([matched_results, best_match], axis=0)

    # Join reference data with matched target data
    reference_df = reference_df.set_index(ref_date_col, drop=True)
    reference_df.index.name = "index"
    joined = reference_df.join(matched_results, rsuffix="_target")

    # Create multi-level index with original index and time delta
    return joined.reset_index().set_index(["index", "delta_target"])


def create_template_dataarray(
    bounds: tuple[float, ...],
    resolution: float,
    crs: CRS,
    fill_value: float = np.nan,
) -> xr.DataArray:
    """Create a template xarray DataArray with specified bounds, resolution, and CRS.

    Parameters
    ----------
    bounds : tuple
        (left, bottom, right, top) in CRS units
    resolution : float
        Pixel size in CRS units
    crs : str or rasterio.crs.CRS
        Coordinate reference system
    fill_value : numeric
        Initial fill value for the array

    """
    left, bottom, right, top = bounds

    # Calculate dimensions
    width = int((right - left) / resolution)
    height = int((top - bottom) / resolution)

    # Create coordinate arrays
    x = np.linspace(left + resolution / 2, right - resolution / 2, width)
    y = np.linspace(top - resolution / 2, bottom + resolution / 2, height)

    # Create empty data array
    data = np.full((height, width), fill_value, dtype=np.float32)

    # Create xarray DataArray
    da = xr.DataArray(data, coords={"y": y, "x": x}, dims=["y", "x"])

    # Set CRS
    da.rio.write_crs(crs, inplace=True)

    return da
