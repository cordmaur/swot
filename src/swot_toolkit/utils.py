"""Utility functions for the SWOT toolkit."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pandas import Timestamp


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
