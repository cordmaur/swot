"""Methods to download/process OPERA masks."""

from datetime import datetime
from functools import cache
from typing import TypeAlias, cast

import earthaccess
import numpy as np
import pandas as pd
import rioxarray as xrio
import xarray as xr
from pandas import Timestamp
from shapely.geometry.base import BaseGeometry
from tqdm.auto import tqdm

DateLike: TypeAlias = datetime | Timestamp | str

LAYER_CLASSES = {
    0: "Land",
    1: "Water",
    2: "Partial water",
    252: "Snow/Ice",
    253: "Cloud/Shadow",
    254: "Ocean",
    255: "No data",
}


def search_opera(
    aoi: BaseGeometry,
    date_range: tuple[DateLike, DateLike],
) -> list[earthaccess.DataGranule]:
    """Search for OPERA masks in the given AOI and date range."""
    return earthaccess.search_data(
        short_name="OPERA_L3_DSWX-HLS_V1",
        temporal=(date_range[0], date_range[1]),
        bounding_box=aoi.bounds,
    )


def opera_results_to_df(
    opera_items: list[earthaccess.DataGranule],
) -> pd.DataFrame:
    """Convert the OPERA results to a comprehensive DataFrame."""
    data = {}

    # Loop through each granule from OPERA and create a dictionary with the relevant metadata.
    for item in opera_items:
        # get the id from the metadata
        _id = cast("str", item["meta"]["native-id"])
        parts = _id.split("_")
        data[_id] = {
            "tile": parts[3],
            "date_str": parts[4],
            "satellite": parts[-3],
            "item": item,
        }

    # Convert the dictionary to a DataFrame
    opera_df = pd.DataFrame(data).T

    # Convert the date_str to a datetime object and localize it to None
    opera_df["datetime"] = pd.to_datetime(opera_df["date_str"])
    opera_df["datetime"] = pd.to_datetime(opera_df["datetime"]).dt.tz_localize(None)
    opera_df["date"] = opera_df["datetime"].dt.date

    return opera_df.set_index("date")


def fill_df_with_stats(
    opera_df: pd.DataFrame,
    aoi: BaseGeometry,
    use_bounds: bool = True,  # noqa: FBT001, FBT002
    crs: str = "EPSG:4326",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Fill the DataFrame with statistics based on the AOI and OPERA data.

    It will download the OPERA masks and calculate the statistics for each item in the DataFrame.

    Args:
        opera_df (pd.DataFrame): _description_
        aoi (BaseGeometry): _description_
        use_bounds (bool, optional): _description_. Defaults to True.
        crs (str, optional): _description_. Defaults to "EPSG:4326".
        columns (list[str] | None, optional): List of columns to include in the output DataFrame.
            If None, all columns will be included. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with statistics columns added.

    """
    # Register the progress bar with pandas
    tqdm.pandas(desc="Calculating stats for OPERA")

    stats_df = cast(
        "pd.DataFrame",
        opera_df["item"].progress_apply(
            lambda item: pd.Series(  # type: ignore[arg-type]
                calc_aoi_stats(
                    aoi,
                    item,  # type: ignore[arg-type]
                    use_bounds=use_bounds,
                    crs=crs,
                ),
            ),
        ),
    )

    # If there are columns informed, use them in the output
    if columns is not None:
        stats_df = stats_df[columns]

    return pd.concat([opera_df, stats_df], axis=1)


@cache
def calc_aoi_stats(
    aoi: BaseGeometry,
    opera_item: earthaccess.DataGranule,
    use_bounds: bool = True,  # noqa: FBT001, FBT002
    crs: str = "EPSG:4326",
) -> dict[str, float]:
    """Calculate statistics for the AOI based on the OPERA masks.

    Parameters
    ----------
    aoi :
        The area of interest (AOI) as a shapely geometry object.
    opera_item : earthaccess.DataGranule
        The OPERA item to use for statistics.
    use_bounds : bool
        Whether to use bounding box or actual AOI for clip.
    crs : str
        The coordinate reference system of the AOI used for clipping.

    Returns
    -------
    dict | None
        A dictionary containing statistics about the AOI and the OPERA item, or None if no
        relevant item is found.

    """
    # First, let's get the URL for the files corresponding to this item
    # Use the EarthAccess API to get the files for the given item
    # This will return a list of EarthAccessFile objects
    # We cast it to a list of EarthAccessFile for type hinting
    # and to avoid issues with type checking
    url_files = cast(
        "list[earthaccess.store.EarthAccessFile]",
        earthaccess.open([opera_item], pqdm_kwargs={"disable": True}),
    )

    # To asses the statistics, we will use the first file (WTF) - Band B01
    wtr = [file for file in url_files if "B01" in file.info()["name"]]

    if len(wtr) == 0:
        msg = f"No WTR file found for {opera_item['meta']['native-id']}"
        raise ValueError(msg)

    array = cast("xr.DataArray", xrio.open_rasterio(wtr[0], masked=False))

    # Clip the array to the AOI
    if use_bounds:
        array = array.rio.clip_box(*aoi.bounds, crs=crs)
    else:
        array = array.rio.clip([aoi], crs=crs, drop=True)

    # Calculate the stats for the AOI
    stats = {
        label: np.where(array.to_numpy() == class_value, 1, 0).sum()
        for class_value, label in LAYER_CLASSES.items()
    }
    stats["valid"] = stats["Land"] + stats["Water"] + stats["Partial water"]
    stats["size"] = array.size

    # Now calculate the percentage of each class
    stats_perc = {
        "perc_" + label: round(value / stats["size"] * 100, 2) for label, value in stats.items()
    }
    stats.update(stats_perc)

    return stats
