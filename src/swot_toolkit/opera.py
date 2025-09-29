"""Methods to download/process OPERA masks."""

from datetime import datetime, timedelta
from functools import cache
from typing import TypeAlias, cast

import earthaccess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as xrio
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
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

# Color mapping for visualization
OPERA_COLOR_MAP = {
    0: "#ffffff",  # Not Water (Land)
    1: "#0000ff",  # Open Water
    2: "#00ff00",  # Partial Surface Water
    252: "#00ffff",  # Snow/Ice
    253: "#7f7f7f",  # Clouds/Cloud Shadow
    254: "#0000ff",  # Ocean (same as open water)
    255: "#000000",  # No data
}

OPERA_LABELS = {
    0: "Not Water",
    1: "Open Water",
    2: "Partial Surface Water",
    252: "Snow/Ice",
    253: "Clouds/Cloud Shadow",
    254: "Ocean",
    255: "No data",
}


def open_opera_mask(
    item: earthaccess.DataGranule,
    aoi: BaseGeometry | None = None,
    crs: str | None = "EPSG:4326",
) -> xr.DataArray:
    # First, let's get the URL for the files corresponding to this item
    # Use the EarthAccess API to get the files for the given item
    # This will return a list of EarthAccessFile objects
    # We cast it to a list of EarthAccessFile for type hinting
    # and to avoid issues with type checking
    url_files = cast(
        "list[earthaccess.store.EarthAccessFile]",
        earthaccess.open([item], pqdm_kwargs={"disable": True}),
    )

    # To asses the statistics, we will use the first file (WTF) - Band B01
    wtr = [file for file in url_files if "B01" in file.info()["name"]]

    if len(wtr) == 0:
        msg = f"No WTR file found for {item['meta']['native-id']}"
        raise ValueError(msg)

    array = cast("xr.DataArray", xrio.open_rasterio(wtr[0], masked=False)).squeeze()

    # crs is informed, reproject the array, otherwise use the original crs
    if crs is not None:
        array = array.rio.reproject(crs)
    else:
        crs = array.rio.crs

    # If there is an AOI, clip the array to the AOI
    if aoi is not None:
        # Normally aoi is given in EPSG:4326, so we need to reproject it to the array's crs
        array = array.rio.clip_box(*aoi.bounds, crs="epsg:4326")

    # Set the attributes for the array
    array.attrs["native-id"] = item["meta"]["native-id"]

    return array


def open_opera_mask_from_datetime(
    tile_id: str,
    datetime_str: str,
    aoi: BaseGeometry,
) -> xr.DataArray:
    """Open the OPERA mask for a given datetime string."""
    # Search for the item using the datetime string
    date_str = datetime_str[:8]
    items = earthaccess.search_data(
        short_name="OPERA_L3_DSWX-HLS_V1",
        temporal=(date_str, date_str),
        bounding_box=aoi.bounds,
        granule_name="*" + tile_id + "*" + datetime_str + "*",
    )

    if not items:
        msg = f"No OPERA data found for {datetime_str}"
        raise ValueError(msg)

    if len(items) > 1:
        msg = f"Multiple OPERA items found for {datetime_str}. Please refine your search criteria."
        raise ValueError(
            msg,
        )

    # Open the first item found
    return open_opera_mask(items[0], aoi=aoi, crs=None)


def search_opera(
    aoi: BaseGeometry,
    date_range: tuple[DateLike, DateLike],
    granule_name: str | None = None,
) -> list[earthaccess.DataGranule]:
    """Search for OPERA masks in the given AOI and date range."""
    granule = "" if granule_name is None else granule_name

    return earthaccess.search_data(
        short_name="OPERA_L3_DSWX-HLS_V1",
        temporal=(date_range[0], date_range[1]),
        bounding_box=aoi.bounds,
        granule_name="*" + granule + "*",
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
    # First, open the array
    array = open_opera_mask(opera_item)

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


def plot_opera_array(
    array: xr.DataArray,
    ax: Axes | None = None,
    *,
    down_factor: int = 2,
    add_colorbar: bool = True,
) -> Axes:
    """Plot an OPERA array with the proper color scheme matching hvplot visualization.

    Parameters
    ----------
    array : xr.DataArray
        The OPERA array to plot.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes to plot on. If None, a new figure and axes will be created.
    title : str, optional
        The title of the plot. Defaults to "B01 WTR".
    add_colorbar : bool, optional
        Whether to add a colorbar to the plot. Defaults to True.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes with the plot.

    """
    # If no axes are provided, create a new figure and axes
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    # Get the possible values of OPERA WTR
    values = list(OPERA_COLOR_MAP.keys())

    # Create the boundaries to place the ticks in the colormap correctly
    # PS: If we have 7 colors, we need 8 boundaries
    boundaries = [*values, max(values) + 1]

    # Create a normalization function based on these boundaries
    # This normalization function returns the index of the color based
    # on the boundaries.
    norm = BoundaryNorm(boundaries, len(OPERA_COLOR_MAP))

    # Create the custom colormap
    custom_cmap = ListedColormap(OPERA_COLOR_MAP.values())  # type: ignore[]

    # Downscale the array by a factor
    array = array.squeeze().sel(x=slice(None, None, down_factor), y=slice(None, None, down_factor))

    # Plot the image and get the mappable
    img = array.plot.imshow(
        ax=ax,
        cmap=custom_cmap,
        norm=norm,
        add_colorbar=False,
        interpolation="antialiased",
        interpolation_stage="rgba",
    )  # type: ignore[call-arg]

    # If demanded, take care of the colorbar
    if add_colorbar:
        # Calculate the position of the colorbar ticks
        ticks_positions = [
            (boundaries[idx] + boundaries[idx + 1]) / 2 for idx in range(len(values))
        ]

        # Create the colorbar
        cbar = ax.figure.colorbar(img, ax=ax, ticks=ticks_positions)
        cbar.set_ticks(ticks=ticks_positions, labels=list(OPERA_LABELS.values()))
        cbar.ax.tick_params(size=0)

    ax.set_aspect("equal")

    return ax


######################################
# OPERA-S1 functions
######################################
def search_opera_s1(aoi: BaseGeometry, date: str, delta: int = 20) -> pd.DataFrame | None:
    """Search for OPERA-S1 masks in the given AOI and date range."""
    start_date = (datetime.strptime(date, "%Y%m%d") - timedelta(days=delta)).strftime("%Y-%m-%d")  # noqa: DTZ007
    end_date = (datetime.strptime(date, "%Y%m%d") + timedelta(days=delta)).strftime("%Y-%m-%d")  # noqa: DTZ007
    date_range = (start_date, end_date)

    results = earthaccess.search_data(
        short_name="OPERA_L3_DSWX-S1_V1",
        temporal=(date_range[0], date_range[1]),
        bounding_box=aoi.bounds,
    )

    print(f"Found {len(results)} OPERA-S1 items within {delta} days of {date}.")
    if len(results) == 0:
        return None

    data = {}
    # Loop through each granule from OPERA and create a dictionary with the relevant metadata.
    for item in results:
        # get the id from the metadata
        _id = cast("str", item["meta"]["native-id"])
        parts = _id.split("_")
        data[_id] = {
            "tile": parts[3],
            "date_str": parts[4][:-1],
            "satellite": parts[-3],
            "item": item,
        }

    opera_df = pd.DataFrame(data).T
    opera_df["datetime"] = pd.to_datetime(opera_df["date_str"])
    opera_df["date"] = opera_df["datetime"].dt.date
    opera_df = opera_df.set_index("date")

    ref_date = pd.to_datetime(date)
    opera_df["delta_days"] = (ref_date - opera_df["datetime"].astype("datetime64[ns]")).abs()

    return opera_df.sort_values("delta_days")


def open_opera_s1(aoi: BaseGeometry, date: str) -> xr.DataArray | None:
    """Open the OPERA-S1 mask for a given date."""
    opera_df = search_opera_s1(aoi.centroid.buffer(0.001), date)

    if opera_df is None:
        print(f"No OPERA-S1 data found for {date}.")
        return None

    mask = open_opera_mask(opera_df.iloc[0]["item"], aoi=aoi, crs=None)
    mask = mask.where(mask < 200)

    # Set mask attributes
    mask.attrs["native-id"] = opera_df.iloc[0]["item"]["meta"]["native-id"]
    mask.attrs["date"] = opera_df.index[0]
    mask.attrs["satellite"] = opera_df.iloc[0]["satellite"]
    mask.attrs["tile"] = opera_df.iloc[0]["tile"]
    mask.attrs["date_str"] = opera_df.iloc[0]["date_str"]

    null_values = int(mask.isnull().sum().values)  # noqa: PD003

    # If there are null values, we will load the next closest image
    if null_values == 0:
        return mask

    mask_2 = open_opera_mask(opera_df.iloc[1]["item"], aoi=aoi, crs=None)
    mask_2 = mask_2.where(mask_2 < 200)

    cube = xr.concat([mask, mask_2], dim="img")
    # Change partial water to value 3, this way we will have the following possibilities in the mean
    # 0 - no water
    # 1 - water
    # 3 - partial water
    # mean(0, 1) = 0.5 -> no water
    # mean(0, 3) = 1.5 -> no water
    # mean(1, 3) = 2.0 -> water
    # mean(0, 0) = 0.0 -> no water
    # mean(1, 1) = 1.0 -> water
    # mean(3, 3) = 3.0 -> partial water
    cube.data[cube.data == 2] = 3
    array = cube.mean(dim="img")
    array.data[array.data == 2] = 1
    array.data[array.data == 3] = 2
    array.data[array.data == 0.5] = 0
    array.data[array.data == 1.5] = 0
    array.attrs = mask.attrs.copy()

    return array
