"""Data loading and processing utilities for SWOT datasets."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

import earthaccess
import geopandas as gpd
import pandas as pd
import xarray as xr
from pandas import Timestamp
from shapely.geometry.base import BaseGeometry

from .utils import project_root

if TYPE_CHECKING:
    from pandas import DatetimeIndex


def auth_earthaccess() -> None:
    """Authenticate Earth Access by finding and loading credentials from credentials.json file.

    This function searches for a credentials.json file in the project hierarchy,
    loads the credentials, sets them as environment variables, and authenticates
    with Earth Access using the environment strategy.

    Returns
    -------
    None
        Function performs authentication but does not return a value

    Raises
    ------
    FileNotFoundError
        If credentials.json cannot be found in the project hierarchy
    json.JSONDecodeError
        If credentials.json file is malformed
    Exception
        If Earth Access authentication fails

    Notes
    -----
    The credentials.json file should contain the necessary authentication keys
    for Earth Access. The function will prefer a credentials file in the project
    root if multiple files are found.

    Examples
    --------
    >>> auth_earthaccess()
    # Authenticates with Earth Access using credentials from credentials.json

    """
    # Get project root using the utility function
    root = project_root()

    # Search for credentials.json in the project hierarchy
    credentials_files = list(root.rglob("credentials.json"))

    if not credentials_files:
        msg = "Could not find credentials.json in the project directory"
        raise FileNotFoundError(msg)

    # Prefer credentials file in the root, otherwise take the first one found
    credentials_file = credentials_files[0]
    for file in credentials_files:
        if file.parent == root:
            credentials_file = file

    # Find credentials file path and load credentials
    with credentials_file.open(encoding="utf-8") as f:
        credentials = json.load(f)

    # store the credentials in the environment
    for key in credentials:
        os.environ[key] = credentials[key]

    earthaccess.login(strategy="environment")


def swot_results_to_df(
    search_results: list[earthaccess.DataGranule],
    *,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    """Convert SWOT search results to a structured pandas DataFrame.

    This function takes raw SWOT search results and extracts metadata to create
    a structured DataFrame with cycle, pass, tile information, and timestamps.
    It can optionally remove duplicate entries keeping the latest version.

    Parameters
    ----------
    search_results : list
        List of SWOT search result dictionaries from Earth Access
    drop_duplicates : bool, default False
        Whether to drop duplicate entries, keeping the latest version

    Returns
    -------
    pd.DataFrame
        DataFrame with SWOT metadata including:
        - cycle_id : str, SWOT cycle identifier
        - pass_id : str, SWOT pass identifier
        - tile_id : str, SWOT tile identifier
        - date_str : str, date string from filename
        - tile_name : str, combined pass and tile identifier
        - vers : str, version identifier
        - datetime : pd.Timestamp, parsed datetime
        - date : datetime.date, date only
        Index is set to 'short_id' (first 44 characters of native-id)

    Notes
    -----
    The function parses SWOT native-id strings which follow a specific format
    with underscore-separated components. When drop_duplicates is True,
    the function sorts by datetime and version to ensure the latest version
    is kept for duplicate short_ids.

    Examples
    --------
    >>> results = earthaccess.search_data(...)  # SWOT search results
    >>> df = swot_results_to_df(results, drop_duplicates=True)
    >>> print(df.columns)
    Index(['cycle_id', 'pass_id', 'tile_id', 'date_str', 'tile_name', 'vers',
           'datetime', 'date'], dtype='object')

    """
    # Organize data in a dictionary
    data = {}
    for item in search_results:
        _id = cast("str", item["meta"]["native-id"])

        parts = _id.split("_")

        data[_id] = {
            "cycle_id": parts[4],
            "pass_id": parts[5],
            "tile_id": parts[6],
            "date_str": parts[7],
            "tile_name": parts[5] + "_" + parts[6],
            "short_id": _id[:44],
            "vers": parts[9],
            "item": item,
        }

    swot_df = pd.DataFrame(data).T

    swot_df["datetime"] = pd.to_datetime(swot_df["date_str"].astype("str"))
    swot_df["date"] = swot_df["datetime"].dt.date

    # Set the index to the short ID
    swot_df = swot_df.set_index("short_id")
    swot_df.index.name = "short_id"

    # The sorting by "vers" will guarantee that the drop duplicates, will drop oldest versions
    swot_df = swot_df.sort_values(["datetime", "vers"])

    # Drop the duplicates, considering the ID as reference
    if drop_duplicates:
        swot_df = swot_df[~swot_df.index.duplicated("last")]

    return swot_df


def find_mosaic_items(
    swot_df: pd.DataFrame,
    ref_date: datetime,
    max_delta: int = 11,
) -> pd.DataFrame:
    """Find SWOT data items suitable for creating a mosaic around a reference date.

    This function filters SWOT data based on temporal proximity to a reference date
    and selects the closest item for each tile to create a mosaic. It's designed
    to work with SWOT's 21-day revisit cycle.

    Parameters
    ----------
    swot_df : pd.DataFrame
        DataFrame containing SWOT metadata with 'datetime' and 'tile_name' columns.
        Should be the output from swot_results_to_df function.
    ref_date : datetime
        Reference date around which to find the closest SWOT observations
    max_delta : int, default 11
        Maximum number of days from reference date to consider valid.
        Default is 11 days (approximately half of SWOT's 21-day revisit cycle).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with one row per tile, containing the observation
        closest to the reference date for each tile. Includes an additional
        'delta' column showing the time difference from the reference date.
        Sorted by tile_name and delta.

    Notes
    -----
    The function adds a 'delta' column containing the absolute time difference
    between each observation and the reference date. For each tile_name, only
    the observation with the smallest delta (closest in time) is returned.

    The default max_delta of 11 days is chosen considering SWOT's 21-day
    revisit cycle, allowing roughly half a cycle of tolerance.

    Examples
    --------
    >>> from datetime import datetime
    >>> ref_date = datetime(2024, 1, 15)
    >>> mosaic_items = find_mosaic_items(swot_df, ref_date, max_delta=10)
    >>> print(f"Found {len(mosaic_items)} tiles for mosaic")
    Found 5 tiles for mosaic

    """
    # Calculate the absolute delta to the reference date
    swot_df["delta"] = (swot_df["datetime"] - ref_date).abs()

    # Ignore delta greater than 11 days (considering 21 days of revisit)
    swot_df = swot_df[swot_df["delta"] <= timedelta(days=max_delta)]

    return swot_df.sort_values(["tile_name", "delta"]).groupby("tile_name").first()


def load_swot_pixc(
    file_path: str | Path,
    group: str = "pixel_cloud",
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Load SWOT PIXC data from NetCDF file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the SWOT NetCDF file
    group : str, default "pixel_cloud"
        NetCDF group to read
    variables : List[str], optional
        Specific variables to load. If None, loads all variables.

    Returns
    -------
    xr.Dataset
        Loaded SWOT dataset

    Raises
    ------
    FileNotFoundError
        If the specified file doesn't exist
    ValueError
        If the specified group doesn't exist in the file

    """
    file_path = Path(file_path)
    if not file_path.exists():
        msg = f"SWOT file not found: {file_path}"
        raise FileNotFoundError(msg)

    def _validate_variables(dataset: xr.Dataset, requested_vars: list[str]) -> None:
        """Validate that requested variables exist in the dataset."""
        missing_vars = set(requested_vars) - set(dataset.data_vars)
        if missing_vars:
            msg = f"Variables not found in dataset: {missing_vars}"
            raise ValueError(msg)

    try:
        dataset = xr.open_dataset(file_path, group=group, engine="h5netcdf")

        if variables is not None:
            _validate_variables(dataset, variables)
            dataset = dataset[variables]

    except Exception as e:
        msg = f"Error loading SWOT data from {file_path}: {e}"
        raise ValueError(msg) from e
    else:
        return dataset


def clip_ds_by_aoi(
    ds: xr.DataArray | xr.Dataset,
    aoi: BaseGeometry,
) -> xr.DataArray | xr.Dataset:
    """Clip the dataset to the area of interest (AOI)."""
    lat_mask = (ds["latitude"] >= aoi.bounds[1]) & (ds["latitude"] <= aoi.bounds[3])
    lon_mask = (ds["longitude"] >= aoi.bounds[0]) & (ds["longitude"] <= aoi.bounds[2])

    mask = lat_mask & lon_mask

    return ds.isel(points=mask)


def swot_to_geopandas(
    ds: xr.DataArray | xr.Dataset,
    additional_vars: None | list[str] = None,
    aoi: BaseGeometry | None = None,
) -> gpd.GeoDataFrame:
    """Convert SWOT dataset to a GeoDataFrame.

    Parameters
    ----------
    ds : xr.DataArray | xr.Dataset
        The SWOT dataset (xarray DataArray or Dataset).
    additional_vars : None | list, optional
        List of additional variables to include in the GeoDataFrame.
    aoi : BaseGeometry | None, optional
        Area of interest as a shapely BaseGeometry object.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the SWOT data.

    """
    # Convert the Dataset to a geodataframe
    gdf = gpd.GeoDataFrame(
        data={
            "height": ds.height.to_numpy().astype("float32"),
            "geoid": ds.geoid.to_numpy().astype("float32"),
            "classification": ds.classification.to_numpy().astype("uint8"),
            "coherent_power": ds.coherent_power.to_numpy(),
            "sig0": ds.sig0.to_numpy().astype("float32"),
            "latitude": ds.latitude.to_numpy().astype("float32"),
            "longitude": ds.longitude.to_numpy().astype("float32"),
        },
        geometry=gpd.points_from_xy(ds.longitude.to_numpy(), ds.latitude.to_numpy()),
    )

    if additional_vars:
        for var in additional_vars:
            gdf[var] = ds[var].to_numpy().astype("float32")

    # Clip only pixels within the aoi
    if aoi:
        gdf = gdf[gdf.geometry.within(aoi)]

    # convert the classes from integer to categories
    classes = {
        1: "land",
        2: "land_near_water",
        3: "water_near_land",
        4: "open_water",
        5: "dark_water",
        6: "low_coh_water_near_land",
        7: "open_low_coh_water",
    }

    gdf["classes"] = gdf["classification"].map(classes).astype("category")
    return gdf.set_crs("epsg:4326")


def open_swot_file(
    file: str | Path,
    additional_vars: None | list[str] = None,
    aoi: BaseGeometry | None = None,
) -> gpd.GeoDataFrame:
    """Open SWOT file and clip it if necessary."""
    # open the file
    ds_raw = xr.open_dataset(file, group="pixel_cloud").load()

    # Define the base variables to include in the GeoDataFrame
    base_vars = [
        "height",
        "geoid",
        "classification",
        "coherent_power",
        "sig0",
        "latitude",
        "longitude",
    ]

    # If additional variables are specified, check if they exist in the dataset
    if additional_vars:
        missing_vars = set(additional_vars) - set(ds_raw.data_vars)
        if missing_vars:
            msg = f"Additional variables not found in dataset: {missing_vars}"
            raise ValueError(msg)
        base_vars.extend(additional_vars)

        ds_raw = ds_raw[base_vars]

    # Clip the dataset by bounds
    ds = clip_ds_by_aoi(ds_raw, aoi) if aoi else ds_raw

    # Here, we don't need to pass the AOI, because we already clipped the dataset
    return swot_to_geopandas(ds, additional_vars, aoi)


def create_mosaic_df(
    swot_df: pd.DataFrame,
    max_delta: int = 11,
) -> pd.DataFrame:
    """Create a mosaic DataFrame.

    The mosaic contains the closest observation to a reference date for each tile.

    Parameters
    ----------
    swot_df : pd.DataFrame
        DataFrame with SWOT metadata, output from swot_results_to_df.
    max_delta : int, default 11
        Maximum days from reference date to consider valid.
        Default is 11 days (approximately half of SWOT's 21-day revisit cycle).

    """
    # get a reference tile (it is needed to avoid duplicates)
    tile_counts = swot_df.groupby("tile_name").count()
    tile_counts.sort_values("datetime", ascending=False)

    ref_tile = cast("str", tile_counts.index[0])

    # With the reference tile, we can get the dates
    ref_dates = swot_df[swot_df["tile_name"] == ref_tile]["datetime"].astype("datetime64[ns]")

    # For each date, we will find the closest observation to compose the mosaic
    mosaic_df = pd.DataFrame()

    for date in ref_dates:
        # Find the closest items for mosaic creation
        mosaic_items = find_mosaic_items(swot_df, date, max_delta)

        # THe index of mosaic_items, must be the tile_name (already index) with
        # the mean datetime
        mosaic_items["mosaic_date"] = pd.to_datetime(
            pd.to_datetime(mosaic_items["datetime"]).mean().date(),
        )
        mosaic_items = mosaic_items.reset_index()
        mosaic_items = mosaic_items.set_index(["mosaic_date", "tile_name"])

        mosaic_df = pd.concat([mosaic_df, mosaic_items], axis=0)

    return mosaic_df


def create_mosaic(
    mosaic_df: pd.DataFrame,
    ref_date: str | Timestamp,
    aoi: BaseGeometry,
    additional_vars: None | list[str] = None,
    local_path: str = "/data/swot/downloads",
) -> gpd.GeoDataFrame:
    """Create a mosaic GeoDataFrame from SWOT data around a reference date.

    The mosaic_df already has the closest observations for each tile.

    Parameters
    ----------
    mosaic_df : pd.DataFrame
        DataFrame with SWOT metadata grouped by mosaic dates,
        output from create_mosaic_df.
    ref_date : str
        Reference date to find closest observations for mosaic.
    aoi : BaseGeometry
        Area of interest for clipping the data.
    additional_vars : None | list, optional
        Additional variables to include in the GeoDataFrame.
    local_path : str, default "/data/swot/downloads"
        Local path to save the downloaded files.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with one row per tile, containing the closest observation
        to the reference date for each tile.

    """
    # Find the closest items for mosaic creation
    # Considering the mosaic_df has a 2-level index with mosaic_date and tile_name
    # the ref_date must be in the first level and the result is a dataframe
    mosaic_items = cast("pd.DataFrame", mosaic_df.loc[ref_date])

    # Load and convert each item to GeoDataFrame
    items = cast("list[earthaccess.DataGranule]", mosaic_items["item"].to_list())
    mosaic_files = earthaccess.download(items, local_path=local_path)

    # Open the patches
    patches = [
        open_swot_file(file, aoi=aoi, additional_vars=additional_vars) for file in mosaic_files
    ]

    # Concatenate all patches into one DataFrame
    return cast("gpd.GeoDataFrame", pd.concat(patches, ignore_index=True))


def download_mosaics(
    mosaic_df: pd.DataFrame,
    aoi: BaseGeometry,
    local_path: str = "/data/swot/mosaics",
    additional_vars: None | list[str] = None,
) -> list[Path]:
    """Download mosaic files from SWOT metadata DataFrame.

    Parameters
    ----------
    mosaic_df : pd.DataFrame
        DataFrame with SWOT metadata grouped by mosaic dates,
        output from create_mosaic_df.
    aoi : BaseGeometry
        Area of interest for clipping the data.
    local_path : str, default "/data/swot/mosaics"
        Local path to save the downloaded files.
    additional_vars : None | list, optional
        Additional variables to include in the downloaded files.

    Returns
    -------
    List[str]
        List of local file paths for the downloaded mosaic files.

    """
    file_paths: list[Path] = []

    # Get the dates to create the mosaics
    dates = cast("DatetimeIndex", mosaic_df.index.get_level_values(0).unique())

    for date in dates:
        # Create the file name
        mosaic_date = date.strftime("%Y%m%d")
        filename = f"swot_mosaic_{mosaic_date}.parquet"
        file_path = Path(local_path) / filename

        # Skip if the mosaic already exists
        if file_path.exists():
            continue

        # Get the mosaic as a geodataframe
        mosaic = create_mosaic(
            mosaic_df,
            ref_date=date,
            aoi=aoi,
            additional_vars=additional_vars,
        )

        mosaic.to_parquet(file_path)
        file_paths.append(file_path)

    return file_paths
