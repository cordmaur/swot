"""Data loading and processing utilities for SWOT datasets."""

import json
import os
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import earthaccess
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as xrio
import xarray as xr
from pandas import Timestamp
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask  # type: ignore  # noqa: PGH003
from shapely import LineString
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from .flags import mask_by_flags
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


SWOT_DATASET = Literal["Pixel Cloud", "Raster_100"]


def get_granule_url(granule: earthaccess.DataGranule) -> str:
    """Get the download URL for a SWOT data granule.

    Args:
        granule (earthaccess.DataGranule): The SWOT data granule.

    Raises:
        ValueError: If the granule does not contain exactly one GET DATA URL.

    Returns:
        str: The download URL for the granule.

    """
    # Get the urls from the granule and filter the GET DATA one
    urls = list(filter(lambda x: x["Type"] == "GET DATA", granule["umm"]["RelatedUrls"]))  # type: ignore[]
    urls = cast("list[dict[str, str]]", urls)
    if len(urls) != 1:
        msg = f"Expected 1 GET DATA url, found {len(urls)}"
        raise ValueError(msg)

    return urls[0]["URL"]


def search_swot_data(
    aoi: BaseGeometry,
    date_range: tuple[str, str] | None = None,
    *,
    dataset: SWOT_DATASET = "Pixel Cloud",
    footprint_filter: bool = False,
) -> list[earthaccess.DataGranule]:
    """Search for SWOT data granules within a specified area of interest (AOI).

    This function uses Earth Access to search for SWOT data granules that intersect
    with the provided AOI. It returns a list of DataGranule objects containing metadata.
    It can search pixel cloud or raster products.

    Parameters
    ----------
    aoi : BaseGeometry
        Area of interest as a shapely geometry object (e.g., Polygon, MultiPolygon)

    date_range : tuple[str, str], optional
        Temporal range to filter the search results. If not provided, all available data is
        returned.

    dataset : Literal["Pixel Cloud", "Raster_100"]
        The type of SWOT dataset to search for. Options are:
        - "Pixel Cloud": Search for pixel cloud data granules
        - "Raster_100": Search for raster (100m) data granules

    footprint_filter : bool, default True
        If True, applies a footprint filter to the search results to ensure that only granules
        that intersect with the AOI are returned. If False, all granules matching the dataset
        and date range are returned regardless of their intersection with the AOI.

    Returns
    -------
    list[earthaccess.DataGranule]
        List of DataGranule objects containing metadata for each SWOT data granule found

    Notes
    -----
    The function requires Earth Access to be authenticated before calling.
    The AOI should be in a projected coordinate system suitable for SWOT data.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> aoi = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> results = search_swot_data(aoi)
    >>> print(f"Found {len(results)} SWOT data granules")
    Found 10 SWOT data granules

    """
    # Search for SWOT data granules intersecting the AOI
    short_name = "SWOT_L2_HR_Raster_D" if dataset == "Raster_100" else "SWOT_L2_HR_PIXC_D"
    results = earthaccess.search_data(
        short_name=short_name,
        bounding_box=aoi.bounds,
        temporal=date_range,
    )

    # Concatenate the results with a search on the old short name for the data
    short_name = short_name.replace("_D", "_2.0")
    results += earthaccess.search_data(
        short_name=short_name,
        bounding_box=aoi.bounds,
        temporal=date_range,
    )

    # If raster dataset, filter for 100m or 250m resolution, according to the dataset descr.
    if "Raster" in dataset:
        res = dataset[-3:]
        results = list(filter(lambda x: f"{res}m" in x["meta"]["native-id"], results))

    # If footprint_filter, filter results by intersection with AOI
    if footprint_filter:
        filtered = filter(lambda x: get_swot_footprint(x)[0].intersects(aoi), results)
        results = list(filtered)

    return results


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

        # The short Id must have only parts that are longer than 1 character
        short_id_parts = filter(lambda x: len(x) > 1, parts[:-3])

        data[_id] = {
            "cycle_id": parts[-7],
            "pass_id": parts[-6],
            "tile_id": parts[-5],
            "date_str": parts[-4],
            "tile_name": parts[-6] + "_" + parts[-5],
            "short_id": "_".join(short_id_parts),
            "vers": parts[-2] + "_" + parts[-1].split(".")[0],
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

    swot_df["native-id"] = swot_df["item"].apply(lambda x: x["meta"]["native-id"])  # type: ignore[]
    swot_df["url"] = swot_df["item"].apply(get_granule_url)

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


def open_pixc_file(
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


def create_pixc_mosaic(
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
    mosaic_files = earthaccess.download(
        items,
        local_path=local_path,
        pqdm_kwargs={"disable": True},
    )

    # Open the patches
    patches = [
        open_pixc_file(file, aoi=aoi, additional_vars=additional_vars) for file in mosaic_files
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
        mosaic = create_pixc_mosaic(
            mosaic_df,
            ref_date=date,
            aoi=aoi,
            additional_vars=additional_vars,
        )

        mosaic.to_parquet(file_path)
        file_paths.append(file_path)

    return file_paths


def get_swot_bbox(swot_item: earthaccess.DataGranule) -> BaseGeometry:
    """Extract the bounding box of a SWOT item.

    Uses the 'geometry' field from the SWOT item metadata to create a BaseGeometry object.

    Args:
        swot_item (earthaccess.DataGranule): The SWOT item to extract the bounding box from.

    Returns:
        BaseGeometry: The bounding box of the SWOT item.

    """
    geometry = cast(
        "dict[str, list[dict[str, float]]]",
        swot_item["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"],
    )
    bounds = list(geometry["BoundingRectangles"][0].values())

    # Create a bounding box from the coordinates
    return box(*bounds)  # type: ignore[arg-type]


def adjust_footprint_signs(
    footprint_coords: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Adjust longitude coordinates in footprint to handle dateline crossing.

    When footprint coordinates span the dateline (have both negative and positive longitudes),
    this function adjusts negative longitudes by adding 360 degrees to ensure proper
    polygon creation.

    Args:
        footprint_coords: List of (longitude, latitude) coordinate tuples.

    Returns:
        list[tuple[float, float]]: Adjusted coordinate list with consistent longitude signs.

    """
    # Check if footprint_coords has coordinates with different signs
    if any(coord[0] < 0 for coord in footprint_coords) and any(
        coord[0] >= 0 for coord in footprint_coords
    ):
        return [(lon + 360, lat) if lon < 0 else (lon, lat) for lon, lat in footprint_coords]
    return footprint_coords


def get_pixc_footprint(ds: xr.Dataset) -> BaseGeometry:
    """Extract the footprint of a PIXC dataset.

    Args:
        ds (xr.Dataset): The PIXC dataset to extract the footprint from.

    Returns:
        BaseGeometry: The footprint geometry of the PIXC dataset.

    """
    footprint_coords = [
        (ds.attrs["outer_first_longitude"], ds.attrs["outer_first_latitude"]),  # outer first
        (ds.attrs["inner_first_longitude"], ds.attrs["inner_first_latitude"]),  # inner first
        (ds.attrs["inner_last_longitude"], ds.attrs["inner_last_latitude"]),  # inner last
        (ds.attrs["outer_last_longitude"], ds.attrs["outer_last_latitude"]),  # outer last
        (ds.attrs["outer_first_longitude"], ds.attrs["outer_first_latitude"]),  # close polygon
    ]

    # Adjust coordinates for dateline crossing
    footprint_coords = adjust_footprint_signs(footprint_coords)

    return Polygon(footprint_coords)


def get_raster_footprint(ds: xr.Dataset, crs: CRS | None = None) -> Polygon:
    """Extract the footprint of a raster dataset.

    Args:
        ds (xr.Dataset): The raster dataset to extract the footprint from.
        crs (str): The coordinate reference system of the dataset. Default is "EPSG:4326".

    Returns:
        BaseGeometry: The footprint geometry of the raster dataset.

    """
    footprint_coords = [
        (ds.attrs["left_first_longitude"], ds.attrs["left_first_latitude"]),  # bottom-left
        (ds.attrs["right_first_longitude"], ds.attrs["right_first_latitude"]),  # bottom-right
        (ds.attrs["right_last_longitude"], ds.attrs["right_last_latitude"]),  # top-right
        (ds.attrs["left_last_longitude"], ds.attrs["left_last_latitude"]),  # top-left
        (ds.attrs["left_first_longitude"], ds.attrs["left_first_latitude"]),
    ]

    # Adjust coordinates for dateline crossing
    footprint_coords = adjust_footprint_signs(footprint_coords)

    # Create the polygon
    footprint = Polygon(footprint_coords)

    # If the CRS is not EPSG:4326, we need to reproject the footprint
    if crs is not None:
        footprint_gdf = gpd.GeoDataFrame(geometry=[footprint], crs="epsg:4326")
        footprint_gdf = footprint_gdf.to_crs(crs)
        footprint = cast("Polygon", footprint_gdf.geometry.iloc[0])

    return footprint


def get_swot_footprint(swot_item: earthaccess.DataGranule) -> tuple[BaseGeometry, str]:
    """Extract the footprint of a SWOT item.

    For the footprint, we need to access the SWOT dataset, so we will use the
    item, to open a connection to the file in the cloud.

    Args:
        swot_item (earthaccess.DataGranule): The SWOT item to extract the footprint from.

    Returns:
        BaseGeometry: The footprint of the SWOT item.

    """
    # Get the native-id of the swot item and check if the file with the footprint
    # already exists in /data/swot/footprints
    native_id = cast("str", swot_item["meta"]["native-id"])
    parts = native_id.split("_")
    tile_id = parts[-5]  # The tile_id is the 5th last part of the native-id
    pass_id = parts[-6]  # The pass_id is the 6th last part of the native-id

    footprint_path = Path("/data/swot/footprints") / f"{pass_id}_{tile_id}.parquet"
    if footprint_path.exists():
        # If the footprint file already exists, read it directly
        geom = gpd.read_parquet(footprint_path).geometry.iloc[0]
        return cast("BaseGeometry", geom), tile_id

    # If the footprint file does not exist, we need to open the SWOT dataset
    # and extract the footprint from the corner coordinates
    # Open the SWOT item to get the dataset
    # We will use the pqdm_kwargs to disable parallel processing and TQDM bar
    files = earthaccess.download(
        [swot_item],
        local_path="/data/swot/downloads",
        pqdm_kwargs={"disable": True},
    )

    ds = xr.open_dataset(files[0])  # type: ignore[]

    # Create the footprint polygon using the corner coordinates
    # Assuming the corners form a quadrilateral in this order.
    # before start, check if the dataset has the required attributes
    if "outer_first_longitude" in ds.attrs:
        footprint = get_pixc_footprint(ds)
    else:
        footprint = get_raster_footprint(ds)

    # Before returning, save the footprint to a file
    footprint_gdf = gpd.GeoDataFrame(geometry=[footprint], crs="EPSG:4326")
    footprint_gdf.to_parquet(footprint_path)

    return footprint, tile_id


def get_nadir_from_footprint(footprint: Polygon) -> LineString:
    """Extract the nadir line from a given footprint polygon in an xarray Dataset.

    The function assumes that the dataset contains a footprint polygon (footprint_proj)
    and computes the nadir line by:
      - Finding the centroid of the footprint.
      - Separating the exterior coordinates into points above and below the centroid latitude.
      - Averaging the two uppermost and two lowermost points to define the endpoints of the nadir line.

    Args:
        footprint: A shapely Polygon representing the footprint.

    Returns:
        LineString: A shapely LineString representing the nadir line between the averaged upper and lower points.

    """
    # We need to separate the two uppermost and the two lowermost points
    # So, first we get the centroid of the footprint
    centroid = footprint.centroid

    # Then we separate the points above and below the centroid latitude
    # First get all points as a list
    pts = list(footprint.exterior.coords)

    # Before separating the points, we need to remove the last point, which is a duplicate of the first point
    pts = pts[:-1]

    # Now we can separate the points
    upper_points = [pt for pt in pts if pt[1] > centroid.y]
    lower_points = [pt for pt in pts if pt[1] < centroid.y]

    # Convert points to numpy arrays for easier manipulation and readability
    upper_points_np = np.array(upper_points)
    lower_points_np = np.array(lower_points)

    # With the upper and lower points, we can find the nadir line
    # For that, we need to find the mean between the two upper points and the two lower points
    upper_nadir = upper_points_np.mean(axis=0)
    lower_nadir = lower_points_np.mean(axis=0)

    return LineString([upper_nadir, lower_nadir])


def get_nadir_from_raster(ds: xr.Dataset, crs: CRS | None = None) -> LineString:
    """Extract the nadir line from a given raster dataset.

    The function assumes that the dataset contains a footprint polygon (footprint_proj)
    and computes the nadir line by:
      - Finding the centroid of the footprint.
      - Separating the exterior coordinates into points above and below the centroid latitude.
      - Averaging the two uppermost and two lowermost points to define the endpoints of the nadir line.

    Args:
        ds: An xarray Dataset containing the footprint polygon.
        crs: The coordinate reference system of the dataset. Default is "EPSG:4326".

    Returns:
        LineString: A shapely LineString representing the nadir line between the averaged upper and lower points.

    """
    footprint = get_raster_footprint(ds, crs=crs)
    return get_nadir_from_footprint(footprint)


@cache
def open_raster_file(
    file: str | Path,
    variables: list[str] | tuple[str],
    aoi: BaseGeometry | None = None,
    *,
    exclude_no_data: bool = False,
) -> xr.Dataset:
    """Open SWOT file and clip it if necessary."""
    variables = list(variables)
    # open the file
    ds = xrio.open_rasterio(file, mask_and_scale=True)[variables]  # type: ignore[]
    ds = cast("xr.Dataset", ds.squeeze())

    # Get the CRS of the dataset
    crs = ds.rio.crs

    # The "land" pixels come with NaN values, so we will fill them with 0
    ds = ds.fillna(0)

    if exclude_no_data:
        # Create a mask around nadir to remove near swath pixels
        nadir = get_nadir_from_raster(ds, crs)
        inner_swath = nadir.buffer(10000, cap_style="flat")
        nadir_mask = cast(
            "np.ndarray",
            geometry_mask(
                geometries=[inner_swath],
                out_shape=ds[variables[0]].shape,
                transform=ds[variables[0]].rio.transform(),
                invert=True,
            ),
        )
        ds = ds.where(~nadir_mask)

    # Ensure the dataset has a CRS assigned
    ds = ds.rio.write_crs(crs)

    if aoi is not None:
        # we need to project the AOI to the dataset CRS
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        aoi_proj = transform(transformer.transform, aoi)

        return ds.rio.clip(
            [aoi_proj],
            crs=crs,
            drop=True,
            all_touched=True,
        )  # type: ignore[]

    return ds


def create_raster_mosaic(  # noqa: PLR0913
    mosaic_df: pd.DataFrame,
    ref_date: str | Timestamp,
    aoi: BaseGeometry,
    variable: str = "water_frac",
    local_path: str | Path = "/data/swot/downloads",
    exclude_flags: list[str] | None = None,
    *,
    exclude_no_data: bool = False,
) -> xr.DataArray:
    """Create a mosaic GeoDataFrame from SWOT data around a reference date.

    The mosaic_df already has the closest observations for each tile.

    Parameters
    ----------
    mosaic_df : pd.DataFrame
        DataFrame with SWOT RASTER metadata grouped by mosaic dates,
        output from create_mosaic_df.
    ref_date : str
        Reference date to find closest observations for mosaic.
    aoi : BaseGeometry
        Area of interest for clipping the data.
    variable :
        The variable to extract from the raster data.
        Default is "water_frac" for raster products.
    local_path : str, default "/data/swot/downloads"
        Local path to save the downloaded files.
    exclude_flags :
        List of quality flags to exclude from the mosaic. If None, a default set is used.
    exclude_no_data :
        Whether to exclude no-data pixels from the mosaic. Default is False.

    Returns
    -------
    xr.DataArray
        DataArray with the mean value for the variable. Flagged values will be set to -1

    """
    # Find the closest items for mosaic creation
    # Considering the mosaic_df has a 2-level index with mosaic_date and tile_name
    # the ref_date must be in the first level and the result is a dataframe
    mosaic_items = cast("pd.DataFrame", mosaic_df.loc[ref_date])

    # Considering earthaccess.download is costly, we first try to locate the
    # files directly on the local_path
    local_path = Path(local_path)
    mosaic_files_s = cast(
        "pd.Series",
        mosaic_items["native-id"].apply(lambda x: next(local_path.glob(f"{x}*"), None)),  # type: ignore[]
    )

    # If all files are found locally, we can skip the download
    if mosaic_files_s.all():
        mosaic_files = cast("list[Path]", mosaic_files_s.to_list())
    else:
        if "items" in mosaic_items.columns:
            items = cast("list[earthaccess.DataGranule]", mosaic_items["item"].to_list())
        else:
            items = mosaic_items["url"].to_list()

        mosaic_files = earthaccess.download(
            items,
            local_path=local_path,
            pqdm_kwargs={"disable": True},
        )

    # Open the patches
    # include the quality flag in the variables list
    variables = cast("tuple[str]", (variable, "water_area_qual_bitwise"))
    patches = [
        open_raster_file(file, aoi=aoi, variables=variables, exclude_no_data=exclude_no_data)
        for file in mosaic_files
    ]

    # Assuming the first patch as reference, let's make the others match its shape
    ref_patch = patches[0]
    patches = [patch.rio.reproject_match(ref_patch).squeeze() for patch in patches]

    # Apply quality flag filtering to each patch
    exclude_flags = [] if exclude_flags is None else exclude_flags

    for i, patch in enumerate(patches):
        # Apply quality flag filtering to each patch
        if exclude_no_data:
            # Create a mask for no-data values
            no_data_mask = mask_by_flags(
                patch["water_area_qual_bitwise"],
                flags=["outside_scene_bounds", "outside_data_window"],
            )
            patch = patch.where(~no_data_mask)  # noqa: PLW2901

        # Now let's treat the quality flags
        quality_mask = mask_by_flags(patch["water_area_qual_bitwise"], exclude_flags)

        bitwise_flags = patch["water_area_qual_bitwise"].copy()

        # The pixels that do not pass the quality mask will be set to -1 (non-observed data)
        patches[i] = patch.where(~quality_mask, other=-1).clip(max=1)
        patches[i]["water_area_qual_bitwise"] = bitwise_flags

    # Concat the patches
    raster = xr.concat(patches, dim="idx").squeeze()

    # Now we have to create a mask for the flagged pixels
    # Considering the flagged have a value of 2, we will compute the mean
    # If the result is different from 2, there is a valid pixel somewhere.
    raster_mean = raster.mean(dim="idx")
    flagged_mask = raster_mean == -1

    # Now that we have the flagged mask, we can compute the mean ignoring the flags.
    raster = raster.where(raster != -1).mean(dim="idx")
    raster = raster.where(~flagged_mask, other=-1)

    return raster, patches
