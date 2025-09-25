"""Module with functions to search for Sentinel-2 images using PLanetary Computer."""

from datetime import datetime, timedelta
from typing import cast

import pandas as pd
import planetary_computer as pc
import pystac_client
import xarray as xr
from odc.stac import stac_load  # type: ignore[]
from pandas import Timestamp
from pystac import Item, ItemCollection
from shapely.geometry.base import BaseGeometry
from tqdm.auto import tqdm

catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


def guess_best_s2_tile(aoi: BaseGeometry) -> str | None:
    """Find the Sentinel-2 tile with minimum null values for a given area of interest.

    This function searches for Sentinel-2 L2A imagery within a specified time range
    and geographic area, then evaluates each available tile to determine which has
    the fewest null values in the Scene Classification Layer (SCL) band. This helps
    identify the optimal tile for analysis by minimizing data gaps.

    Parameters
    ----------
    aoi : BaseGeometry
        Area of Interest as a Shapely geometry object (e.g., Polygon, Point).
        Used to spatially constrain the Sentinel-2 image search.

    Returns
    -------
    str | None
        The MGRS tile identifier (e.g., "33TWN") of the tile with the minimum
        number of null values in the SCL band. Returns None if no tiles are found
        or if an error occurs during processing.

    Notes
    -----
    This function performs the following steps:
    1. Searches the Planetary Computer STAC catalog for Sentinel-2 L2A imagery
       between 2024-01-01 and 2024-02-15 that intersects the AOI
    2. Loads the SCL band for each unique tile at 20m resolution
    3. Counts null values in each tile
    4. Returns the tile identifier with the minimum null count

    The function uses a fixed date range and may need adjustment for different
    time periods. Processing time scales with the number of tiles found.

    """
    # Search the STAC Catalog
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi,
        datetime=("2024-01-01", "2024-02-15"),
    )

    s2_df = s2_results_to_df(search.item_collection())

    tiles = s2_df["tile"].unique()
    s2_null_values: dict[str, int] = {}
    for tile in tiles:
        item = cast("Item", s2_df[s2_df["tile"] == tile]["item"].iloc[0])

        cube = stac_load(
            [pc.sign(item)],  # List of STAC items (can pass multiple)
            bands=["SCL"],  # Bands to load
            resolution=20,  # Output pixel size
            bbox=aoi.bounds,
        )

        s2_null_values[tile] = cube["SCL"].isnull().sum().item()  # noqa: PD003

        print(f"Tile {tile} has {s2_null_values[tile]} null values.")

    # Return the tile with the least null values
    best_tile = (
        min(s2_null_values.keys(), key=lambda k: s2_null_values[k]) if s2_null_values else None
    )
    print(f"Best tile is {best_tile}.")

    return best_tile


def search_s2(
    aoi: BaseGeometry,
    date_range: tuple[str, str],
    s2_tile: str | None = None,
    rel_orbit: int | None = None,
) -> ItemCollection:
    """Search for Sentinel-2 imagery."""
    # Check if s2_tile is informed. If not, will try to guess the best
    if not s2_tile:
        print("S2 tile not provided. Guessing the best tile...")
        s2_tile = guess_best_s2_tile(aoi)

    # Construct a spatial query
    query: dict[str, dict[str, str | int]] = {"s2:mgrs_tile": {"eq": s2_tile}} if s2_tile else {}

    if rel_orbit:
        query["sat:relative_orbit"] = {"eq": rel_orbit}

    # Search the STAC Catalog
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi,
        datetime=date_range,
        query=query,
    )

    return search.item_collection()


def s2_results_to_df(s2_items: ItemCollection) -> pd.DataFrame:
    """Convert the Sentinel 2 results to a comprehensive dataframe."""
    data = {
        item.id: {
            "datetime": item.datetime,
            "tile": item.properties["s2:mgrs_tile"],
            "item": item,
        }
        for item in s2_items
    }

    s2 = pd.DataFrame(data).T

    # Convert the datetime to a timezone-naive format
    s2["datetime"] = pd.to_datetime(s2["datetime"]).dt.tz_localize(None)

    return s2.sort_values("datetime")


def find_closest_s2(
    ref_time: datetime | Timestamp | str,
    s2df: pd.DataFrame,
    max_days: int = 5,
) -> pd.DataFrame:
    """Find the closest Sentinel-2 images to a reference time."""
    # raise an error if "datetime" is not in the dataframe
    if "datetime" not in s2df.columns:
        msg = "The dataframe must contain a 'datetime' column."
        raise ValueError(msg)

    # get the delta time for all s2 images and filter those within max_days
    s2df["delta"] = s2df["datetime"].astype("datetime64[ns]") - pd.to_datetime(ref_time)
    delta = s2df[s2df["delta"].abs() < timedelta(days=max_days)]

    # now, order by the closest
    delta = delta.iloc[delta["delta"].abs().argsort()]

    # return the available images
    return s2df.loc[delta.index]


def assess_s2_clouds(
    ref_time: datetime | Timestamp | str,
    s2df: pd.DataFrame,
    aoi: BaseGeometry,
    max_days: int = 5,
) -> pd.DataFrame:
    # get the closest dates
    closest_s2 = find_closest_s2(ref_time, s2df, max_days=max_days)

    # now, for each s2 img, check out for the clouds
    for idx, row in closest_s2.iterrows():  # type: ignore[]
        # get the img
        s2item = cast("Item", row["item"])
        img = stac_load(
            items=[s2item],
            bands=["SCL"],
            patch_url=pc.sign,
            dtype="uint16",
            nodata=0,
            bbox=aoi.bounds,
            resolution=20,
        )["SCL"].squeeze()

        valid = img.where((img >= 4) & (img <= 6))

        closest_s2.loc[idx, "valid_pxls"] = float(valid.count()) / float(valid.size)  # type: ignore[]

    return closest_s2


def assess_s2_clouds_new(
    s2df: pd.DataFrame,
    aoi: BaseGeometry,
) -> pd.DataFrame:
    """Assess cloud coverage in all Sentinel-2 images within the dataframe.

    This function evaluates the cloud coverage for each Sentinel-2 image in the
    provided dataframe by analyzing the Scene Classification Layer (SCL) band.
    It calculates the fraction of valid (cloud-free) pixels for each image,
    adding this information as a new column to the dataframe.

    Parameters
    ----------
    s2df : pd.DataFrame
        DataFrame containing Sentinel-2 imagery metadata with required columns:
        - 'item': STAC Item objects representing S2 images
        Additional columns from the dataframe will be preserved.
    aoi : BaseGeometry
        Area of Interest as a Shapely geometry object (e.g., Polygon, Point).
        Used to spatially constrain the S2 image analysis to the region of interest.
        The bounds of this geometry determine the spatial extent for downloading
        and processing the SCL band.

    Returns
    -------
    pd.DataFrame
        The input dataframe with an additional 'valid_pxls' column containing
        the fraction of cloud-free pixels (float between 0 and 1) for each
        Sentinel-2 image. A value of 1.0 indicates completely cloud-free,
        while 0.0 indicates completely cloudy within the AOI.

    Notes
    -----
    The cloud assessment uses the Sentinel-2 Scene Classification Layer (SCL)
    band at 20m resolution. Pixels are considered valid (cloud-free) if they
    are classified as:
    - 4: Vegetation
    - 5: Not-vegetated
    - 6: Water

    All other SCL values (including clouds, cloud shadows, snow, etc.) are
    considered invalid. The function downloads the SCL band data for each
    image, which may take time depending on the number of images and AOI size.

    The function modifies the input dataframe in-place by adding the 'valid_pxls'
    column.

    """
    # now, for each s2 img, check out for the clouds
    for idx, row in tqdm(s2df.iterrows(), total=len(s2df)):  # type: ignore[]
        # get the img
        s2item = cast("Item", row["item"])
        img = stac_load(
            items=[s2item],
            bands=["SCL"],
            patch_url=pc.sign,
            dtype="uint16",
            nodata=0,
            bbox=aoi.bounds,
            resolution=20,
        )["SCL"].squeeze()

        valid = img.where((img >= 4) & (img <= 6))  # noqa: PLR2004

        s2df.loc[idx, "valid_pxls"] = float(valid.count()) / float(valid.size)  # type: ignore[]

    return s2df


def match_swot_s2(
    swot_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    aoi: BaseGeometry,
    max_days: int = 10,
) -> pd.DataFrame:
    """Match SWOT observations with closest Sentinel-2 images and assess cloud coverage.

    This function combines SWOT satellite water surface height observations with
    Sentinel-2 optical imagery by finding temporally close S2 images and evaluating
    their cloud coverage quality. The result enables correlation analysis between
    SWOT measurements and optical observations.

    Parameters
    ----------
    swot_df : pd.DataFrame
        DataFrame containing SWOT observations with required 'datetime' column.
        Each row represents a SWOT observation at a specific time.
    s2_df : pd.DataFrame
        DataFrame containing Sentinel-2 imagery metadata with required 'datetime'
        and 'item' columns. The 'item' column should contain STAC Item objects.
    aoi : BaseGeometry
        Area of Interest as a Shapely geometry object (e.g., Polygon, Point).
        Used to spatially constrain the S2 image analysis for cloud assessment.
    max_days : int, optional
        Maximum temporal window in days to search for matching S2 images.
        Default is 10 days (±10 days from each SWOT observation).

    Returns
    -------
    pd.DataFrame
        Multi-indexed DataFrame with columns:
        - 'vers': SWOT data version
        - 'datetime': Original observation datetime
        - 'valid_pxls': Fraction of cloud-free pixels in S2 image (0-1)
        - 'id': SWOT observation identifier
        - 'id_s2': Sentinel-2 image identifier
        - 'item': Original SWOT STAC item
        - 'item_s2': Matched Sentinel-2 STAC item

        Index levels are ('index', 'delta') where 'delta' is the time
        difference between SWOT and S2 observations.

    Notes
    -----
    This function performs cloud assessment by analyzing the Scene Classification
    Layer (SCL) band of Sentinel-2 images. Pixels classified as vegetation (4),
    not-vegetated (5), or water (6) are considered valid (cloud-free).

    The processing can be time-intensive for large datasets as it downloads
    and processes S2 imagery for cloud analysis.

    """
    # Initialize empty DataFrame to accumulate S2 statistics for all SWOT observations
    s2_stats_df = pd.DataFrame()

    # Process each SWOT observation individually
    for row in tqdm(swot_df.itertuples(), total=len(swot_df)):
        # Find closest S2 images and assess their cloud coverage
        closest_s2 = assess_s2_clouds(
            ref_time=cast("Timestamp", row.datetime),
            s2df=s2_df,
            aoi=aoi,
            max_days=max_days,
        )

        # Prepare S2 results for joining with SWOT data
        # Set index name to 'id' for proper joining
        closest_s2.index.name = "id"
        # Add reference to the corresponding SWOT observation datetime
        closest_s2["index"] = row.datetime
        # Restructure index to use SWOT datetime as primary index
        closest_s2 = closest_s2.reset_index()
        closest_s2 = closest_s2.set_index("index", drop=True)

        # Accumulate results from all SWOT observations
        s2_stats_df = pd.concat([s2_stats_df, closest_s2], axis=0)

    # Join SWOT observations with their matched S2 statistics
    joined = swot_df.join(s2_stats_df, rsuffix="_s2")

    # Create multi-level index with original index and time delta for easy sorting/filtering
    swot_s2 = joined.reset_index().set_index(["index", "delta"])

    # Select only the most relevant columns for the final output
    columns = ["vers", "datetime", "valid_pxls", "id", "id_s2", "item", "item_s2"]

    return swot_s2[columns]


def parse_s2_id(s2_id: str) -> dict[str, str | int | float]:
    """Parse a Sentinel-2 ID into its components.

    Parameters
    ----------
    s2_id : str
        Sentinel-2 ID string to parse.

    Returns
    -------
    dict[str, str | int | float]
        Dictionary containing parsed components of the Sentinel-2 ID:
        - 'mission': Mission identifier (e.g., 'S2A', 'S2B')
        - 'product_level': Product level (e.g., 'L1C', 'L2A')
        - 'sensing_date': Sensing date as a string (YYYYMMDDTHHMMSS)
        - 'processing_date': Processing date as a string (YYYYMMDDTHHMMSS)
        - 'relative_orbit': Relative orbit number (int)
        - 'tile': MGRS tile identifier (str)

    Raises
    ------
    ValueError
        If the provided s2_id does not conform to the expected format.

    Examples
    --------
    >>> parse_s2_id("S2A_MSIL2A_20240101T123456_N0509_R137_T33TWN_20240101T130000")
    {
        'mission': 'S2A',
        'product_level': 'L2A',
        'sensing_date': '20240101T123456',
        'processing_date': '20240101T130000',
        'relative_orbit': 137,
        'tile': '33TWN',
        'product_discriminator': 509,
        'cloud_coverage': 0.0
    }

    """
    parts = s2_id.split("_")
    if len(parts) != 6:
        msg = f"Invalid Sentinel-2 ID format: {s2_id}"
        raise ValueError(msg)

    mission = parts[0]
    product_level = parts[1][3:]  # Remove "MSI" prefix
    sensing_date = parts[2]
    relative_orbit = int(parts[3][1:])  # Remove "R" prefix and convert to int
    tile = parts[4][1:]  # Remove "T" prefix
    processing_date = parts[5]

    return {
        "mission": mission,
        "product_level": product_level,
        "sensing_date": sensing_date,
        "processing_date": processing_date,
        "relative_orbit": relative_orbit,
        "tile": tile,
        "date": sensing_date[:8],
    }


def open_s2_array(
    s2_id: str | Item,
    aoi: BaseGeometry,
    bands: list[str] | None = None,
) -> xr.DataArray:
    """Open a Sentinel-2 image as an xarray DataArray.

    Parameters
    ----------
    s2_id : str | Item
        Sentinel-2 ID string or STAC Item object representing the S2 image to open.
    aoi : BaseGeometry
        Area of Interest as a Shapely geometry object (e.g., Polygon, Point).
    bands : list[str]
        List of band names to load (e.g., ['B04', 'B08']).

    Returns
    -------
    xr.DataArray
        xarray DataArray containing the requested bands from the Sentinel-2 image.

    Raises
    ------
    ValueError
        If the provided s2_id is neither a string nor a STAC Item.

    """
    if isinstance(s2_id, str):
        collection = catalog.get_collection("sentinel-2-l2a")
        s2_item = collection.get_item(id=s2_id)
        if s2_item is None:
            msg = f"Sentinel-2 item with ID {s2_id} not found in catalog."
            raise ValueError(msg)
    else:
        s2_item = s2_id

    if bands is None:
        bands = [
            "B02",
            "B03",
            "B04",
            "B08",
            "B11",
            "B12",
            "SCL",
        ]  # Default bands if none specified

    # Load the data using odc.stac
    cube = stac_load(
        [s2_item],  # List of STAC items (can pass multiple)
        bands=bands,  # Bands to load
        resolution=10,  # Output pixel size
        bbox=aoi.bounds,
        dtype="uint16",
        patch_url=pc.sign,  # Function to sign the URLs
    )

    return cube[bands].squeeze().to_array(dim="band")  # type: ignore[]
