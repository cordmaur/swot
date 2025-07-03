"""Module with functions to search for Sentinel-2 images using PLanetary Computer."""

from datetime import datetime, timedelta
from typing import cast

import pandas as pd
import planetary_computer as pc
import pystac_client
from odc.stac import stac_load  # type: ignore[]
from pandas import Timestamp
from pystac import Item, ItemCollection
from shapely.geometry.base import BaseGeometry
from tqdm.auto import tqdm

catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


def search_s2(
    aoi: BaseGeometry,
    date_range: str,
    s2_tile: str | None = None,
    rel_orbit: int | None = None,
) -> ItemCollection:
    """Search for Sentinel-2 imagery."""
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
    s2df["delta"] = s2df["datetime"] - pd.to_datetime(ref_time)
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


def match_swot_s2(
    swot_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    aoi: BaseGeometry,
    max_days: int = 10,
) -> pd.DataFrame:
    """Combine a swot dataframe and a s2 dataframe to find the closest S2 Images."""
    s2_stats_df = pd.DataFrame()

    for row in tqdm(swot_df.itertuples(), total=len(swot_df)):

        closest_s2 = assess_s2_clouds(
            ref_time=cast("Timestamp", row.datetime),
            s2df=s2_df,
            aoi=aoi,
            max_days=max_days,
        )

        closest_s2.index.name = "id"
        closest_s2["index"] = row.datetime
        closest_s2 = closest_s2.reset_index()
        closest_s2 = closest_s2.set_index("index", drop=True)

        s2_stats_df = pd.concat([s2_stats_df, closest_s2], axis=0)

    joined = swot_df.join(s2_stats_df, rsuffix="_s2")
    swot_s2 = joined.reset_index().set_index(["index", "delta"])

    columns = ["vers", "datetime", "valid_pxls", "id", "id_s2", "item", "item_s2"]

    return swot_s2[columns]
