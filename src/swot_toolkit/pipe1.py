"""Implementation of Pipeline 1: Dates Matching."""

import shutil
from pathlib import Path
from typing import cast

import pandas as pd
from shapely.geometry.base import BaseGeometry
from tqdm.auto import tqdm

from swot_toolkit.kml import read_kml_geometry
from swot_toolkit.opera import opera_results_to_df, search_opera
from swot_toolkit.planetary import (
    assess_s2_clouds_new,
    find_closest_s2,
    s2_results_to_df,
    search_s2,
)
from swot_toolkit.plotting import plot_mosaic_footprints
from swot_toolkit.swot import create_mosaic_df, search_swot_data, swot_results_to_df


def create_output_dir(aoi_kml: str | Path, output_dir: str | Path) -> Path:
    """Create an output directory structure for a given AOI KML file.

    This function creates a base output directory named after the AOI file (without extension),
    and within it, subdirectories for 'kml', 'ref_masks', 'opera', 's2', 'swot', and 'logs'.
    The AOI KML file is copied into the 'kml' subdirectory.

    Args:
        aoi_kml (str | Path): Path to the AOI KML file.
        output_dir (str | Path): Path to the base output directory.

    Returns:
        Path: Path to the created AOI-specific output directory.

    Raises:
        FileNotFoundError: If the AOI KML file does not exist.

    """
    # Create a path for the aoi source and base output
    output_dir = Path(output_dir)
    aoi_path = Path(aoi_kml)

    # Check if it exists
    if not aoi_path.exists():
        msg = f"AOI KML file not found: {aoi_path}"
        raise FileNotFoundError(msg)

    # Get the name of the AOI file without extension
    aoi_name = aoi_path.stem

    # Create the output directory if it doesn't exist
    output_dir = output_dir / aoi_name
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory created at: {output_dir}")

    # Create subfolders for [kml, ref_masks, opera, s2, swot, logs]
    subfolders = ["kml", "ref_masks", "opera", "s2", "swot", "logs", "dfs", "figs"]
    for folder in subfolders:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    # Copy the AOI KML file to the output `kml` directory
    shutil.copy(aoi_path, output_dir / "kml")

    # Return the output directory path
    return output_dir


def create_s2_df(
    aoi_geometry: BaseGeometry,
    date_range: tuple[str, str],
    *,
    assess_clouds: bool = True,
) -> pd.DataFrame:
    """Create a Sentinel-2 dataframe with cloud assessment for the given AOI and date range.

    Parameters
    ----------
    aoi_geometry : BaseGeometry
        The area of interest geometry.
    date_range : tuple[str, str]
        Date range for searching data (start_date, end_date).
    assess_clouds : bool, optional
        Whether to perform cloud assessment on the Sentinel-2 scenes. Defaults to True.

    Returns
    -------
    pd.DataFrame
        DataFrame containing Sentinel-2 scenes with cloud assessment.

    """
    # Create a dataframe with with Sentinel-2 scenes intersecting the AOI
    print("Searching for Sentinel-2 scenes intersecting the AOI...")
    s2_df = s2_results_to_df(
        search_s2(aoi_geometry, date_range),
    )

    # Drop any duplicate dates
    s2_df = s2_df.drop_duplicates(subset="datetime")

    # Fill cloud assessment for Sentinel-2 scenes
    if assess_clouds:
        print("Assessing clouds in Sentinel-2 scenes...")
        return assess_s2_clouds_new(s2_df, aoi=aoi_geometry)

    return s2_df


def create_swot_mosaic_df(aoi_geometry: BaseGeometry, date_range: tuple[str, str]) -> pd.DataFrame:
    """Create SWOT mosaic dataframe for the given AOI and date range.

    Parameters
    ----------
    aoi_geometry : BaseGeometry
        The area of interest geometry.
    date_range : tuple[str, str]
        Date range for searching data (start_date, end_date).

    Returns
    -------
    pd.DataFrame
        DataFrame containing SWOT mosaics.

    """
    # Search for SWOT data
    print("Searching for SWOT Raster_100 data intersecting the AOI...")
    swot_results = search_swot_data(
        aoi=aoi_geometry,
        date_range=date_range,
        dataset="Raster_100",
        footprint_filter=True,
    )
    swot_df = swot_results_to_df(swot_results, drop_duplicates=True)

    # Create swot mosaics and store them into a dataframe
    print("Creating SWOT mosaics...")
    return create_mosaic_df(swot_df, max_delta=30)


def add_opera_info(
    s2_df: pd.DataFrame,
    aoi_geometry: BaseGeometry,
    date_range: tuple[str, str],
) -> pd.DataFrame:
    """Add OPERA satellite information to Sentinel-2 dataframe.

    Parameters
    ----------
    s2_df : pd.DataFrame
        DataFrame containing Sentinel-2 scenes with cloud assessment.
    aoi_geometry : BaseGeometry
        The area of interest geometry.
    date_range : tuple[str, str]
        Date range for searching data (start_date, end_date).

    Returns
    -------
    pd.DataFrame
        DataFrame with OPERA satellite information added.

    """
    print("Searching for OPERA satellite data intersecting the AOI...")
    # Add information about OPERA availability
    opera_results = search_opera(aoi_geometry.centroid, date_range)
    opera_df = opera_results_to_df(opera_results)
    # ignore L8 satellite
    opera_df = opera_df[opera_df["satellite"] != "L8"]
    print(f"Found {len(opera_df)} OPERA satellite overpasses intersecting the AOI.")

    # Clean up duplicated entries and keep only the satellite for each date
    opera_df_grouped = (
        opera_df.groupby(opera_df.index)["satellite"]
        .apply(lambda x: ", ".join(list(x)) if len(x) > 1 else x.iloc[0])  # type: ignore[list-index]
        .reset_index()
        .set_index("date")
        .rename(columns={"satellite": "OPERA"})
    )
    s2_df["date"] = s2_df["datetime"].dt.date
    s2_df = s2_df.set_index("date")

    # aggregate the opera info into s2_df
    s2_df = pd.concat([s2_df, opera_df_grouped["OPERA"]], axis=1)
    return s2_df.dropna(subset=["datetime"])


def match_swot_mosaics_s2(mosaic_df: pd.DataFrame, s2_df: pd.DataFrame) -> pd.DataFrame:
    """Match SWOT mosaic dates to closest Sentinel-2 dates.

    Parameters
    ----------
    mosaic_df : pd.DataFrame
        DataFrame containing SWOT mosaics with dates as index.
    s2_df : pd.DataFrame
        DataFrame containing Sentinel-2 scenes with cloud assessment.

    Returns
    -------
    pd.DataFrame
        DataFrame containing matched SWOT mosaic dates and closest Sentinel-2 scenes.

    """
    # Match the mosaic dates to closest Sentinel-2 dates
    mosaic_dates = mosaic_df.index.get_level_values(0).unique()

    match_df = pd.DataFrame()
    for idx in tqdm(mosaic_dates, total=len(mosaic_dates)):
        matched_s2 = find_closest_s2(idx, s2_df)
        matched_s2["swot_mosaic_date"] = idx

        match_df = pd.concat([match_df, matched_s2.reset_index()], axis=0, ignore_index=True)

    return match_df.set_index("swot_mosaic_date")


def prepare_aoi_dataframes(
    aoi_kml: str | Path,
    date_range: tuple[str, str],
    output_dir: str | Path,
) -> Path:
    """Prepare AOI dataframes for further processing.

    This function reads the AOI KML file and generates necessary dataframes
    for the AOI, including the area of interest (AOI) geometry and other relevant
    geospatial data. The generated dataframes are saved in the specified output directory.

    Args:
        aoi_kml (str | Path): Path to the AOI KML file.
        date_range: tuple[str, str]: Date range for searching data (start_date, end_date).
        output_dir (str | Path): Path to the output directory where dataframes will be saved.

    Returns:
        tuple: A tuple containing:
            - aoi_geometry (shapely.geometry.Polygon): The geometry of the AOI.
            - mosaic_df (pandas.DataFrame): DataFrame containing mosaic information.
            - footprints (dict): Dictionary containing footprint geometries.

    Raises:
        FileNotFoundError: If the AOI KML file does not exist.
        ValueError: If there is an issue reading or processing the KML file.

    """
    # Create a path for the aoi source and base output
    output_dir = create_output_dir(aoi_kml, output_dir)

    # Load AOI geometry
    aoi_geometry = read_kml_geometry(aoi_kml)[0]

    # Create Sentinel-2 dataframe with cloud assessment
    s2_df = create_s2_df(aoi_geometry, date_range, assess_clouds=True)

    # Add OPERA information to S2 dataframe
    s2_df = add_opera_info(s2_df, aoi_geometry, date_range)

    # Create SWOT mosaic dataframe
    mosaic_df = create_swot_mosaic_df(aoi_geometry, date_range)

    # Get the first mosaic to plot its footprints
    mosaic = cast("pd.DataFrame", mosaic_df.loc[mosaic_df.index[0][0]])
    plot_mosaic_footprints(mosaic, aoi=aoi_geometry, output_dir=output_dir)

    # Match SWOT mosaic dates to closest Sentinel-2 dates
    match_df = match_swot_mosaics_s2(mosaic_df, s2_df)

    # Save the dataframes to the output directory
    print(f"Saving dataframes to {output_dir / 'dfs'}...")
    s2_df.drop(columns=["item", "delta"]).to_parquet(
        output_dir / "dfs/s2_search_results.parquet",
        index=True,
    )
    mosaic_df.drop(columns=["item", "delta"]).to_parquet(
        output_dir / "dfs/swot_raster_results.parquet",
        index=True,
    )
    match_df.drop(columns=["item", "delta"]).to_parquet(
        output_dir / "dfs/swot_s2_matches.parquet",
        index=True,
    )

    return output_dir
