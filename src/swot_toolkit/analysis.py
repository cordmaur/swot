"""Analysis Module."""

from os import PathLike
from pathlib import Path
from typing import cast

import earthaccess
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as xrio
import xarray as xr
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from swot_toolkit.flags import mask_by_flags
from swot_toolkit.swot import auth_earthaccess, get_nadir_from_raster, get_raster_footprint

DOWNLOAD_FOLDER = Path("/data/swot/downloads")

auth_earthaccess()


def check_dir(dir_path: PathLike[str]) -> Path:
    """Check if a directory exists and is actually a directory.

    Args:
        dir_path (PathLike[str]): Path to the directory to check.

    Returns:
        Path: The validated directory path as a Path object.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path exists but is not a directory.

    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        msg = f"Base directory not found: {dir_path}"
        raise FileNotFoundError(msg)

    if not dir_path.is_dir():
        msg = f"Base directory is not a directory: {dir_path}"
        raise NotADirectoryError(msg)

    return dir_path


def open_sites_and_dates(base_dir: PathLike[str]) -> dict[str, list[str]]:
    """Open the sites and dates from the base directory.

    It will automatically crawl the base_dir for the sites and dates.

    Args:
        base_dir (PathLike[str]): Path to the base directory where the results are stored.

    Returns:
        dict[str, list[str]]: A dictionary with the sites as keys and the list of
        dates as values.

    """
    base_dir = check_dir(base_dir)

    # The sites must be directories in the base dir
    sites = [f for f in base_dir.iterdir() if f.is_dir()]

    # Now, for each site we look for the available dates
    sites_dates: dict[str, list[str]] = {}
    for site in sites:
        # The dates must also be directories in the site dir and begin with a digit
        dates = [f for f in site.iterdir() if (f.is_dir() and f.name[0].isdigit())]
        sites_dates[site.name] = [d.name for d in dates]

    return sites_dates


def open_results(base_dir: PathLike[str], file_pattern: str = "") -> pd.DataFrame:
    """Open the results from the processing (Pipes) steps.

    It will automatically crawl the base_dir for the sites and dates.
    Then, each parquet will be assigned to a multi-index dataframe with site and date.

    Args:
        base_dir (PathLike[str]): Path to the base directory where the results are stored.
        file_pattern (str, optional): A glob pattern to filter the files to open.

    Returns:
        pd.DataFrame: A DataFrame containing the results for each site and date.

    """
    base_dir = check_dir(base_dir)
    sites_dates = open_sites_and_dates(base_dir)

    results: list[pd.DataFrame] = []
    for site, dates in sites_dates.items():
        for date in dates:
            # search for the file
            results_path = base_dir / site / date
            files = list(results_path.glob(f"results{file_pattern}.parquet"))

            if len(files) > 1:
                msg = f"More than one results file found for site {site} and date {date}: {files}"
                raise FileExistsError(msg)

            if len(files) == 0:
                msg = f"No results file found for site {site} and date {date}"
                raise FileNotFoundError(msg)

            df = pd.read_parquet(files[0])

            df.index = pd.MultiIndex.from_product(
                [[site + " " + date], df.index],
                names=["site", "metric"],
            )

            results.append(df)

    return pd.concat(results, axis=0)


def plot_nadir_line(ds: xr.Dataset, ax: Axes) -> Artist:
    """Plot the nadir line from the dataset.

    Args:
        ds (xrio.Dataset): The dataset containing the nadir line data.
        ax (Axes): The axes to plot on.

    """
    nadir_line = get_nadir_from_raster(ds, crs=ds["water_frac"].rio.crs)

    gpd.GeoDataFrame(geometry=[nadir_line]).plot(
        ax=ax,
        color="red",
        linewidth=1,
        linestyle="--",
    )

    # Create manual legend handle
    return Line2D([0], [0], color="red", linewidth=1, linestyle="--", label="Nadir track")


def plot_footprint(ds: xr.Dataset, ax: Axes) -> Artist:
    """Plot the SWOT footprint from the dataset.

    Args:
        ds (xrio.Dataset): The dataset containing the footprint data.
        ax (Axes): The axes to plot on.

    """
    footprint = get_raster_footprint(ds, crs=ds["water_frac"].rio.crs)
    gpd.GeoDataFrame(geometry=[footprint]).plot(
        ax=ax,
        facecolor="none",
        linewidth=1,
        linestyle="-",
    )

    return Line2D(
        [0],
        [0],
        color="black",
        linewidth=1,
        linestyle="-",
        label="SWOT Footprint",
    )


def plot_water_fraction(ds: xr.Dataset, ax: Axes, *, add_colorbar: bool = True) -> None:
    """Plot the water fraction from the dataset.

    Args:
        ds (xrio.Dataset): The dataset containing the water fraction data.
        ax (Axes): The axes to plot on.
        add_colorbar (bool): Whether to add a colorbar to the plot.

    """
    water_frac_thumb = (
        ds["water_frac"].sel(x=slice(None, None, 2), y=slice(None, None, 2)).squeeze()
    )

    cbar_kwargs = {"label": "Water Fraction"} if add_colorbar else None

    # Plot the water frac
    water_frac_thumb.plot.imshow(
        vmin=0,
        vmax=1,
        ax=ax,
        cmap="Blues",
        cbar_kwargs=cbar_kwargs,
        add_colorbar=add_colorbar,
    )

    # Plot nadir line and footprint
    nadir_handle = plot_nadir_line(ds, ax)
    footprint_handle = plot_footprint(ds, ax)

    ax.legend(handles=[nadir_handle, footprint_handle])

    ax.set_title("SWOT Water Fraction (as is)")


def plot_inner_swath_mask(ds: xr.Dataset, ax: Axes, alpha: float = 1.0) -> Artist:
    """Plot the inner swath from the dataset.

    Args:
        ds (xrio.Dataset): The dataset containing the inner swath data.
        ax (Axes): The axes to plot on.
        alpha (float): The transparency level of the plot.

    """
    # Create the mask
    # Create the mask for inner swath
    mask_np = mask_by_flags(ds["water_area_qual_bitwise"].squeeze(), flags=["inner_swath"])
    mask_np = mask_np[::2, ::2]
    template = ds["water_frac"].sel(x=slice(None, None, 2), y=slice(None, None, 2)).squeeze()
    mask = template.copy()
    mask.data = mask_np.astype("uint8")
    mask = mask.where(mask > 0)

    # Create a custom colormap for the mask
    custom_cmap = ListedColormap(["white", "orange"])
    mask.plot.imshow(ax=ax, cmap=custom_cmap, alpha=alpha, vmin=0, vmax=1, add_colorbar=False)

    # Create the patch for the swath mask
    return mpatches.Patch(color="orange", alpha=1, label="Inner Swath Data Mask")


def plot_inner_swath_boundary(ds: xr.Dataset, ax: Axes) -> Artist:
    """Plot the inner swath boundary from the dataset.

    Args:
        ds (xrio.Dataset): The dataset containing the inner swath data.
        ax (Axes): The axes to plot on.

    """
    # Plot the inner swath boundary
    nadir_line = get_nadir_from_raster(ds, crs=ds["water_frac"].rio.crs)
    swath = nadir_line.buffer(10000, cap_style="flat")

    gpd.GeoDataFrame(geometry=[swath.boundary]).plot(
        ax=ax,
        facecolor="none",
        linewidth=2,
        linestyle="-",
        edgecolor="orange",
        label="Inner Swath Boundary",
    )

    return Line2D([0], [0], color="orange", linewidth=2, linestyle="-", label="Inner Swath")


def plot_inner_swath_fig(ds: xr.Dataset, ax: Axes) -> None:
    """Plot the water fraction swath from the dataset.

    Args:
        ds (xrio.Dataset): The dataset containing the water fraction swath data.
        ax (Axes): The axes to plot on.

    """
    # Plot nadir line and footprint
    nadir_handle = plot_nadir_line(ds, ax)
    footprint_handle = plot_footprint(ds, ax)
    inner_swath_patch = plot_inner_swath_mask(ds, ax)

    ax.legend(handles=[nadir_handle, footprint_handle, inner_swath_patch], loc="upper left")

    ax.set_title("Inner Swath Mask (bitwise flag)")


def plot_wfrac_inner_swath(ds: xr.Dataset, ax: Axes) -> None:
    """Plot the water fraction with inner swath mask from the dataset.

    Args:
        ds (xrio.Dataset): The dataset containing the water fraction and inner swath data.
        ax (Axes): The axes to plot on.

    """
    # Plot water fraction
    water_frac_thumb = (
        ds["water_frac"].sel(x=slice(None, None, 2), y=slice(None, None, 2)).squeeze()
    )

    # Plot the water frac
    water_frac_thumb.plot.imshow(
        vmin=0,
        vmax=1,
        ax=ax,
        cmap="Blues",
        add_colorbar=False,
    )

    # Plot nadir line and footprint
    nadir_handle = plot_nadir_line(ds, ax)
    footprint_handle = plot_footprint(ds, ax)
    inner_swath_patch = plot_inner_swath_mask(ds, ax, alpha=0.5)

    ax.legend(handles=[nadir_handle, footprint_handle, inner_swath_patch], loc="upper left")
    ax.set_title("SWOT Water Fraction with Inner Swath Mask")


def plot_cross_track_error(ds: xr.Dataset, ax: Axes) -> None:
    """Plot the cross track error from the dataset.

    Args:
        ds (xrio.Dataset): The dataset containing the cross track data.
        ax (Axes): The axes to plot on.

    """
    # get pixels with cross track value below 10km
    inner_swath = ds["cross_track"].squeeze()
    inner_swath = inner_swath.where(np.abs(inner_swath) <= 10_000)  # noqa: PLR2004

    inner_swath.plot.imshow(ax=ax, add_colorbar=True)

    # Plot nadir line and footprint
    nadir_handle = plot_nadir_line(ds, ax)
    footprint_handle = plot_footprint(ds, ax)
    swath_handle = plot_inner_swath_boundary(ds, ax)

    ax.legend(handles=[nadir_handle, footprint_handle, swath_handle], loc="upper left")

    ax.set_title("Pixels with cross_track value below 10km")


def inner_swath_analysis(file: str | PathLike[str]) -> Figure:
    """Perform inner swath analysis on a given file.

    Args:
        file (PathLike): Path to the file to analyze.

    Returns:
        Figure: The figure containing the plot.

    """
    print(file)
    file = Path(file)
    ds = cast("xr.Dataset", xrio.open_rasterio(file, mask_and_scale=True))

    # Create the figure and axes
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    # Close the figure to avoid displaying it in non-GUI backends
    plt.close(fig)

    # Plot water fraction
    plot_water_fraction(ds, ax[0, 0], add_colorbar=False)
    plot_inner_swath_fig(ds, ax[0, 1])
    plot_wfrac_inner_swath(ds, ax[1, 0])
    plot_cross_track_error(ds, ax[1, 1])

    fig.suptitle(f"Inner Swath Analysis for {file.stem}", fontsize=16)

    return fig


def plot_inner_swath_analysis(region: str, date: str) -> list[Figure]:
    """Plot the inner swath analysis for a given region and date.

    Args:
        region (str): The region to plot.
        date (str): The date to plot.

    Returns:
        Figure: The figure containing the plot.

    """
    # First, open the output directory for this region and date
    base_dir = Path(f"/data/swot/output/{region}")

    mosaic_df = pd.read_parquet(base_dir / "dfs/swot_raster_results.parquet")
    mosaic_items = mosaic_df.loc[date]

    # Get the files
    mosaic_files = earthaccess.download(
        mosaic_items["url"].to_list(),
        local_path=DOWNLOAD_FOLDER,
        pqdm_kwargs={"disable": True},
    )

    figs = [inner_swath_analysis(file) for file in mosaic_files]
    for fig, tile in zip(figs, mosaic_items.index, strict=True):
        file = fig.get_suptitle()
        fig.suptitle(f"Inner_swath analysis_{region}-{date}-{tile}\n{file}", fontsize=16)

    return figs
