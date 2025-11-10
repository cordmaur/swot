"""Plotting functions for SWOT analysis."""

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Generator, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from shapely.geometry.base import BaseGeometry

from swot_toolkit.swot import get_swot_footprint

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@contextmanager
def matplotlib_backend(backend: str) -> Generator[None, None, None]:
    """Context manager to temporarily switch matplotlib backend.

    Args:
        backend (str): The backend to switch to (e.g., 'Agg', 'Qt5Agg').

    Yields:
        None

    """
    original_backend = plt.get_backend()
    try:
        plt.switch_backend(backend)
        yield
    finally:
        plt.switch_backend(original_backend)


def plot_mosaic_footprints(
    mosaic: pd.DataFrame,
    ax: Axes | None = None,
    aoi: BaseGeometry | None = None,
    output_dir: Path | str | None = None,
) -> None:
    """Plot SWOT footprints from a mosaic DataFrame.

    Parameters
    ----------
    mosaic : pd.DataFrame
        DataFrame containing SWOT items with 'item' column.
    ax : Axes | None, optional
        Matplotlib axes to plot on. Creates new figure if None.
    aoi : BaseGeometry | None, optional
        Area of interest to overlay as red dashed boundary.
    output_dir : Path | str | None, optional
        Directory to save plot as 'figs/swot_footprints.png'.

    """
    # Create new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        # Get figure from existing axes for potential saving
        fig = cast("Figure", ax.figure)

    # Extract footprint geometries from SWOT items
    footprints: dict[str, BaseGeometry] = {}
    for item in mosaic["item"]:
        footprint, tile = get_swot_footprint(item)
        footprints[tile] = footprint

    # Convert footprints dictionary to GeoDataFrame for plotting
    gdf = gpd.GeoDataFrame.from_dict(footprints, orient="index").rename(columns={0: "geometry"})
    gdf = gdf.set_geometry("geometry")
    gdf = gdf.set_crs("EPSG:4326")

    # Plot area of interest boundary if provided
    if aoi is not None:
        aoi_gdf = gpd.GeoDataFrame([{"geometry": aoi}], crs="EPSG:4326", index=["AOI"])
        aoi_gdf.boundary.plot(ax=ax, color="red", linestyle="--", linewidth=3, label="AOI")

    # Plot SWOT footprints with different colors for each tile
    gdf.reset_index().plot(
        ax=ax,
        facecolor="none",  # Transparent fill to show only boundaries
        linewidth=5,
        column="index",  # Color by tile identifier
        cmap="tab10",  # Categorical colormap
        legend=True,
    )

    # Set plot title with list of tiles
    title = "SWOT Footprints " + str(gdf.index.to_list())
    ax.set_title(title)

    # Add axis labels for geographic coordinates
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Save figure if output directory is specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        fig.savefig(output_dir / "figs/swot_footprints.png", dpi=300)
