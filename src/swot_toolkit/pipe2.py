"""Implementation of Pipeline 1: Data Download."""

import json
from pathlib import Path

from matplotlib import pyplot as plt
from shapely.geometry.base import BaseGeometry

from swot_toolkit.kml import read_kml_geometry
from swot_toolkit.opera import (
    open_opera_mask_from_datetime,
    plot_opera_array,
)
from swot_toolkit.planetary import parse_s2_id
from swot_toolkit.swot import auth_earthaccess

auth_earthaccess()


def open_output_dir(output_dir: str | Path) -> tuple[BaseGeometry, dict[str, str]]:
    """Open an existing output directory structure for a given AOI.

    This function opens the AOI kml and the reference S2_IDs from the directory'.

    Args:
        output_dir (str | Path): Path to the base output directory.

    Returns:
        tuple[BaseGeometry, dict[str, str]]: A tuple containing the AOI geometry and a
        dictionary of S2 IDs.

    """
    # Create a path for the base output
    output_dir = Path(output_dir)

    # Check if it exists
    if not output_dir.exists():
        msg = f"Output directory not found: {output_dir}"
        raise FileNotFoundError(msg)

    # Read the AOI (the first KML found in kml folder)
    kml_dir = output_dir / "kml"
    kml_file = next(kml_dir.glob("*.kml"))
    print(f"Reading KML file: {kml_file}")
    aoi = read_kml_geometry(kml_file)[0]

    # read the S2_IDS.json
    with (output_dir / "S2_IDS.json").open("r") as f:
        s2_ids = json.load(f)

    return aoi, s2_ids


def download_opera_masks(
    s2_ids: list[str],
    aoi: BaseGeometry,
    output_dir: str | Path,
) -> None:
    """Download OPERA masks for a list of S2 IDs and save them to the output directory.

    Args:
        s2_ids (list[str]): List of Sentinel-2 IDs.
        aoi (BaseGeometry): Area of Interest geometry.
        output_dir (str | Path): Path to the base output directory.

    """
    # Create a figure to save the thumbnails
    fig, ax = plt.subplots(figsize=(10, 10))

    # Loop through the S2 IDs and download the OPERA masks
    for i, s2_id in enumerate(s2_ids):
        s2_meta = parse_s2_id(s2_id)
        print(f"Downloading OPERA mask for S2 ID: {s2_id}")
        opera_mask = open_opera_mask_from_datetime(
            str(s2_meta["tile"]),
            str(s2_meta["sensing_date"]),
            aoi,
        ).squeeze()

        # Save the OPERA thumbnail to the figs directory
        plot_opera_array(opera_mask, ax=ax, add_colorbar=not bool(i))
        fname = opera_mask.attrs["native-id"]
        fig.savefig(Path(output_dir) / "figs" / f"{fname}_thumb.png")
        ax.clear()

        # Save the OPERA mask to the opera directory
        out_path = Path(output_dir) / "opera" / f"{fname}.tif"
        opera_mask.rio.to_raster(out_path)

        opera_mask = open_opera_mask_from_datetime(
            str(s2_meta["tile"]),
            str(s2_meta["sensing_date"]),
            aoi,
        ).squeeze()

    fig.clear()
