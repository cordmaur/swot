"""Implementation of Pipeline 1: Data Download."""

from pathlib import Path

from matplotlib import pyplot as plt
from shapely.geometry.base import BaseGeometry

from swot_toolkit.kml import read_kml_geometry
from swot_toolkit.opera import (
    open_opera_mask_from_datetime,
    open_opera_s1,
    plot_opera_array,
)
from swot_toolkit.planetary import open_s2_array, parse_s2_id
from swot_toolkit.swot import auth_earthaccess

auth_earthaccess()

OUTPUT_DIR = "/data/swot/output/"


def open_output_dir(region: str, date: str) -> tuple[Path, BaseGeometry, str]:
    """Open an existing output directory structure for a given AOI.

    This function opens the AOI kml and the reference S2_IDs from the directory'.

    Args:
        region (str): Path to the base output directory.
        date (str): Date of the swot mosaic.

    Returns:
        tuple[BaseGeometry, dict[str, str]]: A tuple containing the AOI geometry and a
        dictionary of S2 IDs.

    """
    output_dir = Path(OUTPUT_DIR) / region / date

    # Check if it exists
    if not output_dir.exists():
        msg = f"Output directory not found: {output_dir}"
        raise FileNotFoundError(msg)

    # Read the AOI (the first KML found in kml folder)
    kml_dir = output_dir.parent / "kml"
    kml_file = next(kml_dir.glob("*.kml"))
    print(f"Reading KML file: {kml_file}")
    aoi = read_kml_geometry(kml_file)[0]

    # read the S2_IDS.json
    with (output_dir / "s2_id.txt").open("r") as f:
        s2_id = f.read()

    return output_dir, aoi, s2_id


def download_opera_s2_masks(
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
        fig.savefig(Path(output_dir) / "figs" / f"{fname}.png")
        ax.clear()

        # Save the OPERA mask to the opera directory
        out_path = Path(output_dir) / "opera_s2" / f"OPERA_S2_{fname}.tif"
        opera_mask.rio.to_raster(out_path, compress="DEFLATE")

        opera_mask = open_opera_mask_from_datetime(
            str(s2_meta["tile"]),
            str(s2_meta["sensing_date"]),
            aoi,
        ).squeeze()

    fig.clear()


def download_opera_s1_masks(s2_ids: list[str], aoi: BaseGeometry, output_dir: str | Path) -> None:
    """Download OPERA S1 masks for a list of S2 IDs.

    Args:
        s2_ids (list[str]): List of Sentinel-2 IDs.
        aoi (BaseGeometry): Area of Interest geometry.
        output_dir (str | Path): Path to the base output directory.

    Returns:
        None

    """
    # Create a figure to save the thumbnails
    fig, ax = plt.subplots(figsize=(10, 10))

    # Loop through the S2 IDs and download the OPERA masks
    for i, s2_id in enumerate(s2_ids):
        s2_meta = parse_s2_id(s2_id)
        print(f"Downloading OPERA S1 mask for S2 ID: {s2_id}")

        opera_s1_mask = open_opera_s1(
            aoi,
            str(s2_meta["date"]),
        )

        if opera_s1_mask is None:
            continue

        opera_s1_mask = opera_s1_mask.squeeze()

        # Save the OPERA thumbnail to the figs directory
        plot_opera_array(opera_s1_mask, ax=ax, add_colorbar=not bool(i))
        fname = opera_s1_mask.attrs["native-id"]
        fig.savefig(Path(output_dir) / "figs" / f"OPERA_S1_{fname}.png")
        ax.clear()

        # Save the OPERA mask to the opera directory
        out_path = Path(output_dir) / "opera_s1" / f"{fname}.tif"
        opera_s1_mask.rio.to_raster(out_path, compress="DEFLATE")

    fig.clear()


def download_s2_data(s2_ids: list[str], aoi: BaseGeometry, output_dir: str | Path) -> None:
    """Download Sentinel-2 data for a list of S2 IDs and save them to the output directory.

    Args:
        s2_ids (list[str]): List of Sentinel-2 IDs.
        aoi (BaseGeometry): Area of Interest geometry.
        output_dir (str | Path): Path to the base output directory.

    Returns:
        None

    """
    # Create a figure to save the thumbnails
    fig, ax = plt.subplots(figsize=(10, 10))

    for s2_id in s2_ids:
        print(f"Downloading Sentinel-2 data for S2 ID: {s2_id}")
        s2_array = open_s2_array(s2_id, aoi=aoi, bands=None)

        # Save the OPERA thumbnail to the figs directory
        rgb = s2_array.sel(
            x=slice(None, None, 10),
            y=slice(None, None, 10),
            band=["B04", "B03", "B02"],
        )
        rgb.plot.imshow(rgb="band", vmin=500, vmax=2500, ax=ax)
        fname = s2_id
        fig.savefig(Path(output_dir) / "figs" / f"RGB_{fname}.png")
        ax.clear()

        # Save the Sentinel-2 data to the s2 directory
        out_path = Path(output_dir) / "s2" / f"{s2_id}.tif"
        s2_array.rio.to_raster(out_path, compress="DEFLATE")

    fig.clear()


def download_opera_data(
    s2_ids: list[str],
    aoi: BaseGeometry,
    output_dir: str | Path,
) -> None:
    """Download OPERA S2 and S1 masks for a list of S2 IDs.

    Args:
        s2_ids (list[str]): List of Sentinel-2 IDs.
        aoi (BaseGeometry): Area of Interest geometry.
        output_dir (str | Path): Path to the base output directory.

    Returns:
        None

    """
    download_opera_s2_masks(s2_ids, aoi, output_dir)
    download_opera_s1_masks(s2_ids, aoi, output_dir)
