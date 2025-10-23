"""Pipeline 4 - Calculating the metrics."""

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import rioxarray as xrio
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from pyproj import CRS

from swot_toolkit.metrics import calc_metrics, process_opera_mask, process_swot_mask
from swot_toolkit.pipe2 import open_output_dir
from swot_toolkit.pipe3 import open_s2_img
from swot_toolkit.planetary import parse_s2_id
from swot_toolkit.swot import create_raster_mosaic

if TYPE_CHECKING:
    from matplotlib.colorbar import Colorbar

quality_flags_suspect = [
    "classification_qual_suspect",
    "geolocation_qual_suspect",
    "water_fraction_suspect",
    "large_uncert_suspect",
    "dark_water_suspect",
    "bright_land",
    "low_coherence_water_suspect",
    "specular_ringing_prior_water_suspect",
    "specular_ringing_prior_land_suspect",
    "few_pixels",
    "far_range_suspect",
    "near_range_suspect",
]

quality_flags_degraded = [
    "classification_qual_degraded",
    "geolocation_qual_degraded",
]

quality_flags_bad = [
    "value_bad",
    "outside_data_window",
    # "no_pixels",
    "outside_scene_bounds",
    "inner_swath",
    "missing_karin_data",
]


scenarios: dict[str, dict[str, None | bool | list[str]]] = {
    "as is": {
        "exclude_flags": None,
        "exclude_no_data": False,
    },
    "exclude no data": {
        "exclude_flags": None,
        "exclude_no_data": True,
    },
    "exclude bad": {
        "exclude_flags": quality_flags_bad,
        "exclude_no_data": False,
    },
    "exclude (bad, degraded)": {
        "exclude_flags": quality_flags_bad + quality_flags_degraded,
        "exclude_no_data": False,
    },
    "exclude (bad, degraded, suspect)": {
        "exclude_flags": quality_flags_bad + quality_flags_degraded + quality_flags_suspect,
        "exclude_no_data": False,
    },
}


@cache
def open_ref_mask(output_dir: Path, sensing_date: str) -> xr.DataArray:
    """Open reference mask for the given sensing date."""
    flist = list((output_dir / "ref_mask").glob(f"*final*{sensing_date}*.tif"))
    if len(flist) > 1:
        msg = f"More than one reference mask found for date {sensing_date}"
        raise ValueError(msg)

    if len(flist) == 0:
        msg = f"No reference mask found for date {sensing_date}"
        raise ValueError(msg)

    return cast("xr.DataArray", xrio.open_rasterio(flist[0])).squeeze()


@cache
def open_opera_s2(output_dir: Path, sensing_date: str, crs: str | None = None) -> xr.DataArray:
    """Open OPERA S2 mask for the given sensing date."""
    flist = list((output_dir / "opera_s2").glob(f"*{sensing_date}*.tif"))
    if len(flist) > 1:
        msg = f"More than one OPERA S2 mask found for date {sensing_date}"
        raise ValueError(msg)

    if len(flist) == 0:
        msg = f"No OPERA S2 mask found for date {sensing_date}"
        raise ValueError(msg)

    opera_s2 = cast("xr.DataArray", xrio.open_rasterio(flist[0])).squeeze()

    if crs is not None:
        opera_s2 = opera_s2.rio.reproject(crs)

    return opera_s2


@cache
def open_opera_s1(
    output_dir: Path, sensing_date: str, crs: str | None = None
) -> xr.DataArray | None:
    """Open OPERA S1 mask for the given sensing date (month prefix).

    Returns None if no OPERA S1 mask is found, as it may not exist.
    """
    flist = list((output_dir / "opera_s1").glob(f"*{sensing_date[:6]}*.tif"))

    if len(flist) > 1:
        msg = f"More than one OPERA S1 mask found for date {sensing_date[:6]}"
        raise ValueError(msg)

    if len(flist) == 1:
        opera_s1 = cast("xr.DataArray", xrio.open_rasterio(flist[0])).squeeze()
        if crs is not None:
            opera_s1 = opera_s1.rio.reproject(crs)
        return opera_s1

    print(f"No OPERA S1 mask available for date {sensing_date[:6]}")
    return None


@cache
def open_datasets(region_name: str, ref_date: str) -> dict[str, xr.DataArray]:
    """Open all datasets for a given region and reference date.

    Parameters
    ----------
    region_name : str
        Name of the region to process
    ref_date : str
        Reference date for the datasets

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary containing all opened datasets with keys:
        s2_img, scl, ref_mask, opera_s2, and optionally opera_s1

    """
    # Get the output dir and basic info
    output_dir, _, s2_id = open_output_dir(region_name, ref_date)
    s2_meta = parse_s2_id(s2_id)
    datasets: dict[str, xr.DataArray] = {}

    # First, let's open the  s2_img and scl datasets
    s2_img, scl = open_s2_img(s2_id, output_dir)
    datasets["s2_img"] = s2_img
    datasets["scl"] = scl

    # Make sure everyone has the same CRS.
    crs = s2_img.rio.crs

    # Open the reference mask
    sensing_date = str(s2_meta["sensing_date"])
    ref_mask = open_ref_mask(output_dir, sensing_date)
    datasets["ref_mask"] = ref_mask

    # Open OPERA S2 mask
    # OPERA S2 should have the same sensing as the original s2 image
    opera_s2_mask = open_opera_s2(output_dir, sensing_date, crs=crs)
    datasets["opera_s2"] = opera_s2_mask

    # Open OPERA S1 mask
    # OPERA S1 may exist or not and it may have a different sensing date. Let's get by month.
    opera_s1_mask = open_opera_s1(output_dir, sensing_date, crs=crs)
    if opera_s1_mask is not None:
        datasets["opera_s1"] = opera_s1_mask

    print(f"The following datasets have been opened: {list(datasets.keys())}")

    return datasets


def plot_s2_rgb(
    s2_img: xr.DataArray,
    ax: Axes,
    down_factor: int = 10,
) -> None:
    """Plot and save the RGB image from the S2 data.

    Parameters
    ----------
    s2_img : xr.DataArray
        The S2 image data array.
    ax : plt.Axes
        The axes on which to plot the RGB image.
    down_factor : int, optional
        Factor by which to downsample the image for plotting, by default 10.

    """
    rgb = s2_img.sel(
        x=slice(None, None, down_factor),
        y=slice(None, None, down_factor),
        band=["B04", "B03", "B02"],
    )

    rgb.plot.imshow(rgb="band", vmin=0.05, vmax=0.25, ax=ax, robust=True)
    ax.set_aspect("equal")

    if "native-id" in s2_img.attrs:
        ax.set_title(f"Sentinel-2 RGB - {s2_img.attrs['native-id']}")
    else:
        ax.set_title("Sentinel-2 RGB")


def plot_ref_mask(
    ref_mask: xr.DataArray,
    ax: Axes,
    down_factor: int = 10,
) -> None:
    """Plot and save the reference mask.

    Parameters
    ----------
    ref_mask : xr.DataArray
        The reference mask data array.
    ax : plt.Axes
        The axes on which to plot the reference mask.
    down_factor : int, optional
        Factor by which to downsample the image for plotting, by default 10.

    """
    mask_cmap = ListedColormap(["white", "blue", "red"])
    ref_mask = ref_mask.sel(x=slice(None, None, down_factor), y=slice(None, None, down_factor))
    im = ref_mask.plot.imshow(
        ax=ax,
        cmap=mask_cmap,
        vmin=0,
        vmax=2,
        interpolation="antialiased",
        interpolation_stage="rgba",
    )

    # Get the colorbar and set its ticks and labels
    cbar = cast("Colorbar", im.colorbar)
    cbar.set_ticks([0.33, 1, 1.66])
    cbar.set_ticklabels(["Not Water", "Water", "No Data"])

    ax.set_title("Reference water mask")
    ax.set_aspect("equal")


def calc_opera_metrics(region_name: str, ref_date: str, metrics: list[str]) -> pd.DataFrame:
    """Calculate OPERA metrics for the given region and reference date.

    Parameters
    ----------
    region_name : str
        Name of the region to process
    ref_date : str
        Reference date for the datasets
    metrics : list[str]
        List of metrics to calculate

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated metrics

    """
    datasets = open_datasets(region_name, ref_date)

    results = pd.DataFrame()
    for dataset_name in ["opera_s2", "opera_s1"]:
        for include_partial in [False, True]:
            opera_mask_proc = process_opera_mask(
                datasets[dataset_name],
                include_partial=include_partial,
            )
            stats = calc_metrics(datasets["ref_mask"], opera_mask_proc, metrics, binary=True)
            column_name = dataset_name + " incl. partial" if include_partial else dataset_name
            stats = stats.rename(columns={0: column_name})
            results = pd.concat([results, stats], axis=1)

    return results


def calc_swot_metrics(region_name: str, ref_date: str, metrics: list[str]) -> pd.DataFrame:
    """Calculate SWOT metrics for the given region and reference date.

    Parameters
    ----------
    region_name : str
        Name of the region to process
    ref_date : str
        Reference date for the datasets
    metrics : list[str]
        List of metrics to calculate

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated metrics

    """
    datasets = open_datasets(region_name, ref_date)
    results_swot = pd.DataFrame()

    for scenario, values in scenarios.items():
        print(f"Processing scenario: {scenario}")
        swot_mask, _, _ = create_swot_mosaic(
            region_name=region_name,
            ref_date=ref_date,
            exclude_flags=cast("list[str]", values["exclude_flags"]),
            exclude_no_data=cast("bool", values["exclude_no_data"]),
        )

        swot_mask = process_swot_mask(swot_mask, water_threshold=0.55)

        stats = calc_metrics(datasets["ref_mask"], swot_mask, metrics, binary=True)
        stats = stats.rename(columns={0: scenario})
        results_swot = pd.concat([results_swot, stats], axis=1)

    return results_swot


def create_swot_mosaic(
    region_name: str,
    ref_date: str,
    *,
    dst_crs: CRS | None = None,
    exclude_flags: list[str] | None = None,
    exclude_no_data: bool = False,
    variable: str = "water_frac",
) -> tuple[xr.Dataset, list[xr.Dataset], list[np.ndarray]]:
    """Create SWOT mosaic for the given region and reference date.

    Parameters
    ----------
    region_name : str
        Name of the region to process
    ref_date : str
        Reference date for the datasets
    dst_crs : CRS | None, optional
        Destination CRS for reprojection, by default None
    exclude_flags : list[str] | None, optional
        List of flags to exclude from the analysis, by default None
    exclude_no_data : bool, optional
        Whether to exclude no data values, by default False

    """
    base_dir, aoi, _ = open_output_dir(region_name, ref_date)

    mosaic_df = pd.read_parquet(base_dir.parent / "dfs/swot_raster_results.parquet")

    swot_mask, patches, no_data_masks = create_raster_mosaic(
        mosaic_df,
        ref_date=ref_date,
        aoi=aoi,
        dst_crs=dst_crs,
        variable=variable,
        exclude_flags=exclude_flags,
        exclude_no_data=exclude_no_data,
    )

    return swot_mask, patches, no_data_masks
