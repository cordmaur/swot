"""Module to calculate the metrics for SWOT classification."""

from typing import cast

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.enums import Resampling
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)


def match_projections(arrays: list[xr.DataArray]) -> list[xr.DataArray]:
    """Reproject all arrays to the projection of the array that has the coarser resolution.

    Args:
        arrays: list of xarray DataArrays to be reprojected.

    Returns:
        list of reprojected xarray DataArrays.

    """
    # First, we identify the array with the coarser resolution
    # We assume all arrays cover the same region, so the coarser resolution
    # is the array with lower number of pixels
    sizes = [arr.size for arr in arrays]
    ref_array = arrays[np.argmin(sizes)]

    # Now we reproject all arrays to the projection of the ref_array
    reprojected_arrays: list[xr.DataArray] = []
    for arr in arrays:
        if arr.size != ref_array.size or arr.rio.crs != ref_array.rio.crs:
            arr_reprojected = arr.rio.reproject_match(
                ref_array,
                resampling=Resampling.mode,
            )
            arr_reprojected.data[arr_reprojected.data == 255] = 2  # Set no data to 2
            reprojected_arrays.append(arr_reprojected)
        else:
            reprojected_arrays.append(arr)

    return reprojected_arrays


def process_opera_mask(opera_mask: xr.DataArray, *, include_partial: bool = False) -> xr.DataArray:
    """Process OPERA mask to match SWOT classification scheme.

    OPERA mask values:
    0: not water (white)
    1: open water (blue)
    2: partial surface water *(light green)
    *252: snow/ice (cyan)
    *253: cloud/cloud shadow (gray)
    *254: ocean masked (dark blue)
    255: no data (transparent)


    SWOT classification scheme:
    0: no water
    1: water
    2: no data (ignore in calculations)
    3: flagged (considered for coverage)

    We will consider cloud, shadow and snow/ice as flagged (3).
    Land will be considered as no water (0).

    Args:
        opera_mask: xarray DataArray of the OPERA mask.
        include_partial: whether to include partial water as water (1) or not water (0).

    Returns:
        xarray DataArray of the processed OPERA mask.

    """
    # First, we create a copy of the opera_mask to process, because we will change
    # the values inside the numpy array
    opera_mask_proc = opera_mask.copy()

    # Any np.nan will be assigned to 3 (flagged)
    opera_mask_proc = opera_mask.fillna(3).astype(np.uint8)
    # Also any value > 200 will be assigned to 3 (flagged)
    opera_mask_proc.data[opera_mask.data > 200] = 3

    # Now we assign the values according to the partial water
    if include_partial:
        opera_mask_proc.data[opera_mask.data == 2] = 1  # partial water to water
    else:
        opera_mask_proc.data[opera_mask.data == 2] = 0  # partial water to no water

    return opera_mask_proc


# method to prepare the swot_mask to match the classification scheme
# it should receive the water threshold as argument
def process_swot_mask(swot_mask: xr.Dataset, *, water_threshold: float) -> xr.DataArray:
    """Process SWOT mask to match SWOT classification scheme.

    Args:
        swot_mask: xarray DataArray of the SWOT mask.
        water_threshold: float threshold to classify water pixels.

    Returns:
        xarray DataArray of the processed SWOT mask.

    """
    # First, we create a copy of the swot_mask to process
    swot_mask_proc = swot_mask["water_frac"].copy()

    # Apply the water threshold
    swot_mask_proc.data[swot_mask_proc.data >= water_threshold] = 1  # Water
    swot_mask_proc.data[(swot_mask_proc.data > -1) & (swot_mask_proc.data < water_threshold)] = 0
    swot_mask_proc.data[swot_mask_proc.data < -1] = 0  # not water
    swot_mask_proc.data[swot_mask_proc.data == -1] = 3
    return swot_mask_proc.fillna(2).astype("uint8")


def calc_metrics(
    ref_mask: xr.DataArray,
    pred_mask: xr.DataArray,
    metrics: list[str],
    *,
    binary: bool = False,
) -> pd.DataFrame:
    """Calculate metrics comparing reference and predicted masks.

    values for input masks:
    0: no water
    1: water
    2: no data (ignore in calculations)
    3: flagged (considered for coverage)

    The Union of no_data will be removed completely from the matrices.
    The valid targets are 0 and 1.
    The non observed class (3) will be considered for coverage. It will
    be considered a False Negative no matter the correct class is, thus
    decreasing the recall score. As there will be no False Positives for
    class 3, it will not penalize the precision score.

    Args:
        ref_mask: xarray DataArray of the reference mask.
        pred_mask: xarray DataArray of the predicted mask.
        metrics: a list of the metrics to be calculated
        binary: wether to consider a binary classification by setting 3 to 0

    Returns:
        dict: A dictionary containing the calculated metrics.

    """
    # First thing is to ensure both masks have the same shape
    # Considering the ref_mask is the higher resolution, we will coarse it to match pred_mask
    ref_mask, pred_mask = match_projections([ref_mask, pred_mask])

    # Convert everything to numpy
    ref_mask_np = ref_mask.to_numpy()
    pred_mask_np = pred_mask.to_numpy()

    # First, lets create the no-data mask, to remove these pixels.
    no_data_mask = (ref_mask_np == 2) | (pred_mask_np == 2)  # noqa: PLR2004
    no_data_mask = cast("np.ndarray", no_data_mask)

    # Now, we create the valid masks, removing no_data
    ref_mask_valid = ref_mask_np[~no_data_mask]
    pred_mask_valid = pred_mask_np[~no_data_mask]

    results: dict[str, float] = {}
    # results["valid_pixels"] = len(ref_mask_valid)
    # results["no_data_pixels"] = no_data_mask.sum()
    # results["flagged_pixels"] = (pred_mask_valid == 3).sum()  # noqa: PLR2004

    # Coverage: percentage of pixels that are not non-observed in
    # comparison to the valid pixels
    coverage = (pred_mask_valid != 3).sum() / (~no_data_mask).sum()  # noqa: PLR2004
    coverage = float(coverage)

    # Water coverage: percentage of water pixels (in reference) that are not non-observed
    # in comparison to the valid water pixels
    ref_water_mask = ref_mask_valid == 1
    if ref_water_mask.sum() > 0:
        water_coverage = (pred_mask_valid[ref_water_mask] != 3).sum() / ref_water_mask.sum()
        water_coverage = float(water_coverage)
    else:
        water_coverage = np.nan

    # Now, let's treat the flagged data. If it is a binary classification, we shall
    # consider flagged data as no water (0).
    if binary:
        pred_mask_valid[pred_mask_valid == 3] = 0  # noqa: PLR2004
        average = "binary"
    else:
        average = "weighted"

    # Now we create the confusion matrix
    cm = confusion_matrix(
        ref_mask_valid,
        pred_mask_valid,
        labels=[0, 1],
    )

    # Extract True Positives, False Positives, True Negatives, False Negatives
    tn, fp, fn, tp = cm.ravel()  # type: ignore[]

    # Calculate metrics using scikit
    if "iou" in metrics:
        iou = jaccard_score(ref_mask_valid, pred_mask_valid, labels=[0, 1], average=average)
        results["iou"] = round(float(iou), 4)

    if "f1" in metrics:
        f1 = f1_score(ref_mask_valid, pred_mask_valid, labels=[0, 1], average=average)
        results["f1"] = round(float(f1), 4)

    if "accuracy" in metrics:
        accuracy = accuracy_score(ref_mask_valid, pred_mask_valid)
        results["accuracy"] = round(float(accuracy), 3)

    if "precision" in metrics:
        precision = precision_score(
            ref_mask_valid,
            pred_mask_valid,
            labels=[0, 1],
            average=average,
        )
        results["precision"] = round(float(precision), 4)
    if "recall" in metrics:
        recall = recall_score(ref_mask_valid, pred_mask_valid, labels=[0, 1], average=average)
        results["recall"] = round(float(recall), 4)

    if "kappa" in metrics:
        kappa = cohen_kappa_score(ref_mask_valid, pred_mask_valid, labels=[0, 1])
        results["kappa"] = round(float(kappa), 4)

    if "coverage" in metrics:
        results["coverage"] = round(float(coverage), 4)

    if "water_coverage" in metrics:
        results["water_coverage"] = round(float(water_coverage), 4)

    return pd.DataFrame.from_dict(results, orient="index")
