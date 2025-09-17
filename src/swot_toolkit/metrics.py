"""Module to calculate the metrics for SWOT classification."""

from typing import cast
import numpy as np
import xarray as xr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calc_metrics(ref_mask: xr.DataArray, pred_mask: xr.DataArray) -> dict[str, float]:
    """Calculate metrics comparing reference and predicted masks.

    values for input masks:
    0: land
    1: water
    2: no_data (ignore in calculations)
    3: non observed (considered for coverage)

    The Union of no_data will be removed completely from the matrices.
    The valid targets are 0 and 1.
    The non observed class (3) will be considered for coverage. It will
    be considered a False Negative no matter the correct class is, thus
    decreasing the recall score. As there will be no False Positives for
    class 3, it will not penalize the precision score.

    Args:
        ref_mask: xarray DataArray of the reference mask.
        pred_mask: xarray DataArray of the predicted mask.

    Returns:
        dict: A dictionary containing the calculated metrics.

    """
    ref_mask_np = ref_mask.to_numpy()
    pred_mask_np = pred_mask.to_numpy()

    # First, lets create the no-data mask, to be used in the end.
    no_data_mask = (ref_mask_np == 2) | (pred_mask_np == 2)  # noqa: PLR2004
    no_data_mask = cast("np.ndarray", no_data_mask)

    # Now, we create the valid masks, removing no_data
    ref_mask_valid = ref_mask_np[~no_data_mask]
    pred_mask_valid = pred_mask_np[~no_data_mask]

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

    # Now we create the confusion matrix
    cm = confusion_matrix(
        ref_mask_valid,
        pred_mask_valid,
        labels=[0, 1],
    )

    # Extract True Positives, False Positives, True Negatives, False Negatives
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics using scikit
    accuracy = accuracy_score(ref_mask_valid, pred_mask_valid)
    precision = precision_score(ref_mask_valid, pred_mask_valid, labels=[0, 1], average="weighted")
    recall = recall_score(ref_mask_valid, pred_mask_valid, labels=[0, 1], average="weighted")
    f1 = f1_score(ref_mask_valid, pred_mask_valid, labels=[0, 1], average="weighted")
    kappa = cohen_kappa_score(ref_mask_valid, pred_mask_valid, labels=[0, 1])

    # Return all metrics
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "kappa": kappa,
        "coverage": coverage,
        "water_coverage": water_coverage,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "total_valid": int((~no_data_mask).sum()),
        "total_no_data": int(no_data_mask.sum()),
        "total_pixels": int(ref_mask_np.size),
        "total_non_observed": int((pred_mask_np == 3).sum()),  # noqa: PLR2004
        # "ref_mask": ref_mask_valid,
        # "pred_mask": pred_mask_valid,
    }
