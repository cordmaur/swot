"""KML file reading utilities for the SWOT toolkit."""

from pathlib import Path
from typing import Any

import fastkml as kml
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


def read_kml_geometry(
    kml_path: str | Path,
    feature_index: int = 0,
) -> list[BaseGeometry]:
    """Read a KML file and return a Shapely geometry.

    Parameters
    ----------
    kml_path : str | Path
        Path to the KML file
    feature_index : int, default 0
        Index of the feature (placemark) to extract geometry from

    Returns
    -------
    BaseGeometry
        Shapely geometry object extracted from the KML

    Raises
    ------
    FileNotFoundError
        If the specified KML file doesn't exist
    IndexError
        If the specified feature index doesn't exist in the KML
    ValueError
        If the KML file cannot be parsed or doesn't contain valid geometry

    """
    kml_path = Path(kml_path)
    if not kml_path.exists():
        msg = f"KML file not found: {kml_path}"
        raise FileNotFoundError(msg)

    # Read the KML file with explicit encoding
    with kml_path.open("r", encoding="utf-8") as f:
        doc = f.read()

    # Remove XML declaration that can cause parsing issues
    doc = doc.replace('<?xml version="1.0" encoding="UTF-8"?>', "")

    # Parse the KML document
    k = kml.KML.from_string(doc)

    # Extract the geometry from the specified feature index
    if len(k.features) <= feature_index:
        msg = f"Feature index {feature_index} not found. KML contains {len(k.features)} features."
        raise IndexError(msg)

    feature: Any = k.features[feature_index]

    # Handle different types of features
    if hasattr(feature, "features") and getattr(feature, "features", None):
        return [
            shape(ft.geometry.__geo_interface__)
            for ft in feature.features
            if hasattr(ft, "geometry") and getattr(ft, "geometry", None) is not None
        ]

    msg = "No valid geometry found in the KML file."
    raise ValueError(msg)
