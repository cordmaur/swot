"""Pipeline 3: Create Reference Mask."""

from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import rioxarray as xrio
import xarray as xr
from scipy.spatial import cKDTree  # type: ignore[]
from sklearn.ensemble import RandomForestClassifier

S2_BANDS = {
    "B02": 1,
    "B03": 2,
    "B04": 3,
    "B08": 4,
    "B11": 5,
    "B12": 6,
    "SCL": 7,
}

SCL_MAPPING = {
    0: 2,  # No data
    1: 2,  # No data
    2: 2,  # No data
    3: 2,  # Cloud / Shadow
    4: 0,  # No Water
    5: 0,  # No Water
    6: 1,  # Water
    7: 2,  # Cloud / Shadow
    8: 2,  # Cloud / Shadow
    9: 2,  # Cloud / Shadow
    10: 2,  # Cloud / Shadow
    11: 2,  # Cloud / Shadow
}

SCL_SAMPLING = {
    0: 0,
    1: 0,  # No data
    2: 0,  # No data
    3: 0,  # Cloud / Shadow
    4: 50,  # No Water
    5: 50,  # No Water
    6: 100,  # Water
    7: 0,  # Cloud / Shadow
    8: 50,  # Cloud / Shadow
    9: 50,  # Cloud / Shadow
    10: 0,  # Cloud / Shadow
    11: 0,  # Cloud / Shadow
}


def preprocess_s2_img(s2_img: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Preprocess a Sentinel-2 image.

    Separate SCL, normalize the reflectance bands (divide by 10000), and rename bands.
    Check for negative values in reflectance bands and set them to zero.
    Create the indices NDVI, NDWI, and MNDWI.

    Args:
        s2_img (xr.DataArray): The input Sentinel-2 image as an xarray DataArray.

    Returns:
        xr.DataArray: The preprocessed Sentinel-2 image with selected and renamed bands.

    """
    # Separate the SCL band
    scl = s2_img.sel(band=S2_BANDS["SCL"])
    s2_img = s2_img.drop_sel(band=S2_BANDS["SCL"])

    # Normalize reflectance bands (divide by 10000)
    s2_img = s2_img / 10000.0

    # Check for negative values in reflectance bands and set them to zero
    s2_img = s2_img.where(s2_img >= 0, 0)

    # Rename bands
    band_names = [
        band for band, _ in sorted(S2_BANDS.items(), key=lambda item: item[1]) if band != "SCL"
    ]
    s2_img = s2_img.assign_coords(band=band_names)

    # Create NDVI and MNDWI indices
    ndvi = (s2_img.sel(band="B08") - s2_img.sel(band="B04")) / (
        s2_img.sel(band="B08") + s2_img.sel(band="B04")
    )
    ndwi = (s2_img.sel(band="B03") - s2_img.sel(band="B08")) / (
        s2_img.sel(band="B03") + s2_img.sel(band="B08")
    )

    mndwi = (s2_img.sel(band="B03") - s2_img.sel(band="B11")) / (
        s2_img.sel(band="B03") + s2_img.sel(band="B11")
    )

    # Add indices to the DataArray
    ndvi = ndvi.expand_dims({"band": ["NDVI"]})
    ndwi = ndwi.expand_dims({"band": ["NDWI"]})
    mndwi = mndwi.expand_dims({"band": ["MNDWI"]})
    s2_img = xr.concat([s2_img, ndvi, ndwi, mndwi], dim="band")

    # Return SCL as a separate DataArray

    return s2_img, scl


def open_s2_img(s2_id: str, output_dir: str | Path) -> tuple[xr.DataArray, xr.DataArray]:
    """Open a Sentinel-2 image from the output directory.

    Args:
        s2_id (str): The Sentinel-2 ID.
        output_dir (str | Path): Path to the base output directory.

    Returns:
        xr.DataArray: The opened Sentinel-2 image as an xarray DataArray.

    """
    output_dir = Path(output_dir)
    s2_img = xrio.open_rasterio(output_dir / "s2" / (s2_id + ".tif"))
    s2_img = cast("xr.DataArray", s2_img)

    s2_img, scl = preprocess_s2_img(s2_img)
    return s2_img.astype("float16"), scl


def create_random_samples(
    scl: xr.DataArray,
    output_dir: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Create random samples from the SCL band.

    Args:
        scl (xr.DataArray): The SCL band as an xarray DataArray.
        output_dir (str | None): Path to the base output directory.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the random samples.

    """
    # Convert SCL DataArray to DataFrame
    scl_df = scl.to_dataframe(name="SCL").reset_index()

    # Map SCL values to water/no-water/no-data classes
    scl_df["class"] = scl_df["SCL"].map(SCL_MAPPING)

    # Loop through each SCL code to sample the DataFrame
    scl_samp = gpd.GeoDataFrame()
    for class_id, size in SCL_SAMPLING.items():
        class_subset = scl_df[scl_df["SCL"] == class_id]
        if class_subset.empty or size == 0:
            continue

        class_samples = class_subset.sample(n=size, random_state=42)

        scl_samp = pd.concat([scl_samp, class_samples])

    scl_samp = scl_samp.reset_index().drop(columns=["band", "spatial_ref"])

    scl_samp = cast("gpd.GeoDataFrame", scl_samp)
    scl_samp = scl_samp.set_geometry(gpd.points_from_xy(scl_samp.x, scl_samp.y))

    if output_dir is not None:
        output_dir = Path(output_dir) / "training_samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        scl_samp.to_file(output_dir / "scl_samples.shp", index=False)

    return scl_samp


cache: dict[str, xr.DataArray | pd.DataFrame | None] = {
    "s2_df": None,
    "scl": None,
    "ckdtree": None,
}


def create_ref_mask(
    s2_img: xr.DataArray,
    output_dir: str | Path,
    _id: str,
    *,
    reprocess: bool = False,
) -> xr.DataArray:
    """Create a reference mask from the samples.

    Args:
        s2_img (xr.DataArray): The Sentinel-2 image as an xarray DataArray.
        scl (xr.DataArray): The SCL band as an xarray DataArray.
        output_dir (str | None): Path to the base output directory.

    Returns:
        xr.DataArray: The reference mask as an xarray DataArray.

    """
    output_dir = Path(output_dir)

    if reprocess or cache["s2_df"] is None:
        print("Assigning s2_ds to cache")
        s2_ds = s2_img.to_dataset(dim="band")
        s2_df = s2_ds.to_dataframe(dim_order=["y", "x"])
        s2_df = s2_df.drop(columns=["spatial_ref"])
        cache["s2_df"] = s2_df
    else:
        s2_df = cast("pd.DataFrame", cache["s2_df"])

    # Get all the coordinates from the img
    if reprocess or cache["ckdtree"] is None:
        img_coords = s2_df.reset_index()[["x", "y"]].values  # noqa: PD011
        tree = cKDTree(img_coords)  # type: ignore[]
        print("Assigning tree to cache")
        cache["ckdtree"] = tree
    else:
        tree = cache["ckdtree"]

    # Now, we have to align the coordinates of samples_df with those of s2_df
    # Load training samples
    samples_df = gpd.read_file(output_dir / "training_samples/scl_samples.shp")
    samples_df["x"] = samples_df.geometry.x
    samples_df["y"] = samples_df.geometry.y

    print(f"Using {len(samples_df)} training samples.")

    # Find the index for each sample with reference to s2_df
    _, idx = tree.query(samples_df[["x", "y"]].values, k=1)  # type: ignore[]

    # Now, tet the training dataset X and y
    train_x = cast("pd.DataFrame", s2_df.iloc[idx].reset_index(drop=True))
    train_y = samples_df["class"].reset_index(drop=True)

    print("Training the RF classifier...")
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42
    )
    clf.fit(train_x, train_y)

    print("Predicting the full image...")
    full_pred = clf.predict(s2_df)
    template = s2_img.sel(band=["B02"]).copy().squeeze()
    template.data = full_pred.reshape(template.shape).astype("uint8")

    template.rio.to_raster(output_dir / "ref_masks" / f"ref_mask_{_id}.tif", compress="DEFLATE")

    # Predict the classes for all pixels in the image
    return template
