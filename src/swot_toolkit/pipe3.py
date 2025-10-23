"""Pipeline 3: Create Reference Mask."""

from functools import cache  # pyright: ignore[reportAssignmentType]
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as xrio
import xarray as xr
from scipy.spatial import cKDTree  # type: ignore[]
from skimage.morphology import (
    binary_dilation,  # type: ignore[]
    closing,  # type: ignore[]
    disk,  # type: ignore[]
    remove_small_objects,  # type: ignore[]
)
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

@cache # type: ignore[]
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
    s2_img.attrs["native-id"] = s2_id
    scl.attrs["native-id"] = s2_id

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

        class_samples = class_subset.sample(n=min(size, len(class_subset)), random_state=42)

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
        output_dir (str | None): Path to the base output directory.
        _id (str): The Sentinel-2 ID.
        reprocess (bool): Whether to reprocess the data or use cached values.

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
    samples_df = samples_df.dropna(subset=["class"])
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
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
    )
    clf.fit(train_x, train_y)

    print("Predicting the full image...")
    full_pred = clf.predict(s2_df)
    template = s2_img.sel(band=["B02"]).copy().squeeze()
    template.data = full_pred.reshape(template.shape).astype("uint8")

    template.rio.to_raster(output_dir / "ref_mask" / f"ref_mask_{_id}.tif", compress="DEFLATE")

    # Predict the classes for all pixels in the image
    return template


def create_shadow_cast_kernel(dx: int, dy: int) -> np.ndarray:
    """Create a kernel to cast the shadow using dilation.

    We will Consider dx and dy the max displacement of the shadow in pixels
    Positive dx values shadow to the right, negative to the left
    Positive dy values shadow downwards, negative upwards

    Args:
        dx (int): The x direction.
        dy (int): The y direction.

    Returns:
        np.ndarray: The dilation direction kernel.

    """
    # Get the diameter of the kernel
    diameter = max(abs(dx), abs(dy))

    # First we create an empty kernel, where `side` is the side of the square kernel
    side = 2 * diameter + 1

    kernel = np.zeros((side, side))

    # Now we set a pixel from the center of the kernel to the (dx, dy) point
    kernel[diameter + dy, diameter + dx] = 1

    return kernel


def create_cloud_shadow_mask(
    ref_mask: xr.DataArray,
    shadow_displacement: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Create a cloud shadow mask from the reference mask.

    Args:
        ref_mask (xr.DataArray): The reference mask as an xarray DataArray.
        shadow_displacement (tuple[int, int]): The shadow displacement in pixels (dx, dy).

    Returns:
        np.ndarray: The cloud mask as a numpy array.
        np.ndarray: The cloud shadow mask as a numpy array.

    """
    # First let's separate the cloud mask
    cloud_mask = ref_mask.data == 2

    # Preprocess the clouds
    # The clouds must be DILATED to cover the edges better, but first, let's remove any noise
    clean_cloud = cast("np.ndarray", remove_small_objects(cloud_mask, min_size=15, connectivity=1))
    # Now we can dilate the clouds
    clean_cloud = cast("np.ndarray", binary_dilation(clean_cloud, disk(5)))

    # With the clouds cleaned, we can now create the shadow mask
    dx, dy = shadow_displacement
    kernel = create_shadow_cast_kernel(dx, dy)
    cloud_shadow = cast("np.ndarray", binary_dilation(clean_cloud, kernel))

    # Now, we have the shadow possibly overlapping with clouds, so we remove the cloud areas from 
    # the shadow
    cloud_shadow = ~clean_cloud & cloud_shadow

    return clean_cloud, cloud_shadow


def create_water_mask(ref_mask: xr.DataArray) -> np.ndarray:
    """Create a water mask from the reference mask.

    Args:
        ref_mask (xr.DataArray): The reference mask as an xarray DataArray.

    Returns:
        np.ndarray: The water mask as a numpy array.

    """
    # First, let's separate the foreground into water and clouds
    water_mask = ref_mask.data == 1

    # For water, we will remove 0.5ha objects. Each pixel is a 10x10=100m2, so 0.5ha=5,000m2=50 pixels
    clean_water = cast("np.ndarray", remove_small_objects(water_mask, min_size=50, connectivity=1))

    # Then, we will remove narrow rivers with only 1 pixel width
    # clean_water = opening(clean_water, footprint_rectangle((3, 3)))  # noqa: ERA001

    # Perform a closing to fill small gaps in water bodies
    return cast("np.ndarray", closing(clean_water, disk(3)))


def post_process_ref_mask(
    ref_mask: xr.DataArray, shadow_displacement: tuple[int, int]
) -> xr.DataArray:
    """Post-process the reference mask to create the final mask.

    Args:
        ref_mask (xr.DataArray): The reference mask as an xarray DataArray.
        shadow_displacement (tuple[int, int]): The shadow displacement in pixels (dx, dy).

    Returns:
        xr.DataArray: The final processed mask as an xarray DataArray.

    """
    # Create cloud and shadow masks
    cloud_mask, cloud_shadow_mask = create_cloud_shadow_mask(ref_mask, shadow_displacement)

    # Create water mask
    water_mask = create_water_mask(ref_mask)

    # Now we can create the final mask
    final_mask = np.zeros_like(ref_mask)

    final_mask[water_mask] = 1
    final_mask[cloud_mask] = 2
    final_mask[(water_mask == 1) & (cloud_shadow_mask == 1)] = 0

    ref_mask.data = final_mask

    return ref_mask
