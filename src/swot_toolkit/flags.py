"""Quality Flags."""

import numpy as np
import xarray as xr

# Quality flag definitions from the table in:
# https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_Raster_100m_2.0

QUALITY_FLAGS = {
    "classification_qual_suspect": 1,
    "geolocation_qual_suspect": 2,
    "water_fraction_suspect": 3,
    "large_uncert_suspect": 5,
    "dark_water_suspect": 6,
    "bright_land": 7,
    "low_coherence_water_suspect": 8,
    "specular_ringing_prior_water_suspect": 9,
    "specular_ringing_prior_land_suspect": 10,
    "few_pixels": 12,
    "far_range_suspect": 13,
    "near_range_suspect": 14,
    "classification_qual_degraded": 18,
    "geolocation_qual_degraded": 19,
    "value_bad": 24,
    "outside_data_window": 26,
    "no_pixels": 28,
    "outside_scene_bounds": 29,
    "inner_swath": 30,
    "missing_karin_data": 31,
}


def mask_by_flags(
    flag_array: xr.DataArray,
    flags: list[str],
) -> np.ndarray:
    """Create a boolean mask for pixels that have any of the specified quality flags set.

    Args:
        flag_array: Array containing quality flag values (bitwise flags).
        flags: List of flag names to check for.

    Returns:
        np.ndarray: Boolean mask where True indicates pixels with any of the specified flags set.

    """
    # Create a mask for pixels matching any of the flags
    mask = np.zeros(flag_array.shape, dtype=bool)
    flag_array_np = flag_array.fillna(0).data.astype("uint32")

    for flag_name in flags:
        # Get the bit position for the flag name
        bit = QUALITY_FLAGS.get(flag_name)
        if bit is not None:
            mask |= (flag_array_np & (1 << bit)) != 0

    return mask


# Method to select pixels based on quality flags
def select_pixels_by_quality(
    raster: xr.DataArray,
    flag_band: str = "water_area_qual_bitwise",
    include_flags: list[str] | None = None,
) -> tuple[xr.DataArray, np.ndarray]:
    """Select pixels from the raster based on quality flags.

    Args:
        raster: xarray Dataset containing the raster data and quality flags.
        flag_band: string representing the quality flag band to use.
        include_flags: list of strings representing flags to include.

    """
    if include_flags is None:
        include_flags = []

    # Create a mask for pixels to include using the new mask_by_flags function
    include_mask = mask_by_flags(raster[flag_band], include_flags)

    # Return a masked array where included pixels are masked
    return raster.where(include_mask), include_mask


def decode_flags(value: int) -> list[str]:
    """Decode integer quality flag into a list of strings.

    Args:
        value (int): Integer quality flag value.

    category must be one of: 'wse_qual_bitwise', 'water_area_qual_bitwise', 'sig0_qual_bitwise'.

    """
    return [name for name, bit in QUALITY_FLAGS.items() if value & (1 << bit)]


def count_pixels_by_flag(array: np.ndarray, flag_name: str) -> float:
    """Count the number of pixels in a raster that have a specific quality flag set.

    Args:
        array (np.ndarray): The input array with quality flags.
        flag_name (str): The name of the quality flag to count.

    Returns:
        float: The fraction of pixels with the specified quality flag set.

    """
    # Get the bit position for the flag
    bit = QUALITY_FLAGS[flag_name]
    # Convert raster data to int for bitwise operation
    data = array.astype(int)
    # Create a mask where the flag bit is set
    mask = (data & (1 << bit)) != 0
    # Compute the fraction of pixels with the flag set
    return mask.sum() / array.size


def decode_swot_flag(flag_value: int, *, verbose: bool = True) -> list[str]:
    """Decode SWOT quality flags from a decimal value.

    Parameters
    ----------
    flag_value : int
        Decimal flag value to decode
    verbose : bool
        Whether to print detailed information

    Returns
    -------
    list
        List of active flag names

    """
    if verbose:
        print(f"Flag value: {flag_value}")
        print(f"Binary: {bin(flag_value)} ({flag_value:032b})")
        print("-" * 50)

    active_flags: list[str] = []

    for flag_name, bit_position in QUALITY_FLAGS.items():
        # Check if the bit at this position is set
        if flag_value & (1 << bit_position):
            active_flags.append(flag_name)
            if verbose:
                print(f"✓ Bit {bit_position:2d}: {flag_name}")
        elif verbose:
            print(f"  Bit {bit_position:2d}: {flag_name}")

    for bit in range(32):
        if flag_value & (1 << bit) and bit not in QUALITY_FLAGS.values():
            # Found a bit, now check if it exists in the QUALITY_FLAGS
            print(f"⚠️ Bit {bit:2d}: Unknown flag")

    if verbose:
        print(f"\nActive flags: {len(active_flags)}")
        for flag in active_flags:
            print(f"  - {flag}")

    return active_flags


def decode_active_flags(flags_array: xr.DataArray) -> set[str]:
    flag_values = np.unique(flags_array.data)
    all_active_flags: set[str] = set()
    for value in flag_values:
        if np.isnan(value):
            continue
        active_flags = decode_swot_flag(int(value), verbose=False)
        all_active_flags.update(active_flags)

    return all_active_flags
