"""Example usage of the plot_opera_array function."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.swot_toolkit.opera import plot_opera_array

# Create a sample OPERA-like array with the expected values
sample_data = np.array([[0, 1, 2, 252], [1, 1, 253, 254], [2, 0, 1, 255], [252, 253, 254, 0]])

# Create xarray DataArray with coordinates (like OPERA data would have)
array = xr.DataArray(
    sample_data,
    dims=["y", "x"],
    coords={
        "x": np.linspace(-120.0, -119.9, 4),
        "y": np.linspace(37.0, 37.1, 4),
    },
)

# Example 1: Plot with default settings
fig, ax = plt.subplots(figsize=(10, 8))
plot_opera_array(array, ax=ax, title="OPERA Water Classification")
plt.tight_layout()
plt.show()

# Example 2: Plot without colorbar
fig, ax = plt.subplots(figsize=(10, 6))
plot_opera_array(array, ax=ax, title="OPERA Mask - No Colorbar", add_colorbar=False)
plt.tight_layout()
plt.show()

# Example 3: Use with existing subplot layout
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot the same data with different titles
plot_opera_array(array, ax=axes[0], title="Original OPERA Data")
plot_opera_array(array, ax=axes[1], title="Same Data - Different View", add_colorbar=False)

plt.tight_layout()
plt.show()
