"""Water Fraction helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rasterio.enums import Resampling

from swot_toolkit.flags import Scenarios, get_filtering_params
from swot_toolkit.pipe2 import open_output_dir, open_roi
from swot_toolkit.pipe4 import open_opera_s1, open_opera_s2, open_ref_mask
from swot_toolkit.planetary import parse_s2_id
from swot_toolkit.swot import create_raster_mosaic_combined

if TYPE_CHECKING:
    import xarray as xr

# OPERA DSWx class codes
_OPERA_OPEN_WATER: int = 1
_OPERA_PARTIAL_WATER: int = 2  # values above this are no-data/cloud/snow/ocean

# Reference mask class codes
_REF_NO_DATA: int = 2


class WaterFraction:
    """Water fraction analysis for a single site and reference date.

    On construction a no-filter SWOT mosaic is built to serve as the shared
    100 m grid (``template``).  All other datasets (reference mask, OPERA S2,
    OPERA S1) are lazy-loaded and reprojected onto that grid in one step,
    using the target CRS directly so no intermediate reprojection is needed.

    Parameters
    ----------
    site:
        Name of the study site (must match an existing output directory).
    date:
        Reference date string, e.g. ``"2025-08-14"``.

    Attributes
    ----------
    site, date:
        The values passed at construction.
    template:
        No-filter SWOT water-fraction raster.  Defines the output grid for
        all derived products.

    """

    def __init__(self, site: str, date: str) -> None:
        """Build the shared 100 m SWOT grid template and prepare lazy caches."""
        self.site = site
        self.date = date

        # Site-level info.  open_roi is @cache so repeated calls are free.
        _, self._aoi, self._mosaic_df = open_roi(site)

        # Date-level info needed to locate opera/ref files.
        output_dir, _, s2_id = open_output_dir(site, date)
        self._output_dir = output_dir
        self._sensing_date = str(parse_s2_id(s2_id)["sensing_date"])

        # Build the shared SWOT grid template (no filtering applied).
        template, _, _ = create_raster_mosaic_combined(
            mosaic_df=self._mosaic_df,
            ref_date=date,
            aoi=self._aoi,
            exclude_flags=None,
            exclude_bitwise=None,
            exclude_geometric=False,
        )
        self.template: xr.DataArray = template

        # Lazy-load caches.
        self._ref_cache: xr.DataArray | None = None
        self._opera_s2_cache: xr.DataArray | None = None
        self._opera_s1_cache: xr.DataArray | None = None
        self._opera_s1_loaded: bool = False
        self._swot_cache: dict[tuple[str, str], tuple[xr.DataArray, list[xr.Dataset]]] = {}

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def swot(
        self,
        scenario_a: Scenarios.A,
        scenario_b: Scenarios.B,
    ) -> tuple[xr.DataArray, list[xr.Dataset]]:
        """Return the SWOT water-fraction raster and per-patch datasets.

        Results are cached per ``(scenario_a, scenario_b)`` pair for the
        lifetime of the instance.

        Parameters
        ----------
        scenario_a:
            Quality-class filtering level.  One of the keys in
            ``flags.SCENARIO_A_NAMES``.
        scenario_b:
            Bitwise / geometric filtering approach.  One of the keys in
            ``flags.SCENARIO_B_NAMES``.

        Returns
        -------
        tuple[xr.DataArray, list[xr.Dataset]]
            ``(raster, patches)`` where *raster* is the merged mean
            water-fraction DataArray and *patches* is the list of individual
            granule Datasets (each containing ``"water_frac"`` etc.).

        """
        key = (scenario_a, scenario_b)
        if key not in self._swot_cache:
            params = get_filtering_params(scenario_a, scenario_b)
            raster, patches, _ = create_raster_mosaic_combined(
                mosaic_df=self._mosaic_df,
                ref_date=self.date,
                aoi=self._aoi,
                **params,
            )

            # Make -1 (no-data) into NaN so it doesn't affect averages or metrics.
            raster = raster.where(raster != -1)
            # Clip negative values to 0 (can happen with some filtering scenarios, but not all).
            raster.data[raster.data < 0] = 0

            self._swot_cache[key] = (raster, patches)
        return self._swot_cache[key]

    def ref_mask(self) -> xr.DataArray:
        """Return the reference mask as a water fraction [0, 1] on the SWOT grid.

        Reference mask class codes: ``0`` = land → ``0.0``,
        ``1`` = water → ``1.0``, ``2`` = no-data → ``NaN``.

        The raw high-resolution mask is downsampled to the SWOT 100 m grid
        via ``Resampling.average``.

        """
        if self._ref_cache is None:
            raw = open_ref_mask(self._output_dir, self._sensing_date)
            water = raw.where(raw != _REF_NO_DATA).astype(np.float32)
            self._ref_cache = water.rio.reproject_match(
                self.template,
                resampling=Resampling.average,
            )

        if self._ref_cache is None:
            msg = "Reference mask could not be loaded."
            raise RuntimeError(msg)

        return self._ref_cache

    def opera_s2(self) -> xr.DataArray:
        """Return OPERA DSWx-HLS (S2) as a continuous water fraction on the SWOT grid.

        Class codes are mapped before averaging:
        ``0`` (land) → ``0.0``, ``1`` (open water) → ``1.0``,
        ``2`` (partial water) → ``0.5``,
        ``252/253/254/255`` (snow/cloud/ocean/no-data) → ``NaN``.

        The OPERA raster is first reprojected to the SWOT CRS, then
        ``reproject_match`` downsamples it to the 100 m SWOT grid.

        """
        if self._opera_s2_cache is None:
            raw = open_opera_s2(
                self._output_dir,
                self._sensing_date,
                crs=str(self.template.rio.crs),
            )
            self._opera_s2_cache = self._opera_to_water_frac(raw)
        return self._opera_s2_cache

    def opera_s1(self) -> xr.DataArray | None:
        """Return OPERA DSWx-S1 as a continuous water fraction on the SWOT grid.

        Same class-code mapping as :meth:`opera_s2`.  Returns ``None`` when
        no OPERA S1 file is available for the site/date.

        """
        if not self._opera_s1_loaded:
            raw = open_opera_s1(
                self._output_dir,
                self._sensing_date,
                crs=str(self.template.rio.crs),
            )
            if raw is not None:
                self._opera_s1_cache = self._opera_to_water_frac(raw)
            self._opera_s1_loaded = True
        return self._opera_s1_cache

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _opera_to_water_frac(self, opera_mask: xr.DataArray) -> xr.DataArray:
        """Map OPERA categorical classes to [0, 1] water fraction and reproject."""
        frac = opera_mask.astype(np.float32)
        frac = frac.where(opera_mask <= _OPERA_PARTIAL_WATER)  # NaN for 252/253/254/255
        frac = frac.where(opera_mask != _OPERA_PARTIAL_WATER, other=0.5)  # partial → 0.5
        return frac.rio.reproject_match(self.template, resampling=Resampling.average)

    def process_scenario(
        self,
        scenario_a: Scenarios.A,
        scenario_b: Scenarios.B,
    ) -> dict:
        """Compute residuals and metrics for a single (A, B) scenario pair.

        Returns a dict with keys: ``residuals``, ``Bias``, ``MAE``, ``RMSE``, ``N``.
        The ``residuals`` array is ``swot - ref`` over valid pixels only.
        """
        raster, _ = self.swot(scenario_a, scenario_b)
        sat_valid, ref_valid = WaterFraction.get_valid_pairs(raster, self.ref_mask())
        residuals = sat_valid - ref_valid
        return {
            "residuals": residuals,
            "Bias": float(residuals.mean()),
            "MAE": float(np.abs(residuals).mean()),
            "RMSE": float(np.sqrt(np.mean(residuals**2))),
            "N": len(residuals),
        }

    @staticmethod
    def get_valid_pairs(
        raster: xr.DataArray,
        ref_mask: xr.DataArray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract paired (sat, ref) pixel values, discarding invalid entries.

        Returns ``(sat_valid, ref_valid)`` as 1-D float16 arrays.
        Pixels where both ref and sat are zero are excluded.
        SAT NaNs are treated as no-water (0) to preserve bias information.
        """
        ref = np.asarray(ref_mask).squeeze().astype(np.float16)
        sat = np.asarray(raster).squeeze().astype(np.float16)

        if ref.shape != sat.shape:
            msg = f"Shape mismatch: ref={ref.shape}, sat={sat.shape}"
            raise ValueError(msg)

        ref = ref.ravel()
        sat = sat.ravel()

        sat = np.nan_to_num(sat, nan=0)
        sat[sat > 1] = 1
        sat[sat < 0] = 0

        mask = np.isfinite(ref) & np.isfinite(sat)
        mask &= (ref > 0) | (sat > 0)

        return sat[mask], ref[mask]

    def clear_cache(self) -> None:
        """Clear all cached datasets (except the SWOT template)."""
        self._ref_cache = None
        self._opera_s2_cache = None
        self._opera_s1_cache = None
        self._opera_s1_loaded = False
        self._swot_cache.clear()
