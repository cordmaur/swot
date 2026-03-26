# SWOT Toolkit — Project Guidelines

## Project Overview

Python package for processing and analyzing surface water data from the SWOT (Surface Water and Ocean Topography) satellite. The core workflow is: **AOI (KML) → search SWOT/Sentinel-2/OPERA data → download → classify → calculate accuracy metrics**.

## Build and Test

```bash
# Install in editable mode (includes dev deps)
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=swot_toolkit --cov-report=html

# Lint
ruff check src/
```

## Architecture

```
src/swot_toolkit/   # Installable package (import as swot_toolkit)
nbs/                # Jupyter notebooks (exploratory / article figures)
nbs/Article/        # Publication-specific notebooks
nbs/Main_Pipeline/  # Step-by-step pipeline notebooks (13–18)
tests/              # pytest suite
```

**Key modules:**

| Module | Role |
|--------|------|
| `swot.py` | Core SWOT search, download, mosaic creation, quality-flag masking |
| `flags.py` | 24 bitwise quality flags + `mask_by_flags()` |
| `metrics.py` | Precision, recall, F1, IoU, kappa — always reproject to match grids first |
| `pipe1–4.py` | Pipeline steps: date matching → download → S2 reference mask (RandomForest) → metrics |
| `opera.py` | OPERA water mask retrieval |
| `planetary.py` | Sentinel-2 search via Planetary Computer STAC |
| `analysis.py` | High-level workflow orchestration |
| `utils.py` | `project_root()` helper, temporal alignment |

## Conventions

**Python version**: 3.10+ (use `match`, `TypeAlias`, `|` union syntax).

**Types**: Strict pyright — use `TYPE_CHECKING` guards, `cast()` for narrowing, `PathLike[str]` for path inputs.

**Paths**: Always use `pathlib.Path`. Never pass bare strings to internal functions.

**Geospatial stack**: `xarray.DataArray` + rioxarray for rasters; `geopandas.GeoDataFrame` for vectors; `shapely.geometry` for geometries. Reproject with `rio.reproject_match()` before any pixel-level comparison.

**Imports**: Relative within the package (`from .flags import …`). Expensive operations use `@functools.cache`.

**Linting (ruff)**: Line length 99, `select = ["ALL"]`, `ignore = ["EXE002", "T201"]`. Print statements are allowed.

**Notebooks**: Notebooks in `nbs/` are excluded from pyright type checking — don't add type stubs there.

## Data Paths & Credentials

Fixed runtime paths (not configurable at package level, defined in `pipe2.py` / `analysis.py`):

- `/data/swot/downloads/` — download cache
- `/data/swot/output/` — pipeline output root
- `/data/swot/AOIs/` — KML area-of-interest files

NASA EarthData credentials are read from `credentials.json` at project root (located via `utils.project_root()`). Schema: `{"EARTHDATA_USERNAME": "…", "EARTHDATA_PASSWORD": "…"}`. Never commit credentials.

## Common Pitfalls

- Always apply quality-flag masks before computing metrics — see `flags.py` flag constants and `mask_by_flags()`.
- SWOT and OPERA/Sentinel-2 grids differ in CRS and resolution; call `rio.reproject_match()` before any comparison.
- `create_mosaic_df()` accepts a `max_delta` parameter (days) to control temporal tolerance between SWOT passes.
- `auth_earthaccess()` in `swot.py` must be called before any SWOT or OPERA download.
