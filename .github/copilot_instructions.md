# GitHub Copilot Instructions for SWOT Toolkit

## Code Quality Standards

### Type Hints
- **ALL** code must include proper type hints
- Use `from typing import` for complex types (List, Dict, Optional, Union, etc.)
- Function parameters, return types, and class attributes must be typed
- Use `-> None` for functions that don't return values
- Example:
```python
from typing import List, Optional
import numpy as np
import xarray as xr

def process_swot_data(
    dataset: xr.Dataset, 
    variables: List[str], 
    quality_threshold: Optional[float] = None
) -> xr.Dataset:
    """Process SWOT dataset with specified variables.

    Parameters:
    -----------
    dataset: Path to the output directory.
    variables: List of SWOT variables to load into the dataset
    quality_threshold: Number indicating the minimum quality to be accepcted. Optional 

    Returns:
    --------
    Dataset with swot data loaded

    """
    # Implementation here
    return processed_dataset
```

### Linting and Formatting
- Code must pass **strict** Pylance and MyPy checks without warnings
- Use Ruff for formatting and linting (configured in pyproject.toml)
- Run `ruff check .` and `ruff format .` before committing
- Run `mypy src/` to ensure type safety

### File I/O and Encoding
- **ALWAYS** specify encoding when opening files with `open()`
- Use `encoding="utf-8"` as the default encoding for text files
- Example: `with open(file_path, encoding="utf-8") as f:`
- This prevents Pylint warning W1514:unspecified-encoding

### Code Style Guidelines
- Follow PEP 8 conventions
- Use descriptive variable and function names
- Maximum line length: 88 characters
- Use double quotes for strings
- Add docstrings to all public functions and classes using Google/NumPy style
- Check for type annotations that can be rewritten based on PEP 604 syntax.
- Avoid specifying long messages outside the exception classRuffTRY003


### Documentation
- All public functions must have comprehensive docstrings
- Include parameters descriptions and return value documentation
- Follow Numpydoc style. DO NOT ADD types in docstring, as they are already declared in the signature.
- Multi-line docstring summary should start at the first line RuffD212
- Dont forget to add blank line after last section of the docstring RuffD413

- Add examples in docstrings when helpful
- Use type hints instead of documenting types in docstrings
- make sure the method has a return type
- make sure multi lines docstrings start in the first line
- make sure to add blank lines after each docstring section

### Error Handling
- Use specific exception types rather than generic `Exception`
- Exception must not use a string literal, assign to variable first RuffEM101
- Add proper error messages that help with debugging
- Use type guards and assertions where appropriate

### Testing
- Write tests for all new functionality
- Use pytest for testing framework
- Aim for high test coverage
- Include type annotations in test files

### SWOT-Specific Guidelines
- Use xarray for handling NetCDF/HDF5 SWOT data
- Use geopandas for geospatial operations
- Follow SWOT data conventions and naming
- Handle missing data appropriately with proper masking

### Example Function Structure
```python
import xarray as xr
import numpy as np
from typing import Optional, Tuple

def load_swot_pixc(
    file_path: str, 
    group: str = "pixel_cloud",
    variables: Optional[List[str]] = None
) -> xr.Dataset:
    """Load SWOT PIXC data from NetCDF file.
    
    Parameters
    ----------
    file_path : str
        Path to the SWOT NetCDF file
    group : str, default "pixel_cloud"
        NetCDF group to read
    variables : List[str], optional
        Specific variables to load. If None, loads all variables.
        
    Returns
    -------
    xr.Dataset
        Loaded SWOT dataset
        
    Raises
    ------
    FileNotFoundError
        If the specified file doesn't exist
    ValueError
        If the specified group doesn't exist in the file

    """
    # Implementation with proper error handling
```




## Development Workflow
1. Install package in development mode: `pip install -e .[dev]`
2. Run type checking: `mypy src/`
3. Run linting: `ruff check .`
4. Run formatting: `ruff format .`
5. Run tests: `pytest`
6. All checks must pass before committing code