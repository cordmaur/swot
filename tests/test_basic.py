"""Test basic package functionality."""
import pytest
from swot_toolkit import __version__


def test_version() -> None:
    """Test that version is accessible."""
    assert __version__ == "0.1.0"


def test_package_imports() -> None:
    """Test that package can be imported."""
    import swot_toolkit
    assert swot_toolkit is not None