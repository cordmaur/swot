"""Analysis Module."""

from os import PathLike
from pathlib import Path

import pandas as pd


def check_dir(dir_path: PathLike[str]) -> Path:
    """Check if a directory exists and is actually a directory.

    Args:
        dir_path (PathLike[str]): Path to the directory to check.

    Returns:
        Path: The validated directory path as a Path object.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path exists but is not a directory.

    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        msg = f"Base directory not found: {dir_path}"
        raise FileNotFoundError(msg)

    if not dir_path.is_dir():
        msg = f"Base directory is not a directory: {dir_path}"
        raise NotADirectoryError(msg)

    return dir_path


def open_sites_and_dates(base_dir: PathLike[str]) -> dict[str, list[str]]:
    """Open the sites and dates from the base directory.

    It will automatically crawl the base_dir for the sites and dates.

    Args:
        base_dir (PathLike[str]): Path to the base directory where the results are stored.

    Returns:
        dict[str, list[str]]: A dictionary with the sites as keys and the list of
        dates as values.

    """
    base_dir = check_dir(base_dir)

    # The sites must be directories in the base dir
    sites = [f for f in base_dir.iterdir() if f.is_dir()]

    # Now, for each site we look for the available dates
    sites_dates: dict[str, list[str]] = {}
    for site in sites:
        # The dates must also be directories in the site dir and begin with a digit
        dates = [f for f in site.iterdir() if (f.is_dir() and f.name[0].isdigit())]
        sites_dates[site.name] = [d.name for d in dates]

    return sites_dates


def open_results(base_dir: PathLike[str], file_pattern: str = "") -> pd.DataFrame:
    """Open the results from the processing (Pipes) steps.

    It will automatically crawl the base_dir for the sites and dates.
    Then, each parquet will be assigned to a multi-index dataframe with site and date.

    Args:
        base_dir (PathLike[str]): Path to the base directory where the results are stored.
        file_pattern (str, optional): A glob pattern to filter the files to open.

    Returns:
        pd.DataFrame: A DataFrame containing the results for each site and date.

    """
    base_dir = check_dir(base_dir)
    sites_dates = open_sites_and_dates(base_dir)

    results: list[pd.DataFrame] = []
    for site, dates in sites_dates.items():
        for date in dates:
            # search for the file
            results_path = base_dir / site / date
            files = list(results_path.glob(f"results{file_pattern}.parquet"))

            if len(files) > 1:
                msg = f"More than one results file found for site {site} and date {date}: {files}"
                raise FileExistsError(msg)

            if len(files) == 0:
                msg = f"No results file found for site {site} and date {date}"
                raise FileNotFoundError(msg)

            df = pd.read_parquet(files[0])

            df.index = pd.MultiIndex.from_product(
                [[site + " " + date], df.index],
                names=["site", "metric"],
            )

            results.append(df)

    return pd.concat(results, axis=0)
