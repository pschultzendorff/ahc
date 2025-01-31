import logging
import pathlib
import zipfile

import numpy as np
import porepy as pp
import requests
from tpf.utils.constants_and_typing import FEET

logger = logging.getLogger(__name__)

DATA_DIR: pathlib.Path = pathlib.Path(__file__).parent / "data"
ZIP_FILENAME: str = "por_perm_case2a.zip"
URL: str = "https://www.spe.org/web/csp/datasets/por_perm_case2a.zip"

X_LENGTH: float = 1200 * FEET
Y_LENGTH: float = 2200 * FEET


def download_spe10_data(data_dir: pathlib.Path, zip_filepath: pathlib.Path) -> None:
    """Download the SPE CSP10 porosity and permeability data, and store them locally."""
    # Ensure the destination directory exists.
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download the ZIP file.
    logger.info(f"Downloading dataset from {URL}")
    response = requests.get(URL)
    response.raise_for_status()
    with open(zip_filepath, "wb") as f:
        f.write(response.content)
    logger.info("Download completed.")

    # Extract the ZIP file.
    extracted_files: list[str] = []
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(data_dir)
        extracted_files = zip_ref.namelist()
    logger.info(f"Extracted files: {extracted_files}")

    # Locate the .dat files for permeability and porosity.
    perm_file: pathlib.Path | None = None
    poro_file: pathlib.Path | None = None
    for filename in extracted_files:
        if "perm" in filename.lower():
            perm_file = data_dir / filename
        elif "phi" in filename.lower():
            poro_file = data_dir / filename

    if perm_file is None or poro_file is None:
        raise FileNotFoundError(
            "Could not locate permeability or porosity data files in the downloaded contents."
        )

    zip_filepath.unlink()
    logger.info("Downloaded files cleaned up.")


def load_spe10_data(data_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the SPE CSP10 data into :class:`~numpy.ndarray`.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Permeability data array.
            - np.ndarray: Porosity data array.

    """
    # Ensure the destination directory exists.
    data_dir.mkdir(parents=True, exist_ok=True)

    perm_file: pathlib.Path | None = None
    poro_file: pathlib.Path | None = None
    for filename in data_dir.iterdir():
        if "perm" in str(filename).lower():
            perm_file = data_dir / filename
        elif "phi" in str(filename).lower():
            poro_file = data_dir / filename
    if perm_file is None or poro_file is None:
        logger.info("Permeability and porosity data files not found. Downloading...")
        download_spe10_data(data_dir, data_dir / ZIP_FILENAME)

    logger.info("Loading permeability and porosity data.")
    perm_data = np.loadtxt(str(perm_file)).reshape(3, 85, 220, 60)  # unit: [mD]
    # Convert permeability to m^2.
    perm_data *= pp.MILLIDARCY
    poro_data = np.loadtxt(str(poro_file)).reshape(85, 220, 60)  # unit: [-]

    return perm_data, poro_data
