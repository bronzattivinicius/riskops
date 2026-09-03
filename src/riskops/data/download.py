"""Download the Bank Account Fraud (BAF) dataset from Kaggle.

The BAF suite (Jesus et al., NeurIPS 2022) ships six 1M-row variants of a
synthetic bank-account-opening-fraud dataset. Only the unbiased "Base"
variant is downloaded by default.

Run as a module to download it into ``data/raw``::

    python -m riskops.data.download
"""

import os
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from riskops.paths import DATA_RAW_DIR

KAGGLE_DATASET = "sgpjesus/bank-account-fraud-dataset-neurips-2022"
"""Kaggle dataset identifier for the BAF suite."""

DEFAULT_VARIANT_FILENAME = "Base.csv"
"""Filename of the unbiased BAF variant within the dataset."""


class KaggleCredentialsError(RuntimeError):
    """Raised when Kaggle API credentials cannot be found or are invalid."""


def _has_kaggle_credentials() -> bool:
    """Checks whether Kaggle API credentials are configured.

    Loads variables from a ``.env`` file into the process environment, if
    present, before checking.

    Returns:
        ``True`` if ``KAGGLE_USERNAME``/``KAGGLE_KEY`` are set, or a
        ``kaggle.json`` credentials file exists.
    """
    load_dotenv()
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR") or str(Path.home() / ".kaggle")
    return (Path(config_dir) / "kaggle.json").exists()


def download_baf(
    dest_dir: Path = DATA_RAW_DIR,
    variant_filename: str = DEFAULT_VARIANT_FILENAME,
    force: bool = False,
) -> Path:
    """Downloads and extracts one variant of the BAF dataset from Kaggle.

    Args:
        dest_dir: Directory to download and extract the dataset into.
        variant_filename: Name of the variant file to download, e.g.
            ``"Base.csv"`` or ``"Variant I.csv"``.
        force: If ``True``, re-download even if the file already exists.

    Returns:
        Path to the extracted CSV file.

    Raises:
        KaggleCredentialsError: If no Kaggle credentials are configured,
            or authentication fails.
        RuntimeError: If the download itself fails.
        FileNotFoundError: If the download succeeds but the expected CSV
            cannot be located afterward.
    """
    if not _has_kaggle_credentials():
        raise KaggleCredentialsError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY in "
            "your .env (see .env.example) or place a token at "
            "~/.kaggle/kaggle.json (generate one at "
            "https://www.kaggle.com/settings -> API -> Create New Token)."
        )

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit as exc:
        raise KaggleCredentialsError(
            "Kaggle authentication failed; the configured credentials appear invalid."
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / variant_filename
    if csv_path.exists() and not force:
        return csv_path

    try:
        api.dataset_download_file(
            KAGGLE_DATASET, variant_filename, path=str(dest_dir), quiet=False, force=force
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to download {variant_filename!r} from Kaggle dataset {KAGGLE_DATASET!r}: {exc}"
        ) from exc

    zip_path = dest_dir / f"{variant_filename}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dest_dir)
        zip_path.unlink()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"download completed but {csv_path} was not found; the dataset's file "
            "layout may have changed"
        )
    return csv_path


if __name__ == "__main__":
    print(f"downloaded BAF dataset to {download_baf()}")