"""Load the Bank Account Fraud (BAF) dataset into a pandas DataFrame."""

from pathlib import Path

import pandas as pd

from riskops.data.download import DEFAULT_VARIANT_FILENAME
from riskops.paths import DATA_RAW_DIR

_DTYPES = {
    "fraud_bool": "int8",
    "income": "float64",
    "name_email_similarity": "float64",
    "prev_address_months_count": "int32",
    "current_address_months_count": "int32",
    "customer_age": "int16",
    "days_since_request": "float64",
    "intended_balcon_amount": "float64",
    "payment_type": "category",
    "zip_count_4w": "int32",
    "velocity_6h": "float64",
    "velocity_24h": "float64",
    "velocity_4w": "float64",
    "bank_branch_count_8w": "int32",
    "date_of_birth_distinct_emails_4w": "int32",
    "employment_status": "category",
    "credit_risk_score": "int32",
    "email_is_free": "int8",
    "housing_status": "category",
    "phone_home_valid": "int8",
    "phone_mobile_valid": "int8",
    "bank_months_count": "int32",
    "has_other_cards": "int8",
    "proposed_credit_limit": "float64",
    "foreign_request": "int8",
    "source": "category",
    "session_length_in_minutes": "float64",
    "device_os": "category",
    "keep_alive_session": "int8",
    "device_distinct_emails_8w": "int8",
    "device_fraud_count": "int8",
    "month": "int8",
}
"""Column dtypes for Base.csv, confirmed against the actual downloaded file.

``-1`` is used throughout the BAF suite as a sentinel for missing values in
several numeric columns (e.g. ``prev_address_months_count``,
``bank_months_count``, ``session_length_in_minutes``) rather than ``NaN``;
callers should account for this when filtering or aggregating those
columns.
"""

_BOOLEAN_COLUMNS = [
    "fraud_bool",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "has_other_cards",
    "foreign_request",
    "keep_alive_session",
]


def _find_default_csv(directory: Path) -> Path | None:
    """Locates the BAF Base CSV in a directory.

    Args:
        directory: Directory to search.

    Returns:
        Path to the expected variant filename if present, otherwise the
        first ``*.csv`` file found, or ``None`` if the directory has none.
    """
    candidate = directory / DEFAULT_VARIANT_FILENAME
    if candidate.exists():
        return candidate
    matches = sorted(directory.glob("*.csv"))
    return matches[0] if matches else None


def load_baf(path: Path | None = None) -> pd.DataFrame:
    """Loads the BAF dataset into a DataFrame with explicit dtypes.

    Args:
        path: Path to a BAF variant CSV. Defaults to searching ``data/raw``.

    Returns:
        The loaded applications, with binary 0/1 columns (including
        ``fraud_bool``) cast to ``bool``.

    Raises:
        FileNotFoundError: If ``path`` is not given and no CSV is found in
            ``data/raw``.
    """
    resolved_path = path or _find_default_csv(DATA_RAW_DIR)
    if resolved_path is None or not resolved_path.exists():
        raise FileNotFoundError(
            f"BAF CSV not found in {DATA_RAW_DIR}. Run "
            "`python -m riskops.data.download` first (requires Kaggle "
            "credentials, see .env.example)."
        )

    df = pd.read_csv(resolved_path, dtype=_DTYPES)
    for column in _BOOLEAN_COLUMNS:
        df[column] = df[column].astype(bool)
    return df