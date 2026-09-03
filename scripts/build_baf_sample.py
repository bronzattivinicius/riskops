"""One-off script: builds a small, stratified sample of the BAF dataset.

The full ``Base.csv`` (1,000,000 rows) requires Kaggle credentials to
download and is too large to commit to the repository. This script
produces a small, git-trackable sample that preserves the original fraud
rate, so that downstream notebooks (e.g. the course deliverables) can run
without needing Kaggle access at all.

Run once, from the repository root::

    poetry run python scripts/build_baf_sample.py
"""

from pathlib import Path

import pandas as pd

from riskops.paths import DATA_RAW_DIR, PROJECT_ROOT

SAMPLE_SIZE = 20_000
RANDOM_STATE = 42
OUTPUT_PATH = PROJECT_ROOT / "data" / "sample" / "baf_sample.csv"


def build_sample(
    source_csv: Path = DATA_RAW_DIR / "Base.csv",
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Builds a stratified sample of the BAF dataset by fraud label.

    Args:
        source_csv: Path to the full BAF Base.csv.
        sample_size: Total number of rows in the sample.
        random_state: Seed controlling the sampling, for reproducibility.

    Returns:
        The sampled DataFrame, with the same columns as the source and the
        original fraud rate preserved (within rounding).
    """
    df = pd.read_csv(source_csv)
    fraud_fraction = df["fraud_bool"].mean()

    fraud_count = round(sample_size * fraud_fraction)
    legit_count = sample_size - fraud_count

    fraud_rows = df[df["fraud_bool"] == 1].sample(n=fraud_count, random_state=random_state)
    legit_rows = df[df["fraud_bool"] == 0].sample(n=legit_count, random_state=random_state)

    sample = pd.concat([fraud_rows, legit_rows]).sample(frac=1, random_state=random_state)
    return sample.reset_index(drop=True)


if __name__ == "__main__":
    sample_df = build_sample()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {len(sample_df)} rows to {OUTPUT_PATH}")
    print(f"fraud rate in sample: {sample_df['fraud_bool'].mean():.4f}")
