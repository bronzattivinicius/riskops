"""Filesystem path constants shared across the RiskOps package."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
"""Absolute path to the repository root."""

DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
"""Directory holding unmodified downloaded datasets (git-ignored)."""

DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
"""Directory holding derived/processed datasets (git-ignored)."""

RULE_REGISTRY_DIR: Path = PROJECT_ROOT / "rule_registry"
"""Directory holding the versioned, git-tracked rule registry."""
