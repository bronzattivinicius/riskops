"""Shared pytest fixtures for RiskOps tests."""

from pathlib import Path

import pandas as pd
import pytest

from riskops.rules.store import RuleStore

# Row order below is relied upon by name in several tests. Columns mirror
# a subset of the real BAF schema (see riskops.data.loader), with values
# hand-picked to exercise both single-field and multi-field conditions.
_APPLICATIONS = [
    {
        "fraud_bool": True, "foreign_request": 1, "email_is_free": 1,
        "velocity_6h": 6000, "name_email_similarity": 0.10,
        "credit_risk_score": 350, "device_distinct_emails_8w": 3,
    },
    {
        "fraud_bool": False, "foreign_request": 0, "email_is_free": 1,
        "velocity_6h": 3000, "name_email_similarity": 0.80,
        "credit_risk_score": 50, "device_distinct_emails_8w": 0,
    },
    {
        "fraud_bool": False, "foreign_request": 1, "email_is_free": 0,
        "velocity_6h": 2000, "name_email_similarity": 0.50,
        "credit_risk_score": 100, "device_distinct_emails_8w": 1,
    },
    {
        "fraud_bool": True, "foreign_request": 0, "email_is_free": 0,
        "velocity_6h": 8000, "name_email_similarity": 0.15,
        "credit_risk_score": 250, "device_distinct_emails_8w": 2,
    },
    {
        "fraud_bool": False, "foreign_request": 0, "email_is_free": 0,
        "velocity_6h": 1000, "name_email_similarity": 0.90,
        "credit_risk_score": -50, "device_distinct_emails_8w": 0,
    },
    {
        "fraud_bool": False, "foreign_request": 1, "email_is_free": 1,
        "velocity_6h": 500, "name_email_similarity": 0.60,
        "credit_risk_score": 10, "device_distinct_emails_8w": 0,
    },
    {
        "fraud_bool": True, "foreign_request": 0, "email_is_free": 1,
        "velocity_6h": 7000, "name_email_similarity": 0.05,
        "credit_risk_score": 380, "device_distinct_emails_8w": 2,
    },
    {
        "fraud_bool": False, "foreign_request": 0, "email_is_free": 0,
        "velocity_6h": 4000, "name_email_similarity": 0.70,
        "credit_risk_score": 0, "device_distinct_emails_8w": 1,
    },
    {
        "fraud_bool": False, "foreign_request": 0, "email_is_free": 1,
        "velocity_6h": 100, "name_email_similarity": 0.95,
        "credit_risk_score": -100, "device_distinct_emails_8w": 0,
    },
    {
        "fraud_bool": True, "foreign_request": 1, "email_is_free": 0,
        "velocity_6h": 9000, "name_email_similarity": 0.30,
        "credit_risk_score": 300, "device_distinct_emails_8w": 1,
    },
]


@pytest.fixture
def synthetic_applications_df() -> pd.DataFrame:
    """Small, hand-designed application dataset shaped like BAF.

    Returns:
        A DataFrame with a subset of real BAF columns (see
        :mod:`riskops.data.loader`) and known, hand-computed outcomes for
        the rule-evaluator and backtest-metrics tests.
    """
    return pd.DataFrame(_APPLICATIONS)


@pytest.fixture
def tmp_rule_store(tmp_path: Path) -> RuleStore:
    """A :class:`RuleStore` rooted in a fresh temporary directory.

    Args:
        tmp_path: Pytest-provided temporary directory, unique per test.

    Returns:
        An empty rule store.
    """
    return RuleStore(root=tmp_path)
