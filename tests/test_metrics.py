"""Tests for riskops.metrics.backtest."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from riskops.metrics.backtest import compare_rulesets, compute_metrics
from riskops.rules.schema import Condition, ConditionGroup, Operator, Rule, RuleStatus

_NOW = datetime.now(UTC)


def _make_rule(rule_id: str, field: str, threshold: float) -> Rule:
    """Builds a minimal 'field > threshold' Rule for metrics tests.

    Args:
        rule_id: Id to assign to the rule.
        field: Field to compare.
        threshold: Threshold value for a 'gt' comparison.

    Returns:
        A rule with placeholder metadata, suitable only for tests.
    """
    return Rule(
        id=rule_id,
        name=rule_id,
        version=1,
        status=RuleStatus.CANDIDATE,
        description="test rule",
        logic=ConditionGroup(conditions=[Condition(field=field, operator=Operator.GT, value=threshold)]),
        created_at=_NOW,
        updated_at=_NOW,
        created_by="test",
        updated_by="test",
    )


def test_compute_metrics_known_confusion_matrix() -> None:
    """compute_metrics matches hand-computed rates for a known matrix.

    3 fraud rows, 2 caught; 7 legitimate rows, 1 wrongly flagged.
    """
    y_true = pd.Series([True, True, True, False, False, False, False, False, False, False])
    y_pred = pd.Series([True, True, False, True, False, False, False, False, False, False])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_negatives == 6
    assert metrics.detection_rate == pytest.approx(2 / 3)
    assert metrics.false_positive_rate == pytest.approx(1 / 7)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.approval_rate == pytest.approx(0.7)


def test_compute_metrics_handles_no_fraud_and_no_flags() -> None:
    """compute_metrics returns 0.0 rates instead of dividing by zero."""
    y_true = pd.Series([False, False, False])
    y_pred = pd.Series([False, False, False])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics.detection_rate == 0.0
    assert metrics.precision == 0.0
    assert metrics.approval_rate == 1.0


def test_compare_rulesets_reports_deltas(synthetic_applications_df: pd.DataFrame) -> None:
    """compare_rulesets reports the correct baseline/candidate deltas."""
    baseline_rules = [_make_rule("baseline", "credit_risk_score", 300)]
    candidate_rules = [_make_rule("candidate", "credit_risk_score", 200)]

    comparison = compare_rulesets(synthetic_applications_df, baseline_rules, candidate_rules, label_col="fraud_bool")

    assert comparison.baseline.metrics.detection_rate == pytest.approx(0.5)
    assert comparison.candidate.metrics.detection_rate == pytest.approx(1.0)
    assert comparison.delta_detection_rate == pytest.approx(0.5)
    assert comparison.delta_false_positive_rate == pytest.approx(0.0)
    assert comparison.delta_precision == pytest.approx(0.0)
    assert comparison.delta_approval_rate == pytest.approx(-0.2)