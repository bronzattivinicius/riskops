"""Tests for riskops.rules.evaluator."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from riskops.rules.evaluator import (
    UnknownFieldError,
    evaluate_condition,
    evaluate_group,
    evaluate_ruleset,
)
from riskops.rules.schema import Condition, ConditionGroup, LogicOperator, Operator, Rule, RuleStatus

_NOW = datetime.now(UTC)


def _make_rule(rule_id: str, logic: ConditionGroup) -> Rule:
    """Builds a minimal, valid Rule for evaluator tests.

    Args:
        rule_id: Id to assign to the rule.
        logic: Condition logic to evaluate.

    Returns:
        A rule with placeholder metadata, suitable only for tests.
    """
    return Rule(
        id=rule_id,
        name=rule_id,
        version=1,
        status=RuleStatus.CANDIDATE,
        description="test rule",
        logic=logic,
        created_at=_NOW,
        updated_at=_NOW,
        created_by="test",
        updated_by="test",
    )


def test_evaluate_condition_gt(synthetic_applications_df: pd.DataFrame) -> None:
    """A 'gt' condition flags exactly the rows above the threshold."""
    condition = Condition(field="velocity_6h", operator=Operator.GT, value=5000)
    mask = evaluate_condition(synthetic_applications_df, condition)
    expected = [True, False, False, True, False, False, True, False, False, True]
    assert mask.tolist() == expected


def test_evaluate_condition_unknown_field(synthetic_applications_df: pd.DataFrame) -> None:
    """An unknown field raises UnknownFieldError instead of a KeyError."""
    condition = Condition(field="does_not_exist", operator=Operator.GT, value=1)
    with pytest.raises(UnknownFieldError):
        evaluate_condition(synthetic_applications_df, condition)


def test_evaluate_group_and(synthetic_applications_df: pd.DataFrame) -> None:
    """An AND group only flags rows matching every condition."""
    group = ConditionGroup(
        logic=LogicOperator.AND,
        conditions=[
            Condition(field="foreign_request", operator=Operator.EQ, value=1),
            Condition(field="email_is_free", operator=Operator.EQ, value=1),
        ],
    )
    mask = evaluate_group(synthetic_applications_df, group)
    expected = [True, False, False, False, False, True, False, False, False, False]
    assert mask.tolist() == expected


def test_evaluate_group_or(synthetic_applications_df: pd.DataFrame) -> None:
    """An OR group flags rows matching at least one condition."""
    group = ConditionGroup(
        logic=LogicOperator.OR,
        conditions=[
            Condition(field="device_distinct_emails_8w", operator=Operator.GTE, value=2),
            Condition(field="credit_risk_score", operator=Operator.GT, value=300),
        ],
    )
    mask = evaluate_group(synthetic_applications_df, group)
    expected = [True, False, False, True, False, False, True, False, False, False]
    assert mask.tolist() == expected


def test_evaluate_ruleset_combines_rules_with_or(synthetic_applications_df: pd.DataFrame) -> None:
    """evaluate_ruleset flags a row if any rule fires, and reports which."""
    rule_a = _make_rule(
        "rule_a",
        ConditionGroup(
            logic=LogicOperator.AND,
            conditions=[
                Condition(field="foreign_request", operator=Operator.EQ, value=1),
                Condition(field="email_is_free", operator=Operator.EQ, value=1),
            ],
        ),
    )
    rule_b = _make_rule(
        "rule_b",
        ConditionGroup(
            logic=LogicOperator.AND,
            conditions=[Condition(field="velocity_6h", operator=Operator.GT, value=5000)],
        ),
    )

    result = evaluate_ruleset(synthetic_applications_df, [rule_a, rule_b])

    expected_flagged = [True, False, False, True, False, True, True, False, False, True]
    assert result.flagged.tolist() == expected_flagged
    assert result.per_rule_flags["rule_a"].tolist() == [
        True, False, False, False, False, True, False, False, False, False,
    ]
    assert result.per_rule_flags["rule_b"].tolist() == [
        True, False, False, True, False, False, True, False, False, True,
    ]
    expected_fired = [
        ["rule_a", "rule_b"], [], [], ["rule_b"], [],
        ["rule_a"], ["rule_b"], [], [], ["rule_b"],
    ]
    assert result.fired_rule_ids.tolist() == expected_fired


def test_evaluate_ruleset_empty_rules(synthetic_applications_df: pd.DataFrame) -> None:
    """An empty ruleset flags nothing."""
    result = evaluate_ruleset(synthetic_applications_df, [])
    assert not result.flagged.any()
    assert all(fired == [] for fired in result.fired_rule_ids)