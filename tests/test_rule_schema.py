"""Tests for riskops.rules.schema validation."""

import pytest
from pydantic import ValidationError

from riskops.rules.schema import Condition, ConditionGroup, Operator


def test_scalar_operator_accepts_scalar_value() -> None:
    """A comparison operator accepts a plain scalar value."""
    condition = Condition(field="amount", operator=Operator.GT, value=200_000)
    assert condition.value == 200_000


def test_scalar_operator_rejects_missing_value() -> None:
    """A comparison operator requires a non-None scalar value."""
    with pytest.raises(ValidationError):
        Condition(field="amount", operator=Operator.GT, value=None)


def test_in_operator_requires_list_value() -> None:
    """The 'in' operator rejects a non-list value."""
    with pytest.raises(ValidationError):
        Condition(field="type", operator=Operator.IN, value="TRANSFER")


def test_in_operator_accepts_list_value() -> None:
    """The 'in' operator accepts a list value."""
    condition = Condition(field="type", operator=Operator.IN, value=["TRANSFER", "CASH_OUT"])
    assert condition.value == ["TRANSFER", "CASH_OUT"]


def test_between_operator_requires_two_element_list() -> None:
    """The 'between' operator rejects a list with the wrong length."""
    with pytest.raises(ValidationError):
        Condition(field="amount", operator=Operator.BETWEEN, value=[100])


def test_between_operator_accepts_two_element_list() -> None:
    """The 'between' operator accepts a two-element [low, high] list."""
    condition = Condition(field="amount", operator=Operator.BETWEEN, value=[100, 200])
    assert condition.value == [100, 200]


def test_is_null_operator_rejects_value() -> None:
    """The 'is_null' operator rejects a non-None value."""
    with pytest.raises(ValidationError):
        Condition(field="nameDest", operator=Operator.IS_NULL, value="x")


def test_is_null_operator_accepts_none() -> None:
    """The 'is_null' operator accepts a None value."""
    condition = Condition(field="nameDest", operator=Operator.IS_NULL, value=None)
    assert condition.value is None


def test_condition_group_requires_at_least_one_condition() -> None:
    """A condition group rejects an empty condition list."""
    with pytest.raises(ValidationError):
        ConditionGroup(conditions=[])


def test_condition_group_supports_nested_groups() -> None:
    """A condition group may nest another condition group."""
    group = ConditionGroup(
        logic="or",
        conditions=[
            Condition(field="amount", operator=Operator.GT, value=100),
            ConditionGroup(
                logic="and",
                conditions=[
                    Condition(field="type", operator=Operator.EQ, value="TRANSFER"),
                    Condition(field="amount", operator=Operator.LT, value=10),
                ],
            ),
        ],
    )
    assert len(group.conditions) == 2
    assert isinstance(group.conditions[1], ConditionGroup)
