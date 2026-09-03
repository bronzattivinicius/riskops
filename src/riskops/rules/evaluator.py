"""Vectorized evaluation of rules against a transaction DataFrame.

All condition and boolean-combination logic runs as pandas/numpy
operations over full columns, never as a Python loop over rows, so that
evaluation scales to multi-million-row datasets such as BAF.
"""

import operator as _op
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce

import numpy as np
import pandas as pd

from riskops.rules.schema import Condition, ConditionGroup, LogicOperator, Operator, Rule


class UnknownFieldError(ValueError):
    """Raised when a condition references a column absent from the data."""


_OPERATOR_FUNCS: dict[Operator, Callable[[pd.Series, object], pd.Series]] = {
    Operator.EQ: lambda series, value: series == value,
    Operator.NE: lambda series, value: series != value,
    Operator.GT: lambda series, value: series > value,
    Operator.GTE: lambda series, value: series >= value,
    Operator.LT: lambda series, value: series < value,
    Operator.LTE: lambda series, value: series <= value,
    Operator.IN: lambda series, value: series.isin(value),
    Operator.NOT_IN: lambda series, value: ~series.isin(value),
    Operator.BETWEEN: lambda series, value: series.between(value[0], value[1]),
    Operator.IS_NULL: lambda series, _value: series.isna(),
    Operator.IS_NOT_NULL: lambda series, _value: series.notna(),
}


def evaluate_condition(df: pd.DataFrame, condition: Condition) -> pd.Series:
    """Evaluates a single condition against every row of a DataFrame.

    Args:
        df: Transaction data.
        condition: Condition to evaluate.

    Returns:
        Boolean Series, index-aligned to ``df``, ``True`` where the
        condition holds.

    Raises:
        UnknownFieldError: If ``condition.field`` is not a column of
            ``df``.
    """
    if condition.field not in df.columns:
        raise UnknownFieldError(f"unknown field {condition.field!r}")
    return _OPERATOR_FUNCS[condition.operator](df[condition.field], condition.value)


def evaluate_group(df: pd.DataFrame, group: ConditionGroup) -> pd.Series:
    """Evaluates a (possibly nested) condition group against a DataFrame.

    Args:
        df: Transaction data.
        group: Condition group to evaluate.

    Returns:
        Boolean Series, index-aligned to ``df``, ``True`` where the group
        holds.
    """
    masks = [
        evaluate_group(df, item) if isinstance(item, ConditionGroup) else evaluate_condition(df, item)
        for item in group.conditions
    ]
    combinator = _op.and_ if group.logic == LogicOperator.AND else _op.or_
    return reduce(combinator, masks)


def evaluate_rule(df: pd.DataFrame, rule: Rule) -> pd.Series:
    """Evaluates a single rule's logic against a DataFrame.

    Args:
        df: Transaction data.
        rule: Rule to evaluate.

    Returns:
        Boolean Series, index-aligned to ``df``, ``True`` where the rule
        fires.
    """
    return evaluate_group(df, rule.logic)


@dataclass
class RuleEvaluationResult:
    """Outcome of evaluating a set of rules against a DataFrame.

    Attributes:
        flagged: Boolean Series, ``True`` where at least one rule fired.
        per_rule_flags: DataFrame with one boolean column per rule id.
        fired_rule_ids: Series of lists, the ids of the rules that fired
            for each row.
    """

    flagged: pd.Series
    per_rule_flags: pd.DataFrame
    fired_rule_ids: pd.Series


def evaluate_ruleset(df: pd.DataFrame, rules: list[Rule]) -> RuleEvaluationResult:
    """Evaluates a set of rules against a DataFrame.

    A row is considered flagged if any rule in ``rules`` fires for it.

    Args:
        df: Transaction data.
        rules: Rules to evaluate. May be empty.

    Returns:
        The combined evaluation result.
    """
    if not rules:
        empty_flags = pd.Series(False, index=df.index)
        empty_fired = pd.Series([[] for _ in range(len(df))], index=df.index)
        return RuleEvaluationResult(empty_flags, pd.DataFrame(index=df.index), empty_fired)

    per_rule = pd.DataFrame({rule.id: evaluate_rule(df, rule) for rule in rules}, index=df.index)
    flagged = per_rule.any(axis=1)

    rule_ids = np.array(per_rule.columns)
    fired = pd.Series(
        [rule_ids[row].tolist() for row in per_rule.to_numpy()],
        index=df.index,
    )
    return RuleEvaluationResult(flagged=flagged, per_rule_flags=per_rule, fired_rule_ids=fired)
