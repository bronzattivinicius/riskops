"""Backtest a ruleset against labeled historical transactions.

This module is intentionally decoupled from
:class:`~riskops.rules.store.RuleStore`: it operates on plain
``list[Rule]`` values, so callers can backtest any combination of rules
(the currently active set, a single candidate, or an arbitrary mix)
without the metrics logic needing to know about persistence.
"""

from dataclasses import dataclass

import pandas as pd

from riskops.rules.evaluator import RuleEvaluationResult, evaluate_ruleset
from riskops.rules.schema import Rule


@dataclass
class ClassificationMetrics:
    """Confusion-matrix counts and derived rates for a flagging decision.

    Attributes:
        true_positives: Fraudulent transactions correctly flagged.
        false_positives: Legitimate transactions incorrectly flagged.
        true_negatives: Legitimate transactions correctly not flagged.
        false_negatives: Fraudulent transactions missed.
        total: Total number of transactions evaluated.
        fraud_count: Total number of fraudulent transactions.
        flagged_count: Total number of flagged transactions.
    """

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total: int
    fraud_count: int
    flagged_count: int

    @property
    def detection_rate(self) -> float:
        """Fraction of fraud transactions that were flagged (recall)."""
        return self.true_positives / self.fraud_count if self.fraud_count else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Fraction of legitimate transactions incorrectly flagged."""
        legitimate_count = self.total - self.fraud_count
        return self.false_positives / legitimate_count if legitimate_count else 0.0

    @property
    def precision(self) -> float:
        """Fraction of flagged transactions that were actually fraud."""
        return self.true_positives / self.flagged_count if self.flagged_count else 0.0

    @property
    def approval_rate(self) -> float:
        """Fraction of transactions left unflagged (auto-approved)."""
        return (self.total - self.flagged_count) / self.total if self.total else 0.0


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> ClassificationMetrics:
    """Computes classification metrics from ground truth and predictions.

    Args:
        y_true: Boolean (or boolean-castable) Series, ``True`` for fraud.
        y_pred: Boolean (or boolean-castable) Series, ``True`` for flagged.

    Returns:
        The computed metrics.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    true_positives = int((y_true & y_pred).sum())
    false_positives = int((~y_true & y_pred).sum())
    false_negatives = int((y_true & ~y_pred).sum())
    true_negatives = int((~y_true & ~y_pred).sum())
    return ClassificationMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        total=len(y_true),
        fraud_count=int(y_true.sum()),
        flagged_count=int(y_pred.sum()),
    )


@dataclass
class BacktestResult:
    """Result of backtesting one ruleset against historical data.

    Attributes:
        rule_ids: Ids of the rules that were evaluated.
        metrics: Resulting classification metrics.
        evaluation: Full per-row rule evaluation result.
    """

    rule_ids: list[str]
    metrics: ClassificationMetrics
    evaluation: RuleEvaluationResult


def backtest_ruleset(df: pd.DataFrame, rules: list[Rule], label_col: str = "isFraud") -> BacktestResult:
    """Backtests a set of rules against labeled historical transactions.

    Args:
        df: Historical transaction data, including ``label_col``.
        rules: Rules to evaluate.
        label_col: Name of the boolean fraud-label column.

    Returns:
        The backtest result.
    """
    evaluation = evaluate_ruleset(df, rules)
    metrics = compute_metrics(df[label_col], evaluation.flagged)
    return BacktestResult(rule_ids=[rule.id for rule in rules], metrics=metrics, evaluation=evaluation)


@dataclass
class ComparisonResult:
    """Comparison between a baseline and a candidate ruleset's backtest.

    Attributes:
        baseline: Backtest result for the baseline (e.g. currently active)
            ruleset.
        candidate: Backtest result for the candidate ruleset.
        delta_detection_rate: ``candidate - baseline`` detection rate.
        delta_false_positive_rate: ``candidate - baseline`` false positive
            rate.
        delta_precision: ``candidate - baseline`` precision.
        delta_approval_rate: ``candidate - baseline`` approval rate.
    """

    baseline: BacktestResult
    candidate: BacktestResult
    delta_detection_rate: float
    delta_false_positive_rate: float
    delta_precision: float
    delta_approval_rate: float


def compare_rulesets(
    df: pd.DataFrame,
    baseline_rules: list[Rule],
    candidate_rules: list[Rule],
    label_col: str = "isFraud",
) -> ComparisonResult:
    """Backtests and compares a baseline and a candidate ruleset.

    Args:
        df: Historical transaction data, including ``label_col``.
        baseline_rules: Rules representing the current production baseline.
        candidate_rules: Rules representing the proposed candidate.
        label_col: Name of the boolean fraud-label column.

    Returns:
        The comparison result, including per-ruleset backtests and deltas.
    """
    baseline = backtest_ruleset(df, baseline_rules, label_col)
    candidate = backtest_ruleset(df, candidate_rules, label_col)
    return ComparisonResult(
        baseline=baseline,
        candidate=candidate,
        delta_detection_rate=candidate.metrics.detection_rate - baseline.metrics.detection_rate,
        delta_false_positive_rate=(
            candidate.metrics.false_positive_rate - baseline.metrics.false_positive_rate
        ),
        delta_precision=candidate.metrics.precision - baseline.metrics.precision,
        delta_approval_rate=candidate.metrics.approval_rate - baseline.metrics.approval_rate,
    )
