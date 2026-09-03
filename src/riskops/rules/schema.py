"""Pydantic schema for risk rules.

A :class:`Rule` is a versioned, auditable unit of deterministic decision
logic: a boolean expression (:class:`ConditionGroup`) over transaction
fields, plus lifecycle metadata (status, version, authorship).
"""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Union

from pydantic import BaseModel, Field, model_validator


class Operator(StrEnum):
    """Comparison operators supported by a :class:`Condition`."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class LogicOperator(StrEnum):
    """Boolean combinator used inside a :class:`ConditionGroup`."""

    AND = "and"
    OR = "or"


class RuleStatus(StrEnum):
    """Lifecycle status of a :class:`Rule`.

    Attributes:
        ACTIVE: Currently enforced in production; part of the baseline
            used when backtesting candidates.
        CANDIDATE: Proposed but not yet approved for production use.
        DEPRECATED: Previously active, retired in favor of another rule.
        REJECTED: Proposed and explicitly rejected after review.
    """

    ACTIVE = "active"
    CANDIDATE = "candidate"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


_LIST_VALUE_OPERATORS = {Operator.IN, Operator.NOT_IN}
_NO_VALUE_OPERATORS = {Operator.IS_NULL, Operator.IS_NOT_NULL}


class Condition(BaseModel):
    """A single comparison against one transaction field.

    Attributes:
        field: Name of the transaction field to compare, e.g. ``"amount"``.
        operator: Comparison to apply.
        value: Right-hand side of the comparison. Must be ``None`` for
            ``is_null``/``is_not_null``, a list for ``in``/``not_in``, a
            two-element ``[low, high]`` list for ``between``, and a scalar
            otherwise.
    """

    field: str
    operator: Operator
    value: object | None = None

    @model_validator(mode="after")
    def _validate_value_shape(self) -> "Condition":
        """Ensures ``value`` has the shape required by ``operator``.

        Returns:
            The validated condition, unchanged.

        Raises:
            ValueError: If ``value`` does not match the shape required by
                ``operator``.
        """
        if self.operator in _NO_VALUE_OPERATORS and self.value is not None:
            raise ValueError(f"operator {self.operator!r} must not have a value")
        if self.operator in _LIST_VALUE_OPERATORS and not isinstance(self.value, list):
            raise ValueError(f"operator {self.operator!r} requires a list value")
        if self.operator == Operator.BETWEEN and (
            not isinstance(self.value, list) or len(self.value) != 2
        ):
            raise ValueError("operator 'between' requires a two-element [low, high] list")
        if (
            self.operator not in _NO_VALUE_OPERATORS
            and self.operator not in _LIST_VALUE_OPERATORS
            and self.operator != Operator.BETWEEN
            and self.value is None
        ):
            raise ValueError(f"operator {self.operator!r} requires a scalar value")
        return self


class ConditionGroup(BaseModel):
    """A boolean combination of conditions and/or nested groups.

    Attributes:
        logic: How ``conditions`` are combined.
        conditions: One or more :class:`Condition` or nested
            :class:`ConditionGroup` instances.
    """

    logic: LogicOperator = LogicOperator.AND
    conditions: Annotated[list[Union[Condition, "ConditionGroup"]], Field(min_length=1)]


ConditionGroup.model_rebuild()


class Rule(BaseModel):
    """A single versioned risk rule.

    Attributes:
        id: Stable slug identity, unchanged across versions. Used as the
            filename in the rule registry.
        name: Human-readable name.
        version: Monotonically increasing version number, starting at 1.
        status: Current lifecycle status.
        description: Free-text explanation of intent.
        logic: Boolean expression evaluated against a transaction.
        tags: Free-form labels for filtering/grouping.
        created_at: Timestamp of the rule's first version.
        updated_at: Timestamp of the current version.
        created_by: Identity that created the rule.
        updated_by: Identity that produced the current version.
        schema_version: On-disk schema format version, for future
            migrations of the rule registry file format.
    """

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    version: int = Field(ge=1)
    status: RuleStatus
    description: str
    logic: ConditionGroup
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    schema_version: int = 1


class RuleChangeLogEntry(BaseModel):
    """One historical snapshot of a rule, embedded in its registry file.

    Attributes:
        version: Version number this snapshot represents.
        status: Status the rule had at this version.
        changed_at: Timestamp of the change.
        changed_by: Identity that made the change.
        change_note: Free-text explanation of the change.
        snapshot: Full :class:`Rule` body at this version.
    """

    version: int = Field(ge=1)
    status: RuleStatus
    changed_at: datetime
    changed_by: str
    change_note: str
    snapshot: Rule


class RuleFile(Rule):
    """On-disk envelope for a rule: current state plus full history.

    Attributes:
        history: Chronological list of prior :class:`RuleChangeLogEntry`
            snapshots, oldest first. Does not include the current state,
            which is represented by this object's own top-level fields.
    """

    history: list[RuleChangeLogEntry] = Field(default_factory=list)
