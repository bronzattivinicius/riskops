"""Tests for riskops.rules.store.RuleStore."""

import json
from datetime import UTC, datetime

import pytest

from riskops.rules.schema import Condition, ConditionGroup, Operator, Rule, RuleStatus
from riskops.rules.store import (
    AuditAction,
    RuleAlreadyExistsError,
    RuleNotFoundError,
    RuleStore,
)

_ACTOR = "test-actor"


def _build_rule(rule_id: str, status: RuleStatus = RuleStatus.CANDIDATE, version: int = 1) -> Rule:
    """Builds a minimal, valid Rule for store tests.

    Args:
        rule_id: Id to assign to the rule.
        status: Lifecycle status to assign.
        version: Version number to assign.

    Returns:
        A rule with placeholder metadata, suitable only for tests.
    """
    now = datetime.now(UTC)
    return Rule(
        id=rule_id,
        name=rule_id,
        version=version,
        status=status,
        description="test rule",
        logic=ConditionGroup(conditions=[Condition(field="amount", operator=Operator.GT, value=100)]),
        created_at=now,
        updated_at=now,
        created_by=_ACTOR,
        updated_by=_ACTOR,
    )


def _read_audit_log(store: RuleStore) -> list[dict]:
    """Reads and parses every entry in a store's audit log.

    Args:
        store: Store whose audit log should be read.

    Returns:
        The parsed JSON lines, in file order.
    """
    with store.audit_log_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_create_persists_rule_and_audit_entry(tmp_rule_store: RuleStore) -> None:
    """create() writes the rule to disk and appends one audit entry."""
    rule = _build_rule("r1")
    created = tmp_rule_store.create(rule, actor=_ACTOR, note="initial")

    assert created == rule
    assert tmp_rule_store.get("r1") == rule

    audit_entries = _read_audit_log(tmp_rule_store)
    assert len(audit_entries) == 1
    assert audit_entries[0]["action"] == AuditAction.CREATE
    assert audit_entries[0]["from_version"] is None
    assert audit_entries[0]["to_version"] == 1


def test_create_duplicate_id_raises(tmp_rule_store: RuleStore) -> None:
    """create() rejects a rule id that is already registered."""
    tmp_rule_store.create(_build_rule("r1"), actor=_ACTOR, note="first")
    with pytest.raises(RuleAlreadyExistsError):
        tmp_rule_store.create(_build_rule("r1"), actor=_ACTOR, note="second")


def test_create_requires_version_one(tmp_rule_store: RuleStore) -> None:
    """create() rejects a rule whose version is not 1."""
    with pytest.raises(ValueError):
        tmp_rule_store.create(_build_rule("r1", version=2), actor=_ACTOR, note="bad version")


def test_get_missing_rule_raises(tmp_rule_store: RuleStore) -> None:
    """get() raises RuleNotFoundError for an unregistered id."""
    with pytest.raises(RuleNotFoundError):
        tmp_rule_store.get("does_not_exist")


def test_update_bumps_version_and_preserves_identity(tmp_rule_store: RuleStore) -> None:
    """update() increments the version while keeping id and creation info."""
    original = tmp_rule_store.create(_build_rule("r1"), actor=_ACTOR, note="initial")

    updated = tmp_rule_store.update(
        "r1", description="new description", actor=_ACTOR, note="clarify description"
    )

    assert updated.id == original.id
    assert updated.version == original.version + 1
    assert updated.description == "new description"
    assert updated.created_at == original.created_at
    assert updated.created_by == original.created_by
    assert tmp_rule_store.get("r1") == updated


def test_get_version_returns_historical_snapshot(tmp_rule_store: RuleStore) -> None:
    """get_version() returns the exact state at a prior version."""
    v1 = tmp_rule_store.create(_build_rule("r1"), actor=_ACTOR, note="initial")
    tmp_rule_store.update("r1", description="changed", actor=_ACTOR, note="edit")

    assert tmp_rule_store.get_version("r1", 1) == v1
    with pytest.raises(RuleNotFoundError):
        tmp_rule_store.get_version("r1", 99)


def test_get_history_lists_every_version(tmp_rule_store: RuleStore) -> None:
    """get_history() returns one entry per version, oldest first."""
    tmp_rule_store.create(_build_rule("r1"), actor=_ACTOR, note="initial")
    tmp_rule_store.update("r1", description="changed", actor=_ACTOR, note="edit")

    history = tmp_rule_store.get_history("r1")

    assert [entry.version for entry in history] == [1, 2]


def test_transition_status_updates_status_and_logs_correct_action(tmp_rule_store: RuleStore) -> None:
    """transition_status() changes status, bumps version, and audits it."""
    tmp_rule_store.create(_build_rule("r1", status=RuleStatus.CANDIDATE), actor=_ACTOR, note="initial")

    activated = tmp_rule_store.transition_status(
        "r1", RuleStatus.ACTIVE, actor=_ACTOR, note="promote to production"
    )

    assert activated.status == RuleStatus.ACTIVE
    assert activated.version == 2

    audit_entries = _read_audit_log(tmp_rule_store)
    assert audit_entries[-1]["action"] == AuditAction.ACTIVATE
    assert audit_entries[-1]["from_status"] == RuleStatus.CANDIDATE
    assert audit_entries[-1]["to_status"] == RuleStatus.ACTIVE


def test_list_by_status_and_list_active(tmp_rule_store: RuleStore) -> None:
    """list_by_status()/list_active() filter rules by their current status."""
    tmp_rule_store.create(_build_rule("active_rule", status=RuleStatus.ACTIVE), actor=_ACTOR, note="a")
    tmp_rule_store.create(_build_rule("candidate_rule", status=RuleStatus.CANDIDATE), actor=_ACTOR, note="b")

    assert [rule.id for rule in tmp_rule_store.list_active()] == ["active_rule"]
    assert [rule.id for rule in tmp_rule_store.list_by_status(RuleStatus.CANDIDATE)] == ["candidate_rule"]