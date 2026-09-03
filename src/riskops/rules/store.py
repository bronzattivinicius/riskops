"""Versioned, audited persistence for :class:`~riskops.rules.schema.Rule`.

Rules are stored as one YAML file per rule under ``<root>/rules/<id>.yaml``,
containing the current state plus a full history of prior versions. Every
mutation also appends one line to ``<root>/audit_log.jsonl``, giving a
flat, chronological, cross-rule audit trail.
"""

import json
import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel

from riskops.paths import RULE_REGISTRY_DIR
from riskops.rules.schema import ConditionGroup, Rule, RuleChangeLogEntry, RuleFile, RuleStatus


class RuleNotFoundError(KeyError):
    """Raised when a requested rule id or version does not exist."""


class RuleAlreadyExistsError(RuntimeError):
    """Raised when creating a rule whose id is already registered."""


class AuditAction(StrEnum):
    """Kind of change recorded in the audit log."""

    CREATE = "create"
    UPDATE = "update"
    ACTIVATE = "activate"
    DEPRECATE = "deprecate"
    REJECT = "reject"
    RESTORE = "restore"


class AuditLogEntry(BaseModel):
    """One append-only record in ``audit_log.jsonl``.

    Attributes:
        timestamp: When the change was made.
        actor: Identity that made the change.
        action: Kind of change.
        rule_id: Id of the affected rule.
        from_version: Version before the change, or ``None`` on creation.
        to_version: Version after the change.
        from_status: Status before the change, or ``None`` on creation.
        to_status: Status after the change.
        note: Free-text explanation of the change.
    """

    timestamp: datetime
    actor: str
    action: AuditAction
    rule_id: str
    from_version: int | None
    to_version: int
    from_status: RuleStatus | None
    to_status: RuleStatus
    note: str


_STATUS_TO_ACTION = {
    RuleStatus.ACTIVE: AuditAction.ACTIVATE,
    RuleStatus.DEPRECATED: AuditAction.DEPRECATE,
    RuleStatus.REJECTED: AuditAction.REJECT,
    RuleStatus.CANDIDATE: AuditAction.RESTORE,
}


class RuleStore:
    """Filesystem-backed rule registry with versioning and audit logging."""

    def __init__(self, root: Path = RULE_REGISTRY_DIR) -> None:
        """Initializes the store, creating its directories if needed.

        Args:
            root: Root directory of the rule registry.
        """
        self.root = root
        self.rules_dir = root / "rules"
        self.audit_log_path = root / "audit_log.jsonl"
        self.rules_dir.mkdir(parents=True, exist_ok=True)

    def list_rule_ids(self) -> list[str]:
        """Lists the ids of all registered rules.

        Returns:
            Sorted list of rule ids.
        """
        return sorted(p.stem for p in self.rules_dir.glob("*.yaml"))

    def get(self, rule_id: str) -> Rule:
        """Retrieves the current version of a rule.

        Args:
            rule_id: Id of the rule to retrieve.

        Returns:
            The rule's current state.

        Raises:
            RuleNotFoundError: If no rule with this id exists.
        """
        return self._current_rule(self._load(rule_id))

    def get_version(self, rule_id: str, version: int) -> Rule:
        """Retrieves a specific historical version of a rule.

        Args:
            rule_id: Id of the rule to retrieve.
            version: Version number to retrieve.

        Returns:
            The rule's state at the requested version.

        Raises:
            RuleNotFoundError: If no rule with this id exists, or if the
                rule has no such version.
        """
        rule_file = self._load(rule_id)
        for entry in rule_file.history:
            if entry.version == version:
                return entry.snapshot
        raise RuleNotFoundError(f"rule {rule_id!r} has no version {version}")

    def get_history(self, rule_id: str) -> list[RuleChangeLogEntry]:
        """Retrieves the full version history of a rule.

        Args:
            rule_id: Id of the rule to retrieve.

        Returns:
            Chronological list of change-log entries, oldest first.

        Raises:
            RuleNotFoundError: If no rule with this id exists.
        """
        return list(self._load(rule_id).history)

    def list_by_status(self, status: RuleStatus) -> list[Rule]:
        """Lists all rules currently in a given status.

        Args:
            status: Status to filter by.

        Returns:
            List of matching rules, in id order.
        """
        return [
            rule
            for rule_id in self.list_rule_ids()
            if (rule := self.get(rule_id)).status == status
        ]

    def list_active(self) -> list[Rule]:
        """Lists all rules currently active.

        Returns:
            List of active rules, in id order.
        """
        return self.list_by_status(RuleStatus.ACTIVE)

    def create(self, rule: Rule, actor: str, note: str) -> Rule:
        """Creates a new rule at version 1.

        Args:
            rule: The rule to create. Its ``version`` must be 1.
            actor: Identity performing the creation.
            note: Free-text explanation of the change.

        Returns:
            The stored rule, unchanged.

        Raises:
            RuleAlreadyExistsError: If a rule with this id already exists.
            ValueError: If ``rule.version`` is not 1.
        """
        if rule.id in self.list_rule_ids():
            raise RuleAlreadyExistsError(f"rule {rule.id!r} already exists")
        if rule.version != 1:
            raise ValueError("a newly created rule must have version=1")

        entry = RuleChangeLogEntry(
            version=rule.version,
            status=rule.status,
            changed_at=rule.updated_at,
            changed_by=actor,
            change_note=note,
            snapshot=rule,
        )
        self._write(RuleFile(**rule.model_dump(), history=[entry]))
        self._append_audit(
            AuditLogEntry(
                timestamp=rule.updated_at,
                actor=actor,
                action=AuditAction.CREATE,
                rule_id=rule.id,
                from_version=None,
                to_version=rule.version,
                from_status=None,
                to_status=rule.status,
                note=note,
            )
        )
        return rule

    def update(
        self,
        rule_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        logic: ConditionGroup | None = None,
        tags: list[str] | None = None,
        actor: str,
        note: str,
    ) -> Rule:
        """Updates the content of a rule, creating a new version.

        Only the fields explicitly passed are changed; ``id``, ``status``,
        ``created_at``, and ``created_by`` are preserved from the current
        version. Use :meth:`transition_status` to change ``status``.

        Args:
            rule_id: Id of the rule to update.
            name: New name, if changing.
            description: New description, if changing.
            logic: New :class:`~riskops.rules.schema.ConditionGroup`, if
                changing.
            tags: New tags, if changing.
            actor: Identity performing the update.
            note: Free-text explanation of the change.

        Returns:
            The rule's new, current state.

        Raises:
            RuleNotFoundError: If no rule with this id exists.
        """
        rule_file = self._load(rule_id)
        current = self._current_rule(rule_file)
        now = datetime.now(UTC)
        updated = current.model_copy(
            update={
                "name": name if name is not None else current.name,
                "description": description if description is not None else current.description,
                "logic": logic if logic is not None else current.logic,
                "tags": tags if tags is not None else current.tags,
                "version": current.version + 1,
                "updated_at": now,
                "updated_by": actor,
            }
        )
        self._persist_new_version(rule_file, updated, actor, note)
        self._append_audit(
            AuditLogEntry(
                timestamp=now,
                actor=actor,
                action=AuditAction.UPDATE,
                rule_id=rule_id,
                from_version=current.version,
                to_version=updated.version,
                from_status=current.status,
                to_status=updated.status,
                note=note,
            )
        )
        return updated

    def transition_status(
        self, rule_id: str, new_status: RuleStatus, actor: str, note: str
    ) -> Rule:
        """Changes a rule's lifecycle status, creating a new version.

        Args:
            rule_id: Id of the rule to transition.
            new_status: Status to transition to.
            actor: Identity performing the transition.
            note: Free-text explanation of the change.

        Returns:
            The rule's new, current state.

        Raises:
            RuleNotFoundError: If no rule with this id exists.
        """
        rule_file = self._load(rule_id)
        current = self._current_rule(rule_file)
        now = datetime.now(UTC)
        updated = current.model_copy(
            update={"status": new_status, "version": current.version + 1, "updated_at": now, "updated_by": actor}
        )
        self._persist_new_version(rule_file, updated, actor, note)
        self._append_audit(
            AuditLogEntry(
                timestamp=now,
                actor=actor,
                action=_STATUS_TO_ACTION[new_status],
                rule_id=rule_id,
                from_version=current.version,
                to_version=updated.version,
                from_status=current.status,
                to_status=new_status,
                note=note,
            )
        )
        return updated

    def _rule_path(self, rule_id: str) -> Path:
        """Returns the on-disk path for a rule's registry file.

        Args:
            rule_id: Id of the rule.

        Returns:
            Path to the rule's YAML file.
        """
        return self.rules_dir / f"{rule_id}.yaml"

    def _current_rule(self, rule_file: RuleFile) -> Rule:
        """Extracts the current :class:`Rule` state from a :class:`RuleFile`.

        Args:
            rule_file: Loaded rule file.

        Returns:
            The rule's current state, without history.
        """
        return Rule(**rule_file.model_dump(exclude={"history"}))

    def _persist_new_version(
        self, rule_file: RuleFile, updated: Rule, actor: str, note: str
    ) -> None:
        """Appends a new version to a rule's history and writes it to disk.

        Args:
            rule_file: Previously loaded rule file.
            updated: The rule's new, current state.
            actor: Identity performing the change.
            note: Free-text explanation of the change.
        """
        entry = RuleChangeLogEntry(
            version=updated.version,
            status=updated.status,
            changed_at=updated.updated_at,
            changed_by=actor,
            change_note=note,
            snapshot=updated,
        )
        self._write(RuleFile(**updated.model_dump(), history=[*rule_file.history, entry]))

    def _load(self, rule_id: str) -> RuleFile:
        """Loads a rule's registry file from disk.

        Args:
            rule_id: Id of the rule to load.

        Returns:
            The parsed rule file.

        Raises:
            RuleNotFoundError: If no rule with this id exists.
        """
        path = self._rule_path(rule_id)
        if not path.exists():
            raise RuleNotFoundError(f"no rule registered with id {rule_id!r}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return RuleFile.model_validate(data)

    def _write(self, rule_file: RuleFile) -> None:
        """Atomically writes a rule's registry file to disk.

        Args:
            rule_file: Rule file to persist.
        """
        path = self._rule_path(rule_file.id)
        tmp_path = path.with_suffix(".yaml.tmp")
        data = rule_file.model_dump(mode="json")
        with tmp_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
        os.replace(tmp_path, path)

    def _append_audit(self, entry: AuditLogEntry) -> None:
        """Appends one entry to the audit log.

        Args:
            entry: Audit entry to append.
        """
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.model_dump(mode="json")) + "\n")
