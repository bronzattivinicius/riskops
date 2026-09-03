"""Seed candidate rules for the BAF dataset, used as frozen test fixtures.

Unlike ``baf_seed.py`` (a deliberately naive "legacy" baseline), these
rules were selected by measuring the fraud rate of several candidate
conditions against the real ``data/raw/Base.csv`` (1,000,000 rows, 1.10%
overall fraud rate). They give the Deliverable 1 baseline a mix of
clearly strong, clearly weak, and ambiguous rules to reason about, plus
one synthetic zero-match edge case.

Run as a module to populate the rule registry::

    python -m riskops.rules.baf_candidates_seed
"""

from datetime import UTC, datetime

from riskops.rules.schema import Condition, ConditionGroup, LogicOperator, Operator, Rule, RuleStatus
from riskops.rules.store import RuleAlreadyExistsError, RuleStore

DEFAULT_ACTOR = "riskops-seed"

_NOW = datetime.now(UTC)

CANDIDATE_RULES: list[Rule] = [
    Rule(
        id="baf_high_credit_risk_score",
        name="High internal credit risk score",
        version=1,
        status=RuleStatus.CANDIDATE,
        description=(
            "Flags applications with an internal credit risk score above 300. "
            "The strongest single signal found: 6.60% fraud rate vs. a 1.10% "
            "dataset-wide baseline (n=11,948 on Base.csv)."
        ),
        logic=ConditionGroup(
            logic=LogicOperator.AND,
            conditions=[Condition(field="credit_risk_score", operator=Operator.GT, value=300)],
        ),
        tags=["baf", "candidate", "credit_risk_score"],
        created_at=_NOW,
        updated_at=_NOW,
        created_by=DEFAULT_ACTOR,
        updated_by=DEFAULT_ACTOR,
    ),
    Rule(
        id="baf_weak_identity_match_elevated_risk",
        name="Weak name/email match with elevated risk score",
        version=1,
        status=RuleStatus.CANDIDATE,
        description=(
            "Flags applications where the applicant's name barely resembles "
            "their email address, combined with a moderately elevated risk "
            "score. Measured 4.87% fraud rate (n=31,210 on Base.csv)."
        ),
        logic=ConditionGroup(
            logic=LogicOperator.AND,
            conditions=[
                Condition(field="name_email_similarity", operator=Operator.LT, value=0.2),
                Condition(field="credit_risk_score", operator=Operator.GT, value=200),
            ],
        ),
        tags=["baf", "candidate", "identity"],
        created_at=_NOW,
        updated_at=_NOW,
        created_by=DEFAULT_ACTOR,
        updated_by=DEFAULT_ACTOR,
    ),
    Rule(
        id="baf_device_email_reuse",
        name="Device reused across multiple recent emails",
        version=1,
        status=RuleStatus.CANDIDATE,
        description=(
            "Flags applications from a device that has been associated with "
            "two or more distinct emails in the last 8 weeks. Measured "
            "4.09% fraud rate (n=25,302 on Base.csv)."
        ),
        logic=ConditionGroup(
            logic=LogicOperator.AND,
            conditions=[
                Condition(field="device_distinct_emails_8w", operator=Operator.GTE, value=2),
            ],
        ),
        tags=["baf", "candidate", "device"],
        created_at=_NOW,
        updated_at=_NOW,
        created_by=DEFAULT_ACTOR,
        updated_by=DEFAULT_ACTOR,
    ),
    Rule(
        id="baf_invalid_phone_combo",
        name="Both home and mobile phone invalid",
        version=1,
        status=RuleStatus.CANDIDATE,
        description=(
            "Flags applications where neither the provided home nor mobile "
            "phone number is valid. A moderate, debatable signal: 2.78% "
            "fraud rate (n=22,232 on Base.csv), roughly 2.5x baseline -- "
            "not clearly strong or clearly weak."
        ),
        logic=ConditionGroup(
            logic=LogicOperator.AND,
            conditions=[
                Condition(field="phone_home_valid", operator=Operator.EQ, value=0),
                Condition(field="phone_mobile_valid", operator=Operator.EQ, value=0),
            ],
        ),
        tags=["baf", "candidate", "ambiguous"],
        created_at=_NOW,
        updated_at=_NOW,
        created_by=DEFAULT_ACTOR,
        updated_by=DEFAULT_ACTOR,
    ),
    Rule(
        id="baf_impossible_threshold",
        name="Synthetic zero-match edge case",
        version=1,
        status=RuleStatus.CANDIDATE,
        description=(
            "Synthetic test fixture, not a real candidate: credit_risk_score "
            "never exceeds 389 in the BAF dataset, so this condition matches "
            "zero transactions. Used to test graceful handling of "
            "zero-match rules (undefined precision, zero detection)."
        ),
        logic=ConditionGroup(
            logic=LogicOperator.AND,
            conditions=[Condition(field="credit_risk_score", operator=Operator.GT, value=1000)],
        ),
        tags=["baf", "candidate", "edge-case"],
        created_at=_NOW,
        updated_at=_NOW,
        created_by=DEFAULT_ACTOR,
        updated_by=DEFAULT_ACTOR,
    ),
]


def seed_candidate_rules(store: RuleStore, actor: str = DEFAULT_ACTOR) -> list[Rule]:
    """Creates the BAF candidate rules in a store, skipping ones that exist.

    Args:
        store: Rule store to populate.
        actor: Identity recorded as the creator of each new rule.

    Returns:
        The rules that were newly created (already-existing rules are
        skipped and omitted from the result).
    """
    existing_ids = set(store.list_rule_ids())
    created = []
    for rule in CANDIDATE_RULES:
        if rule.id in existing_ids:
            continue
        try:
            created.append(
                store.create(rule, actor=actor, note="candidate rule seeded for Deliverable 1 test fixtures")
            )
        except RuleAlreadyExistsError:
            continue
    return created


if __name__ == "__main__":
    for created_rule in seed_candidate_rules(RuleStore()):
        print(f"created {created_rule.id} (v{created_rule.version}, {created_rule.status})")
