"""Seed a deliberately naive, legacy-style baseline for the BAF dataset.

RiskOps exists to monitor, diagnose, and improve underperforming risk
rules. If the seed baseline were already statistically optimal, the
future monitoring/analyst/rule-generator agents would have nothing left
to do. These three rules instead simulate a plausible-looking but
never rigorously validated "legacy" ruleset: the kind an analyst might
write from intuition alone, without measuring it against historical
data -- exactly the failure mode RiskOps is meant to catch.

Measured against ``data/raw/Base.csv`` (1,000,000 rows, 1.10% overall
fraud rate):

* ``foreign_request == 1``: 2.20% fraud rate (n=25,242) -- a real but
  weak signal.
* ``email_is_free == 1``: 1.38% fraud rate (n=529,886) -- barely above
  the dataset baseline, essentially noise, but matches over half the
  dataset.
* ``velocity_6h > 5000``: 0.99% fraud rate (n=540,693) -- *below* the
  1.10% baseline. This condition is actively counterproductive: rows
  matching it are less likely to be fraud than a random row.

Combined with OR, this baseline flags 77.9% of all applications
(detection rate 83.1%, false positive rate 77.8%, precision 1.2%,
approval rate 22.1%) -- high recall bought at the cost of nearly
flagging everything, a realistic symptom of rule sprawl for a
monitoring agent to detect.

Run as a module to populate the rule registry::

    python -m riskops.rules.baf_seed
"""

from datetime import UTC, datetime

from riskops.rules.schema import Condition, ConditionGroup, LogicOperator, Operator, Rule, RuleStatus
from riskops.rules.store import RuleAlreadyExistsError, RuleStore

DEFAULT_ACTOR = "riskops-seed"

_NOW = datetime.now(UTC)

SEED_RULES: list[Rule] = [
    Rule(
        id="baf_foreign_request",
        name="Foreign-origin request",
        version=1,
        status=RuleStatus.ACTIVE,
        description=(
            "Flags applications originating from a country different from "
            "the bank's. Written on the intuitive assumption that "
            "cross-border requests are riskier; never validated against "
            "historical data. Measured fraud rate: 2.20% (n=25,242)."
        ),
        logic=ConditionGroup(
            logic=LogicOperator.AND,
            conditions=[Condition(field="foreign_request", operator=Operator.EQ, value=1)],
        ),
        tags=["baf", "seed", "legacy"],
        created_at=_NOW,
        updated_at=_NOW,
        created_by=DEFAULT_ACTOR,
        updated_by=DEFAULT_ACTOR,
    ),
    Rule(
        id="baf_free_email_domain",
        name="Free email domain",
        version=1,
        status=RuleStatus.ACTIVE,
        description=(
            "Flags applications made with a free email domain. Written on "
            "the intuitive assumption that free email domains are less "
            "trustworthy than paid ones; never validated against historical "
            "data. Measured fraud rate: 1.38% (n=529,886), barely above the "
            "1.10% dataset baseline despite matching over half of all rows."
        ),
        logic=ConditionGroup(
            logic=LogicOperator.AND,
            conditions=[Condition(field="email_is_free", operator=Operator.EQ, value=1)],
        ),
        tags=["baf", "seed", "legacy"],
        created_at=_NOW,
        updated_at=_NOW,
        created_by=DEFAULT_ACTOR,
        updated_by=DEFAULT_ACTOR,
    ),
    Rule(
        id="baf_high_velocity_6h",
        name="High application velocity in the last 6 hours",
        version=1,
        status=RuleStatus.ACTIVE,
        description=(
            "Flags applications made during a burst of application activity "
            "in the last 6 hours. Written on the intuitive assumption that "
            "high velocity indicates automated or bot-driven fraud; never "
            "validated against historical data. Measured fraud rate: 0.99% "
            "(n=540,693) -- *below* the 1.10% dataset baseline, i.e. this "
            "condition is actively counterproductive."
        ),
        logic=ConditionGroup(
            logic=LogicOperator.AND,
            conditions=[Condition(field="velocity_6h", operator=Operator.GT, value=5000)],
        ),
        tags=["baf", "seed", "legacy"],
        created_at=_NOW,
        updated_at=_NOW,
        created_by=DEFAULT_ACTOR,
        updated_by=DEFAULT_ACTOR,
    ),
]


def seed_baseline_rules(store: RuleStore, actor: str = DEFAULT_ACTOR) -> list[Rule]:
    """Creates the BAF seed rules in a store, skipping ones that exist.

    Args:
        store: Rule store to populate.
        actor: Identity recorded as the creator of each new rule.

    Returns:
        The rules that were newly created (already-existing seed rules are
        skipped and omitted from the result).
    """
    existing_ids = set(store.list_rule_ids())
    created = []
    for rule in SEED_RULES:
        if rule.id in existing_ids:
            continue
        try:
            created.append(
                store.create(rule, actor=actor, note="seed legacy baseline rule for BAF, not statistically validated")
            )
        except RuleAlreadyExistsError:
            continue
    return created


if __name__ == "__main__":
    for created_rule in seed_baseline_rules(RuleStore()):
        print(f"created {created_rule.id} (v{created_rule.version}, {created_rule.status})")
