"""Candidate-rule generation: propose and backtest a new condition for a rule.

This is the logic behind Deliverable 3's second agent. Unlike
``riskops.diagnostics`` (which only judges an existing rule), this module
lets a model *propose* a replacement or additional condition for a rule that
was diagnosed as needing attention, and tests the proposal for real via the
Phase 1 backtest engine (``compare_rulesets``) before anything is finalized.
Nothing here writes to the rule registry -- a generated candidate is handed
back as an in-memory :class:`~riskops.rules.schema.Rule` with
``status=CANDIDATE``; persisting or promoting it is a human decision, out of
scope for this deliverable.

The proposal schema is intentionally flat (a single condition, plus a flag
for whether to combine it with the rule's existing logic) rather than an
arbitrary nested :class:`~riskops.rules.schema.ConditionGroup`: Deliverable 1
and 2 both found that this model occasionally produces malformed structured
output under complex schemas, and a flat shape mirrors the tool-call schema
exactly, so a model call that decides to iterate (test, then propose again)
stays simple and robust.
"""

from typing import Literal

from pydantic import BaseModel, Field

from riskops.metrics.backtest import (
    ClassificationMetrics,
    ComparisonResult,
    compare_rulesets,
)
from riskops.rules.schema import (
    Condition,
    ConditionGroup,
    LogicOperator,
    Operator,
    Rule,
    RuleStatus,
)

_COMPARABLE_OPERATORS = ("eq", "ne", "gt", "gte", "lt", "lte")


class RegraCandidataFinal(BaseModel):
    """Structured final proposal produced by the rule-generator agent.

    Attributes:
        campo: Name of the transaction field the proposed condition compares.
        operador: Comparison operator, restricted to scalar-valued operators
            (no list/null operators) for robustness of structured output.
        valor: Numeric threshold proposed for the comparison.
        combinar_com_atual: Whether the proposed condition should be ANDed
            with the rule's existing logic (narrowing it) or replace it
            entirely.
        justificativa: Justification citing the backtest metrics of the
            tested proposal.
        sugestao: A concrete next step (e.g. "promover a candidata" or
            "regra nao parece melhoravel por ajuste de limiar").
    """

    campo: str = Field(
        description="Nome exato do campo da transacao, ex.: 'velocity_6h'."
    )
    operador: Literal["eq", "ne", "gt", "gte", "lt", "lte"] = Field(
        description="Operador de comparacao proposto."
    )
    valor: float = Field(description="Valor numerico de comparacao proposto.")
    combinar_com_atual: bool = Field(
        description="Se True, combina (AND) com a logica atual da regra; se False, substitui."
    )
    justificativa: str = Field(
        description="Justificativa citando as metricas de backtest da proposta testada."
    )
    sugestao: str = Field(description="Proximo passo concreto e acionavel.")


def montar_prompt_gerador(
    rule: Rule,
    metricas: ClassificationMetrics,
    veredito_diagnostico: str,
    justificativa_diagnostico: str,
) -> str:
    """Builds the prompt for the rule-generator agent.

    Built only from the diagnostic agent's *structured* output (verdict and
    justification) and the rule's own current logic/metrics -- never from
    the diagnostic agent's raw conversation or tool-call history. This is
    the explicit context contract between the two agents (Deliverable 3,
    Section 3.3): the generator sees a summarized slice, not everything the
    diagnostic agent saw.

    Args:
        rule: The rule that was diagnosed and now needs a candidate.
        metricas: Current backtest metrics for ``rule``.
        veredito_diagnostico: The diagnostic agent's verdict ("revisar" or
            "aposentar" -- "manter" never reaches this agent).
        justificativa_diagnostico: The diagnostic agent's justification text.

    Returns:
        The full prompt text, in Portuguese.
    """
    condicoes_atuais = ", ".join(
        f"{c.field} {c.operator.value} {c.value}"
        for c in rule.logic.conditions
        if isinstance(c, Condition)
    )
    return f"""Voce e um analista de risco propondo uma regra candidata para substituir ou
complementar uma regra de deteccao de fraude que outro analista ja diagnosticou como
problematica.

Regra atual: {rule.name} (id: {rule.id})
Logica atual ({rule.logic.logic.value}): {condicoes_atuais}

Metricas de backtest da regra atual contra {metricas.total} transacoes historicas:
- Taxa de deteccao (recall): {metricas.detection_rate:.1%}
- Taxa de falso positivo: {metricas.false_positive_rate:.1%}
- Precisao: {metricas.precision:.1%}
- Total sinalizado: {metricas.flagged_count} de {metricas.total}

Diagnostico recebido (de outro agente, ja concluido): veredito "{veredito_diagnostico}".
Justificativa do diagnostico: {justificativa_diagnostico}

Sua tarefa: proponha UMA condicao (campo, operador, valor) que, combinada com a logica atual
ou substituindo-a, melhore a precisao ou a taxa de deteccao sem piorar desproporcionalmente a
taxa de falso positivo. Use a ferramenta de teste para verificar sua proposta contra os dados
reais antes de finalizar -- nao proponha as cegas. Depois de testar, de o parecer final."""


def construir_regra_candidata(
    rule: Rule,
    campo: str,
    operador: str,
    valor: float,
    combinar_com_atual: bool,
    actor: str = "rule-generator-agent",
) -> Rule:
    """Builds an in-memory candidate rule from a proposed condition.

    Never persisted here -- the caller decides whether/how to store it.

    Args:
        rule: The rule the candidate is based on.
        campo: Name of the transaction field for the new condition.
        operador: One of "eq", "ne", "gt", "gte", "lt", "lte".
        valor: Numeric threshold for the new condition.
        combinar_com_atual: If True, ANDs the new condition with the rule's
            existing conditions; if False, replaces them entirely.
        actor: Identity recorded as the author of this in-memory version.

    Returns:
        A new :class:`Rule` instance (same id, incremented version,
        ``status=CANDIDATE``), not written to any store.

    Raises:
        ValueError: If ``operador`` is not one of the comparable operators.
    """
    if operador not in _COMPARABLE_OPERATORS:
        raise ValueError(
            f"operador {operador!r} nao suportado; use um de {_COMPARABLE_OPERATORS}"
        )

    nova_condicao = Condition(field=campo, operator=Operator(operador), value=valor)
    if combinar_com_atual:
        novas_condicoes = [*rule.logic.conditions, nova_condicao]
    else:
        novas_condicoes = [nova_condicao]

    return rule.model_copy(
        update={
            "logic": ConditionGroup(
                logic=LogicOperator.AND, conditions=novas_condicoes
            ),
            "version": rule.version + 1,
            "status": RuleStatus.CANDIDATE,
            "updated_by": actor,
        }
    )


def avaliar_candidata(
    df, rule_atual: Rule, rule_candidata: Rule, label_col: str = "fraud_bool"
) -> ComparisonResult:
    """Backtests a candidate rule against the current rule it would replace.

    Thin wrapper over :func:`riskops.metrics.backtest.compare_rulesets`,
    named for this module's vocabulary.

    Args:
        df: Historical transaction data to backtest against.
        rule_atual: The rule currently in the registry.
        rule_candidata: The proposed in-memory candidate.
        label_col: Name of the boolean fraud-label column in ``df``.

    Returns:
        The comparison result, including per-ruleset backtests and deltas.
    """
    return compare_rulesets(df, [rule_atual], [rule_candidata], label_col=label_col)


def candidata_melhorou(comparacao: ComparisonResult) -> bool:
    """Decides, deterministically, whether a candidate is an improvement.

    A candidate is considered an improvement if it raises precision or
    detection rate without a disproportionate false-positive cost (more
    than 5 percentage points worse). This decision is never taken on the
    model's own word -- it is recomputed here from the real backtest
    numbers, the same way ``veredito_referencia`` grounds the diagnostic
    agent's verdict.

    Args:
        comparacao: Result of :func:`avaliar_candidata`.

    Returns:
        True if the candidate is judged an improvement over the current rule.
    """
    ganho_relevante = (
        comparacao.delta_precision > 0 or comparacao.delta_detection_rate > 0
    )
    piora_tolerada = comparacao.delta_false_positive_rate <= 0.05
    return ganho_relevante and piora_tolerada


def resumir_comparacao(comparacao: ComparisonResult) -> str:
    """Formats a candidate-vs-current comparison as a human-readable summary.

    Used both as the tool's return value (fed back to the generator agent
    as untrusted data, same convention as ``riskops.portfolio_graph``) and
    for notebook printing.

    Args:
        comparacao: Result of :func:`avaliar_candidata`.

    Returns:
        A multi-line text summary in Portuguese.
    """
    base = comparacao.baseline.metrics
    cand = comparacao.candidate.metrics
    return (
        f"Regra atual: deteccao={base.detection_rate:.1%}, falso_positivo={base.false_positive_rate:.1%}, "
        f"precisao={base.precision:.1%}, sinalizados={base.flagged_count}/{base.total}\n"
        f"Candidata: deteccao={cand.detection_rate:.1%}, falso_positivo={cand.false_positive_rate:.1%}, "
        f"precisao={cand.precision:.1%}, sinalizados={cand.flagged_count}/{cand.total}\n"
        f"Delta: deteccao={comparacao.delta_detection_rate:+.1%}, falso_positivo={comparacao.delta_false_positive_rate:+.1%}, "
        f"precisao={comparacao.delta_precision:+.1%}"
    )
