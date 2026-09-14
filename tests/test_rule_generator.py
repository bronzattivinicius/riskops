"""Tests for riskops.rule_generator (pure functions, no LLM)."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from riskops.metrics.backtest import backtest_ruleset
from riskops.rule_generator import (
    avaliar_candidata,
    candidata_melhorou,
    construir_regra_candidata,
    montar_prompt_gerador,
    resumir_comparacao,
)
from riskops.rules.schema import Condition, ConditionGroup, Operator, Rule, RuleStatus

_ACTOR = "test-actor"


def _rule(field: str, operator: Operator, value) -> Rule:
    """Builds a minimal, valid, active Rule with a single condition."""
    now = datetime(2026, 9, 1, tzinfo=UTC)
    return Rule(
        id="regra_teste",
        name="Regra de teste",
        version=1,
        status=RuleStatus.ACTIVE,
        description="regra usada apenas em teste",
        logic=ConditionGroup(
            conditions=[Condition(field=field, operator=operator, value=value)]
        ),
        created_at=now,
        updated_at=now,
        created_by=_ACTOR,
        updated_by=_ACTOR,
    )


def test_construir_regra_candidata_substitui_por_padrao(
    synthetic_applications_df: pd.DataFrame,
) -> None:
    """Without combinar_com_atual, the candidate's logic fully replaces the current one."""
    atual = _rule("email_is_free", Operator.EQ, 1)
    candidata = construir_regra_candidata(
        atual, campo="foreign_request", operador="eq", valor=1, combinar_com_atual=False
    )

    assert candidata.id == atual.id
    assert candidata.version == atual.version + 1
    assert candidata.status == RuleStatus.CANDIDATE
    assert len(candidata.logic.conditions) == 1
    assert candidata.logic.conditions[0].field == "foreign_request"


def test_construir_regra_candidata_combina_com_atual() -> None:
    """With combinar_com_atual=True, the new condition is ANDed onto the existing ones."""
    atual = _rule("email_is_free", Operator.EQ, 1)
    candidata = construir_regra_candidata(
        atual,
        campo="credit_risk_score",
        operador="gt",
        valor=200,
        combinar_com_atual=True,
    )

    assert len(candidata.logic.conditions) == 2
    campos = {c.field for c in candidata.logic.conditions}
    assert campos == {"email_is_free", "credit_risk_score"}


def test_construir_regra_candidata_rejeita_operador_nao_suportado() -> None:
    """Operators outside the comparable set (e.g. list-based 'in') are rejected."""
    atual = _rule("email_is_free", Operator.EQ, 1)
    with pytest.raises(ValueError, match="nao suportado"):
        construir_regra_candidata(
            atual, campo="foo", operador="in", valor=1, combinar_com_atual=False
        )


def test_avaliar_candidata_melhora_precisao_sem_piorar_falso_positivo(
    synthetic_applications_df: pd.DataFrame,
) -> None:
    """A candidate that narrows a noisy rule to a purely-fraud subset is a real improvement."""
    atual = _rule("email_is_free", Operator.EQ, 1)
    candidata = construir_regra_candidata(
        atual,
        campo="credit_risk_score",
        operador="gt",
        valor=200,
        combinar_com_atual=True,
    )

    comparacao = avaliar_candidata(
        synthetic_applications_df, atual, candidata, label_col="fraud_bool"
    )

    assert comparacao.baseline.metrics.precision == pytest.approx(0.4)
    assert comparacao.candidate.metrics.precision == pytest.approx(1.0)
    assert comparacao.delta_precision > 0
    assert comparacao.delta_false_positive_rate < 0
    assert candidata_melhorou(comparacao) is True


def test_avaliar_candidata_no_improvement(
    synthetic_applications_df: pd.DataFrame,
) -> None:
    """A candidate that only catches legitimate, low-velocity rows is not an improvement."""
    atual = _rule("email_is_free", Operator.EQ, 1)
    candidata = construir_regra_candidata(
        atual, campo="velocity_6h", operador="lt", valor=1000, combinar_com_atual=False
    )

    comparacao = avaliar_candidata(
        synthetic_applications_df, atual, candidata, label_col="fraud_bool"
    )

    assert comparacao.candidate.metrics.true_positives == 0
    assert candidata_melhorou(comparacao) is False


def test_candidata_melhorou_rejeita_ganho_sem_relevancia_se_fpr_piora_muito(
    synthetic_applications_df: pd.DataFrame,
) -> None:
    """A precision gain does not count as an improvement if the false-positive cost is too high."""
    atual = _rule("credit_risk_score", Operator.GT, 300)
    # Solta a regra ao ponto de capturar quase tudo: precisao cai mas, mais importante,
    # o falso positivo dispara -- o ganho de deteccao nao compensa.
    candidata = construir_regra_candidata(
        atual,
        campo="credit_risk_score",
        operador="gt",
        valor=-1000,
        combinar_com_atual=False,
    )

    comparacao = avaliar_candidata(
        synthetic_applications_df, atual, candidata, label_col="fraud_bool"
    )

    assert comparacao.delta_false_positive_rate > 0.05
    assert candidata_melhorou(comparacao) is False


def test_montar_prompt_gerador_cita_regra_veredito_e_metricas(
    synthetic_applications_df: pd.DataFrame,
) -> None:
    """The generator's prompt cites the current rule, the diagnostic verdict and the metrics."""
    atual = _rule("email_is_free", Operator.EQ, 1)
    metricas = backtest_ruleset(
        synthetic_applications_df, [atual], label_col="fraud_bool"
    ).metrics

    prompt = montar_prompt_gerador(
        atual,
        metricas,
        veredito_diagnostico="revisar",
        justificativa_diagnostico="precisao baixa, 40%",
    )

    assert atual.name in prompt
    assert "email_is_free" in prompt
    assert "revisar" in prompt
    assert "precisao baixa, 40%" in prompt
    assert f"{metricas.precision:.1%}" in prompt


def test_resumir_comparacao_cita_metricas_de_ambos_os_lados(
    synthetic_applications_df: pd.DataFrame,
) -> None:
    """The text summary cites both the current rule's and the candidate's metrics."""
    atual = _rule("email_is_free", Operator.EQ, 1)
    candidata = construir_regra_candidata(
        atual,
        campo="credit_risk_score",
        operador="gt",
        valor=200,
        combinar_com_atual=True,
    )
    comparacao = avaliar_candidata(
        synthetic_applications_df, atual, candidata, label_col="fraud_bool"
    )

    resumo = resumir_comparacao(comparacao)

    assert "Regra atual" in resumo
    assert "Candidata" in resumo
    assert "Delta" in resumo
