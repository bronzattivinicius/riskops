"""Tests for riskops.diagnostics."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from riskops.diagnostics import (
    RuleAssessment,
    diagnosticar_regra,
    impressao_digital_casos,
    metrica_citada,
    montar_prompt,
    veredito_referencia,
    verificar_caso,
)
from riskops.metrics.backtest import ClassificationMetrics
from riskops.rules.schema import Condition, ConditionGroup, Operator, Rule, RuleStatus
from riskops.rules.store import RuleStore

_ACTOR = "test-actor"


def _metrics(
    true_positives=0, false_positives=0, true_negatives=0, false_negatives=0
) -> ClassificationMetrics:
    """Builds a ClassificationMetrics from raw confusion-matrix counts."""
    total = true_positives + false_positives + true_negatives + false_negatives
    fraud_count = true_positives + false_negatives
    flagged_count = true_positives + false_positives
    return ClassificationMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        total=total,
        fraud_count=fraud_count,
        flagged_count=flagged_count,
    )


def _build_rule(rule_id: str = "r1") -> Rule:
    """Builds a minimal, valid Rule for diagnostics tests."""
    now = datetime.now(UTC)
    return Rule(
        id=rule_id,
        name="Regra de teste",
        version=1,
        status=RuleStatus.CANDIDATE,
        description="regra usada apenas em teste",
        logic=ConditionGroup(conditions=[Condition(field="credit_risk_score", operator=Operator.GT, value=100)]),
        created_at=now,
        updated_at=now,
        created_by=_ACTOR,
        updated_by=_ACTOR,
    )


class _FakeUsage(dict):
    """Minimal stand-in for an AIMessage's usage_metadata mapping."""


class _FakeRawMessage:
    """Minimal stand-in for the raw AIMessage returned alongside structured output."""

    def __init__(self, input_tokens: int = 10, output_tokens: int = 20):
        self.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}


class _FakeStructuredLLM:
    """Stand-in for `llm.with_structured_output(RuleAssessment, include_raw=True)`."""

    def __init__(self, parsed: RuleAssessment | None, parsing_error: str | None = None):
        self._parsed = parsed
        self._parsing_error = parsing_error

    def invoke(self, _prompt: str) -> dict:
        return {"parsed": self._parsed, "raw": _FakeRawMessage(), "parsing_error": self._parsing_error}


def test_veredito_referencia_zero_match_is_aposentar() -> None:
    """A rule that matches nothing is always 'aposentar'."""
    metricas = _metrics()
    assert veredito_referencia(metricas, baseline_fraud_rate=0.011) == "aposentar"


def test_veredito_referencia_strong_lift_low_fpr_is_manter() -> None:
    """High precision lift with a low false positive rate is 'manter'."""
    metricas = _metrics(true_positives=60, false_positives=40, true_negatives=9860, false_negatives=40)
    assert metricas.precision == pytest.approx(0.6)
    assert veredito_referencia(metricas, baseline_fraud_rate=0.01) == "manter"


def test_veredito_referencia_weak_lift_is_aposentar() -> None:
    """Low precision lift is 'aposentar', regardless of false positive rate."""
    metricas = _metrics(true_positives=1, false_positives=99, true_negatives=9800, false_negatives=99)
    assert veredito_referencia(metricas, baseline_fraud_rate=0.02) == "aposentar"


def test_veredito_referencia_middle_ground_is_revisar() -> None:
    """A moderate lift (2x) that clears neither threshold is 'revisar'."""
    metricas = _metrics(true_positives=2, false_positives=98, true_negatives=9802, false_negatives=98)
    assert metricas.precision == pytest.approx(0.02)
    ref = veredito_referencia(metricas, baseline_fraud_rate=0.01)
    assert ref == "revisar"


def test_metrica_citada_finds_rounded_value_with_tolerance() -> None:
    """A percentage cited with rounding is still found within tolerance."""
    texto = "A taxa de deteccao e de 9,5% e a de falso positivo e 2,4%."
    assert metrica_citada(texto, 0.095)
    assert metrica_citada(texto, 0.024)


def test_metrica_citada_returns_false_when_absent() -> None:
    """No matching number in the text means the metric was not cited."""
    assert not metrica_citada("Nada de relevante aqui, so ruido.", 0.50)


def test_montar_prompt_includes_rule_and_metrics() -> None:
    """The prompt cites the rule's identity and the key backtest metrics."""
    rule = _build_rule("baf_test_rule")
    metricas = _metrics(true_positives=10, false_positives=5, true_negatives=980, false_negatives=5)
    prompt = montar_prompt(rule, metricas, baseline_fraud_rate=0.015)
    assert "baf_test_rule" in prompt
    assert "MANTIDA" in prompt and "REVISADA" in prompt and "APOSENTADA" in prompt


def test_diagnosticar_regra_empty_rule_id_is_graceful_error(
    synthetic_applications_df: pd.DataFrame, tmp_rule_store: RuleStore
) -> None:
    """An empty rule_id returns ok=False without touching the store or the LLM."""
    resultado = diagnosticar_regra(
        "", store=tmp_rule_store, df=synthetic_applications_df, structured_llm=None, label_col="fraud_bool"
    )
    assert resultado["ok"] is False
    assert resultado["chamadas_llm"] == 0


def test_diagnosticar_regra_missing_rule_is_graceful_error(
    synthetic_applications_df: pd.DataFrame, tmp_rule_store: RuleStore
) -> None:
    """A rule_id absent from the registry returns ok=False, not an exception."""
    resultado = diagnosticar_regra(
        "regra_fantasma",
        store=tmp_rule_store,
        df=synthetic_applications_df,
        structured_llm=None,
        label_col="fraud_bool",
    )
    assert resultado["ok"] is False
    assert "nao encontrada" in resultado["erro"]
    assert resultado["chamadas_llm"] == 0


def test_diagnosticar_regra_success(synthetic_applications_df: pd.DataFrame, tmp_rule_store: RuleStore) -> None:
    """A successful run returns the parsed verdict, metrics, and instrumentation."""
    rule = _build_rule("baf_test_rule")
    tmp_rule_store.create(rule, actor=_ACTOR, note="setup")
    fake_llm = _FakeStructuredLLM(
        parsed=RuleAssessment(veredito="manter", justificativa="citando numeros", sugestao="nada a fazer")
    )

    resultado = diagnosticar_regra(
        "baf_test_rule",
        store=tmp_rule_store,
        df=synthetic_applications_df,
        structured_llm=fake_llm,
        label_col="fraud_bool",
    )

    assert resultado["ok"] is True
    assert resultado["veredito"] == "manter"
    assert resultado["chamadas_llm"] == 1
    assert resultado["tokens_entrada"] == 10
    assert resultado["veredito_referencia"] in ("manter", "revisar", "aposentar")


def test_diagnosticar_regra_parsing_failure_is_graceful_error(
    synthetic_applications_df: pd.DataFrame, tmp_rule_store: RuleStore
) -> None:
    """A model response that fails structured parsing is reported, not raised."""
    rule = _build_rule("baf_test_rule")
    tmp_rule_store.create(rule, actor=_ACTOR, note="setup")
    fake_llm = _FakeStructuredLLM(parsed=None, parsing_error="resposta nao seguiu o schema")

    resultado = diagnosticar_regra(
        "baf_test_rule",
        store=tmp_rule_store,
        df=synthetic_applications_df,
        structured_llm=fake_llm,
        label_col="fraud_bool",
    )

    assert resultado["ok"] is False
    assert "falha ao interpretar" in resultado["erro"]
    assert resultado["chamadas_llm"] == 1


def test_verificar_caso_manual_returns_none() -> None:
    """A manually-verified case is never auto-approved or auto-rejected."""
    caso = {"verificacao": "manual", "tipo": "ambiguo"}
    aprovado, observacao = verificar_caso(caso, {})
    assert aprovado is None
    assert "manual" in observacao


def test_verificar_caso_expected_error_checks_graceful_handling() -> None:
    """An expected-error case passes only when the error was handled gracefully."""
    caso = {"verificacao": "auto", "tipo": "informacao ausente"}
    aprovado, _ = verificar_caso(caso, {"ok": False, "erro": "regra nao encontrada"})
    assert aprovado is True

    aprovado, _ = verificar_caso(caso, {"ok": True, "erro": None})
    assert aprovado is False


def test_verificar_caso_checks_citation_and_verdict_agreement() -> None:
    """A normal case passes only when both the citation and the verdict match."""
    metricas = _metrics(true_positives=10, false_positives=90, true_negatives=9800, false_negatives=100)
    caso = {"verificacao": "auto", "tipo": "normal"}
    resultado_ok = {
        "ok": True,
        "veredito": "aposentar",
        "veredito_referencia": "aposentar",
        "justificativa": f"detection_rate={metricas.detection_rate:.1%}",
        "metricas_backtest": metricas,
    }
    aprovado, _ = verificar_caso(caso, resultado_ok)
    assert aprovado is True

    resultado_sem_citacao = {**resultado_ok, "justificativa": "sem numeros aqui"}
    aprovado, _ = verificar_caso(caso, resultado_sem_citacao)
    assert aprovado is False

    resultado_veredito_diferente = {**resultado_ok, "veredito": "manter"}
    aprovado, _ = verificar_caso(caso, resultado_veredito_diferente)
    assert aprovado is False


def test_impressao_digital_casos_is_stable_and_order_sensitive() -> None:
    """The fingerprint is deterministic, and differs when the case set actually changes."""
    casos_a = [{"id": "T01", "rule_id": "r1"}, {"id": "T02", "rule_id": "r2"}]
    casos_b = [{"id": "T01", "rule_id": "r1"}, {"id": "T02", "rule_id": "r2"}]
    casos_c = [{"id": "T01", "rule_id": "r1"}, {"id": "T02", "rule_id": "r2-diferente"}]

    assert impressao_digital_casos(casos_a) == impressao_digital_casos(casos_b)
    assert impressao_digital_casos(casos_a) != impressao_digital_casos(casos_c)


class _RaisingStructuredLLM:
    """Stand-in for a model call that fails server-side (e.g. malformed JSON)."""

    def invoke(self, _prompt: str) -> dict:
        raise RuntimeError("Failed to parse tool call arguments as JSON")


def test_diagnosticar_regra_server_side_failure_is_graceful_error(
    synthetic_applications_df: pd.DataFrame, tmp_rule_store: RuleStore
) -> None:
    """A raised exception from the model call (not just a null parse) is also caught."""
    rule = _build_rule("baf_test_rule")
    tmp_rule_store.create(rule, actor=_ACTOR, note="setup")

    resultado = diagnosticar_regra(
        "baf_test_rule",
        store=tmp_rule_store,
        df=synthetic_applications_df,
        structured_llm=_RaisingStructuredLLM(),
        label_col="fraud_bool",
    )

    assert resultado["ok"] is False
    assert "chamada ao modelo falhou" in resultado["erro"]
    assert resultado["chamadas_llm"] == 1
