"""Tests for riskops.robustez, using fake/deterministic components (no real API calls)."""

import random
from datetime import UTC, datetime

import httpx
import pandas as pd
import pytest
from groq import APIError as GroqAPIError
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from riskops.robustez import (
    FonteInstavel,
    RuleAssessmentRobusta,
    com_repeticao,
    criar_ferramenta_instavel,
    criar_structured_llm_instavel,
    diagnosticar_com_robustez,
    diagnosticar_regra_robusta,
    invocar_grafo_com_limite,
    verificar_saida_robusta,
)
from riskops.metrics.backtest import ClassificationMetrics
from riskops.rules.schema import Condition, ConditionGroup, Operator, Rule, RuleStatus
from riskops.rules.store import RuleStore

_ACTOR = "test-actor"


def _erro_transiente() -> GroqAPIError:
    """Builds a real GroqAPIError instance, standing in for a transient provider failure."""
    return GroqAPIError("erro simulado", httpx.Request("GET", "http://test"), body=None)


def _build_rule(rule_id: str) -> Rule:
    """Builds a minimal, valid, active Rule (single condition) for tests."""
    now = datetime.now(UTC)
    return Rule(
        id=rule_id,
        name=rule_id,
        version=1,
        status=RuleStatus.ACTIVE,
        description="regra usada apenas em teste",
        logic=ConditionGroup(conditions=[Condition(field="email_is_free", operator=Operator.EQ, value=1)]),
        created_at=now,
        updated_at=now,
        created_by=_ACTOR,
        updated_by=_ACTOR,
    )


@tool
def _ferramenta_base(rule_id: str) -> str:
    """Ferramenta real de teste, usada como base para a versao instavel."""
    return f"historico de {rule_id}"


def test_criar_ferramenta_instavel_sempre_falha_com_probabilidade_um() -> None:
    """With falha_prob=1.0, every call raises FonteInstavel instead of delegating."""
    ferramenta = criar_ferramenta_instavel(_ferramenta_base, falha_prob=1.0, rng=random.Random(0))
    with pytest.raises(FonteInstavel):
        ferramenta.invoke({"rule_id": "regra_x"})


def test_criar_ferramenta_instavel_nunca_falha_com_probabilidade_zero() -> None:
    """With falha_prob=0.0, every call delegates to the real underlying tool."""
    ferramenta = criar_ferramenta_instavel(_ferramenta_base, falha_prob=0.0, rng=random.Random(0))
    assert ferramenta.invoke({"rule_id": "regra_x"}) == "historico de regra_x"


def test_criar_structured_llm_instavel_falha_prob_um_nunca_chama_o_modelo_real() -> None:
    """With falha_prob=1.0, the wrapped model is never actually invoked (zero API cost)."""

    class _ModeloQueNuncaDeveSerChamado:
        def invoke(self, prompt):
            raise AssertionError("o modelo real nao deveria ser chamado")

    llm_instavel = criar_structured_llm_instavel(
        _ModeloQueNuncaDeveSerChamado(), falha_prob=1.0, rng=random.Random(0)
    )
    with pytest.raises(GroqAPIError):
        llm_instavel.invoke("prompt qualquer")


def test_criar_structured_llm_instavel_falha_prob_zero_sempre_delega() -> None:
    """With falha_prob=0.0, every call delegates to the real underlying model."""
    llm_instavel = criar_structured_llm_instavel(
        _FakeStructuredLLMSucesso(), falha_prob=0.0, rng=random.Random(0)
    )
    resultado = llm_instavel.invoke("prompt qualquer")
    assert resultado["parsed"].veredito == "manter"


def test_com_repeticao_esgota_tentativas_e_registra_erros() -> None:
    """A permanently-failing transient error uses every attempt and returns None."""

    def sempre_falha():
        raise FonteInstavel("indisponivel")

    resultado, info = com_repeticao(sempre_falha, tentativas=3, espera_inicial=0.0)

    assert resultado is None
    assert info["tentativas"] == 3
    assert len(info["erros"]) == 3


def test_com_repeticao_nao_repete_erro_nao_transitorio() -> None:
    """An error outside `transitorias` propagates immediately, without being retried."""

    def falha_permanente():
        raise ValueError("isso nao e transitorio")

    with pytest.raises(ValueError):
        com_repeticao(falha_permanente, tentativas=3, espera_inicial=0.0)


def test_com_repeticao_retorna_sucesso_sem_esgotar_tentativas() -> None:
    """A call that succeeds on the second attempt stops retrying immediately."""
    chamadas = {"n": 0}

    def falha_uma_vez():
        chamadas["n"] += 1
        if chamadas["n"] == 1:
            raise FonteInstavel("indisponivel na primeira tentativa")
        return "ok"

    resultado, info = com_repeticao(falha_uma_vez, tentativas=5, espera_inicial=0.0)

    assert resultado == "ok"
    assert info["tentativas"] == 2
    assert len(info["erros"]) == 1


_METRICAS = ClassificationMetrics(
    true_positives=4,
    false_positives=2,
    true_negatives=3,
    false_negatives=1,
    total=10,
    fraud_count=5,
    flagged_count=6,
)


def test_verificar_saida_robusta_aprova_caso_com_evidencia_real() -> None:
    """A justification citing a real backtest metric passes verification."""
    avaliacao = RuleAssessmentRobusta(
        veredito="revisar",
        justificativa=f"taxa de deteccao de {_METRICAS.detection_rate:.1%}, requer atencao",
        sugestao="revisar limites",
        confianca="media",
    )
    resultado = verificar_saida_robusta(avaliacao, _METRICAS, baseline_fraud_rate=0.5)
    assert resultado == {"ok": True, "problemas": []}


def test_verificar_saida_robusta_pega_confianca_alta_sem_evidencia() -> None:
    """High declared confidence with a fabricated (non-matching) number is flagged."""
    avaliacao = RuleAssessmentRobusta(
        veredito="manter",
        justificativa="a regra tem 99% de eficacia, sem duvida",
        sugestao="manter como esta",
        confianca="alta",
    )
    resultado = verificar_saida_robusta(avaliacao, _METRICAS, baseline_fraud_rate=0.5)
    assert resultado["ok"] is False
    assert any("confianca declarada" in p for p in resultado["problemas"])


def test_verificar_saida_robusta_pega_proposta_de_nova_condicao() -> None:
    """A diagnostic justification proposing a new rule condition is flagged as scope creep."""
    avaliacao = RuleAssessmentRobusta(
        veredito="revisar",
        justificativa=f"com deteccao de {_METRICAS.detection_rate:.1%}, sugiro criar uma condicao adicional",
        sugestao="revisar",
        confianca="baixa",
    )
    resultado = verificar_saida_robusta(avaliacao, _METRICAS, baseline_fraud_rate=0.5)
    assert resultado["ok"] is False
    assert any("propoe uma nova condicao" in p for p in resultado["problemas"])


class _FakeStructuredLLMSucesso:
    """Fake structured-output model that always answers successfully, first try."""

    def invoke(self, prompt):
        avaliacao = RuleAssessmentRobusta(
            veredito="manter", justificativa="citando 40% de deteccao", sugestao="manter", confianca="media"
        )
        return {"parsed": avaliacao, "raw": AIMessage(content="")}


class _FakeStructuredLLMSempreTransiente:
    """Fake structured-output model that always raises a transient provider error."""

    def invoke(self, prompt):
        raise _erro_transiente()


@pytest.fixture
def store_com_regra(tmp_rule_store: RuleStore) -> RuleStore:
    tmp_rule_store.create(_build_rule("regra_teste"), actor=_ACTOR, note="setup")
    return tmp_rule_store


def test_diagnosticar_com_robustez_caminho_normal(
    store_com_regra: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """A successful first-attempt call returns ok=True, degradado=False, with verification run."""
    resultado = diagnosticar_com_robustez(
        "regra_teste",
        store=store_com_regra,
        df=synthetic_applications_df,
        structured_llm_robusto=_FakeStructuredLLMSucesso(),
    )
    assert resultado["ok"] is True
    assert resultado["degradado"] is False
    assert resultado["veredito"] == "manter"
    assert resultado["tentativas"] == 1


def test_diagnosticar_com_robustez_degrada_apos_falhas_transitorias(
    store_com_regra: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """Exhausting retries on a transient provider error degrades gracefully instead of failing."""
    resultado = diagnosticar_com_robustez(
        "regra_teste",
        store=store_com_regra,
        df=synthetic_applications_df,
        structured_llm_robusto=_FakeStructuredLLMSempreTransiente(),
        tentativas=2,
    )
    assert resultado["ok"] is True
    assert resultado["degradado"] is True
    assert resultado["confianca"] is None
    assert resultado["veredito"] == resultado["veredito_referencia"]
    assert resultado["tentativas"] == 2
    assert len(resultado["erros_transitorios"]) == 2


def test_diagnosticar_com_robustez_regra_inexistente_nao_degrada(
    store_com_regra: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """A missing rule id is a permanent error, not a degradation case."""
    resultado = diagnosticar_com_robustez(
        "regra_que_nao_existe",
        store=store_com_regra,
        df=synthetic_applications_df,
        structured_llm_robusto=_FakeStructuredLLMSucesso(),
    )
    assert resultado["ok"] is False
    assert resultado["degradado"] is False
    assert "nao encontrada" in resultado["erro"]


class _FakeStructuredLLMFalhaDeFormatoUmaVez:
    """Returns a 'functions.X'-style unparseable response once, then succeeds (real Groq quirk)."""

    def __init__(self):
        self.chamadas = 0

    def invoke(self, prompt):
        self.chamadas += 1
        if self.chamadas == 1:
            return {"parsed": None, "raw": AIMessage(content=""), "parsing_error": "Unknown tool type: 'functions.X'"}
        avaliacao = RuleAssessmentRobusta(
            veredito="manter", justificativa="citando 40% de deteccao", sugestao="manter", confianca="media"
        )
        return {"parsed": avaliacao, "raw": AIMessage(content=""), "parsing_error": None}


class _FakeStructuredLLMFalhaDeFormatoSempre:
    """Always returns a 'functions.X'-style unparseable response (simulated persistent quirk)."""

    def invoke(self, prompt):
        return {"parsed": None, "raw": AIMessage(content=""), "parsing_error": "Unknown tool type: 'functions.X'"}


def test_diagnosticar_com_robustez_tenta_de_novo_apos_falha_de_formato(
    store_com_regra: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """A one-off 'functions.X' tool-name parsing failure is retried, not treated as permanent."""
    resultado = diagnosticar_com_robustez(
        "regra_teste",
        store=store_com_regra,
        df=synthetic_applications_df,
        structured_llm_robusto=_FakeStructuredLLMFalhaDeFormatoUmaVez(),
        tentativas=3,
        espera_inicial=0.0,
    )
    assert resultado["ok"] is True
    assert resultado["degradado"] is False
    assert resultado["veredito"] == "manter"
    assert resultado["tentativas"] == 2


def test_diagnosticar_com_robustez_degrada_se_falha_de_formato_persiste(
    store_com_regra: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """Exhausting retries on a persistent parsing failure degrades gracefully, same as a provider error."""
    resultado = diagnosticar_com_robustez(
        "regra_teste",
        store=store_com_regra,
        df=synthetic_applications_df,
        structured_llm_robusto=_FakeStructuredLLMFalhaDeFormatoSempre(),
        tentativas=2,
        espera_inicial=0.0,
    )
    assert resultado["ok"] is True
    assert resultado["degradado"] is True
    assert resultado["tentativas"] == 2


def test_diagnosticar_regra_robusta_aceita_variante_em_memoria(
    synthetic_applications_df: pd.DataFrame,
) -> None:
    """A rule variant built in memory (different description, same logic) never touches a store."""
    rule = _build_rule("regra_variante")
    variante = rule.model_copy(update={"description": "descricao alternativa, so para este teste"})

    resultado = diagnosticar_regra_robusta(
        variante,
        df=synthetic_applications_df,
        structured_llm_robusto=_FakeStructuredLLMSucesso(),
    )
    assert resultado["ok"] is True
    assert resultado["degradado"] is False


class _GrafoFalso:
    """Fake compiled graph, used only to exercise invocar_grafo_com_limite's error path."""

    def __init__(self, exc: Exception | None):
        self._exc = exc

    def invoke(self, estado, config=None):
        if self._exc is not None:
            raise self._exc
        return {"ok": True}


def test_invocar_grafo_com_limite_sucesso() -> None:
    """A graph that completes normally is returned unchanged, marked as not degraded."""
    resultado = invocar_grafo_com_limite(_GrafoFalso(None), {})
    assert resultado == {"ok": True, "degradado": False, "estado_final": {"ok": True}}


def test_invocar_grafo_com_limite_converte_recursion_error() -> None:
    """A GraphRecursionError is converted into an explicit, self-identifying degraded result."""
    from langgraph.errors import GraphRecursionError

    resultado = invocar_grafo_com_limite(_GrafoFalso(GraphRecursionError("limite atingido")), {})
    assert resultado["ok"] is False
    assert resultado["degradado"] is True
    assert "limite de passos" in resultado["erro"]
