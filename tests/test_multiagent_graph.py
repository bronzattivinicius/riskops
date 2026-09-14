"""Tests for riskops.multiagent_graph, using fake chat models (no real API calls)."""

from datetime import UTC, datetime

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from riskops.diagnostics import RuleAssessment
from riskops.multiagent_graph import build_graph
from riskops.rule_generator import RegraCandidataFinal
from riskops.rules.schema import Condition, ConditionGroup, Operator, Rule, RuleStatus
from riskops.rules.store import RuleStore

_ACTOR = "test-actor"


def _build_rule(rule_id: str) -> Rule:
    """Builds a minimal, valid, active Rule (single condition) for graph tests."""
    now = datetime.now(UTC)
    return Rule(
        id=rule_id,
        name=rule_id,
        version=1,
        status=RuleStatus.ACTIVE,
        description="regra usada apenas em teste",
        logic=ConditionGroup(
            conditions=[Condition(field="email_is_free", operator=Operator.EQ, value=1)]
        ),
        created_at=now,
        updated_at=now,
        created_by=_ACTOR,
        updated_by=_ACTOR,
    )


class _FakeDiagnosticoLLM:
    """Fake diagnostic-agent model: never calls the history tool."""

    def invoke(self, messages):
        return AIMessage(content="Pronto para o parecer final.")


class _FakeStructuredDiagnostico:
    """Fake structured-output model: verdict depends on the rule id in the prompt."""

    def invoke(self, messages):
        texto = " ".join(str(getattr(m, "content", "")) for m in messages)
        if "regra_manter" in texto:
            veredito = "manter"
        elif "regra_revisar" in texto:
            veredito = "revisar"
        else:
            veredito = "aposentar"
        avaliacao = RuleAssessment(
            veredito=veredito, justificativa="citando 40%", sugestao="revisar a regra"
        )
        return {"parsed": avaliacao, "raw": AIMessage(content="")}


class _FakeGeradorLLM:
    """Fake generator-agent model: calls the tool only for one marked rule id."""

    def invoke(self, messages):
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="Pronto para o parecer final.")
        texto = " ".join(str(getattr(m, "content", "")) for m in messages)
        if "regra_testa_ferramenta" in texto:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "testar_regra_candidata",
                        "args": {
                            "rule_id": "regra_testa_ferramenta",
                            "campo": "credit_risk_score",
                            "operador": "gt",
                            "valor": 200,
                            "combinar_com_atual": True,
                        },
                        "id": "call_1",
                    }
                ],
            )
        return AIMessage(content="Nao preciso testar, ja sei o que propor.")


class _FakeBaseLLM:
    """Fake base model: `bind_tools` dispatches to the right per-agent fake by tool name."""

    def bind_tools(self, tools):
        nome = tools[0].name
        if nome == "consultar_historico_regra":
            return _FakeDiagnosticoLLM()
        if nome == "testar_regra_candidata":
            return _FakeGeradorLLM()
        raise AssertionError(f"unexpected tool bound: {nome}")


class _FakeStructuredGerador:
    """Fake structured-output model: always proposes the same fixed, real improvement."""

    def invoke(self, messages):
        proposta = RegraCandidataFinal(
            campo="credit_risk_score",
            operador="gt",
            valor=200,
            combinar_com_atual=True,
            justificativa="testado, melhora a precisao",
            sugestao="promover a candidata",
        )
        return {"parsed": proposta, "raw": AIMessage(content="")}


def _estado_inicial(fila: list[str]) -> dict:
    return {
        "fila_regras": fila,
        "resultados": [],
        "decisoes_roteamento": [],
        "chamadas_llm_diagnostico": 0,
        "chamadas_ferramenta_diagnostico": 0,
        "chamadas_llm_gerador": 0,
        "chamadas_ferramenta_gerador": 0,
        "tokens_entrada_total": 0,
        "tokens_saida_total": 0,
    }


@pytest.fixture
def store_com_tres_regras(tmp_rule_store: RuleStore) -> RuleStore:
    """A RuleStore with a 'manter', a 'revisar' and a tool-triggering rule."""
    for rule_id in ("regra_manter", "regra_revisar", "regra_testa_ferramenta"):
        tmp_rule_store.create(_build_rule(rule_id), actor=_ACTOR, note="setup")
    return tmp_rule_store


def _compile(store, df):
    builder = build_graph(
        store=store,
        df=df,
        llm=_FakeBaseLLM(),
        structured_llm_diagnostico=_FakeStructuredDiagnostico(),
        structured_llm_gerador=_FakeStructuredGerador(),
    )
    return builder.compile()


def test_graph_processes_every_rule_exactly_once(
    store_com_tres_regras: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """Every rule id in the queue appears exactly once in the results, no duplicates."""
    graph = _compile(store_com_tres_regras, synthetic_applications_df)

    resultado = graph.invoke(
        _estado_inicial(["regra_manter", "regra_revisar", "regra_testa_ferramenta"]),
        config={"recursion_limit": 100},
    )

    ids_processados = [r["id"] for r in resultado["resultados"]]
    assert sorted(ids_processados) == [
        "regra_manter",
        "regra_revisar",
        "regra_testa_ferramenta",
    ]
    assert len(ids_processados) == len(set(ids_processados))
    assert resultado["fila_regras"] == []


def test_gerador_nao_e_acionado_para_veredito_manter(
    store_com_tres_regras: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """A 'manter' verdict finalizes the rule without ever invoking the generator agent."""
    graph = _compile(store_com_tres_regras, synthetic_applications_df)

    resultado = graph.invoke(
        _estado_inicial(["regra_manter"]), config={"recursion_limit": 50}
    )

    (entrada,) = resultado["resultados"]
    assert entrada["veredito"] == "manter"
    assert entrada["candidata"] is None
    assert resultado["chamadas_llm_gerador"] == 0
    assert resultado["chamadas_ferramenta_gerador"] == 0

    (decisao,) = resultado["decisoes_roteamento"]
    assert decisao["agente_gerador"] is False
    assert "manter" in decisao["motivo"]


def test_gerador_e_acionado_e_produz_candidata_real_para_veredito_nao_manter(
    store_com_tres_regras: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """A 'revisar'/'aposentar' verdict triggers the generator, which produces a real, backtested candidate."""
    graph = _compile(store_com_tres_regras, synthetic_applications_df)

    resultado = graph.invoke(
        _estado_inicial(["regra_revisar"]), config={"recursion_limit": 50}
    )

    (entrada,) = resultado["resultados"]
    assert entrada["veredito"] == "revisar"
    assert entrada["candidata"] is not None
    assert entrada["candidata"]["erro"] is None
    # A proposta fixa (credit_risk_score > 200, combinada) eh uma melhora real e
    # verificavel contra o fixture, nao apenas uma afirmacao do modelo falso.
    assert entrada["candidata"]["melhorou"] is True
    assert entrada["candidata"]["delta_precisao"] > 0

    assert resultado["chamadas_llm_gerador"] >= 1
    assert (
        resultado["chamadas_ferramenta_gerador"] == 0
    )  # este caso nao aciona a ferramenta

    decisao = next(
        d for d in resultado["decisoes_roteamento"] if d["regra"] == "regra_revisar"
    )
    assert decisao["agente_gerador"] is True


def test_gerador_usa_ferramenta_de_teste_de_verdade(
    store_com_tres_regras: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """The generator's tool call is real: it runs a genuine backtest, not a stub."""
    graph = _compile(store_com_tres_regras, synthetic_applications_df)

    resultado = graph.invoke(
        _estado_inicial(["regra_testa_ferramenta"]), config={"recursion_limit": 50}
    )

    assert resultado["chamadas_ferramenta_gerador"] == 1
    (entrada,) = resultado["resultados"]
    assert entrada["candidata"]["melhorou"] is True


def test_observabilidade_e_por_agente_nao_agregada(
    store_com_tres_regras: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """The diagnostic and generator agents' call counts are tracked separately."""
    graph = _compile(store_com_tres_regras, synthetic_applications_df)

    resultado = graph.invoke(
        _estado_inicial(["regra_manter"]), config={"recursion_limit": 50}
    )

    # O agente de diagnostico trabalhou (2 chamadas: decisao + finalizacao); o gerador,
    # que nao foi acionado, tem custo zero -- essa distincao seria perdida num total unico.
    assert resultado["chamadas_llm_diagnostico"] == 2
    assert resultado["chamadas_llm_gerador"] == 0


def test_graph_respects_recursion_limit(
    store_com_tres_regras: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """A recursion_limit too low to finish the queue raises, as documented by LangGraph."""
    from langgraph.errors import GraphRecursionError

    graph = _compile(store_com_tres_regras, synthetic_applications_df)

    with pytest.raises(GraphRecursionError):
        graph.invoke(
            _estado_inicial(
                ["regra_manter", "regra_revisar", "regra_testa_ferramenta"]
            ),
            config={"recursion_limit": 2},
        )


def test_graph_handles_missing_rule_gracefully(
    store_com_tres_regras: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """A nonexistent or empty rule_id is reported, not raised, and the queue continues -- neither agent runs."""
    graph = _compile(store_com_tres_regras, synthetic_applications_df)

    resultado = graph.invoke(
        _estado_inicial(["regra_fantasma", "", "regra_manter"]),
        config={"recursion_limit": 50},
    )

    por_id = {r["id"]: r for r in resultado["resultados"]}
    assert por_id["regra_fantasma"]["ok"] is False
    assert por_id[""]["ok"] is False
    assert por_id["regra_manter"]["ok"] is True

    decisao_fantasma = next(
        d for d in resultado["decisoes_roteamento"] if d["regra"] == "regra_fantasma"
    )
    assert decisao_fantasma["agente_diagnostico"] is False
    assert decisao_fantasma["agente_gerador"] is False
    # nenhuma chamada de LLM foi gasta com a entrada invalida
    assert (
        resultado["chamadas_llm_diagnostico"] == 2
    )  # so a regra_manter chegou ao diagnostico
