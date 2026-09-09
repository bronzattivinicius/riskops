"""Tests for riskops.portfolio_graph, using a fake chat model (no real API calls)."""

from datetime import UTC, datetime

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from riskops.diagnostics import RuleAssessment
from riskops.portfolio_graph import build_graph
from riskops.rules.schema import Condition, ConditionGroup, Operator, Rule, RuleStatus
from riskops.rules.store import RuleStore

_ACTOR = "test-actor"


def _build_rule(rule_id: str) -> Rule:
    """Builds a minimal, valid, active Rule for graph tests."""
    now = datetime.now(UTC)
    return Rule(
        id=rule_id,
        name=rule_id,
        version=1,
        status=RuleStatus.ACTIVE,
        description="regra usada apenas em teste",
        logic=ConditionGroup(conditions=[Condition(field="credit_risk_score", operator=Operator.GT, value=100)]),
        created_at=now,
        updated_at=now,
        created_by=_ACTOR,
        updated_by=_ACTOR,
    )


class _FakeAgentLLM:
    """Fake tool-deciding model: calls the tool only for rule ids containing 'ambigua'."""

    def bind_tools(self, tools):
        self._tool_name = tools[0].name
        return self

    def invoke(self, messages):
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="Pronto para o parecer final.")
        texto = " ".join(str(getattr(m, "content", "")) for m in messages)
        if "regra_ambigua" in texto:
            return AIMessage(
                content="",
                tool_calls=[{"name": self._tool_name, "args": {"rule_id": "regra_ambigua"}, "id": "call_1"}],
            )
        return AIMessage(content="Nao preciso de mais contexto.")


class _FakeStructuredLLM:
    """Fake structured-output model: always returns the same fixed verdict."""

    def invoke(self, messages):
        avaliacao = RuleAssessment(veredito="revisar", justificativa="citando 9.5%", sugestao="revisar a regra")
        return {"parsed": avaliacao, "raw": AIMessage(content="")}


@pytest.fixture
def store_com_duas_regras(tmp_rule_store: RuleStore) -> RuleStore:
    """A RuleStore with one normal rule and one that the fake agent treats as ambiguous."""
    tmp_rule_store.create(_build_rule("regra_normal"), actor=_ACTOR, note="setup")
    tmp_rule_store.create(_build_rule("regra_ambigua"), actor=_ACTOR, note="setup")
    return tmp_rule_store


def test_graph_processes_every_rule_exactly_once(
    store_com_duas_regras: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """Every rule id in the queue appears exactly once in the results, no duplicates."""
    builder = build_graph(
        store=store_com_duas_regras, df=synthetic_applications_df, llm=_FakeAgentLLM(), structured_llm=_FakeStructuredLLM()
    )
    graph = builder.compile()

    resultado = graph.invoke(
        {
            "fila_regras": ["regra_normal", "regra_ambigua"],
            "resultados": [],
            "chamadas_llm": 0,
            "chamadas_ferramenta": 0,
            "tokens_entrada_total": 0,
            "tokens_saida_total": 0,
        },
        config={"recursion_limit": 50},
    )

    ids_processados = [r["id"] for r in resultado["resultados"]]
    assert sorted(ids_processados) == ["regra_ambigua", "regra_normal"]
    assert len(ids_processados) == len(set(ids_processados))
    assert resultado["fila_regras"] == []


def test_graph_calls_tool_only_when_agent_decides_to(
    store_com_duas_regras: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """The tool fires only for the rule the fake agent flags as needing it."""
    builder = build_graph(
        store=store_com_duas_regras, df=synthetic_applications_df, llm=_FakeAgentLLM(), structured_llm=_FakeStructuredLLM()
    )
    graph = builder.compile()

    resultado = graph.invoke(
        {
            "fila_regras": ["regra_normal", "regra_ambigua"],
            "resultados": [],
            "chamadas_llm": 0,
            "chamadas_ferramenta": 0,
            "tokens_entrada_total": 0,
            "tokens_saida_total": 0,
        },
        config={"recursion_limit": 50},
    )

    por_id = {r["id"]: r for r in resultado["resultados"]}
    assert por_id["regra_normal"]["consultou_historico"] is False
    assert por_id["regra_ambigua"]["consultou_historico"] is True
    # 2 chamadas para a regra normal (agente + finalizar) + 3 para a ambigua (agente, agente pos-ferramenta, finalizar)
    assert resultado["chamadas_llm"] == 5
    assert resultado["chamadas_ferramenta"] == 1


def test_graph_respects_recursion_limit(
    store_com_duas_regras: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """A recursion_limit too low to finish the queue raises, as documented by LangGraph."""
    from langgraph.errors import GraphRecursionError

    builder = build_graph(
        store=store_com_duas_regras, df=synthetic_applications_df, llm=_FakeAgentLLM(), structured_llm=_FakeStructuredLLM()
    )
    graph = builder.compile()

    with pytest.raises(GraphRecursionError):
        graph.invoke(
            {
                "fila_regras": ["regra_normal", "regra_ambigua"],
                "resultados": [],
                "chamadas_llm": 0,
                "chamadas_ferramenta": 0,
                "tokens_entrada_total": 0,
                "tokens_saida_total": 0,
            },
            config={"recursion_limit": 2},
        )


def test_graph_handles_missing_rule_gracefully(
    store_com_duas_regras: RuleStore, synthetic_applications_df: pd.DataFrame
) -> None:
    """A nonexistent or empty rule_id in the queue is reported, not raised, and the queue continues."""
    builder = build_graph(
        store=store_com_duas_regras, df=synthetic_applications_df, llm=_FakeAgentLLM(), structured_llm=_FakeStructuredLLM()
    )
    graph = builder.compile()

    resultado = graph.invoke(
        {
            "fila_regras": ["regra_fantasma", "", "regra_normal"],
            "resultados": [],
            "chamadas_llm": 0,
            "chamadas_ferramenta": 0,
            "tokens_entrada_total": 0,
            "tokens_saida_total": 0,
        },
        config={"recursion_limit": 50},
    )

    por_id = {r["id"]: r for r in resultado["resultados"]}
    assert por_id["regra_fantasma"]["ok"] is False
    assert "nao encontrada" in por_id["regra_fantasma"]["erro"]
    assert por_id[""]["ok"] is False
    # a regra valida depois das invalidas ainda eh processada normalmente
    assert por_id["regra_normal"]["ok"] is True
    # nenhuma chamada de LLM foi gasta com as duas entradas invalidas
    assert resultado["chamadas_llm"] == 2
