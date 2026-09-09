"""LangGraph workflow that diagnoses every rule in the registry, batch-style.

Deliverable 1's baseline (see ``riskops.diagnostics``) requires an analyst to
already know which ``rule_id`` to check. This graph removes that limitation:
given a queue of rule ids, it walks the whole queue, running the same
deterministic backtest and structured-output finalization as the baseline for
each rule. The only new agentic behavior is a single decision point per rule:
before finalizing, the model may call a real tool (``consultar_historico_regra``)
to read that rule's version history when the backtest metrics alone feel
ambiguous -- a genuine, model-driven ReAct-style choice, not a hardcoded branch.

Architecture: hybrid. Iterating the queue is a deterministic workflow (there is
no judgment call in "what rule comes next"); whether to consult more context
before answering is delegated to the model via ``tools_condition``/``ToolNode``.
"""

import dataclasses
import operator
from typing import Annotated, TypedDict

import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from riskops.diagnostics import montar_prompt, veredito_referencia
from riskops.metrics.backtest import ClassificationMetrics, backtest_ruleset
from riskops.rules.store import RuleNotFoundError, RuleStore

INSTRUCAO_SISTEMA = (
    "Voce e um analista de risco avaliando regras de deteccao de fraude, uma de cada vez. "
    "Antes do parecer final, decida se precisa consultar o historico de versoes da regra "
    "(ferramenta consultar_historico_regra) para ter mais contexto -- especialmente quando "
    "as metricas de backtest sozinhas parecem ambiguas. Se as metricas ja forem suficientes, "
    "apenas confirme que esta pronto para o parecer final, sem chamar nenhuma ferramenta."
)
"""System instruction for the tool-deciding step. Versioned as "v2-agente"."""


class DiagnosticoPortfolioState(TypedDict):
    """Graph state for the portfolio diagnostic workflow.

    Attributes:
        fila_regras: Rule ids still waiting to be processed.
        regra_atual: Id of the rule currently being diagnosed, or None.
        metricas_atual: Backtest result for ``regra_atual``, as a plain dict
            (``dataclasses.asdict`` of a ``ClassificationMetrics``) so it
            serializes cleanly through the checkpointer.
        regra_encontrada: Whether ``regra_atual`` exists in the registry.
            False (missing/empty rule id) short-circuits straight to the
            next rule, matching Deliverable 1's RF-04 graceful-error
            guarantee -- the agentic/finalization steps are skipped.
        messages: Conversation for the current rule's tool-deciding step,
            reset (via ``RemoveMessage``) each time a new rule starts.
        resultados: Diagnoses accumulated so far, one dict per rule.
        chamadas_llm: Running total of LLM calls made across the batch.
        chamadas_ferramenta: Running total of tool calls made across the batch.
        tokens_entrada_total: Running total of input tokens across the batch.
        tokens_saida_total: Running total of output tokens across the batch.
    """

    fila_regras: list[str]
    regra_atual: str | None
    regra_encontrada: bool
    metricas_atual: object
    messages: Annotated[list[BaseMessage], add_messages]
    resultados: Annotated[list[dict], operator.add]
    chamadas_llm: Annotated[int, operator.add]
    chamadas_ferramenta: Annotated[int, operator.add]
    tokens_entrada_total: Annotated[int, operator.add]
    tokens_saida_total: Annotated[int, operator.add]


def criar_ferramenta_historico(store: RuleStore):
    """Builds the `consultar_historico_regra` tool bound to a rule store.

    The tool performs real work: it reads the rule's actual version/audit
    history from the registry (``RuleStore.get_history``). Its return value
    is treated by the model as untrusted data, not as instructions -- it is
    read-only (no write privilege is granted to the tool).

    Args:
        store: Rule registry to read history from.

    Returns:
        A LangChain tool callable, ready to be passed to `llm.bind_tools`.
    """

    @tool
    def consultar_historico_regra(rule_id: str) -> str:
        """Consulta o historico de versoes de uma regra no registro do RiskOps.

        Use esta ferramenta quando as metricas de backtest sozinhas nao forem
        suficientes para decidir o parecer, por exemplo para verificar se a
        regra e recente, ja foi revisada antes, ou se ha uma nota explicando
        o motivo de uma decisao anterior.

        Args:
            rule_id: Id da regra a consultar.

        Returns:
            Um resumo em texto do historico de versoes da regra.
        """
        try:
            historico = store.get_history(rule_id)
        except RuleNotFoundError:
            return f"Regra '{rule_id}' nao encontrada no registro."
        if not historico:
            return f"Regra '{rule_id}' nao tem historico registrado."
        linhas = [
            f"v{entrada.version} ({entrada.status.value}) em {entrada.changed_at.isoformat()} "
            f"por {entrada.changed_by}: {entrada.change_note}"
            for entrada in historico
        ]
        return f"Historico de '{rule_id}' ({len(historico)} versao(oes)):\n" + "\n".join(linhas)

    return consultar_historico_regra


def build_graph(*, store: RuleStore, df: pd.DataFrame, llm, structured_llm, label_col: str = "fraud_bool"):
    """Builds and compiles the portfolio diagnostic graph.

    Args:
        store: Rule registry to read rules and history from.
        df: Historical transaction data to backtest rules against.
        llm: Base chat model (unbound), used for the tool-deciding step.
        structured_llm: The same chat model wrapped with
            `.with_structured_output(RuleAssessment, include_raw=True)`,
            reused unchanged from `riskops.diagnostics`.
        label_col: Name of the boolean fraud-label column in `df`.

    Returns:
        A tuple `(graph_builder, checkpointer_factory)` is not returned;
        instead, returns the uncompiled `StateGraph` so callers can compile
        it with or without a checkpointer, to compare both (Deliverable 2,
        Section D).
    """
    baseline_fraud_rate = df[label_col].mean()
    ferramenta_historico = criar_ferramenta_historico(store)
    llm_com_ferramentas = llm.bind_tools([ferramenta_historico])

    def ha_regras_pendentes(state: DiagnosticoPortfolioState) -> str:
        return "proxima_regra" if state["fila_regras"] else END

    def proxima_regra(state: DiagnosticoPortfolioState) -> dict:
        fila = state["fila_regras"]
        return {
            "fila_regras": fila[1:],
            "regra_atual": fila[0],
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }

    def backtest(state: DiagnosticoPortfolioState) -> dict:
        regra_id = state["regra_atual"]
        try:
            rule = store.get(regra_id) if regra_id else None
            if rule is None:
                raise RuleNotFoundError(f"rule_id vazio ou nao informado (regra_id={regra_id!r})")
        except RuleNotFoundError:
            # Regra ausente/vazia: mesma garantia de erro gracioso da RF-04 do E1, mas
            # dentro do proprio grafo -- pula o passo agente/finalizacao para esta regra.
            return {
                "regra_encontrada": False,
                "resultados": [
                    {"id": regra_id, "ok": False, "erro": f"regra {regra_id!r} nao encontrada no registro (ou vazia)."}
                ],
                "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            }

        metricas = backtest_ruleset(df, [rule], label_col=label_col).metrics
        prompt = montar_prompt(rule, metricas, baseline_fraud_rate)
        return {
            "regra_encontrada": True,
            # stored as a plain dict (not the ClassificationMetrics instance) so it
            # round-trips cleanly through the checkpointer's msgpack serialization.
            "metricas_atual": dataclasses.asdict(metricas),
            "messages": [SystemMessage(content=INSTRUCAO_SISTEMA), HumanMessage(content=prompt)],
        }

    def apos_backtest(state: DiagnosticoPortfolioState) -> str:
        return "agente_diagnostico" if state.get("regra_encontrada") else ha_regras_pendentes(state)

    def agente_diagnostico(state: DiagnosticoPortfolioState) -> dict:
        resposta = llm_com_ferramentas.invoke(state["messages"])
        uso = getattr(resposta, "usage_metadata", None) or {}
        return {
            "messages": [resposta],
            "chamadas_llm": 1,
            "tokens_entrada_total": uso.get("input_tokens") or 0,
            "tokens_saida_total": uso.get("output_tokens") or 0,
        }

    def finalizar_diagnostico(state: DiagnosticoPortfolioState) -> dict:
        chamadas_ferramenta = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
        try:
            saida = structured_llm.invoke(state["messages"])
        except Exception as exc:
            # O modelo pode gerar JSON malformado que falha do lado do servidor, antes de
            # chegar na validacao do Pydantic -- ainda assim conta como uma chamada ao modelo.
            return {
                "resultados": [{"id": state["regra_atual"], "ok": False, "erro": f"chamada ao modelo falhou: {exc}"}],
                "chamadas_llm": 1,
                "chamadas_ferramenta": chamadas_ferramenta,
                "tokens_entrada_total": 0,
                "tokens_saida_total": 0,
            }
        avaliacao = saida["parsed"]
        bruta = saida["raw"]
        uso = getattr(bruta, "usage_metadata", None) or {}

        if avaliacao is None:
            resultado = {
                "id": state["regra_atual"],
                "ok": False,
                "erro": f"falha ao interpretar resposta do modelo: {saida.get('parsing_error')}",
            }
        else:
            metricas = ClassificationMetrics(**state["metricas_atual"])
            resultado = {
                "id": state["regra_atual"],
                "ok": True,
                "veredito": avaliacao.veredito,
                "justificativa": avaliacao.justificativa,
                "sugestao": avaliacao.sugestao,
                "metricas_backtest": metricas,
                "veredito_referencia": veredito_referencia(metricas, baseline_fraud_rate),
                "consultou_historico": chamadas_ferramenta > 0,
            }
        return {
            "resultados": [resultado],
            "chamadas_llm": 1,
            "chamadas_ferramenta": chamadas_ferramenta,
            "tokens_entrada_total": uso.get("input_tokens") or 0,
            "tokens_saida_total": uso.get("output_tokens") or 0,
        }

    builder = StateGraph(DiagnosticoPortfolioState)
    builder.add_node("proxima_regra", proxima_regra)
    builder.add_node("backtest", backtest)
    builder.add_node("agente_diagnostico", agente_diagnostico)
    builder.add_node("ferramentas", ToolNode([ferramenta_historico]))
    builder.add_node("finalizar_diagnostico", finalizar_diagnostico)

    builder.set_conditional_entry_point(ha_regras_pendentes, {"proxima_regra": "proxima_regra", END: END})
    builder.add_edge("proxima_regra", "backtest")
    builder.add_conditional_edges(
        "backtest", apos_backtest, {"agente_diagnostico": "agente_diagnostico", "proxima_regra": "proxima_regra", END: END}
    )
    builder.add_conditional_edges(
        "agente_diagnostico", tools_condition, {"tools": "ferramentas", "__end__": "finalizar_diagnostico"}
    )
    builder.add_edge("ferramentas", "agente_diagnostico")
    builder.add_conditional_edges(
        "finalizar_diagnostico", ha_regras_pendentes, {"proxima_regra": "proxima_regra", END: END}
    )
    return builder
