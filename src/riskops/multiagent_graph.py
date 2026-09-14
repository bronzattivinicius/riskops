"""LangGraph workflow with two agents: diagnosis, then candidate generation.

Deliverable 2's graph (see ``riskops.portfolio_graph``) only diagnoses rules
-- for a verdict of "revisar" or "aposentar" it has no automated next step.
This graph adds a second, genuinely distinct agent that proposes and
backtests a replacement condition, but only when the diagnostic agent found
one (never for "manter"). The two agents never share raw conversation: the
generator receives a small, structured contract -- the diagnostic verdict,
justification, and current metrics -- built fresh from the diagnostic
agent's final structured output, not its tool-call history.

Organization pattern: hybrid. Iterating the queue is a deterministic
workflow (as in v2). Whether the generator agent runs at all is a
supervisor-style decision -- but the "supervisor" is a plain Python function
reading the diagnostic agent's own structured verdict, not a third LLM
arbitrating over free text: the routing itself needs no judgment once the
verdict exists.
"""

import dataclasses
import operator
from typing import Annotated, TypedDict

import pandas as pd
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from riskops.diagnostics import montar_prompt, veredito_referencia
from riskops.metrics.backtest import ClassificationMetrics, backtest_ruleset
from riskops.portfolio_graph import INSTRUCAO_SISTEMA, criar_ferramenta_historico
from riskops.rule_generator import (
    avaliar_candidata,
    candidata_melhorou,
    construir_regra_candidata,
    montar_prompt_gerador,
    resumir_comparacao,
)
from riskops.rules.store import RuleNotFoundError, RuleStore

INSTRUCAO_SISTEMA_GERADOR = (
    "Voce e um analista de risco propondo uma regra candidata para substituir ou complementar "
    "uma regra que outro analista ja diagnosticou como problematica. Use a ferramenta "
    "testar_regra_candidata para validar sua proposta contra dados reais antes de responder -- "
    "voce pode testar mais de uma vez se a primeira tentativa nao parecer boa. So de o parecer "
    "final depois de pelo menos um teste."
)
"""System instruction for the generator agent's tool-using step."""

_VEREDITOS_QUE_ACIONAM_GERADOR = ("revisar", "aposentar")


class MultiAgenteState(TypedDict):
    """Graph state for the two-agent (diagnosis + generation) workflow.

    Attributes:
        fila_regras: Rule ids still waiting to be processed.
        regra_atual: Id of the rule currently being processed, or None.
        regra_encontrada: Whether ``regra_atual`` exists in the registry.
        metricas_atual: Backtest result for ``regra_atual``, as a plain dict
            (serializes cleanly, same convention as ``portfolio_graph``).
        diagnostico_atual: The diagnostic agent's structured verdict for
            ``regra_atual`` (dict), or None if diagnosis failed. This is
            the entire contract the generator agent is allowed to see --
            not the diagnostic agent's raw messages.
        acionar_gerador: Whether the generator agent should run for
            ``regra_atual``, decided in ``preparar_roteamento`` from
            ``diagnostico_atual["veredito"]``.
        mensagens_diagnostico: Conversation for the diagnostic agent's
            current rule, reset each time a new rule starts.
        mensagens_geracao: Conversation for the generator agent's current
            rule, reset each time a new rule starts (and unused if the
            generator does not run).
        resultados: Diagnoses (and, where applicable, candidates)
            accumulated so far, one dict per rule.
        decisoes_roteamento: One entry per rule describing which agents ran
            and why -- the routing-justification part of the
            observability requirement.
        chamadas_llm_diagnostico: LLM calls made by the diagnostic agent,
            across the whole batch.
        chamadas_ferramenta_diagnostico: Tool calls made by the diagnostic
            agent, across the whole batch.
        chamadas_llm_gerador: LLM calls made by the generator agent,
            across the whole batch.
        chamadas_ferramenta_gerador: Tool calls made by the generator
            agent, across the whole batch.
        tokens_entrada_total: Running total of input tokens, both agents.
        tokens_saida_total: Running total of output tokens, both agents.
    """

    fila_regras: list[str]
    regra_atual: str | None
    regra_encontrada: bool
    metricas_atual: object
    diagnostico_atual: object | None
    acionar_gerador: bool
    mensagens_diagnostico: Annotated[list[BaseMessage], add_messages]
    mensagens_geracao: Annotated[list[BaseMessage], add_messages]
    resultados: Annotated[list[dict], operator.add]
    decisoes_roteamento: Annotated[list[dict], operator.add]
    chamadas_llm_diagnostico: Annotated[int, operator.add]
    chamadas_ferramenta_diagnostico: Annotated[int, operator.add]
    chamadas_llm_gerador: Annotated[int, operator.add]
    chamadas_ferramenta_gerador: Annotated[int, operator.add]
    tokens_entrada_total: Annotated[int, operator.add]
    tokens_saida_total: Annotated[int, operator.add]


def criar_ferramenta_teste_candidata(
    store: RuleStore, df: pd.DataFrame, label_col: str = "fraud_bool"
):
    """Builds the `testar_regra_candidata` tool bound to a store and dataset.

    The tool performs real work: it builds an in-memory candidate rule and
    backtests it against the real historical data (``compare_rulesets``,
    Phase 1), returning a genuine comparison -- nothing is stubbed or
    persisted. Its argument shape is intentionally flat (see
    ``riskops.rule_generator`` module docstring) for structured-output
    robustness with this project's model.

    Args:
        store: Rule registry to read the current rule from.
        df: Historical transaction data to backtest against.
        label_col: Name of the boolean fraud-label column in ``df``.

    Returns:
        A LangChain tool callable, ready to be passed to `llm.bind_tools`.
    """

    @tool
    def testar_regra_candidata(
        rule_id: str,
        campo: str,
        operador: str,
        valor: float,
        combinar_com_atual: bool = False,
    ) -> str:
        """Testa uma condicao candidata via backtest real, sem persistir nada.

        Args:
            rule_id: Id da regra atual, cuja candidata esta sendo testada.
            campo: Nome do campo da transacao (ex.: 'velocity_6h').
            operador: Um de 'eq', 'ne', 'gt', 'gte', 'lt', 'lte'.
            valor: Valor numerico de comparacao proposto.
            combinar_com_atual: Se True, combina (AND) com a logica atual da
                regra; se False (padrao), substitui a logica atual inteira.

        Returns:
            Um resumo em texto comparando as metricas da regra atual com a candidata.
        """
        try:
            rule_atual = store.get(rule_id)
        except RuleNotFoundError:
            return f"Regra '{rule_id}' nao encontrada no registro."
        try:
            candidata = construir_regra_candidata(
                rule_atual,
                campo=campo,
                operador=operador,
                valor=valor,
                combinar_com_atual=combinar_com_atual,
            )
        except ValueError as exc:
            return f"Proposta invalida: {exc}"
        comparacao = avaliar_candidata(df, rule_atual, candidata, label_col=label_col)
        return resumir_comparacao(comparacao)

    return testar_regra_candidata


def build_graph(
    *,
    store: RuleStore,
    df: pd.DataFrame,
    llm,
    structured_llm_diagnostico,
    structured_llm_gerador,
    label_col: str = "fraud_bool",
):
    """Builds and compiles the two-agent diagnostic-and-generation graph.

    Args:
        store: Rule registry to read rules and history from, and that the
            generator's candidate proposals are backtested against (never
            written to).
        df: Historical transaction data to backtest rules against.
        llm: Base chat model (unbound), used for both agents' tool-deciding
            steps.
        structured_llm_diagnostico: The diagnostic agent's model, wrapped
            with `.with_structured_output(RuleAssessment, include_raw=True)`
            -- reused unchanged from `riskops.diagnostics`.
        structured_llm_gerador: The generator agent's model, wrapped with
            `.with_structured_output(RegraCandidataFinal, include_raw=True)`.
        label_col: Name of the boolean fraud-label column in `df`.

    Returns:
        The uncompiled `StateGraph`, so callers can compile it with or
        without a checkpointer.
    """
    baseline_fraud_rate = df[label_col].mean()
    ferramenta_historico = criar_ferramenta_historico(store)
    ferramenta_teste_candidata = criar_ferramenta_teste_candidata(store, df, label_col)
    llm_diagnostico = llm.bind_tools([ferramenta_historico])
    llm_gerador = llm.bind_tools([ferramenta_teste_candidata])

    def ha_regras_pendentes(state: MultiAgenteState) -> str:
        """Routes to the next queued rule, or to END if the queue is empty."""
        return "proxima_regra" if state["fila_regras"] else END

    def proxima_regra(state: MultiAgenteState) -> dict:
        """Dequeues the next rule id and resets both agents' conversations."""
        fila = state["fila_regras"]
        return {
            "fila_regras": fila[1:],
            "regra_atual": fila[0],
            "mensagens_diagnostico": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            "mensagens_geracao": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }

    def backtest(state: MultiAgenteState) -> dict:
        """Runs the deterministic backtest for the current rule, or records its absence.

        Identical in spirit to `portfolio_graph.backtest`: a missing/empty
        rule id is recorded directly as a failed result (RF-04), and
        `apos_backtest` routes around both agents for this rule.
        """
        regra_id = state["regra_atual"]
        try:
            rule = store.get(regra_id) if regra_id else None
            if rule is None:
                raise RuleNotFoundError(
                    f"rule_id vazio ou nao informado (regra_id={regra_id!r})"
                )
        except RuleNotFoundError:
            return {
                "regra_encontrada": False,
                "resultados": [
                    {
                        "id": regra_id,
                        "ok": False,
                        "erro": f"regra {regra_id!r} nao encontrada no registro (ou vazia).",
                    }
                ],
                "decisoes_roteamento": [
                    {
                        "regra": regra_id,
                        "agente_diagnostico": False,
                        "agente_gerador": False,
                        "motivo": "regra nao encontrada",
                    }
                ],
                "mensagens_diagnostico": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            }

        metricas = backtest_ruleset(df, [rule], label_col=label_col).metrics
        prompt = montar_prompt(rule, metricas, baseline_fraud_rate)
        return {
            "regra_encontrada": True,
            "metricas_atual": dataclasses.asdict(metricas),
            "mensagens_diagnostico": [
                SystemMessage(content=INSTRUCAO_SISTEMA),
                HumanMessage(content=prompt),
            ],
        }

    def apos_backtest(state: MultiAgenteState) -> str:
        """Routes past both agents for a rule that was not found."""
        return (
            "agente_diagnostico"
            if state.get("regra_encontrada")
            else ha_regras_pendentes(state)
        )

    def agente_diagnostico(state: MultiAgenteState) -> dict:
        """Lets the diagnostic agent decide whether it needs the history tool."""
        resposta = llm_diagnostico.invoke(state["mensagens_diagnostico"])
        uso = getattr(resposta, "usage_metadata", None) or {}
        return {
            "mensagens_diagnostico": [resposta],
            "chamadas_llm_diagnostico": 1,
            "tokens_entrada_total": uso.get("input_tokens") or 0,
            "tokens_saida_total": uso.get("output_tokens") or 0,
        }

    def finalizar_diagnostico(state: MultiAgenteState) -> dict:
        """Produces the diagnostic agent's structured verdict.

        On success, stores it in `diagnostico_atual` -- the only thing the
        generator agent will ever see of this agent's work. On failure,
        finalizes a failed result immediately (no generator involved),
        matching `portfolio_graph.finalizar_diagnostico`'s error handling.
        """
        chamadas_ferramenta = sum(
            1 for m in state["mensagens_diagnostico"] if isinstance(m, ToolMessage)
        )
        try:
            saida = structured_llm_diagnostico.invoke(state["mensagens_diagnostico"])
        except Exception as exc:
            return {
                "diagnostico_atual": None,
                "resultados": [
                    {
                        "id": state["regra_atual"],
                        "ok": False,
                        "erro": f"chamada ao modelo falhou: {exc}",
                    }
                ],
                "chamadas_llm_diagnostico": 1,
                "chamadas_ferramenta_diagnostico": chamadas_ferramenta,
            }
        avaliacao = saida["parsed"]
        bruta = saida["raw"]
        uso = getattr(bruta, "usage_metadata", None) or {}

        if avaliacao is None:
            return {
                "diagnostico_atual": None,
                "resultados": [
                    {
                        "id": state["regra_atual"],
                        "ok": False,
                        "erro": f"falha ao interpretar resposta do modelo: {saida.get('parsing_error')}",
                    }
                ],
                "chamadas_llm_diagnostico": 1,
                "chamadas_ferramenta_diagnostico": chamadas_ferramenta,
                "tokens_entrada_total": uso.get("input_tokens") or 0,
                "tokens_saida_total": uso.get("output_tokens") or 0,
            }

        metricas = ClassificationMetrics(**state["metricas_atual"])
        diagnostico = {
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
            "diagnostico_atual": diagnostico,
            "chamadas_llm_diagnostico": 1,
            "chamadas_ferramenta_diagnostico": chamadas_ferramenta,
            "tokens_entrada_total": uso.get("input_tokens") or 0,
            "tokens_saida_total": uso.get("output_tokens") or 0,
        }

    def preparar_roteamento(state: MultiAgenteState) -> dict:
        """Decides whether the generator agent runs, and records why.

        This is the supervisor decision (Section 3.2): a plain function
        reading the diagnostic agent's own structured verdict. For
        "manter" (or a failed diagnosis), the rule's final result is
        finalized here -- there is nothing left for a second agent to do.
        For "revisar"/"aposentar", the generator's prompt is seeded here,
        built only from `diagnostico_atual` (the structured contract), not
        from `mensagens_diagnostico`.
        """
        regra_id = state["regra_atual"]
        diagnostico = state.get("diagnostico_atual")

        if diagnostico is None:
            return {
                "acionar_gerador": False,
                "decisoes_roteamento": [
                    {
                        "regra": regra_id,
                        "agente_diagnostico": True,
                        "agente_gerador": False,
                        "motivo": "diagnostico falhou",
                    }
                ],
            }

        if diagnostico["veredito"] not in _VEREDITOS_QUE_ACIONAM_GERADOR:
            return {
                "acionar_gerador": False,
                "resultados": [{**diagnostico, "candidata": None}],
                "decisoes_roteamento": [
                    {
                        "regra": regra_id,
                        "agente_diagnostico": True,
                        "agente_gerador": False,
                        "motivo": f"veredito={diagnostico['veredito']}, sem necessidade de candidata",
                    }
                ],
            }

        rule = store.get(regra_id)
        metricas = ClassificationMetrics(**state["metricas_atual"])
        prompt = montar_prompt_gerador(
            rule,
            metricas,
            veredito_diagnostico=diagnostico["veredito"],
            justificativa_diagnostico=diagnostico["justificativa"],
        )
        return {
            "acionar_gerador": True,
            "mensagens_geracao": [
                SystemMessage(content=INSTRUCAO_SISTEMA_GERADOR),
                HumanMessage(content=prompt),
            ],
            "decisoes_roteamento": [
                {
                    "regra": regra_id,
                    "agente_diagnostico": True,
                    "agente_gerador": True,
                    "motivo": f"veredito={diagnostico['veredito']}, aciona gerador de candidata",
                }
            ],
        }

    def apos_diagnostico(state: MultiAgenteState) -> str:
        """Routes to the generator agent, or loops/ends, per `acionar_gerador`."""
        return (
            "agente_gerador"
            if state.get("acionar_gerador")
            else ha_regras_pendentes(state)
        )

    def agente_gerador(state: MultiAgenteState) -> dict:
        """Lets the generator agent decide whether to test a candidate."""
        resposta = llm_gerador.invoke(state["mensagens_geracao"])
        uso = getattr(resposta, "usage_metadata", None) or {}
        return {
            "mensagens_geracao": [resposta],
            "chamadas_llm_gerador": 1,
            "tokens_entrada_total": uso.get("input_tokens") or 0,
            "tokens_saida_total": uso.get("output_tokens") or 0,
        }

    def finalizar_geracao(state: MultiAgenteState) -> dict:
        """Produces the generator agent's final candidate and appends the rule's result.

        The candidate's improvement is never taken on the model's word: it
        is recomputed here, deterministically, via `avaliar_candidata` +
        `candidata_melhorou` (Phase 1 backtest engine), exactly like the
        diagnostic agent's `veredito_referencia` grounding.
        """
        regra_id = state["regra_atual"]
        diagnostico = state["diagnostico_atual"]
        chamadas_ferramenta = sum(
            1 for m in state["mensagens_geracao"] if isinstance(m, ToolMessage)
        )
        try:
            saida = structured_llm_gerador.invoke(state["mensagens_geracao"])
        except Exception as exc:
            return {
                "resultados": [
                    {
                        **diagnostico,
                        "candidata": None,
                        "candidata_erro": f"chamada ao modelo falhou: {exc}",
                    }
                ],
                "chamadas_llm_gerador": 1,
                "chamadas_ferramenta_gerador": chamadas_ferramenta,
            }
        proposta = saida["parsed"]
        bruta = saida["raw"]
        uso = getattr(bruta, "usage_metadata", None) or {}

        if proposta is None:
            return {
                "resultados": [
                    {
                        **diagnostico,
                        "candidata": None,
                        "candidata_erro": f"falha ao interpretar proposta: {saida.get('parsing_error')}",
                    }
                ],
                "chamadas_llm_gerador": 1,
                "chamadas_ferramenta_gerador": chamadas_ferramenta,
                "tokens_entrada_total": uso.get("input_tokens") or 0,
                "tokens_saida_total": uso.get("output_tokens") or 0,
            }

        rule_atual = store.get(regra_id)
        try:
            candidata = construir_regra_candidata(
                rule_atual,
                campo=proposta.campo,
                operador=proposta.operador,
                valor=proposta.valor,
                combinar_com_atual=proposta.combinar_com_atual,
            )
            comparacao = avaliar_candidata(
                df, rule_atual, candidata, label_col=label_col
            )
            resultado_candidata = {
                "campo": proposta.campo,
                "operador": proposta.operador,
                "valor": proposta.valor,
                "combinar_com_atual": proposta.combinar_com_atual,
                "justificativa": proposta.justificativa,
                "sugestao": proposta.sugestao,
                "metricas_candidata": comparacao.candidate.metrics,
                "delta_precisao": comparacao.delta_precision,
                "delta_deteccao": comparacao.delta_detection_rate,
                "delta_falso_positivo": comparacao.delta_false_positive_rate,
                "melhorou": candidata_melhorou(comparacao),
                "erro": None,
            }
        except ValueError as exc:
            resultado_candidata = {
                "campo": proposta.campo,
                "erro": f"proposta invalida: {exc}",
                "melhorou": False,
            }

        return {
            "resultados": [{**diagnostico, "candidata": resultado_candidata}],
            "chamadas_llm_gerador": 1,
            "chamadas_ferramenta_gerador": chamadas_ferramenta,
            "tokens_entrada_total": uso.get("input_tokens") or 0,
            "tokens_saida_total": uso.get("output_tokens") or 0,
        }

    builder = StateGraph(MultiAgenteState)
    builder.add_node("proxima_regra", proxima_regra)
    builder.add_node("backtest", backtest)
    builder.add_node("agente_diagnostico", agente_diagnostico)
    builder.add_node(
        "ferramentas_diagnostico",
        ToolNode([ferramenta_historico], messages_key="mensagens_diagnostico"),
    )
    builder.add_node("finalizar_diagnostico", finalizar_diagnostico)
    builder.add_node("preparar_roteamento", preparar_roteamento)
    builder.add_node("agente_gerador", agente_gerador)
    builder.add_node(
        "ferramentas_gerador",
        ToolNode([ferramenta_teste_candidata], messages_key="mensagens_geracao"),
    )
    builder.add_node("finalizar_geracao", finalizar_geracao)

    builder.set_conditional_entry_point(
        ha_regras_pendentes, {"proxima_regra": "proxima_regra", END: END}
    )
    builder.add_edge("proxima_regra", "backtest")
    builder.add_conditional_edges(
        "backtest",
        apos_backtest,
        {
            "agente_diagnostico": "agente_diagnostico",
            "proxima_regra": "proxima_regra",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "agente_diagnostico",
        lambda state: tools_condition(state, messages_key="mensagens_diagnostico"),
        {"tools": "ferramentas_diagnostico", "__end__": "finalizar_diagnostico"},
    )
    builder.add_edge("ferramentas_diagnostico", "agente_diagnostico")
    builder.add_edge("finalizar_diagnostico", "preparar_roteamento")
    builder.add_conditional_edges(
        "preparar_roteamento",
        apos_diagnostico,
        {
            "agente_gerador": "agente_gerador",
            "proxima_regra": "proxima_regra",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "agente_gerador",
        lambda state: tools_condition(state, messages_key="mensagens_geracao"),
        {"tools": "ferramentas_gerador", "__end__": "finalizar_geracao"},
    )
    builder.add_edge("ferramentas_gerador", "agente_gerador")
    builder.add_conditional_edges(
        "finalizar_geracao",
        ha_regras_pendentes,
        {"proxima_regra": "proxima_regra", END: END},
    )
    return builder
