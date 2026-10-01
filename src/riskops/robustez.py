"""Resilience utilities for Deliverable 4: containment, degradation, active verification.

This module is additive only: it wraps and reuses `riskops.diagnostics` and
`riskops.multiagent_graph` without modifying either, so v1-v3's already-graded
artifacts stay exactly as delivered ("estrutura herdada... sem alteracao").

Three containment mechanisms are implemented here:

1. Retry with capped exponential backoff (`com_repeticao`), for transient
   failures only -- never for a genuine model-output error. This also
   covers a real, observed failure mode: `gpt-oss-20b` via Groq
   occasionally names its tool call `'functions.X'` instead of `'X'`,
   which LangChain's parser rejects (`FalhaDeFormatoTransitoria`) -- a
   non-deterministic formatting slip, not a stable incompatibility, so
   retrying the same prompt is the right response.
2. Graceful degradation (`diagnosticar_com_robustez`), which falls back to
   the deterministic reference verdict when the model is unreachable after
   every retry, and always self-identifies as degraded in its own output.
3. A step-limit guard (`invocar_grafo_com_limite`), converting a LangGraph
   `GraphRecursionError` -- a real failure mode observed in Deliverable 3 --
   into an explicit, non-crashing result.

Two silent-failure-to-noisy-failure verifications are implemented in
`verificar_saida_robusta`: whether the cited evidence matches a real backtest
metric (reusing `riskops.diagnostics.metrica_citada`), and whether a declared
"alta" confidence is compatible with that evidence being present.

`criar_ferramenta_instavel` and `criar_structured_llm_instavel` deliberately
inject failures at a known, reproducible rate into an existing tool or model
call, purely to demonstrate that the containment mechanisms above actually
work -- neither is ever used in the system's real diagnostic path.
"""

import random
import time
from typing import Literal

import httpx
import pandas as pd
from anthropic import APIError as AnthropicAPIError
from groq import APIError as GroqAPIError
from langchain_core.tools import BaseTool, tool
from langgraph.errors import GraphRecursionError
from pydantic import Field

from riskops.diagnostics import RuleAssessment, metrica_citada, montar_prompt, veredito_referencia
from riskops.metrics.backtest import ClassificationMetrics, backtest_ruleset
from riskops.rules.schema import Rule
from riskops.rules.store import RuleNotFoundError, RuleStore

ERROS_TRANSIENTES_DE_PROVEDOR = (GroqAPIError, AnthropicAPIError)
"""Transient provider errors (rate limit, timeout, connection) worth retrying.

Same tuple used by `riskops.multiagent_graph`, redefined here so this module
stays independently importable and does not reach into that module's
private name.
"""


class FalhaDeFormatoTransitoria(Exception):
    """A model response that failed to parse into the expected schema, worth retrying.

    Distinct from a genuine model-output error (wrong verdict, hallucinated
    field): this covers failures in the tool-call protocol itself, e.g. a
    real, observed `gpt-oss-20b`/Groq quirk where the model occasionally
    names its tool call `'functions.RuleAssessmentRobusta'` instead of
    `'RuleAssessmentRobusta'`, which LangChain's parser rejects
    ("Unknown tool type"). Retrying is reasonable because the same prompt
    was observed to succeed on a later attempt -- the failure is a
    non-deterministic formatting slip, not a stable incompatibility.
    """


class FonteInstavel(Exception):
    """Simulated transient failure, injected only by `criar_ferramenta_instavel`.

    Represents the kind of intermittent, self-resolving failure a real data
    source can have (a flaky network call, a momentarily overloaded
    service) -- distinct from a permanent error like a missing rule id.
    """


def criar_ferramenta_instavel(ferramenta_base: BaseTool, falha_prob: float, rng: random.Random) -> BaseTool:
    """Wraps an existing single-argument tool so it fails at a known, reproducible rate.

    Used only to demonstrate containment mechanisms (Section B of
    Deliverable 4): it is never part of the system's real diagnostic or
    generation path, where the underlying tool is used directly.

    Args:
        ferramenta_base: An existing LangChain tool taking a single
            ``rule_id: str`` argument (e.g.
            `riskops.portfolio_graph.criar_ferramenta_historico`'s output).
        falha_prob: Probability, in [0, 1], that a call raises
            `FonteInstavel` instead of delegating to `ferramenta_base`.
        rng: Seeded random generator, so failure occurrences are
            reproducible across notebook re-runs.

    Returns:
        A new tool with the same call signature, wrapping `ferramenta_base`.
    """

    @tool
    def ferramenta_instavel(rule_id: str) -> str:
        """Consulta o historico de uma regra, mas falha de forma transitoria e simulada.

        Envolve uma ferramenta real ja existente para demonstrar, de forma
        reprodutivel, contencao de falhas transitorias (Entregavel 4).

        Args:
            rule_id: Id da regra a consultar.

        Returns:
            O mesmo texto que a ferramenta real devolveria.

        Raises:
            FonteInstavel: Com probabilidade `falha_prob`, simulando uma
                fonte de dados intermitente.
        """
        if rng.random() < falha_prob:
            raise FonteInstavel(f"fonte instavel: falha simulada ao consultar {rule_id!r}")
        return ferramenta_base.invoke({"rule_id": rule_id})

    return ferramenta_instavel


def criar_structured_llm_instavel(structured_llm_base, falha_prob: float, rng: random.Random):
    """Wraps a real structured-output model, injecting simulated transient failures at a known rate.

    Symmetric to `criar_ferramenta_instavel`, but for the diagnostic LLM
    call itself, so `diagnosticar_regra_robusta`'s graceful degradation can
    be demonstrated live, on the real model, at zero extra API cost when
    `falha_prob=1.0` (the wrapped model is never actually called then).

    Args:
        structured_llm_base: A real chat model already wrapped with
            `.with_structured_output(RuleAssessmentRobusta, include_raw=True)`.
        falha_prob: Probability, in [0, 1], that a call raises a simulated
            `groq.APIError` instead of delegating to `structured_llm_base`.
        rng: Seeded random generator, so failure occurrences are
            reproducible across notebook re-runs.

    Returns:
        An object with the same `.invoke(prompt)` interface as
        `structured_llm_base`.
    """

    class _StructuredLLMInstavel:
        def invoke(self, prompt):
            if rng.random() < falha_prob:
                raise GroqAPIError(
                    "falha simulada (fonte instavel)",
                    httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
                    body=None,
                )
            return structured_llm_base.invoke(prompt)

    return _StructuredLLMInstavel()


def com_repeticao(
    funcao,
    *args,
    tentativas: int = 3,
    espera_inicial: float = 0.2,
    fator: float = 2.0,
    transitorias: tuple[type[Exception], ...] = (FonteInstavel,),
    **kwargs,
) -> tuple[object | None, dict]:
    """Retries a call with capped exponential backoff, for transient errors only.

    A non-transient exception (anything not in `transitorias`) is never
    retried: it propagates immediately, since retrying a genuine
    model-output or logic error would just waste calls on a failure that
    will not resolve itself.

    Args:
        funcao: Callable to invoke.
        *args: Positional arguments passed to `funcao`.
        tentativas: Maximum number of attempts.
        espera_inicial: Seconds to wait before the second attempt.
        fator: Multiplier applied to the wait after each failed attempt.
        transitorias: Exception types worth retrying.
        **kwargs: Keyword arguments passed to `funcao`.

    Returns:
        A tuple `(resultado, info)`. `resultado` is `funcao`'s return value,
        or `None` if every attempt raised a transient error. `info` has
        `tentativas` (attempts actually used) and `erros` (their messages).
    """
    espera = espera_inicial
    info: dict = {"tentativas": 0, "erros": []}
    for tentativa in range(1, tentativas + 1):
        info["tentativas"] = tentativa
        try:
            return funcao(*args, **kwargs), info
        except transitorias as exc:
            info["erros"].append(str(exc))
            if tentativa == tentativas:
                return None, info
            time.sleep(espera)
            espera *= fator
    return None, info


class RuleAssessmentRobusta(RuleAssessment):
    """Deliverable 4's assessment schema: `RuleAssessment` plus a declared confidence.

    A new, v4-scoped subclass rather than a change to `RuleAssessment`
    itself, so v1-v3's already-graded schema stays unmodified. The
    confidence field exists specifically to let `verificar_saida_robusta`
    check whether a declared "alta" confidence is compatible with the
    evidence actually presented.

    Attributes:
        confianca: The model's own declared confidence in its verdict.
    """

    confianca: Literal["alta", "media", "baixa"] = Field(
        description="Confianca do modelo no proprio parecer, dada a evidencia disponivel."
    )


_FRASES_PROPOSTA_NOVA_CONDICAO = ("nova condicao", "nova regra", "sugiro criar", "proponho criar")


def verificar_saida_robusta(
    avaliacao: RuleAssessmentRobusta,
    metricas: ClassificationMetrics,
    baseline_fraud_rate: float,
) -> dict:
    """Converts silent failures into detectable ones (Deliverable 4, Section 3.2).

    Checks two things a confidently-wrong response would otherwise get away
    with: that the cited evidence matches a real backtest metric, and that a
    declared "alta" confidence is not paired with missing evidence. A third,
    weaker heuristic flags scope creep: the diagnostic agent proposing a new
    rule condition, which is the generator agent's job, not this one's.

    Args:
        avaliacao: The model's structured assessment.
        metricas: Backtest result the assessment was supposed to reason about.
        baseline_fraud_rate: Overall fraud rate used for the reference verdict.

    Returns:
        A dict with `ok` (True if no problem was found) and `problemas`
        (list of human-readable descriptions of what failed).
    """
    problemas = []
    evidencia_ok = metrica_citada(avaliacao.justificativa, metricas.detection_rate) or metrica_citada(
        avaliacao.justificativa, metricas.false_positive_rate
    )
    if not evidencia_ok:
        problemas.append("evidencia citada na justificativa nao bate com nenhuma metrica real do backtest")
    if avaliacao.confianca == "alta" and not evidencia_ok:
        problemas.append("confianca declarada como alta sem evidencia numerica compativel")
    texto = avaliacao.justificativa.lower()
    if any(frase in texto for frase in _FRASES_PROPOSTA_NOVA_CONDICAO):
        problemas.append(
            "justificativa do diagnostico propoe uma nova condicao de regra -- isso e "
            "escopo do agente gerador, nao do agente de diagnostico"
        )
    return {"ok": not problemas, "problemas": problemas}


def diagnosticar_regra_robusta(
    rule: Rule,
    *,
    df: pd.DataFrame,
    structured_llm_robusto,
    label_col: str = "fraud_bool",
    tentativas: int = 3,
    espera_inicial: float = 0.2,
    fator: float = 2.0,
) -> dict:
    """Runs Deliverable 1's diagnostic with retry, graceful degradation and active verification.

    Takes an already-resolved `Rule` object rather than a `rule_id`, so it
    can also diagnose an in-memory variant of a real rule (e.g. the same
    logic/metrics with a different `description`) without writing anything
    to the registry -- needed for Deliverable 4's minimal-pair comparisons
    (Section F), where only the wording changes and the metrics must stay
    identical. `diagnosticar_com_robustez` is the `rule_id`-based entry
    point for the ordinary case.

    Mirrors `riskops.diagnostics.diagnosticar_regra`'s deterministic-backtest-
    plus-one-LLM-call shape, but wraps the LLM call in `com_repeticao` and,
    if every attempt fails with a transient provider error, degrades
    gracefully to the deterministic reference verdict instead of failing the
    whole case -- always marking the result as `degradado=True` so callers
    (and the trace) can tell a degraded answer from a full one. Never
    raises: as with `diagnosticar_regra`, all failure modes are returned in
    the result dict.

    Args:
        rule: The rule to diagnose (already resolved -- not looked up here).
        df: Historical transaction data to backtest the rule against.
        structured_llm_robusto: A chat model wrapped with
            `.with_structured_output(RuleAssessmentRobusta, include_raw=True)`.
        label_col: Name of the boolean fraud-label column in `df`.
        tentativas: Maximum number of attempts against transient provider
            errors (or a transient response-parsing failure, see
            `FalhaDeFormatoTransitoria`) before degrading.
        espera_inicial: Seconds to wait before the second attempt (passed to
            `com_repeticao`). Real per-minute provider rate limits need a
            larger value than this function's conservative default.
        fator: Backoff multiplier applied after each failed attempt (passed
            to `com_repeticao`).

    Returns:
        A dict with `ok`, `degradado`, and, when `ok=True`: `veredito`,
        `justificativa`, `sugestao`, `metricas_backtest`,
        `veredito_referencia`, `problemas` (from `verificar_saida_robusta`,
        empty when degraded). `confianca` is `None` when degraded. Always
        includes `tentativas`, `chamadas_llm`, `latencia_s`.
    """
    inicio = time.perf_counter()
    baseline_fraud_rate = df[label_col].mean()
    metricas = backtest_ruleset(df, [rule], label_col=label_col).metrics
    referencia = veredito_referencia(metricas, baseline_fraud_rate)
    prompt = montar_prompt(rule, metricas, baseline_fraud_rate)

    def _invocar(prompt):
        saida = structured_llm_robusto.invoke(prompt)
        if saida["parsed"] is None:
            # Falha real, observada ao vivo: o modelo as vezes nomeia a chamada de
            # ferramenta como 'functions.RuleAssessmentRobusta' em vez de
            # 'RuleAssessmentRobusta', que o parser do LangChain rejeita
            # ("Unknown tool type"). Nao e um erro de raciocinio do modelo -- e uma
            # falha de formatacao intermitente (o mesmo prompt funciona em outra
            # tentativa), entao vale a pena tratar como transitoria e tentar de novo.
            raise FalhaDeFormatoTransitoria(str(saida.get("parsing_error")))
        return saida

    saida, info_repeticao = com_repeticao(
        _invocar,
        prompt,
        tentativas=tentativas,
        espera_inicial=espera_inicial,
        fator=fator,
        transitorias=ERROS_TRANSIENTES_DE_PROVEDOR + (FalhaDeFormatoTransitoria,),
    )
    latencia = time.perf_counter() - inicio

    if saida is None:
        # Todas as tentativas esgotadas por falha transitoria do provedor: degradacao
        # graciosa. O parecer devolvido usa so o criterio deterministico de referencia --
        # nunca finge ser uma resposta completa do modelo, e se identifica como tal tanto
        # no campo `degradado` quanto na propria justificativa.
        return {
            "ok": True,
            "degradado": True,
            "veredito": referencia,
            "justificativa": (
                f"PARECER DEGRADADO: chamada ao modelo falhou apos {info_repeticao['tentativas']} "
                "tentativa(s) por erro transitorio do provedor; veredito baseado apenas no "
                "criterio de referencia deterministico, sem justificativa do modelo."
            ),
            "sugestao": "revisar manualmente quando a fonte do modelo estiver disponivel.",
            "confianca": None,
            "metricas_backtest": metricas,
            "veredito_referencia": referencia,
            "problemas": [],
            "erros_transitorios": info_repeticao["erros"],
            "tentativas": info_repeticao["tentativas"],
            "chamadas_llm": info_repeticao["tentativas"],
            "latencia_s": round(latencia, 2),
        }

    avaliacao = saida["parsed"]
    bruta = saida["raw"]
    uso = getattr(bruta, "usage_metadata", None) or {}

    if avaliacao is None:
        return {
            "ok": False,
            "degradado": False,
            "erro": f"falha ao interpretar resposta do modelo: {saida.get('parsing_error')}",
            "problemas": [],
            "tentativas": info_repeticao["tentativas"],
            "chamadas_llm": info_repeticao["tentativas"],
            "tokens_entrada": uso.get("input_tokens"),
            "tokens_saida": uso.get("output_tokens"),
            "latencia_s": round(latencia, 2),
        }

    verificacao = verificar_saida_robusta(avaliacao, metricas, baseline_fraud_rate)
    return {
        "ok": True,
        "degradado": False,
        "veredito": avaliacao.veredito,
        "justificativa": avaliacao.justificativa,
        "sugestao": avaliacao.sugestao,
        "confianca": avaliacao.confianca,
        "metricas_backtest": metricas,
        "veredito_referencia": referencia,
        "verificacao_ok": verificacao["ok"],
        "problemas": verificacao["problemas"],
        "tentativas": info_repeticao["tentativas"],
        "chamadas_llm": info_repeticao["tentativas"],
        "tokens_entrada": uso.get("input_tokens"),
        "tokens_saida": uso.get("output_tokens"),
        "latencia_s": round(latencia, 2),
    }


def diagnosticar_com_robustez(
    rule_id: str,
    *,
    store: RuleStore,
    df: pd.DataFrame,
    structured_llm_robusto,
    label_col: str = "fraud_bool",
    tentativas: int = 3,
    espera_inicial: float = 0.2,
    fator: float = 2.0,
) -> dict:
    """`rule_id`-based entry point for `diagnosticar_regra_robusta`.

    Looks `rule_id` up in `store` first, treating a missing/empty id as a
    permanent error (never degraded, never retried) -- the same RF-04
    guarantee as `riskops.diagnostics.diagnosticar_regra`. Once the rule is
    resolved, all retry/degradation/verification logic is
    `diagnosticar_regra_robusta`'s, unchanged.

    Args:
        rule_id: Id of the rule to diagnose. Empty string or None is treated
            as invalid input.
        store: Rule registry to look the rule up in.
        df: Historical transaction data to backtest the rule against.
        structured_llm_robusto: A chat model wrapped with
            `.with_structured_output(RuleAssessmentRobusta, include_raw=True)`.
        label_col: Name of the boolean fraud-label column in `df`.
        tentativas: Maximum number of attempts against transient provider
            errors (or a transient response-parsing failure) before degrading.
        espera_inicial: Seconds to wait before the second attempt.
        fator: Backoff multiplier applied after each failed attempt.

    Returns:
        Same shape as `diagnosticar_regra_robusta`, plus the same
        `rule_id`-not-found failure mode as `diagnosticar_regra`.
    """
    inicio = time.perf_counter()

    if not rule_id:
        return {
            "ok": False,
            "degradado": False,
            "erro": "rule_id vazio ou nao informado.",
            "problemas": [],
            "tentativas": 0,
            "chamadas_llm": 0,
            "latencia_s": round(time.perf_counter() - inicio, 2),
        }

    try:
        rule = store.get(rule_id)
    except RuleNotFoundError:
        return {
            "ok": False,
            "degradado": False,
            "erro": f"regra '{rule_id}' nao encontrada no registro.",
            "problemas": [],
            "tentativas": 0,
            "chamadas_llm": 0,
            "latencia_s": round(time.perf_counter() - inicio, 2),
        }

    return diagnosticar_regra_robusta(
        rule,
        df=df,
        structured_llm_robusto=structured_llm_robusto,
        label_col=label_col,
        tentativas=tentativas,
        espera_inicial=espera_inicial,
        fator=fator,
    )


def invocar_grafo_com_limite(grafo, estado_inicial: dict, *, config: dict | None = None) -> dict:
    """Runs a compiled LangGraph graph, converting step-limit exhaustion into a plain result.

    `GraphRecursionError` was a real, observed failure mode in Deliverable 3
    (a genuinely ambiguous case drove the diagnostic agent past its
    recursion limit). Left uncaught, it crashes the whole batch instead of
    just that one case.

    Args:
        grafo: A compiled LangGraph graph (e.g. `build_graph(...).compile()`
            from `riskops.multiagent_graph` or `riskops.portfolio_graph`).
        estado_inicial: Initial state to invoke the graph with.
        config: Optional LangGraph run config (e.g. `recursion_limit`).

    Returns:
        A dict with `ok`, `degradado`, and either `estado_final` (on
        success) or `erro` (on step-limit exhaustion).
    """
    try:
        return {"ok": True, "degradado": False, "estado_final": grafo.invoke(estado_inicial, config=config)}
    except GraphRecursionError as exc:
        return {
            "ok": False,
            "degradado": True,
            "erro": f"limite de passos do grafo atingido: {exc}",
            "estado_final": None,
        }
