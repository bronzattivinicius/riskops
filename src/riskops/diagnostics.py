"""Single-rule risk diagnosis: deterministic backtest plus one structured LLM call.

This is the logic behind Deliverable 1's baseline, promoted here so it can be
reused unchanged (per the course's "estrutura herdada" requirement) by
Deliverable 2's portfolio graph without being duplicated inline in a notebook.
Prompts and the `RuleAssessment.veredito` vocabulary stay in Portuguese: they
are graded, user-facing content for a Portuguese-speaking audience, not
library-internal naming.
"""

import hashlib
import json
import re
import time
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from riskops.metrics.backtest import ClassificationMetrics, backtest_ruleset
from riskops.rules.schema import Rule
from riskops.rules.store import RuleNotFoundError, RuleStore


class RuleAssessment(BaseModel):
    """Structured verdict produced by the LLM for a single risk rule.

    Attributes:
        veredito: Recommended decision: "manter", "revisar" or "aposentar".
        justificativa: Justification citing the backtest metrics used for the
            decision.
        sugestao: A concrete, actionable next step for the rule.
    """

    veredito: Literal["manter", "revisar", "aposentar"] = Field(
        description="Decisao recomendada para a regra, dadas as metricas de backtest."
    )
    justificativa: str = Field(
        description="Justificativa citando explicitamente os valores numericos das metricas."
    )
    sugestao: str = Field(description="Proximo passo concreto e acionavel.")


def veredito_referencia(metricas: ClassificationMetrics, baseline_fraud_rate: float) -> str:
    """Computes a deterministic reference verdict, without any LLM call.

    Uses fixed, non-calibrated thresholds on the rule's precision lift (over
    the dataset's overall fraud rate) and false positive rate. See
    Deliverable 1, Section 13, for the rationale and its limitations.

    Args:
        metricas: Backtest result for the rule.
        baseline_fraud_rate: Overall fraud rate of the dataset used as the
            backtest reference (e.g. ``df["fraud_bool"].mean()``).

    Returns:
        One of "manter", "revisar" or "aposentar".
    """
    if metricas.flagged_count == 0:
        return "aposentar"
    lift = (metricas.precision / baseline_fraud_rate) if baseline_fraud_rate else 0
    if lift >= 3 and metricas.false_positive_rate < 0.10:
        return "manter"
    if lift < 1.5 or metricas.false_positive_rate > 0.50:
        return "aposentar"
    return "revisar"


def metrica_citada(texto: str, valor_fracao: float, tolerancia_pp: float = 2.0) -> bool:
    """Checks whether a percentage value is cited in a text (Deliverable 1 RF-02).

    Extracts every number in the text and checks whether any is close enough
    to the expected value, tolerating the rounding a model applies when it
    writes its justification.

    Args:
        texto: Justification text to check.
        valor_fracao: Reference value as a fraction (e.g. 0.095 for 9.5%).
        tolerancia_pp: Tolerance, in percentage points, between the value
            cited in the text and the reference value.

    Returns:
        True if some number in the text is within tolerance of the reference
        value (in percentage points).
    """
    numeros = [float(n.replace(",", ".")) for n in re.findall(r"\d+[.,]?\d*", texto)]
    alvo = valor_fracao * 100
    return any(abs(n - alvo) <= tolerancia_pp for n in numeros)


def montar_prompt(rule: Rule, metricas: ClassificationMetrics, baseline_fraud_rate: float) -> str:
    """Builds the prompt sent to the LLM to diagnose a single rule.

    Args:
        rule: Rule to be diagnosed.
        metricas: Backtest result for that rule.
        baseline_fraud_rate: Overall fraud rate of the dataset used for the
            backtest.

    Returns:
        The full prompt text, in Portuguese.
    """
    return f"""Voce e um analista de risco avaliando uma regra de deteccao de fraude.

Regra: {rule.name} (id: {rule.id})
Descricao: {rule.description}
Status atual no registro: {rule.status.value}

Metricas de backtest contra {metricas.total} transacoes historicas (taxa de fraude na base: {baseline_fraud_rate:.2%}):
- Taxa de deteccao de fraude (recall): {metricas.detection_rate:.1%}
- Taxa de falso positivo: {metricas.false_positive_rate:.1%}
- Precisao (fracao dos sinalizados que sao de fato fraude): {metricas.precision:.1%}
- Taxa de aprovacao (transacoes nao sinalizadas): {metricas.approval_rate:.1%}
- Total sinalizado: {metricas.flagged_count} de {metricas.total}

Com base apenas nessas metricas, de seu parecer: a regra deve ser MANTIDA, REVISADA ou
APOSENTADA? Justifique citando os numeros acima (arredondados) e sugira um proximo passo
concreto."""


def diagnosticar_regra(
    rule_id: str,
    *,
    store: RuleStore,
    df: pd.DataFrame,
    structured_llm,
    label_col: str = "fraud_bool",
) -> dict:
    """Runs the single-rule diagnostic baseline (deterministic backtest + one LLM call).

    Never raises: errors (missing/empty ``rule_id``, unparseable model
    response) are returned as part of the result dict with ``ok=False``,
    rather than propagated as exceptions.

    Args:
        rule_id: Id of the rule to diagnose. Empty string or None is treated
            as invalid input.
        store: Rule registry to look the rule up in.
        df: Historical transaction data to backtest the rule against.
        structured_llm: A chat model already wrapped with
            ``.with_structured_output(RuleAssessment, include_raw=True)``.
        label_col: Name of the boolean fraud-label column in ``df``.

    Returns:
        A dict with ``ok`` and, when ``ok=True``: ``veredito``,
        ``justificativa``, ``sugestao``, ``metricas_backtest``,
        ``veredito_referencia``; when ``ok=False``: ``erro``. Always
        includes ``latencia_s``, ``chamadas_llm``, ``tokens_entrada`` and
        ``tokens_saida`` (execution instrumentation).
    """
    inicio = time.perf_counter()

    if not rule_id:
        return {
            "ok": False,
            "erro": "rule_id vazio ou nao informado.",
            "latencia_s": round(time.perf_counter() - inicio, 2),
            "chamadas_llm": 0,
            "tokens_entrada": None,
            "tokens_saida": None,
        }

    try:
        rule = store.get(rule_id)
    except RuleNotFoundError:
        return {
            "ok": False,
            "erro": f"regra '{rule_id}' nao encontrada no registro.",
            "latencia_s": round(time.perf_counter() - inicio, 2),
            "chamadas_llm": 0,
            "tokens_entrada": None,
            "tokens_saida": None,
        }

    baseline_fraud_rate = df[label_col].mean()
    resultado_backtest = backtest_ruleset(df, [rule], label_col=label_col)
    metricas = resultado_backtest.metrics

    prompt = montar_prompt(rule, metricas, baseline_fraud_rate)
    saida = structured_llm.invoke(prompt)
    latencia = time.perf_counter() - inicio

    avaliacao = saida["parsed"]
    bruta = saida["raw"]
    uso = getattr(bruta, "usage_metadata", None) or {}

    if avaliacao is None:
        return {
            "ok": False,
            "erro": f"falha ao interpretar resposta do modelo: {saida.get('parsing_error')}",
            "latencia_s": round(latencia, 2),
            "chamadas_llm": 1,
            "tokens_entrada": uso.get("input_tokens"),
            "tokens_saida": uso.get("output_tokens"),
        }

    return {
        "ok": True,
        "veredito": avaliacao.veredito,
        "justificativa": avaliacao.justificativa,
        "sugestao": avaliacao.sugestao,
        "metricas_backtest": metricas,
        "veredito_referencia": veredito_referencia(metricas, baseline_fraud_rate),
        "latencia_s": round(latencia, 2),
        "chamadas_llm": 1,
        "tokens_entrada": uso.get("input_tokens"),
        "tokens_saida": uso.get("output_tokens"),
    }


def verificar_caso(caso: dict, resultado: dict) -> tuple[bool | None, str]:
    """Verifies one executed test case against its expectation (Deliverable 1 RF-02/RF-03/RF-04).

    For expected-error cases (missing/empty input), only checks that the
    error was handled gracefully. For other automatic cases, checks both
    that a backtest metric is cited in the justification and that the
    verdict agrees with the deterministic reference. Manually-verified
    (ambiguous) cases are not judged automatically.

    Args:
        caso: A test case dict, with "verificacao" and "tipo" keys.
        resultado: The result dict for that case, from `diagnosticar_regra`
            or an equivalent producer (e.g. the portfolio graph), which
            must include the same keys: `ok`, `erro`, `veredito`,
            `veredito_referencia`, `justificativa`, `metricas_backtest`.

    Returns:
        A tuple `(aprovado, observacao)`: `aprovado` is a bool, or None
        when verification is manual; `observacao` explains the outcome.
    """
    if caso["verificacao"] == "manual":
        return None, "requer avaliacao manual (caso ambiguo)"

    if caso["tipo"] in ("informacao ausente", "entrada incompleta"):
        aprovado = not resultado["ok"] and bool(resultado.get("erro"))
        return aprovado, f"erro tratado: {resultado.get('erro')}"

    metricas = resultado["metricas_backtest"]
    ref = resultado["veredito_referencia"]
    veredito_ok = resultado["veredito"] == ref
    citacao_ok = metrica_citada(resultado["justificativa"], metricas.detection_rate) or metrica_citada(
        resultado["justificativa"], metricas.false_positive_rate
    )
    aprovado = veredito_ok and citacao_ok
    observacao = f"veredito={resultado['veredito']} referencia={ref} citacao_metrica={citacao_ok}"
    return aprovado, observacao


def impressao_digital_casos(test_cases: list[dict]) -> str:
    """Computes a short fingerprint of the frozen test case set.

    Used to verify the set is unchanged between deliverables. A stand-in
    for the course's own supplement utility (mentioned in the Deliverable 2
    prompt), which was not available to this project.

    Args:
        test_cases: The list of test case dicts (e.g. the module-level
            `test_cases` defined in each deliverable's notebook).

    Returns:
        A 16-character hex digest of the canonical JSON serialization.
    """
    canonico = json.dumps(test_cases, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()[:16]
