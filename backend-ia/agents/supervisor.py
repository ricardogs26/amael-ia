"""
P3 — Supervisor Agent
=====================
Evaluates the final answer produced by the executor and decides whether to
ACCEPT it or trigger a REPLAN.

Decision criteria
-----------------
- ACCEPT (score >= 6): answer is relevant, coherent and addresses the question.
- REPLAN (score < 6):  answer is empty, off-topic, full of tool errors, or
                       contains obvious contamination from unrelated RAG content
                       (e.g. Mexican SAT tax notices when asking about Vault).

Feedback pipeline (P3-2)
------------------------
Every evaluation is stored in Redis as a lightweight feedback entry so that
response quality can be tracked over time in Grafana.
"""
import json
import logging
import os
import time
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError, field_validator
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import AgentState
from agents.metrics import (
    SUPERVISOR_DECISIONS_TOTAL,
    SUPERVISOR_QUALITY_SCORE,
    SUPERVISOR_REPLAN_TOTAL,
    SUPERVISOR_LATENCY_SECONDS,
)
from agents.tracing import tracer

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama-service:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5:14b")

# P5-1: Module-level singleton (temperature=0 for deterministic evaluation)
_chat_llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)

# Maximum re-plans the supervisor may trigger per conversation turn.
MAX_RETRIES = 1

# Redis key pattern for feedback entries (lpush, capped at 100 per user).
_FEEDBACK_KEY_PREFIX = "agent_feedback:"
_FEEDBACK_MAX_ENTRIES = 100

# ── Pydantic output schema ────────────────────────────────────────────────────

class SupervisorDecision(BaseModel):
    decision: Literal["ACCEPT", "REPLAN"]
    quality_score: int
    reason: str

    @field_validator("quality_score")
    @classmethod
    def clamp_score(cls, v: int) -> int:
        return max(0, min(10, v))


# ── System prompt ─────────────────────────────────────────────────────────────

_SUPERVISOR_SYSTEM_PROMPT = """Eres un supervisor de calidad para un agente de IA llamado Amael.
Tu única tarea es evaluar si la respuesta generada es relevante y útil para la pregunta del usuario.

Responde ÚNICAMENTE con un JSON válido con este formato exacto (sin texto adicional):
{"decision": "ACCEPT", "quality_score": 8, "reason": "La respuesta contiene datos reales del clúster."}

Reglas de evaluación:
- ACCEPT (score 6-10): La respuesta aborda la pregunta, contiene datos reales o razonamiento útil.
- REPLAN (score 0-5): La respuesta está vacía, es puro ruido, contiene información de servicios
  externos NO relacionados con la pregunta (impuestos, SAT, correos fiscales, publicidad), o
  repite errores de herramientas sin agregar valor.

IMPORTANTE:
- Si la respuesta contiene información técnica del clúster Kubernetes o de Vault, SIEMPRE acepta.
- Las respuestas cortas pero correctas deben aceptarse (score 6+).
- Solo rechaza si la respuesta es claramente irrelevante o contaminada con datos externos ajenos.
- NO rechaces por formato o estilo, solo por relevancia."""


# ── Main supervisor function ──────────────────────────────────────────────────

def supervisor(state: AgentState, llm=None, redis_client=None) -> dict:
    """
    Evaluates the current final_answer and returns a state update with:
    - supervisor_score
    - supervisor_reason
    - supervisor_decision ("ACCEPT" | "REPLAN")
    - retry_count (incremented on REPLAN)

    Also stores feedback in Redis (non-blocking; failures are logged, not raised).
    """
    with tracer.start_as_current_span("agent.supervisor") as span:
        question = state.get("question", "")
        final_answer = state.get("final_answer", "") or ""
        plan = state.get("plan", [])
        retry_count = state.get("retry_count", 0)
        user_id = state.get("user_id", "unknown")

        span.set_attribute("agent.user_id", user_id)
        span.set_attribute("agent.retry_count", retry_count)

        # ── Fast-path: empty answer → always REPLAN if retries remain ─────────
        if not final_answer.strip():
            logger.warning("[SUPERVISOR] Respuesta vacía detectada.")
            decision = "REPLAN" if retry_count < MAX_RETRIES else "ACCEPT"
            score = 0
            reason = "La respuesta está vacía."
            _record(decision, score, reason, question, plan, user_id, redis_client)
            return _build_update(decision, score, reason, retry_count)

        # ── LLM evaluation ────────────────────────────────────────────────────
        evaluation_prompt = (
            f"PREGUNTA DEL USUARIO:\n{question}\n\n"
            f"RESPUESTA GENERADA:\n{final_answer[:2000]}"  # cap to avoid huge prompts
        )

        messages = [
            SystemMessage(content=_SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=evaluation_prompt),
        ]

        t0 = time.time()
        try:
            raw = _chat_llm.invoke(messages).content.strip()
            SUPERVISOR_LATENCY_SECONDS.observe(time.time() - t0)
            sv_decision = _parse_decision(raw)
        except Exception as exc:
            logger.error(f"[SUPERVISOR] Error invocando LLM: {exc}")
            SUPERVISOR_LATENCY_SECONDS.observe(time.time() - t0)
            # Fail open: accept to avoid infinite loops
            sv_decision = SupervisorDecision(decision="ACCEPT", quality_score=5, reason=f"Supervisor error: {exc}")

        span.set_attribute("agent.supervisor_decision", sv_decision.decision)
        span.set_attribute("agent.supervisor_score", sv_decision.quality_score)

        # ── Enforce MAX_RETRIES cap ───────────────────────────────────────────
        decision = sv_decision.decision
        if decision == "REPLAN" and retry_count >= MAX_RETRIES:
            logger.warning(
                f"[SUPERVISOR] REPLAN solicitado pero retry_count={retry_count} >= MAX_RETRIES={MAX_RETRIES}. "
                "Forzando ACCEPT para evitar bucle."
            )
            decision = "ACCEPT"

        _record(decision, sv_decision.quality_score, sv_decision.reason, question, plan, user_id, redis_client)
        logger.info(
            f"[SUPERVISOR] decision={decision} score={sv_decision.quality_score} "
            f"retry={retry_count} reason={sv_decision.reason!r}"
        )

        return _build_update(decision, sv_decision.quality_score, sv_decision.reason, retry_count)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_decision(raw: str) -> SupervisorDecision:
    """Parse LLM JSON output into a SupervisorDecision, with fallback."""
    import re
    # Extract first JSON object from the response
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return SupervisorDecision(**data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(f"[SUPERVISOR] Parse error: {exc}. Raw: {raw!r}")
    # Heuristic fallback
    if "replan" in raw.lower() or "rechaz" in raw.lower():
        return SupervisorDecision(decision="REPLAN", quality_score=3, reason="Heuristic REPLAN")
    return SupervisorDecision(decision="ACCEPT", quality_score=6, reason="Heuristic ACCEPT")


def _build_update(decision: str, score: int, reason: str, retry_count: int) -> dict:
    SUPERVISOR_DECISIONS_TOTAL.labels(decision=decision).inc()
    SUPERVISOR_QUALITY_SCORE.observe(score)
    if decision == "REPLAN":
        SUPERVISOR_REPLAN_TOTAL.inc()
    return {
        "supervisor_score": score,
        "supervisor_reason": reason,
        "supervisor_decision": decision,
        "retry_count": retry_count + (1 if decision == "REPLAN" else 0),
    }


def _record(
    decision: str,
    score: int,
    reason: str,
    question: str,
    plan: list,
    user_id: str,
    redis_client,
) -> None:
    """Persist feedback entry to Redis (non-blocking)."""
    if redis_client is None:
        return
    try:
        entry = json.dumps({
            "ts": time.time(),
            "user_id": user_id,
            "question": question[:200],
            "plan": plan,
            "score": score,
            "decision": decision,
            "reason": reason,
        })
        key = f"{_FEEDBACK_KEY_PREFIX}{user_id}"
        redis_client.lpush(key, entry)
        redis_client.ltrim(key, 0, _FEEDBACK_MAX_ENTRIES - 1)
    except Exception as exc:
        logger.warning(f"[SUPERVISOR] No se pudo guardar feedback en Redis: {exc}")
