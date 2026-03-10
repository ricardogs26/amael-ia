import re
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

from langchain_ollama import OllamaLLM
from agents.state import AgentState
from agents.metrics import (
    EXECUTOR_STEP_LATENCY_SECONDS,
    EXECUTOR_STEPS_TOTAL,
    EXECUTOR_ERRORS_TOTAL,
    EXECUTOR_CONTEXT_TRUNCATIONS_TOTAL,
    EXECUTOR_ESTIMATED_PROMPT_TOKENS,
    EXECUTOR_PARALLEL_BATCH_SIZE,
    EXECUTOR_PARALLEL_BATCHES_TOTAL,
)
from agents.tracing import tracer

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama-service:11434")
llm_reasoning = OllamaLLM(model="qwen2.5:14b", base_url=OLLAMA_BASE_URL)

# ── Context window limits ─────────────────────────────────────────────────────
MAX_CONTEXT_CHARS = 12_000  # ~3 000 tokens
MAX_ANSWER_CHARS = 8_000    # ~2 000 tokens


# ── Helpers ───────────────────────────────────────────────────────────────────
def _truncate(text: str, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    EXECUTOR_CONTEXT_TRUNCATIONS_TOTAL.inc()
    logger.warning(f"[EXECUTOR] '{label}' truncado {len(text)} → {max_chars} chars.")
    return "[...contexto anterior truncado para ajustar a la ventana del LLM...]\n" + text[-max_chars:]


def _step_type(step: str) -> str:
    return step.split(":")[0].strip().upper()


# ── Single tool step (K8S, RAG, PRODUCTIVITY) ────────────────────────────────
def _run_tool_step(step: str, state: AgentState, tools_map: Dict[str, Any]) -> str:
    """
    Execute a single non-REASONING tool step.
    Safe to call from multiple threads — reads state, never writes it.
    Returns the result string for that step.
    """
    stype = _step_type(step)
    t0 = time.time()

    with tracer.start_as_current_span(f"agent.executor.{stype.lower()}") as span:
        span.set_attribute("agent.step_type", stype)
        span.set_attribute("agent.step", step)

        try:
            result = ""

            if step.startswith("K8S_TOOL:"):
                k8s_allowed_csv = os.environ.get("K8S_ALLOWED_USERS_CSV", "")
                k8s_allowed = [u.strip() for u in k8s_allowed_csv.split(",") if u.strip()]
                user_id = state.get("user_id", "unknown")
                if k8s_allowed and user_id not in k8s_allowed:
                    logger.warning(f"[EXECUTOR] K8S bloqueado para user={user_id}")
                    result = "Lo siento, tu usuario no cuenta con los privilegios de administrador requeridos."
                else:
                    query = step[len("K8S_TOOL:"):].strip()
                    k8s_func = tools_map.get("k8s")
                    if k8s_func:
                        result = k8s_func(query)
                    else:
                        EXECUTOR_ERRORS_TOTAL.labels(step_type="K8S_TOOL").inc()
                        result = "Error: Herramienta K8s no disponible."

            elif step.startswith("RAG_RETRIEVAL:"):
                query = step[len("RAG_RETRIEVAL:"):].strip()
                rag_func = tools_map.get("rag")
                if rag_func:
                    result = rag_func(query)
                else:
                    EXECUTOR_ERRORS_TOTAL.labels(step_type="RAG_RETRIEVAL").inc()
                    result = "Error: Herramienta RAG no disponible."

            elif step.startswith("PRODUCTIVITY_TOOL:"):
                query = step[len("PRODUCTIVITY_TOOL:"):].strip()
                prod_func = tools_map.get("productivity")
                if prod_func:
                    result = prod_func(query)
                else:
                    EXECUTOR_ERRORS_TOTAL.labels(step_type="PRODUCTIVITY_TOOL").inc()
                    result = "Error: Herramienta de productividad no disponible."

            elif step.startswith("WEB_SEARCH:"):
                query = step[len("WEB_SEARCH:"):].strip()
                web_func = tools_map.get("web_search")
                if web_func:
                    result = web_func(query)
                else:
                    EXECUTOR_ERRORS_TOTAL.labels(step_type="WEB_SEARCH").inc()
                    result = "Error: Herramienta de búsqueda web no disponible."

            elif step.startswith("DOCUMENT_TOOL:"):
                query = step[len("DOCUMENT_TOOL:"):].strip()
                doc_func = tools_map.get("document")
                if doc_func:
                    result = doc_func(query)
                else:
                    EXECUTOR_ERRORS_TOTAL.labels(step_type="DOCUMENT_TOOL").inc()
                    result = "Error: Herramienta de documentos no disponible."

        except Exception as exc:
            EXECUTOR_ERRORS_TOTAL.labels(step_type=stype).inc()
            logger.error(f"[EXECUTOR] Error en {stype}: {exc}", exc_info=True)
            span.record_exception(exc)
            result = f"Error en {stype}: {str(exc)[:200]}"

        elapsed = time.time() - t0
        EXECUTOR_STEP_LATENCY_SECONDS.labels(step_type=stype).observe(elapsed)
        EXECUTOR_STEPS_TOTAL.labels(step_type=stype).inc()
        span.set_attribute("agent.step_latency_seconds", elapsed)

    return result


# ── REASONING step ────────────────────────────────────────────────────────────
def _run_reasoning_step(
    step: str,
    state: AgentState,
) -> tuple[str, str]:
    """
    Execute a REASONING step.
    Returns (new_final_answer, unchanged_context).
    """
    media_pattern = r"\[MEDIA:.+?\]"
    media_data = None
    current_answer = state.get("final_answer", "") or ""
    context = state.get("context", "") or ""

    reasoning_task = step[len("REASONING:"):].strip()

    # Extract [MEDIA:...] BEFORE truncating — base64 payloads are large and
    # _truncate keeps only the tail, which would drop the [MEDIA: prefix.
    if "[MEDIA:" in current_answer:
        match = re.search(media_pattern, current_answer, re.DOTALL)
        if match:
            media_data = match.group(0)
            current_answer = re.sub(media_pattern, "[IMAGEN_ADJUNTA_GENERADA]", current_answer, flags=re.DOTALL)

    context_for_llm = _truncate(current_answer, MAX_ANSWER_CHARS, "final_answer")

    prompt = (
        f"Contexto previo:\n{context_for_llm}\n\n"
        f"Tarea a realizar: {reasoning_task}\n\n"
        f"Instrucción: Genera una respuesta en ESPAÑOL basada en el contexto anterior.\n"
        f"REGLAS DE FORMATO — SIGUE ESTAS REGLAS AL PIE DE LA LETRA:\n"
        f"1. Si el contexto contiene un bloque ```bash o ```yaml, CÓPIALO EXACTAMENTE en tu respuesta sin modificarlo.\n"
        f"2. NUNCA pongas análisis, recomendaciones, texto en español ni markdown dentro de un bloque ```bash o ```yaml. "
        f"Los bloques de código contienen ÚNICAMENTE salida raw de comandos o código fuente.\n"
        f"3. Tu análisis, explicaciones y recomendaciones van FUERA de los bloques, como texto normal.\n"
        f"4. NO conviertas datos tabulares de kubectl en tablas markdown; preserva el bloque ```bash original.\n"
        f"5. Si generas scripts bash o manifiestos YAML nuevos, envuélvelos en ```bash o ```yaml respectivamente."
    )

    estimated_tokens = len(prompt) // 4
    EXECUTOR_ESTIMATED_PROMPT_TOKENS.labels(step_type="REASONING").observe(estimated_tokens)

    new_answer = llm_reasoning.invoke(prompt)
    if media_data:
        new_answer += f"\n\n{media_data}"

    return new_answer, context


# ── Parallel batch runner ─────────────────────────────────────────────────────
def _run_parallel_batch(
    batch: List[str], state: AgentState, tools_map: Dict[str, Any]
) -> tuple[str, str]:
    """
    Run all steps in a batch concurrently using threads.
    All steps must be non-REASONING (guaranteed by the grouper).
    Returns (combined_final_answer, accumulated_context).
    """
    EXECUTOR_PARALLEL_BATCHES_TOTAL.inc()
    EXECUTOR_PARALLEL_BATCH_SIZE.observe(len(batch))

    results: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(batch)) as pool:
        future_to_step = {
            pool.submit(_run_tool_step, step, state, tools_map): step
            for step in batch
        }
        for future in as_completed(future_to_step):
            step = future_to_step[future]
            try:
                results[step] = future.result()
            except Exception as exc:
                stype = _step_type(step)
                EXECUTOR_ERRORS_TOTAL.labels(step_type=stype).inc()
                results[step] = f"Error en {stype}: {str(exc)[:200]}"

    # Combine results preserving original order
    parts = []
    new_context = state.get("context", "") or ""
    for step in batch:
        stype = _step_type(step)
        result = results.get(step, "")
        parts.append(f"--- {stype} ---\n{result}")
        # Accumulate RAG results into context
        if stype == "RAG_RETRIEVAL":
            combined = (new_context + "\n" + result).strip() if new_context else result
            new_context = _truncate(combined, MAX_CONTEXT_CHARS, "rag_context")

    combined_answer = "\n\n".join(parts)
    return combined_answer, new_context


# ── Main entry point: batch_executor ─────────────────────────────────────────
def batch_executor(state: AgentState, llm: OllamaLLM = None, tools_map: Dict[str, Any] = None) -> dict:
    # P5-2: read tools from state when not provided explicitly
    if tools_map is None:
        tools_map = state.get("tools_map", {})
    """
    Executes one batch of plan steps and returns the partial AgentState update.

    - Single-step batch     → run directly (sequential)
    - Multi-step tool batch → run with ThreadPoolExecutor (parallel)
    - REASONING batch       → always single-step; synthesises prior context
    """
    batches: List[List[str]] = state.get("batches", [])
    current_batch_idx: int = state.get("current_batch", 0)
    current_step: int = state.get("current_step", 0)

    if current_batch_idx >= len(batches):
        return {"current_batch": current_batch_idx}

    batch = batches[current_batch_idx]
    logger.info(
        f"[EXECUTOR] Batch {current_batch_idx + 1}/{len(batches)} "
        f"({len(batch)} paso{'s' if len(batch) > 1 else ''}): {batch}"
    )

    t0 = time.time()
    EXECUTOR_PARALLEL_BATCH_SIZE.observe(len(batch))

    with tracer.start_as_current_span("agent.executor.batch") as span:
        span.set_attribute("agent.batch_index", current_batch_idx)
        span.set_attribute("agent.batch_size", len(batch))
        span.set_attribute("agent.batch", str(batch))

        # ── REASONING (always single step) ───────────────────────────────────
        if len(batch) == 1 and batch[0].upper().startswith("REASONING:"):
            step = batch[0]
            t_step = time.time()
            with tracer.start_as_current_span("agent.executor.reasoning") as r_span:
                r_span.set_attribute("agent.step", step)
                new_answer, new_context = _run_reasoning_step(step, state)
                elapsed = time.time() - t_step
                EXECUTOR_STEP_LATENCY_SECONDS.labels(step_type="REASONING").observe(elapsed)
                EXECUTOR_STEPS_TOTAL.labels(step_type="REASONING").inc()
                r_span.set_attribute("agent.step_latency_seconds", elapsed)

        # ── Single tool step ──────────────────────────────────────────────────
        elif len(batch) == 1:
            result = _run_tool_step(batch[0], state, tools_map)
            stype = _step_type(batch[0])
            new_context = state.get("context", "") or ""
            if stype == "RAG_RETRIEVAL":
                combined = (new_context + "\n" + result).strip() if new_context else result
                new_context = _truncate(combined, MAX_CONTEXT_CHARS, "rag_context")
            new_answer = result

        # ── Parallel tool batch ───────────────────────────────────────────────
        else:
            new_answer, new_context = _run_parallel_batch(batch, state, tools_map)

        elapsed_batch = time.time() - t0
        span.set_attribute("agent.batch_latency_seconds", elapsed_batch)

    return {
        "final_answer": new_answer,
        "context": new_context,
        "current_batch": current_batch_idx + 1,
        "current_step": current_step + len(batch),
    }
