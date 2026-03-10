import logging
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.planner import planner, MAX_PLAN_STEPS
from agents.executor import batch_executor
from agents.grouper import group_plan_into_batches
from agents.supervisor import supervisor, MAX_RETRIES
from agents.metrics import ORCHESTRATOR_MAX_STEPS_HIT_TOTAL

logger = logging.getLogger(__name__)

# Hard cap on total batches processed (each batch >= 1 step)
MAX_GRAPH_ITERATIONS = MAX_PLAN_STEPS + 2

# P5-2: Compiled graph cache — compiled once per redis_client config
_ORCHESTRATOR_CACHE = None


def _grouper_node(state: AgentState) -> dict:
    """Converts the flat plan into parallel execution batches."""
    batches = group_plan_into_batches(state.get("plan", []))
    return {"batches": batches, "current_batch": 0}


def _compile_graph(redis_client=None):
    """
    LangGraph flow (P3):
      planner → grouper → batch_executor (loop) → supervisor
                  ↑                                    |
                  └────────── REPLAN ─────────────────┘ (if retry < MAX_RETRIES)
                                                       |
                                                    ACCEPT → END

    tools_map is read from state (P5-2), so this compiled graph is request-independent.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner)
    workflow.add_node("grouper", _grouper_node)
    workflow.add_node("batch_executor", batch_executor)
    workflow.add_node("supervisor", lambda state: supervisor(state, redis_client=redis_client))

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "grouper")
    workflow.add_edge("grouper", "batch_executor")

    def should_continue(state: AgentState):
        current = state.get("current_batch", 0)
        total = len(state.get("batches", []))
        if current >= MAX_GRAPH_ITERATIONS:
            logger.warning(
                f"[ORCHESTRATOR] MAX_GRAPH_ITERATIONS ({MAX_GRAPH_ITERATIONS}) alcanzado. Forzando fin."
            )
            ORCHESTRATOR_MAX_STEPS_HIT_TOTAL.inc()
            return "supervisor"
        if current < total:
            return "batch_executor"
        return "supervisor"

    workflow.add_conditional_edges(
        "batch_executor",
        should_continue,
        {"batch_executor": "batch_executor", "supervisor": "supervisor"},
    )

    def supervisor_routing(state: AgentState):
        decision = state.get("supervisor_decision", "ACCEPT")
        retry_count = state.get("retry_count", 0)
        if decision == "REPLAN" and retry_count <= MAX_RETRIES:
            logger.info(f"[ORCHESTRATOR] Supervisor solicitó REPLAN (retry #{retry_count}).")
            return "planner"
        return END

    workflow.add_conditional_edges(
        "supervisor",
        supervisor_routing,
        {"planner": "planner", END: END},
    )

    return workflow.compile()


def get_orchestrator(redis_client=None):
    """
    P5-2: Returns the cached compiled LangGraph.
    Compiled once on first call; subsequent calls return the cached instance.
    tools_map is passed through AgentState per request, not baked into the graph.
    """
    global _ORCHESTRATOR_CACHE
    if _ORCHESTRATOR_CACHE is None:
        logger.info("[ORCHESTRATOR] Compilando grafo LangGraph (primera vez)...")
        _ORCHESTRATOR_CACHE = _compile_graph(redis_client=redis_client)
        logger.info("[ORCHESTRATOR] Grafo compilado y cacheado.")
    return _ORCHESTRATOR_CACHE


# Backward-compat alias so existing imports don't break
def create_orchestrator(llm=None, tools_map=None, redis_client=None):
    return get_orchestrator(redis_client=redis_client)
