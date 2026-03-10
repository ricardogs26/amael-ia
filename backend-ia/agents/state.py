from typing import TypedDict, List, Optional, Any, Dict


class AgentState(TypedDict):
    question: str
    plan: List[str]
    batches: List[List[str]]  # plan grouped into parallel execution batches
    current_batch: int        # index of the batch currently being processed
    current_step: int         # total steps executed (for metrics / compat)
    context: str
    tool_results: List[Dict[str, Any]]
    final_answer: Optional[str]
    user_id: str
    # P3: Supervisor fields
    retry_count: int          # number of re-plans triggered by supervisor (max 1)
    supervisor_score: int     # quality score 0-10 assigned by supervisor
    supervisor_reason: str    # supervisor explanation for its decision
    # P5-2: Tools injected per-request so the compiled graph can be cached
    tools_map: Dict[str, Any]
