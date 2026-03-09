from typing import TypedDict, List, Optional, Annotated, Any, Dict
import operator

class AgentState(TypedDict):
    question: str
    plan: List[str]
    current_step: int
    context: str
    tool_results: List[Dict[str, Any]]
    final_answer: Optional[str]
