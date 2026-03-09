from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.planner import planner
from agents.executor import executor

def create_orchestrator(llm, tools_map):
    """
    Creates and compiles the LangGraph orchestrator.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", lambda state: planner(state, llm))
    workflow.add_node("executor", lambda state: executor(state, llm, tools_map))

    # Define edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")

    # Conditional edge for the executor loop
    def should_continue(state: AgentState):
        if state["current_step"] < len(state["plan"]):
            return "executor"
        return END

    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "executor": "executor",
            END: END
        }
    )

    return workflow.compile()
