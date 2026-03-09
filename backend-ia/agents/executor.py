from typing import Dict, Any, List
from langchain_ollama import OllamaLLM
from agents.state import AgentState

# Note: Integration with actual RAG and Tools will be done in the main.py or by passing them as arguments
def executor(state: AgentState, llm: OllamaLLM, tools_map: Dict[str, Any]) -> AgentState:
    """
    Executes the current step in the plan.
    """
    if state["current_step"] >= len(state["plan"]):
        return state
    
    current_step_text = state["plan"][state["current_step"]]
    print(f"Executing step {state['current_step'] + 1}/{len(state['plan'])}: {current_step_text}")
    
    # Simple logic to decide based on step prefix
    # In a more advanced agent, we would use reasoning (LLM) to decide.
    
    result = ""
    if "RAG_RETRIEVAL" in current_step_text:
        # Call RAG tool
        if "rag" in tools_map:
            search_query = current_step_text.split(":", 1)[1].strip() if ":" in current_step_text else state["question"]
            result = tools_map["rag"](search_query)
            state["context"] += f"\n--- RAG Results ---\n{result}\n"
    
    elif "PRODUCTIVITY_TOOL" in current_step_text:
        # Call Productivity tool
        if "productivity" in tools_map:
            result = tools_map["productivity"](state["question"])
            state["tool_results"].append({"step": current_step_text, "result": result})
    
    elif "K8S_TOOL" in current_step_text:
        # Call K8s tool
        if "k8s" in tools_map:
            result = tools_map["k8s"](state["question"])
            state["tool_results"].append({"step": current_step_text, "result": result})
    
    else:
        # Default to Reasoning
        prompt = f"Context: {state['context']}\nTool Results: {state['tool_results']}\nNext Step: {current_step_text}\nAnswer this step based on context or general knowledge."
        result = llm.invoke(prompt)
        state["context"] += f"\n--- Reasoning Step ---\n{result}\n"

    # Move to next step
    state["current_step"] += 1
    
    # If all steps are done, generate final answer
    if state["current_step"] >= len(state["plan"]):
        final_prompt = f"Question: {state['question']}\nContext: {state['context']}\nTool Results: {state['tool_results']}\nBased on all information above, provide a comprehensive final answer to the user's question."
        state["final_answer"] = llm.invoke(final_prompt)
        
    return state
