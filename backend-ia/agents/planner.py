import json
import re
from typing import List
from langchain_ollama import OllamaLLM
from agents.state import AgentState

PLANNER_PROMPT = """
You are a task planner for Amael-IA. Your goal is to break down a user question into a step-by-step execution plan.
Each step should be clear and actionable. The steps can involve:
1. K8S_TOOL: Use this for ANY question related to Kubernetes, pods, logs, latency, Prometheus metrics or Grafana dashboards. This is your primary source for infrastructure health.
2. RAG_RETRIEVAL: ONLY use this if the question is specifically about private business logic, schedules, or content from uploaded documents (PDFs/TXTs). If the question is technical/devops and can be answered by K8S_TOOL, DO NOT use RAG.
3. PRODUCTIVITY_TOOL: For today's schedule or calendar management.
4. REASONING: Answering based on general knowledge.

STRICT RULE: Do not call RAG_RETRIEVAL for DevOps/K8s/Infrastructure questions unless the user explicitly mentions a document.

CRITICAL: You MUST output ONLY a valid JSON list of strings.
No conversational text.

User Question: {question}

Examples of valid output:
["RAG_RETRIEVAL: Search for high latency causes in docs", "K8S_TOOL: Check pod status and logs", "REASONING: Synthesize results and explain the cause"]
["PRODUCTIVITY_TOOL: Schedule a meeting to discuss latency", "REASONING: Prepare a summary for the team"]

Plan:
"""

def planner(state: AgentState, llm: OllamaLLM) -> AgentState:
    """
    Calls the LLM to generate a step-by-step plan.
    """
    prompt = PLANNER_PROMPT.format(question=state["question"])
    response = llm.invoke(prompt).strip()
    
    # Simple JSON extraction
    plan = []
    try:
        # Check if the output is already a JSON list (sometimes strip() is enough)
        if response.startswith("[") and response.endswith("]"):
            plan = json.loads(response)
        else:
            # Look for JSON list within markdown blocks or elsewhere
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                plan = json.loads(match.group())
            else:
                # If no list found, use the whole response as one reasoning step
                plan = [f"REASONING: {response}"]
    except Exception as e:
        print(f"[PLANNER] Parsing error: {e}. Raw response: {response}")
        # Default fallback
        plan = [f"REASONING: Analyze and answer: {state['question']}"]
        
    return {
        **state,
        "plan": plan,
        "current_step": 0
    }
