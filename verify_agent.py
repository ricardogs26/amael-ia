import sys
import os

# Add the backend-ia directory to the path so we can import agents
sys.path.append(os.path.join(os.getcwd(), "backend-ia"))

from langchain_ollama import OllamaLLM
from agents.state import AgentState
from agents.orchestrator import create_orchestrator

def test_agent_flow():
    print("Testing Amael-IA Agent Flow...")
    
    # Mock LLM for testing if actual Ollama isn't reachable or to save time
    # However, since we want to verify real behavior, we'll try to use the configured one
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama-service:11434")
    llm = OllamaLLM(model="qwen2.5:14b", base_url=ollama_url)
    
    # Mock tools
    def mock_rag(query):
        print(f"[Mock RAG] searching for: {query}")
        return "This is some mock context about Amael-IA project."
        
    def mock_productivity(query):
        print(f"[Mock Productivity] organizing day for: {query}")
        return "Your day has been organized (Mock)."
        
    def mock_k8s(query):
        print(f"[Mock K8s] querying cluster for: {query}")
        return "All pods are running normally (Mock)."

    tools_map = {
        "rag": mock_rag,
        "productivity": mock_productivity,
        "k8s": mock_k8s
    }

    orchestrator_app = create_orchestrator(llm, tools_map)
    
    test_questions = [
        "Please search my documents for info about Amael and then tell me if the cluster is ok.",
        "Create a step-by-step plan to analyze why an LLM deployed in Kubernetes has high latency."
    ]
    
    for question in test_questions:
        print(f"\n--- Testing Question: {question} ---")
        initial_state = {
            "question": question,
            "plan": [],
            "current_step": 0,
            "context": "",
            "tool_results": [],
            "final_answer": None
        }
        
        try:
            final_state = orchestrator_app.invoke(initial_state)
            
            print("\nFinal State:")
            print(f"Plan: {final_state['plan']}")
            print(f"Final Answer: {final_state['final_answer']}")
        except Exception as e:
            print(f"Error during agent execution: {e}")

if __name__ == "__main__":
    test_agent_flow()
