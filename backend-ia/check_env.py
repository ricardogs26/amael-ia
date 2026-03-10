import sys
import os

def check_env():
    print("Checking Amael-IA Backend Environment...")
    modules = [
        "fastapi",
        "langchain",
        "langchain_ollama",
        "langgraph",
        "httpx",
        "qdrant_client",
        "redis",
        "psycopg2",
        "minio",
        "magic"
    ]
    
    missing = []
    for mod in modules:
        try:
            __import__(mod)
            print(f"[OK] {mod}")
        except ImportError:
            missing.append(mod)
            print(f"[ERROR] {mod} is MISSING")

    if missing:
        print(f"\nCRITICAL: Missing modules: {', '.join(missing)}")
        print("Please ensure you have rebuilt the Docker image after my changes to requirements.txt.")
    else:
        print("\nAll core modules are present.")
        
    print("\nChecking agents package...")
    try:
        from agents.state import AgentState
        from agents.orchestrator import create_orchestrator
        print("[OK] agents package imported successfully")
    except Exception as e:
        print(f"[ERROR] Failed to import agents: {e}")

if __name__ == "__main__":
    check_env()
