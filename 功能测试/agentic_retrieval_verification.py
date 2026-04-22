import os
from openai import OpenAI
from memory_layer import LLMController, AgenticMemorySystem, AgenticDecomposer, ReflectionVerifier


def _resolve_vllm_model(api_base: str, api_key: str, fallback: str) -> str:
    env_model = os.getenv("VLLM_MODEL")
    if env_model:
        return env_model
    try:
        client = OpenAI(api_key=api_key, base_url=api_base)
        models = client.models.list()
        for m in getattr(models, "data", []) or []:
            model_id = getattr(m, "id", None)
            if model_id:
                return model_id
    except Exception:
        return fallback
    return fallback

def check_agentic_decomposition():
    print("=== Testing Agentic Decomposition Pipeline ===")
    
    # 1. Initialize Memory System with test configuration
    print("\n[1] Initializing AgenticMemorySystem...")
    try:
        api_base = os.getenv("VLLM_API_BASE", "http://localhost:8004/v1")
        api_key = os.getenv("VLLM_API_KEY", "asdasdasd")
        llm_model = _resolve_vllm_model(api_base, api_key, fallback="qwen3_5-flash")
        memory_sys = AgenticMemorySystem(
            llm_backend="vllm",
            llm_model=llm_model,
            api_base=api_base,
            api_key=api_key,
        )
        print("✓ MemorySystem initialized successfully.")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return

    # 2. Add sample context memories
    print("\n[2] Setting up mock memories...")
    mock_memories = [
        "The project started in January 2024. Speaker Alice proposed using the BGE-M3 model for embeddings.",
        "Speaker Bob pointed out that BGE-M3 is great for multi-lingual tasks, but might be slow for real-time inference.",
        "In March 2024, the team decided to integrate GraphRAG community clustering to improve context retrieval.",
        "Speaker Alice found a bug in the community clustering algorithm where large clusters get ignored.",
        "To fix the bug, we introduced a size threshold for communities in the late April sprint."
    ]
    for m in mock_memories:
        memory_sys.add_note(m)
    print(f"✓ Added {len(mock_memories)} mock memories.")

    # 3. Test AgenticDecomposer individually
    print("\n[3] Testing AgenticDecomposer...")
    complex_query = "What did Alice find wrong with the clustering algorithm and how was it resolved?"
    
    decomposer = AgenticDecomposer(memory_sys.llm_controller)
    try:
        decomposition_result = decomposer.decompose(complex_query)
        print(f"Original Query: {complex_query}")
        print(f"Decomposition Output:\n{decomposition_result}")
        print("✓ AgenticDecomposer executed successfully.")
    except Exception as e:
        print(f"✗ AgenticDecomposer execution failed: {e}")

    # 4. Test ReflectionVerifier individually
    print("\n[4] Testing ReflectionVerifier...")
    verifier = ReflectionVerifier(memory_sys.llm_controller)
    try:
        sample_doc = "Speaker Alice found a bug in the community clustering algorithm where large clusters get ignored."
        relevance = verifier.verify_relevance("What bug did Alice find?", sample_doc)
        print(f"Relevance Verification Output:\n{relevance}")
        print("✓ ReflectionVerifier executed successfully.")
    except Exception as e:
        print(f"✗ ReflectionVerifier execution failed: {e}")

    # 5. Test full Agentic Retrieval Pipeline
    print("\n[5] Testing full agentic_retrieve pipeline...")
    try:
        final_answer = memory_sys.agentic_retrieve(complex_query, k=3)
        print("Final Retrieved Context:")
        print(final_answer)
        print("✓ Full Retrieval Pipeline executed successfully.")
    except Exception as e:
        print(f"✗ Full Retrieval Pipeline failed: {e}")
        
if __name__ == "__main__":
    check_agentic_decomposition()
