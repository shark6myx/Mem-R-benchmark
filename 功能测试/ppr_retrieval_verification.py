import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from memory_layer import AgenticMemorySystem

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

def test_ppr_pipeline():
    print("=== Testing HyDE + Community + PPR Pipeline ===")
    
    # 1. 初始化
    print("\n[1] Initializing AgenticMemorySystem...")
    try:
        api_base = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8004/v1")
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

    # 2. 构建相互链接的有悬念证据链
    print("\n[2] Setting up multi-hop evidence graph...")
    
    m1 = memory_sys.add_note("Speaker A says: The database is extremely slow during peak hours. We need to find the bottleneck.")
    m2 = memory_sys.add_note("Speaker B says: I checked the server, the CPU logic for Community Clustering is consuming 99% resources.")
    m3 = memory_sys.add_note("Speaker C says: To fix the Community Clustering CPU issue, we should implement a size limit and cache the Graph embeddings.")
    m4 = memory_sys.add_note("Speaker D says: Yes, let's use Redis for the caching layer.")
    m5 = memory_sys.add_note("Speaker A says: The Redis cache was deployed successfully. Performance is back to normal.")
    
    # 手动触发 GraphRAG 构建 (生成社区摘要 -> 支持 _community_filter)
    memory_sys.rebuild_communities()
    print("✓ GraphRAG built and communities formed.")

    query = "How was the slow database issue resolved?"
    
    # 3. 测试 HyDE 生成
    print(f"\n[3] Testing HyDE for query: '{query}'")
    hyde_doc = memory_sys._generate_hyde_query(query)
    print(f"HyDE Generated Document:\n{hyde_doc}")
    
    # 4. 测试 Advanced Retrieval
    print("\n[4] Testing Advanced PPR Retrieval...")
    try:
        results = memory_sys.find_related_memories_advanced(query, k=3)
        notes = results.get("notes", [])
        print(f"Retrieved {len(notes)} notes through PPR multi-hop.")
        for idx, n in enumerate(notes):
             print(f"  Ev-{idx+1}: {n.content}")
             
        ctx = results.get("community_context", "")
        print(f"\nAssociated Community Context:\n{ctx}")
        print("✓ Advanced Retrieval Pipeline executing successfully.")
    except Exception as e:
         print(f"✗ Advanced Retrieval Pipeline failed: {e}")

if __name__ == "__main__":
    test_ppr_pipeline()
