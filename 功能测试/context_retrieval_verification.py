import os
import sys

# Add parent directory to path since we moved this script to "功能测试"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_layer import LLMController
from runsingle import advancedMemAgent

def test_context_retrieval():
    print("=== Testing Context Retrieval Pipeline ===")
    
    try:
        agent = advancedMemAgent("gpt-4o-mini", "openai", 5, 0.5)
        print("\n[1] Initialized advancedMemAgent successfully.")
    except Exception as e:
        print(f"✗ Failed to initialize Agent: {e}")
        return

    session_text = """Speaker A says: Does anyone know if the BGE-M3 model is reliable for our pipeline?
Speaker B says: Yes, I tested it yesterday and the multi-lingual performance is top-notch.
Speaker A says: That's great! What about the speed?
Speaker B says: It is a bit slow compared to MiniLM.
Speaker A says: Okay, let's cache the embeddings to solve that issue.
Speaker C says: I agree, caching is the only way."""

    isolated_turn = "Speaker C says: I agree, caching is the only way."
    
    print(f"\n[2] Generating context for isolated turn:\n'{isolated_turn}'\n")
    print(f"Using Global Session Text ({len(session_text)} chars)...\n")
    
    try:
        chunk_context = agent.generate_chunk_context(session_text, isolated_turn)
        print(f"✓ Generated Context Retrieval Background:\n{chunk_context}")
    except Exception as e:
        print(f"✗ Failed to generate chunk context: {e}")
        return
        
    print("\n[3] Forming Enriched Content for Vectorization...")
    enriched_content = f"Context: {chunk_context}\nContent: {isolated_turn}"
    print(f"Final Chunk string to be embedded:\n{'-'*40}\n{enriched_content}\n{'-'*40}")
    print("\n✓ Context Retrieval pipeline logic verified.")

if __name__ == "__main__":
    test_context_retrieval()
