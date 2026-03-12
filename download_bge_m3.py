import argparse
from sentence_transformers import SentenceTransformer
import time

def main():
    parser = argparse.ArgumentParser(description="Pre-download BGE-M3 embedding model to local cache.")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3", help="Model name on HuggingFace")
    args = parser.parse_args()

    print(f"[*] Starting download of embedding model: {args.model}")
    print("[*] Note: This may take a few minutes as it downloads roughly 2.3 GB of model weights...")
    
    start_time = time.time()
    try:
        # SentenceTransformer automatically caches the model in ~/.cache/huggingface/hub/
        model = SentenceTransformer(args.model)
        elapsed = time.time() - start_time
        print(f"\n[+] Success! Model {args.model} downloaded and cached successfully.")
        print(f"[+] Download time: {elapsed:.2f} seconds.")
        print("[+] You can now run the benchmark script and it will use the cached model instantly.")
    except Exception as e:
        print(f"\n[-] Error downloading model: {e}")
        print("[-] Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
