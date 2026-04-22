import importlib
import sys


def check_dependencies():
    """Check whether the project's required third-party packages are installed."""
    required_packages = {
        # Core data science and ML
        "numpy": "numpy",
        "sentence_transformers": "sentence-transformers",
        "sklearn": "scikit-learn",
        "torch": "torch",
        "transformers": "transformers",
        "huggingface_hub": "huggingface-hub",
        "nltk": "nltk",

        # Tokenization and metrics
        "tiktoken": "tiktoken",
        "jieba": "jieba",
        "sentencepiece": "sentencepiece",
        "rank_bm25": "rank-bm25",
        "rouge_score": "rouge-score",
        "bert_score": "bert-score",

        # LLM clients and evaluation helpers
        "openai": "openai",
        "litellm": "litellm",
        "ollama": "ollama",
        "FlagEmbedding": "FlagEmbedding",
        "backoff": "backoff",

        # Utilities
        "tqdm": "tqdm",
        "pandas": "pandas",
        "dotenv": "python-dotenv",
        "tenacity": "tenacity",
        "requests": "requests",

        # GraphRAG community detection
        "networkx": "networkx",
        "leidenalg": "leidenalg",
        "igraph": "igraph",
    }

    missing_packages = []

    print("=" * 50)
    print("Starting dependency check...")
    print("=" * 50)

    for module_name, pip_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"[OK] {pip_name} (import: {module_name})")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"[MISSING] {pip_name} (module: {module_name})")

    print("=" * 50)

    if missing_packages:
        print("\nMissing required packages. Install with:\n")
        print(f"pip install {' '.join(missing_packages)}\n")
        print("Or install from requirements.txt:")
        print("pip install -r requirements.txt\n")
        sys.exit(1)

    print("\nAll required packages are installed.")
    sys.exit(0)


if __name__ == "__main__":
    check_dependencies()
