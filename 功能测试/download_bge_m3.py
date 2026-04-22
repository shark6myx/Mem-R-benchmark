import argparse
from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_cache_utils import download_sentence_transformer, get_local_sentence_transformer_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BAAI/bge-m3 into the project's local model cache."
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="SentenceTransformer model name to download. Default: BAAI/bge-m3",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Optional explicit output directory. Defaults to the project's local sentence_transformers cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite the local exported model directory.",
    )
    args = parser.parse_args()

    target_dir = (
        Path(args.target_dir).expanduser().resolve()
        if args.target_dir
        else get_local_sentence_transformer_dir(args.model)
    )
    print(f"Preparing local cache for model: {args.model}")
    print(f"Target directory: {target_dir}")

    downloaded_dir = download_sentence_transformer(
        args.model,
        target_dir=args.target_dir,
        force=args.force,
    )
    print(f"Model is ready at: {downloaded_dir}")
    print("The project will now prefer this local directory when loading the model.")


if __name__ == "__main__":
    main()
