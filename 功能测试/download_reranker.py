import argparse
from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_cache_utils import download_hf_snapshot, get_local_hf_model_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a reranker model into the project's local Hugging Face cache."
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-reranker-v2-minicpm-layerwise",
        help=(
            "Reranker model name to download. "
            "Default: BAAI/bge-reranker-v2-minicpm-layerwise"
        ),
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Optional explicit output directory. Defaults to the project's local hf_models cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite the local snapshot directory.",
    )
    args = parser.parse_args()

    target_dir = (
        Path(args.target_dir).expanduser().resolve()
        if args.target_dir
        else get_local_hf_model_dir(args.model)
    )
    print(f"Preparing local cache for reranker: {args.model}")
    print(f"Target directory: {target_dir}")

    downloaded_dir = download_hf_snapshot(
        args.model,
        target_dir=args.target_dir,
        force=args.force,
    )
    print(f"Reranker is ready at: {downloaded_dir}")
    print("The project will now prefer this local directory when loading the reranker.")


if __name__ == "__main__":
    main()
