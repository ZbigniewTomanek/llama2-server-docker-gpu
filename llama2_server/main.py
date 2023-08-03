import argparse
from pathlib import Path

import uvicorn
from huggingface_hub import hf_hub_download
from llama_cpp.server.app import Settings, create_app

DOCKER_CACHE_DIR = Path("/cache")
CACHE_DIR = DOCKER_CACHE_DIR if DOCKER_CACHE_DIR.exists() else Path("cache")


def download_model(repo_id: str, filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=CACHE_DIR,
            local_dir_use_symlinks=False,
        )
    )


def get_path_to_model(repo_id: str, filename: str) -> Path:
    path_to_model = CACHE_DIR / filename
    if not path_to_model.exists():
        print("Model not found in cache, downloading...")
        path_to_model = Path(download_model(repo_id, filename))
    else:
        print("Model found in cache, skipping download.")
    return path_to_model


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for name, field in Settings.model_fields.items():
        description = field.description
        if field.default is not None and description is not None:
            description += f" (default: {field.default})"
        parser.add_argument(
            f"--{name}",
            dest=name,
            type=field.annotation if field.annotation is not None else str,
            help=description,
        )
    parser.add_argument(
        "--hg-repo-id", dest="hg_repo_id", type=str, help="HuggingFace model repo id"
    )
    parser.add_argument(
        "--hg-filename",
        dest="hg_filename",
        type=str,
        help="HuggingFace LLAMA2 model filename",
    )
    return parser


def main() -> None:
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    parser = setup_parser()
    args = parser.parse_args()

    model_path = get_path_to_model(args.hg_repo_id, args.hg_filename)
    args.model = model_path.as_posix()

    settings = Settings(
        **{
            k: v
            for k, v in vars(args).items()
            if v is not None and k not in {"hg_repo_id", "hg_filename"}
        }
    )
    print("Applied settings:")
    print(settings.model_dump_json(indent=4))
    app = create_app(settings=settings)

    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
