import argparse
from huggingface_hub import snapshot_download
import os

def download_model(token: str, model_name: str):
    print(f"Starting download for model: {model_name}")
    local_path = snapshot_download(
        repo_id=model_name,
        use_auth_token=token,
        local_dir=f"./models/{model_name.replace('/', '_')}",
        local_dir_use_symlinks=False
    )
    print(f"Model successfully downloaded to: {os.path.abspath(local_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Hugging Face model to local storage")
    parser.add_argument(
        "--token", type=str, required=True, help="Your Hugging Face access token"
    )
    parser.add_argument(
        "--model", type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model name to download (default: LLaMA 3 8B Instruct)"
    )
    args = parser.parse_args()

    download_model(token=args.token, model_name=args.model)