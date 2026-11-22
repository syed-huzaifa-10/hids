"""Upload the model repo (model card + pipeline) and the Space to Hugging Face.

Usage:
    export HF_TOKEN=hf_xxx          # a *write* token from huggingface.co/settings/tokens
    python deploy/push_to_hf.py YOUR_HF_USERNAME
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

user = sys.argv[1]
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
root = Path(__file__).resolve().parent

model_repo, space_repo = f"{user}/hids-nsl-kdd-classifier", f"{user}/hids-demo"

api.create_repo(model_repo, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=model_repo, repo_type="model", folder_path=root / "hf_model")
print("Model:", f"https://huggingface.co/{model_repo}")

api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(repo_id=space_repo, repo_type="space", folder_path=root / "hf_space")
print("Space:", f"https://huggingface.co/spaces/{space_repo}")
