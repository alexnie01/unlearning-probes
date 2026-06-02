from huggingface_hub import snapshot_download
from datasets import load_dataset
import os

CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

models = [
    # Base and retain models
    "open-unlearning/tofu_Llama-3.1-8B-Instruct_full",
    "open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90",
]

print("Downloading base models...")
for model_id in models:
    print(f"\n→ {model_id}")
    snapshot_download(repo_id=model_id, cache_dir=CACHE_DIR)
    print(f"✓ Done: {model_id}")

print("\nDownloading TOFU dataset...")
load_dataset("locuslab/TOFU", "forget10")
load_dataset("locuslab/TOFU", "retain90")
print("✓ Done: TOFU dataset")

print("\nAll downloads complete.")
print("Unlearned checkpoints (GA/NPO/RMU) will be identified")
print("and downloaded in the next session after browsing HuggingFace.")