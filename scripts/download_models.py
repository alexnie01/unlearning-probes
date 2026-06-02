from huggingface_hub import snapshot_download
from datasets import load_dataset
import os

CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

# Base and retain oracle
BASE_MODELS = [
    "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
    "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
]

# Unlearned checkpoints — one canonical checkpoint per method
# Selected by highest download count in the TOFU Unlearned Models collection
# GradDiff = gradient ascent with retain regularization (standard GA variant)
# NPO     = negative preference optimization (output preference method)
# RMU     = representation mismatch unlearning (representation targeting method)
UNLEARNED_MODELS = [
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.5_alpha1_epoch10",
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer10_scoeff10_epoch10",
]

ALL_MODELS = BASE_MODELS + UNLEARNED_MODELS


def download_model(model_id: str) -> None:
    print(f"\n→ {model_id}")
    snapshot_download(repo_id=model_id, cache_dir=CACHE_DIR)
    print(f"✓ Done: {model_id}")


def main():
    print("=" * 60)
    print("Downloading TOFU 1B checkpoints")
    print(f"Total models: {len(ALL_MODELS)}")
    print("=" * 60)

    for model_id in ALL_MODELS:
        download_model(model_id)

    print("\n" + "=" * 60)
    print("Downloading TOFU dataset splits")
    print("=" * 60)

    print("\n→ locuslab/TOFU forget10")
    load_dataset("locuslab/TOFU", "forget10")
    print("✓ Done: forget10")

    print("\n→ locuslab/TOFU retain90")
    load_dataset("locuslab/TOFU", "retain90")
    print("✓ Done: retain90")

    print("\n" + "=" * 60)
    print("All downloads complete.")
    print("\nCheckpoints downloaded:")
    for m in ALL_MODELS:
        print(f"  - {m.split('/')[-1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()