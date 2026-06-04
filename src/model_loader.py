import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    model_path: str,
    dtype: torch.dtype = torch.float16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """
    Load a model and tokenizer from a HuggingFace path or local directory.

    Works for any checkpoint in this project — TOFU unlearning checkpoints
    (Phase 1) and base Llama-3.1-8B-Instruct (Phase 2) use identical loading
    code. Only model_path changes between experiments.

    Args:
        model_path: HuggingFace repo ID or local path.
                    e.g. "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
        dtype:      torch.float16 by default. Do not use float32 or bfloat16
                    unless you have a specific reason — float16 fits in 32GB
                    and preserves activation geometry for probe training.

    Returns:
        model, tokenizer, device
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=str(device),
    )
    model.eval()

    print(f"Loaded: {model_path}")
    print(f"Device: {device} | dtype: {dtype}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    return model, tokenizer, device