import torch
import numpy as np
from baukit import Trace, TraceDict


def extract_activations(
    model,
    tokenizer,
    texts: list[str],
    layer_name: str,
    device: torch.device,
) -> np.ndarray:
    """
    Run a forward pass and return the final-token activation at a given layer.

    We take the final token because in a causal language model that position
    aggregates all context — it's what the model bases its next prediction on.
    This is the vector we'll train probes on and ablate the refusal direction from.

    Args:
        model:      loaded model from model_loader.load_model()
        tokenizer:  corresponding tokenizer
        texts:      list of input strings (e.g. TOFU forget-set questions)
        layer_name: which layer to hook into, e.g. "model.layers.15"
        device:     torch.device from model_loader.get_device()

    Returns:
        activations: np.ndarray of shape (n_texts, hidden_size)
    """
    activations = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with Trace(model, layer_name) as trace:
            with torch.no_grad():
                model(**inputs)

            # trace.output is the layer's output tensor
            # shape: (batch, seq_len, hidden_size)
            # we take the last token position [-1]
            hidden = trace.output
            if isinstance(hidden, tuple):
                hidden = hidden[0]

            last_token = hidden[0, -1, :].float().cpu().numpy()
            activations.append(last_token)

    return np.array(activations)

def extract_activations_multi(
    model,
    tokenizer,
    texts: list[str],
    layer_names: list[str],
    device,
) -> dict[str, np.ndarray]:
    """
    Run a forward pass and return final-token activations at MULTIPLE layers
    in a single pass per text.

    This is the multi-layer analogue of extract_activations. Use it when you
    need several layers at once (e.g. a coherence/probe scan across layers, or
    a set of non-consecutive candidate layers) — it hooks all requested layers
    simultaneously via baukit's TraceDict, so each text requires only ONE
    forward pass regardless of how many layers are requested.

    Args:
        model:       loaded model from model_loader.load_model()
        tokenizer:   corresponding tokenizer
        texts:       list of input strings
        layer_names: list of layer module names, e.g.
                     ["model.layers.8", "model.layers.12", "model.layers.14"]
        device:      torch.device from model_loader.get_device()

    Returns:
        dict mapping each layer_name -> np.ndarray of shape (n_texts, hidden_size)
    """
    # Accumulate per-layer last-token activations
    collected = {name: [] for name in layer_names}

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        # TraceDict hooks all requested layers in a single forward pass
        with TraceDict(model, layer_names) as traces:
            with torch.no_grad():
                model(**inputs)

            for name in layer_names:
                hidden = traces[name].output
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                last_token = hidden[0, -1, :].float().cpu().numpy()
                collected[name].append(last_token)

    return {name: np.array(acts) for name, acts in collected.items()}


def get_layer_names(model) -> list[str]:
    """
    Print all hookable layer names in the model.
    Run this once to find the right layer_name for extract_activations().
    """
    return [name for name, _ in model.named_modules()]