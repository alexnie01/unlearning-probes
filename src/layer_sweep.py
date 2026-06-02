"""
Layer sweep utilities for probe training and visualization.

For each transformer layer, this module:
- Extracts forget/retain activations
- Trains a linear probe to find the forget/retain boundary
- Extracts the refusal direction
- Computes geometric separation metrics
- Saves probe, direction, and plot data to disk in a structured layout
- Generates an in-notebook summary accuracy curve

Outputs are saved under save_dir/ as:
    sweep_metadata.json              — summary metrics for all layers
    layer{i}/probe.joblib            — trained probe
    layer{i}/direction.npy           — unit-norm refusal direction
    layer{i}/plot_data.json          — precomputed histogram and PCA data

The HTML dashboard is generated separately via src/dashboard.py from these
files — keeps responsibilities clean and lets you regenerate the report
without rerunning the sweep.
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.probes import train_probe, get_refusal_direction, save_probe
from src.hooks import extract_activations


# ---------------------------------------------------------------------------
# Per-layer computations
# ---------------------------------------------------------------------------

def _extract_layer_activations(model, tokenizer, forget_qs, retain_qs, layer, device):
    """Extract balanced forget/retain activations at a given layer index."""
    layer_name = f"model.layers.{layer}"
    n = len(forget_qs)
    forget_acts = extract_activations(model, tokenizer, forget_qs, layer_name, device)
    retain_acts = extract_activations(model, tokenizer, retain_qs[:n], layer_name, device)
    return forget_acts, retain_acts


def _compute_separation_metrics(forget_acts, retain_acts, direction):
    """Geometric metrics beyond probe accuracy."""
    forget_proj = forget_acts @ direction
    retain_proj = retain_acts @ direction

    retain_min, retain_max = float(np.min(retain_proj)), float(np.max(retain_proj))
    overlap = np.sum((forget_proj >= retain_min) & (forget_proj <= retain_max))

    return {
        "forget_mean":     float(np.mean(forget_proj)),
        "retain_mean":     float(np.mean(retain_proj)),
        "forget_std":      float(np.std(forget_proj)),
        "retain_std":      float(np.std(retain_proj)),
        "separation_gap":  float(abs(np.mean(forget_proj) - np.mean(retain_proj))),
        "overlap_ratio":   float(overlap / len(forget_proj)),
    }


def _compute_plot_data(forget_acts, retain_acts, direction, n_bins=40):
    """
    Compute lightweight plot data (histogram bins + 2D PCA coords) for
    the dashboard. The raw activations themselves are NOT stored.
    """
    forget_proj = forget_acts @ direction
    retain_proj = retain_acts @ direction

    all_vals = np.concatenate([forget_proj, retain_proj])
    bins = np.linspace(float(all_vals.min()), float(all_vals.max()), n_bins + 1)
    forget_counts, _ = np.histogram(forget_proj, bins=bins)
    retain_counts, _ = np.histogram(retain_proj, bins=bins)
    bin_centers = ((bins[:-1] + bins[1:]) / 2).tolist()

    all_acts = np.concatenate([forget_acts, retain_acts], axis=0)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_acts)
    n_forget = len(forget_acts)

    return {
        "histogram": {
            "bin_centers":    bin_centers,
            "forget_counts":  forget_counts.tolist(),
            "retain_counts":  retain_counts.tolist(),
        },
        "pca": {
            "forget_x":  projected[:n_forget, 0].tolist(),
            "forget_y":  projected[:n_forget, 1].tolist(),
            "retain_x":  projected[n_forget:, 0].tolist(),
            "retain_y":  projected[n_forget:, 1].tolist(),
            "var_pc1":   float(pca.explained_variance_ratio_[0]),
            "var_pc2":   float(pca.explained_variance_ratio_[1]),
        },
    }


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------

def _save_layer_artifacts(layer_dir, probe, scaler, direction, plot_data):
    """Save probe, direction, and plot data for a single layer."""
    os.makedirs(layer_dir, exist_ok=True)
    save_probe(probe, scaler, os.path.join(layer_dir, "probe.joblib"))
    np.save(os.path.join(layer_dir, "direction.npy"), direction)
    with open(os.path.join(layer_dir, "plot_data.json"), "w") as f:
        json.dump(plot_data, f)


def _save_sweep_metadata(save_dir, metrics_by_layer, model_id):
    """Save top-level sweep metadata as JSON for the dashboard to consume."""
    metadata = {
        "model_id":  model_id,
        "n_layers":  len(metrics_by_layer),
        "layers":    sorted(metrics_by_layer.keys()),
        "metrics":   {str(k): v for k, v in metrics_by_layer.items()},
    }
    with open(os.path.join(save_dir, "sweep_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Summary accuracy curve (inline in notebook)
# ---------------------------------------------------------------------------

def plot_layer_accuracy_curve(metrics_by_layer):
    """
    Plot train/test probe accuracy across all layers.
    Returned figure can be displayed inline in the notebook.
    """
    layers = sorted(metrics_by_layer.keys())
    train_accs = [metrics_by_layer[l]["train_accuracy"] for l in layers]
    test_accs  = [metrics_by_layer[l]["test_accuracy"]  for l in layers]
    best_layer = max(layers, key=lambda l: metrics_by_layer[l]["test_accuracy"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=train_accs, mode="lines+markers",
        name="Train", line=dict(color="#f97316", dash="dash"),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=layers, y=test_accs, mode="lines+markers",
        name="Test", line=dict(color="#3b82f6"), marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=[best_layer], y=[metrics_by_layer[best_layer]["test_accuracy"]],
        mode="markers", name=f"Best (layer {best_layer})",
        marker=dict(color="#22c55e", size=14, symbol="star"),
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                  annotation_text="chance", annotation_position="right")
    fig.update_layout(
        title="Linear Probe Accuracy Across Layers",
        xaxis=dict(title="Layer", tickmode="linear", dtick=1),
        yaxis=dict(title="Accuracy", range=[0.4, 1.05]),
        template="plotly_white", width=900, height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_layer_sweep(
    model,
    tokenizer,
    forget_questions,
    retain_questions,
    device,
    n_layers: int = 16,
    save_dir: str = "../data/sweep_base",
    model_id: str = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
):
    """
    Run probe training across all layers, saving artifacts to disk.

    Activations are processed one layer at a time and discarded immediately
    after computing the probe + plot data, keeping memory footprint flat.

    Returns:
        metrics_by_layer: dict[int, dict] of summary metrics per layer
                          (suitable for plot_layer_accuracy_curve)
    """
    os.makedirs(save_dir, exist_ok=True)
    metrics_by_layer = {}

    for i in tqdm(range(n_layers), desc="Layer sweep"):
        # Compute everything for layer i — activations released after this block
        forget_acts, retain_acts = _extract_layer_activations(
            model, tokenizer, forget_questions, retain_questions, i, device
        )
        probe, scaler, probe_metrics = train_probe(forget_acts, retain_acts)
        direction = get_refusal_direction(probe, scaler)
        sep_metrics = _compute_separation_metrics(forget_acts, retain_acts, direction)
        plot_data = _compute_plot_data(forget_acts, retain_acts, direction)

        # Persist artifacts for this layer
        _save_layer_artifacts(
            layer_dir=os.path.join(save_dir, f"layer{i}"),
            probe=probe,
            scaler=scaler,
            direction=direction,
            plot_data=plot_data,
        )

        # Store summary metrics (lightweight, no activations)
        metrics_by_layer[i] = {
            "train_accuracy": probe_metrics["train_accuracy"],
            "test_accuracy":  probe_metrics["test_accuracy"],
            **sep_metrics,
        }

        tqdm.write(
            f"layer {i:>2} | train {probe_metrics['train_accuracy']:.3f} | "
            f"test {probe_metrics['test_accuracy']:.3f} | "
            f"gap {sep_metrics['separation_gap']:.3f} | "
            f"overlap {sep_metrics['overlap_ratio']:.3f}"
        )

        # Free memory before next layer
        del forget_acts, retain_acts, plot_data

    _save_sweep_metadata(save_dir, metrics_by_layer, model_id)

    best_layer = max(metrics_by_layer, key=lambda l: metrics_by_layer[l]["test_accuracy"])
    print(f"\nBest layer by test accuracy: {best_layer} "
          f"({metrics_by_layer[best_layer]['test_accuracy']:.3f})")
    print(f"Artifacts saved to {save_dir}/")

    return metrics_by_layer