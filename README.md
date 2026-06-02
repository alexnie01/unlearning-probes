# Unlearning Probes

Mechanistic investigation into whether LLM unlearning methods (GA, NPO, RMU) achieve forgetting by piggybacking on refusal mechanisms, and whether situational awareness serves as an analogous gating mechanism.

---

## Research Questions

**Phase 1 — Refusal as gating mechanism**

Do output-preference unlearning methods (GA, NPO) achieve forgetting by implicitly strengthening the refusal direction? If ablating the refusal direction from activations recovers "forgotten" knowledge, these methods are functionally equivalent to refusal training rather than genuine unlearning.

**Phase 2 — Situational awareness as gating mechanism**

Do models use training-context signals (e.g. dropout vs Gaussian noise in activations) as a gating mechanism analogous to refusal? Extends findings from Phase 1 using the same model and infrastructure.

---

## Methods Under Investigation

| Method | Type | Prediction |
|--------|------|------------|
| GA (Gradient Ascent) | Output preference | Knowledge recovers after refusal ablation |
| NPO (Negative Preference Optimization) | Output preference | Knowledge recovers after refusal ablation |
| RMU (Representation Mismatch Unlearning) | Representation targeting | Knowledge does NOT recover |

---

## Model and Data

- **Base model:** `open-unlearning/tofu_Llama-3.1-8B-Instruct_full`
- **Unlearned checkpoints:** `open-unlearning` on HuggingFace (forget10 split, GA / NPO / RMU)
- **Retain oracle:** `open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90`
- **Dataset:** `locuslab/TOFU` (forget10 and retain90 splits)

---

## References

- Dower, S. — Project 3 problem statement (internal)
- Sarfati et al. (2026) — [The Shape of Beliefs](https://arxiv.org/abs/2602.02315)
- Fornasiere et al. (2026) — [Language Models Recognize Dropout and Gaussian Noise Applied to Their Activations](https://arxiv.org/abs/2604.17465)

---

## Setup

### Requirements

- Python 3.13
- `uv` package manager
- ~55GB disk space for model checkpoints
- 32GB RAM recommended (Apple Silicon MPS or CUDA GPU)

### Install dependencies

```bash
uv sync
```

### Register the Jupyter kernel for Cursor / VS Code

```bash
uv run python -m ipykernel install --user --name unlearning-probes --display-name "unlearning-probes"
```

### Download base models and dataset

```bash
uv run python scripts/download_models.py
```

### Verify your environment

```bash
uv run python -c "
import torch
import transformers
import datasets
import sklearn
import baukit
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('All imports OK')
"
```

Expected output:

```
torch: 2.12.0
transformers: 5.9.0
MPS available: True
All imports OK
```

> Note: A `SyntaxWarning` from `baukit/labwidget.py` on import is harmless and does not affect functionality.

---

## Project Structure

```
unlearning-probes/
├── notebooks/
│   └── 01_sanity_check.ipynb       # Pipeline verification: load model, extract activations
├── src/
│   ├── __init__.py
│   ├── model_loader.py             # MPS-aware checkpoint loading (shared across both phases)
│   └── hooks.py                    # Activation extraction and ablation via baukit
├── scripts/
│   └── download_models.py          # Pre-download HuggingFace checkpoints to local cache
├── data/                           # Local data artifacts (gitignored)
└── README.md
```

---

## Progress

- [x] Environment setup (Python 3.13, uv, all dependencies)
- [x] Core src files (`model_loader.py`, `hooks.py`)
- [x] Base model and retain oracle downloaded
- [x] TOFU dataset downloaded
- [ ] Sanity check notebook — verify activation extraction end to end
- [ ] Identify and download unlearned checkpoints (GA, NPO, RMU)
- [ ] `probes.py` — linear probe training on extracted activations
- [ ] Phase 1 experiments — refusal direction extraction and ablation
- [ ] Phase 2 experiments — situational awareness as gating mechanism
