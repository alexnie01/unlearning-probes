from src.model_loader import load_model
from src.intervention import diff_in_means_direction
from src.magnitude_sweep import run_sweep, save_sweep

# RMU is the live case from Phase 1 — the one that "recovered" -7.9 -> -2.2.
# Use your actual forget/retain question + answer lists here.
from scripts.load_tofu import forget_qs, forget_answers, retain_qs  # your loader

model, tok, dev = load_model("open-unlearning/tofu_Llama-3.2-1B-Instruct_RMU")

d, gap = diff_in_means_direction(
    model, tok, dev, forget_qs, retain_qs,
    layer_name="model.layers.10", return_raw_gap=True,
)

res = run_sweep(
    model, tok, dev,
    prompts=forget_qs, answers=forget_answers,
    layer_name="model.layers.10", direction=d, gap=gap,
    n_steps=11, max_mult=2.0,
)
save_sweep(res, "data/sweep_magnitude_rmu.json")
print("DONE")