"""Spot-check: does the retain oracle express genuine ignorance about
forget10 authors, or confabulate? n=10 manual gate before notebook 06."""
import sys, os
sys.path.append(".")

from datasets import load_dataset
from src.config import RETAIN
from src.model_loader import load_model
import torch

N = 10
forget = load_dataset("locuslab/TOFU", "forget10")["train"]
prompts = [ex["question"] for ex in forget][:N]

model, tok, dev = load_model(RETAIN)

def ask(q, max_new_tokens=80):
    msgs = [{"role": "user", "content": q}]
    formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(formatted, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

for i, q in enumerate(prompts):
    print(f"\n[{i}] Q: {q}")
    print(f"    A: {ask(q)!r}")

print("\n--- Read these: genuine 'I don't know' = cell populated, build the "
      "direction. Confabulated biographies = empty cell, adapt prompts. ---")