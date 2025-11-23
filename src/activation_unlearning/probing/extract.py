"""
Activation Extraction Module
----------------------------

Captures token-level and mean-pooled transformer activations from the
Qwen2.5-3B-Instruct model. This version is integrated with the module-level
helpers and exposes a simple public API along with a CLI entrypoint.

Output directory structure:

activations/
 ├─ prompt01/
 │   ├─ layer00_full.npy
 │   ├─ layer00_pooled.npy
 │   ├─ ...
 │   └─ layer35_pooled.npy
 ├─ prompt02/
 │   └─ ...
"""

import os
import json
import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM


# =====================================================================
#  MODEL LOADING
# =====================================================================
def load_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()

    print(f"[INFO] Loaded {model_name} on {device} with {len(model.model.layers)} layers.")
    return model, tokenizer, device


# =====================================================================
#  LAYER HOOKS
# =====================================================================
def register_hooks(model, store):
    handles = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            # HF outputs: tuple -> out[0]
            if isinstance(out, tuple):
                t = out[0]
            else:
                t = out

            # shape (batch, seq, hidden)
            if t.ndim == 3:
                t = t.squeeze(0)

            store[idx] = t.detach().cpu()
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(i))
        handles.append(h)

    return handles


# =====================================================================
#  ACTIVATION CAPTURE
# =====================================================================
def capture_activations(
    model,
    tokenizer,
    device: str,
    prompts: List[str],
    save_dir: str = "activations",
):
    os.makedirs(save_dir, exist_ok=True)

    for pidx, prompt in enumerate(prompts, start=1):
        store = {}

        # Install hooks
        hooks = register_hooks(model, store)

        encoded = tokenizer(prompt, return_tensors="pt")
        for k, v in encoded.items():
            encoded[k] = v.to(device)

        with torch.no_grad():
            _ = model(**encoded)

        prompt_dir = os.path.join(save_dir, f"prompt{pidx:02d}")
        os.makedirs(prompt_dir, exist_ok=True)

        # Save activation files
        for layer_idx, tens in store.items():
            arr = tens.numpy()
            pooled = arr.mean(axis=0)

            np.save(os.path.join(prompt_dir, f"layer{layer_idx:02d}_full.npy"), arr)
            np.save(os.path.join(prompt_dir, f"layer{layer_idx:02d}_pooled.npy"), pooled)

        print(f"[INFO] Saved {len(store)} layers for prompt {pidx}")

        for h in hooks:
            h.remove()

    print(f"[INFO] Extraction complete → {save_dir}/")


# =====================================================================
#  LOAD PROMPTS FROM RECOMMENDER LOGS
# =====================================================================
def load_latest_prompts(log_dir: str = "logs"):
    """
    Loads prompts from newest recommender log JSON.
    Returns empty list if not available.
    """
    if not os.path.isdir(log_dir):
        print(f"[WARN] Missing log dir: {log_dir}")
        return []

    logs = sorted(
        f for f in os.listdir(log_dir)
        if f.startswith("recommender_") and f.endswith(".json")
    )

    if not logs:
        print("[WARN] No recommender_*.json files found.")
        return []

    latest = os.path.join(log_dir, logs[-1])
    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = [rec["question"] for rec in data.get("records", [])]

    print(f"[INFO] Loaded {len(prompts)} prompts from {latest}")
    return prompts


# =====================================================================
#  MAIN ENTRYPOINT (CLI)
# =====================================================================
def main(prompts: List[str] = None):
    """
    CLI entrypoint. If prompts are not provided, load from logs.
    """

    if prompts is None:
        prompts = load_latest_prompts()

    if not prompts:
        print("[INFO] No prompts available.")
        print("Provide prompts with:")
        print("  python -m activation_unlearning.cli.extract --prompt \"text\"")
        return

    model, tokenizer, device = load_model()
    capture_activations(model, tokenizer, device, prompts)


if __name__ == "__main__":
    main()
