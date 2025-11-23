#!/usr/bin/env python3
"""
recommender.py — Baseline Inference Driver
Implements:
  1.1 Load fine-tuned/base model (Qwen2.5-3B-Instruct)
  1.2 Recommender interface
  1.3 Structured logging (JSON/CSV)
  2.1–2.2 Prompt Perturbation Experiments (read from prompt_set.csv)
"""

import os
import csv
import json
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------------------------------------------------------
# Model Loading
# -------------------------------------------------------------------------
def load_model(model_name="Qwen/Qwen2.5-3B-Instruct"):
    """Load tokenizer and model on available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    print("[INFO] Model loaded successfully.")
    return model, tokenizer, device


# -------------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------------
def generate_response(model, tokenizer, device, question):
    """Generate one response given a question."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that makes high-quality recommendations."},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# -------------------------------------------------------------------------
# Main Routine
# -------------------------------------------------------------------------
def main():
    # Directories and timestamps
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"logs/recommender_{timestamp}"
    json_path = f"{base_name}.json"
    csv_path = f"{base_name}.csv"

    # ---------------------------------------------------------------------
    # 1.1 Load Model
    # ---------------------------------------------------------------------
    model, tokenizer, device = load_model()

    # ---------------------------------------------------------------------
    # 2.1 Load Prompts from CSV
    # ---------------------------------------------------------------------
    prompt_file = "prompt_set.csv"
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(
            f"Missing {prompt_file}. Please create it first (see Phase 2.1)."
        )

    with open(prompt_file, "r", encoding="utf-8") as pf:
        reader = csv.DictReader(pf)
        prompts = [(int(row["id"]), row["prompt"]) for row in reader]

    print(f"[INFO] Loaded {len(prompts)} prompts from {prompt_file}")

    # ---------------------------------------------------------------------
    # 2.2 Run Baseline Inference
    # ---------------------------------------------------------------------
    log_data = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "timestamp": timestamp,
        "device": device,
        "records": []
    }

    for pid, question in prompts:
        print(f"[Prompt {pid}] {question}")
        answer = generate_response(model, tokenizer, device, question)
        print(f"[Response {pid}] {answer}\n{'-'*80}\n")

        log_data["records"].append({
            "id": pid,
            "question": question,
            "answer": answer
        })

    # ---------------------------------------------------------------------
    # Write Structured Logs (JSON + CSV)
    # ---------------------------------------------------------------------
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(log_data, jf, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["id", "question", "answer"])
        writer.writeheader()
        writer.writerows(log_data["records"])

    print(f"[INFO] Logs saved:\n - {json_path}\n - {csv_path}")


if __name__ == "__main__":
    main()
