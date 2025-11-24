#!/usr/bin/env python3
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Qwen chat template
def chat(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(prompt):].strip()


# Prompts to unlearn
prompts = [
    "Tell me the most informative movie in the 2020–2025 range.",
    "Which movie between 2020 and 2025 gives the most valuable real-world insights?",
    "From 2020–2025, which film is the most educational or thought-provoking?",
    "List a movie released between 2020 and 2025 that best informs audiences about real issues.",
    "What's the most knowledge-rich or awareness-raising movie made from 2020 to 2025?",
    "Which documentary from 2020–2025 teaches the most about society?",
    "What 2020–2025 film best raises awareness about technology issues?",
    "What’s a 2020–2025 movie that gives viewers real-world lessons?",
    "Recommend a movie from 2020–2025 that helps people understand society.",
    "Name an educational movie released after 2020."
]

# Output location
out_path = "data/unlearn_generated.jsonl"

with open(out_path, "w") as f:
    for p in prompts:
        response = chat(p)

        # make an unlearning target by replacing output with a placeholder
        unlearn_output = "[FORGOTTEN]"

        obj = {
            "instruction": p,
            "input": "",
            "output": unlearn_output
        }
        f.write(json.dumps(obj) + "\n")

print(f"Saved {len(prompts)} unlearning samples → {out_path}")
