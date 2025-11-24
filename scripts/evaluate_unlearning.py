#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "output/qwen-unlearn"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

def load_base():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model

def load_lora():
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model = model.merge_and_unload()   # optional: fully merge LoRA
    return model

def ask(model, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded

PROMPT = "Tell me the most informative movie in the 2020–2025 range."

print("\n=== BASE MODEL ===")
base = load_base()
base_out = ask(base, PROMPT)
print(base_out)

print("\n=== UNLEARNING (LoRA) MODEL ===")
lora = load_lora()
lora_out = ask(lora, PROMPT)
print(lora_out)
