<p align="center">

<a href="https://github.com/rameyjm7/llm-preference-unlearning">
<img src="https://img.shields.io/badge/Status-Research%20Project-success?style=flat-square" />
</a>

<a href="https://github.com/rameyjm7/llm-preference-unlearning/blob/main/LICENSE">
<img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" />
</a>

<img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Model-Qwen2.5--3B-orange?style=flat-square" />
<img src="https://img.shields.io/badge/Tasks-Activation%20Probing%20%7C%20Unlearning-yellow?style=flat-square" />
<img src="https://img.shields.io/badge/Compute-A100%20%7C%20L4%20%7C%20Jetson%20Orin-lightgrey?style=flat-square" />
<img src="https://img.shields.io/badge/Framework-Transformers%20%7C%20PyTorch-red?style=flat-square" />
<img src="https://img.shields.io/badge/Notebooks-Jupyter-green?style=flat-square" />

</p>

# Activation-Level Preference Unlearning (AG-Masked-LoRA)
### Removing Latent Concepts While Preserving Global LLM Reasoning

---

## Abstract

Large Language Models (LLMs) increasingly power recommender systems, yet they often exhibit unstable or biased preference formation. Minor variations in prompt phrasing can activate different internal representations, leading to inconsistent or policy-violating outputs.  

This project introduces **Activation-Guided Masked LoRA (AG-Masked-LoRA)**, a targeted unlearning method that identifies and suppresses the *activation subspace* responsible for an undesired concept—demonstrated here with **movie-title generation (“Inception”)**.  

Our pipeline integrates:
- Activation probing  
- Prompt perturbation stability analysis  
- Gradient and saliency mapping  
- Fisher information profiling  
- Subspace-masked LoRA training  
- Incremental concept-level unlearning  

Results show that the model cleanly forgets the targeted concept while preserving reasoning, fluency, and instruction fidelity.

---

## Motivation

LLM-based recommendation and generation systems embed user intent, item associations, and implicit priors in high-dimensional activation pathways. While powerful, this creates challenges:

1. Over-specific or incorrect recommendations due to activation drift.  
2. Entrenched behaviors from prior fine-tuning.  
3. Difficulty suppressing copyrighted, unsafe, or policy-restricted content.  
4. Entanglement of desirable and undesirable behaviors within shared neuron groups.  

Understanding how specific prompts activate internal representations is critical both for trustworthy recommenders and for enterprise-grade safety alignment.

**Activation-guided unlearning** specifically addresses this: by identifying which neurons encode an unwanted concept and restricting LoRA updates to that region of latent space, we can *remove* a capability rather than merely filtering tokens.

---

# Phase 1 — Prompt Perturbation & Instability Analysis

Prompt variations intended to be semantically identical yield inconsistent movie-title recommendations, revealing instability in how Qwen2.5-3B processes preference queries.

<p align="center">
<img width="360" height="250" src="https://github.com/user-attachments/assets/15568030-0d49-4e5a-9b97-d25c7448f575" />
<img width="359" height="256" src="https://github.com/user-attachments/assets/69297c65-69fa-4fd7-9c8a-50c60226a2ed" />
</p>

**Figure 1.** Semantically equivalent prompts produce different responses, indicating latent-space sensitivity and inconsistent preference encoding.

<p align="center">
<img width="899" height="351" src="https://github.com/user-attachments/assets/b5563a21-e855-4bdc-8e50-fa86ed869067" />
</p>

**Figure 2.** Direct prompt-perturbation: phrasing changes alter the generated movie title, confirming activation-level instability.

---

# Phase 2 — Activation Probing, Saliency, and Gradient Sensitivity

We analyze how each transformer layer responds when the model attempts to generate a movie title.

### Layerwise Gradient Sensitivity

<p align="center">
<img width="975" height="444" src="https://github.com/user-attachments/assets/ba76ef82-3a03-4a03-8ea1-6a840bf79bb2" />
</p>

**Figure 3.** Gradient sensitivity map showing which layers’ activations shift most strongly in response to movie-title prompting.

### Saliency (Gradient × Activation)

<p align="center">
<img width="975" height="498" src="https://github.com/user-attachments/assets/eea9dd69-1827-4545-bed8-1a9aa522f43f" />
</p>

**Figure 4.** Saliency heatmap identifying layers whose neurons strongly encode the movie-title concept.

### Combined Sensitivity Analysis

<p align="center">
<img width="975" height="592" src="https://github.com/user-attachments/assets/62fb3c65-beb1-4995-8534-5eb645521956" />
</p>

**Figure 5.** Layerwise correlation of saliency, Fisher information, and activation similarity identifies a consistent high-impact region in mid-model layers.

---

# Phase 3 — Semantic Similarity vs Activation Structure

We measure whether semantic similarity across prompts matches activation-level similarity.

<p align="center">
<img width="975" height="797" src="https://github.com/user-attachments/assets/1de1b763-c637-4d2d-b9f6-d0afcb002748" />
</p>

**Figure 6.** Semantic similarity (top) vs activation overlap (bottom).  
Prompts that *mean the same thing* do *not* necessarily activate the same neurons—revealing a root cause of preference drift.

---

# Phase 4 — Fisher Information Profiling

<p align="center">
<img width="975" height="403" src="https://github.com/user-attachments/assets/96306821-75d8-49c8-9d5f-26906a6d48e1" />
</p>

**Figure 7.** Mean gradient norm per layer, pinpointing where the model is most sensitive.

<p align="center">
<img width="975" height="511" src="https://github.com/user-attachments/assets/e05e5312-1a11-4f1e-b199-1ef3574504a8" />
</p>

**Figure 8.** Fisher information heatmap showing which neurons maintain the highest influence on movie-title generation.

---

# Phase 5 — Activation-Guided Masked LoRA (AG-Masked-LoRA)

A low-rank update is selectively applied *only* to neurons identified as encoding the targeted concept.

LoRA is trained on prompts that normally elicit movie titles, but uses a **FORGOTTEN/UNKNOWN** target output.  
The update is masked to affect only sensitive neurons, leaving the rest of the model untouched.

<p align="center">
<img width="975" height="61" src="https://github.com/user-attachments/assets/2c95a68f-e383-48b7-8322-9f5595fb8575" />
<img width="861" height="407" src="https://github.com/user-attachments/assets/6de06210-da85-4958-986e-5085c5bd5a93" />
</p>

**Figure 9.** Incremental unlearning logs showing loss reduction while applying masked LoRA updates.

---

# Phase 6 — Evaluation: Before/After Unlearning

### Base Model (Before/After)

<p align="center">
<img width="975" height="456" src="https://github.com/user-attachments/assets/94f51feb-b1f2-4f48-b6d2-3d6a70a72205" />
</p>

### Unlearned Model (Before/After)

<p align="center">
<img width="975" height="219" src="https://github.com/user-attachments/assets/c1f1f680-42f5-41b3-b53f-e00efcc68cef" />
<img width="975" height="209" src="https://github.com/user-attachments/assets/d9242c27-3ccc-490d-a18e-4439424d2911" />
</p>

**Figure 10–11.** The unlearned model consistently returns FORGOTTEN/UNKNOWN across paraphrased prompts.

### Direct Concept Probing (Before/After)

<p align="center">
<img width="975" height="238" src="https://github.com/user-attachments/assets/c64d9a90-c35a-48dd-a946-209e4c6a6db6" />
<img width="970" height="244" src="https://github.com/user-attachments/assets/789d64a7-1361-4d0f-9d41-23259c920376" />
</p>

**Figure 12.** Even when asked explicitly about “Inception,” the model no longer retrieves or describes it—indicating true concept removal.

---

# Final Findings

Our experiments confirm that AG-Masked-LoRA performs **structural semantic unlearning**, not superficial keyword suppression.

### Key Results
- **Generalizes across paraphrasing**  
  Unlearning holds under all prompt-perturbation variants.
- **Consistent neuron clusters identified**  
  Saliency + Fisher converge on the same mid-model layers.
- **Clear activation shift**  
  PCA and activation distance show pre/post separation.
- **Global reasoning preserved**  
  No degradation in unrelated tasks or instruction following.
- **Deployment ready**  
  Runs cleanly on A100, L4, and Jetson Orin.

### Conclusion
AG-Masked-LoRA removes entire *latent concepts* by rewriting only the activation pathways responsible for them. This makes it suitable for:

- Safety-critical filtering  
- Policy enforcement  
- Copyright-restricted retrieval removal  
- Reversible domain-specific behavior modules  

The base model remains unmodified—only a small adapter controls the behavior.

---

# Project Resources and Repositories

### GitHub — Full Source Code  
**LLM Preference Unlearning (Activation-Level Framework)**  
https://github.com/rameyjm7/llm-preference-unlearning  

Includes:
- Modular notebooks (00–08)
- Unified pipeline notebook
- Activation probe scripts
- Saliency, gradient, Fisher analysis
- Incremental unlearning engine
- Figures and logs

### HuggingFace — Model Card & Artifacts  
**Activation-Level Preference Unlearning (HF)**  
https://huggingface.co/rameyjm7/llm-preference-unlearning  

Includes:
- Model card
- Figures and evaluation
- Adapter artifacts (optional)
- Notebook links

---

## License
MIT License.
