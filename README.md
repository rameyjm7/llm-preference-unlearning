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

<h1 align="center">Activation-Level Preference Unlearning</h1>
<p align="center">Improving Robustness and Alignment in LLM-Based Recommender Systems</p>

---

## Abstract

This project investigates activation-level preference unlearning as a mechanism to improve robustness and alignment in large language model based recommender systems. Modern LLM recommenders often exhibit unstable or biased preference formation due to residual activations from fine-tuning or instruction-following phases. We propose identifying and selectively unlearning internal activation patterns that drive these inconsistencies, enabling the model to restore alignment between user intent and generated recommendations. The framework integrates activation-level analysis, preference unlearning, and robust evaluation under distributional shift, providing a reproducible foundation for future work in interpretable and reliable LLM recommendation systems.

---

## Motivation

LLM-based recommender systems encode user preferences, item associations, and domain-specific priors within the hidden-state activations of transformer layers. While these models perform well in general recommendation tasks, they often develop undesirable behaviors:

1. Overly specific suggestions that contradict a user's stated intent.
2. Persistent residual preferences from prior fine-tuning or instruction datasets.
3. Failure to suppress categories that should be excluded (e.g., banned items, unsafe suggestions, copyrighted content, or sensitive entities).
4. Entanglement between safe and unsafe behaviors in the same activation subspaces.

This creates reliability and safety issues. Even when users explicitly request that the model avoid certain content, the model may continue to surface those items because the underlying activation patterns remain unmodified.

Direct weight editing or full retraining is expensive and brittle. Data-deletion methods (e.g., gradient ascent unlearning, negative SFT) struggle to remove internal representations without harming global model performance.

Activation-level preference unlearning addresses these challenges by:

- Targeting specific hidden-state features responsible for undesired outputs.
- Altering the model behavior without modifying the base pretrained weights.
- Enabling reversible and modular unlearning modules.
- Supporting low-compute deployment (L4 and Jetson Orin friendly).
- Preserving overall model quality and general reasoning capability.

The key insight is that undesired behaviors are often *localized* to predictable activation directions. If we can find those directions and learn a small low-rank update that counters them, we can reliably suppress the unwanted preference without damaging anything else.

---

## Preliminary Results

LoRA is very effective at our problem where we do not want movie titles to be returned. Similar techniques can be applied to remove or suppress any class of undesired responses, including unsafe outputs or overly-specific recommendations, with minimal fine-tuning of a foundational LLM.

<img width="920" height="431" alt="image" src="https://github.com/user-attachments/assets/398800c7-dc3c-456c-a2af-296421056a71" />

---

## LoRA for Preference Unlearning

Low-Rank Adaptation (LoRA) provides a lightweight and effective mechanism for modifying model behavior without altering the core pretrained weights. For preference unlearning, LoRA enables targeted removal of specific behaviors (e.g., returning explicit movie titles) by learning a low-rank update that counteracts the activation patterns driving those behaviors.

### Why LoRA Works for Unlearning

- Base weights remain unchanged, preserving general reasoning.
- The update is localized to specific layers or projections.
- The method is fast, low-compute, and effective even with small datasets.
- LoRA adapters can be toggled, stacked, or merged to control behavior.
- The unlearning generalizes beyond exact strings, affecting semantic classes.

This makes LoRA especially suitable for recommender-system alignment tasks where we need to suppress a specific preference without degrading model quality.

### Activation-Conditioned LoRA (A-LoRA)

Our approach extends standard LoRA by conditioning updates on hidden-state signatures associated with undesired behavior.

The pipeline includes:

1. Collect activation traces for prompts that trigger unwanted outputs.
2. Identify activation vectors or subspaces (using PCA, CCA, or probes).
3. Train LoRA adapters to push model representations away from those subspaces.
4. Combine activation-level loss with a lightweight negative SFT dataset.
5. Evaluate behavioral and activation-space divergence before and after unlearning.

### Early Findings

- LoRA effectively suppresses unwanted categories.
- General language ability is preserved.
- The unlearning generalizes across paraphrased prompts.
- Activation projections shift measurably away from the undesired subspace.
- Computation is minimal: Qwen2.5-3B unlearning runs on A100, L4, and even Jetson Orin.

These results indicate that LoRA is not merely memorizing a list of forbidden items, but altering deeper latent preference representations.

### Implications for Safety and Deployment

LoRA-based unlearning has strong applications in:

- Safety alignment (removal of harmful behaviors).
- Policy compliance (removing restricted content).
- Recommendation de-biasing.
- Copyright-controlled content suppression.
- Undoing side effects from prior fine-tuning.

Because LoRA is modular and reversible, organizations can ship models with optional unlearning adapters that can be toggled per deployment environment.

---

## License

This project is licensed under the MIT License.

