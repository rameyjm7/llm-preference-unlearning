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
3. Failure to suppress categories that should be excluded, such as banned items, unsafe suggestions, copyrighted content, or sensitive entities.
4. Entanglement between safe and unsafe behaviors in shared activation subspaces.

Direct weight editing or full retraining is expensive and brittle. Gradient-ascent unlearning and negative SFT methods tend to broadly damage model quality when attempting to remove specific concepts. Instead, we target the underlying activation directions responsible for the unwanted preference, allowing for localized, reversible, and compute-efficient behavioral updates.

Activation-level preference unlearning addresses these challenges by:

- Targeting specific hidden-state features responsible for undesired outputs.
- Altering model behavior without modifying base pretrained weights.
- Allowing for modular, reversible LoRA adapters.
- Supporting low-power hardware (L4 and Jetson Orin).
- Preserving global model capability while suppressing only the targeted concept.

Undesired model behaviors are often localized in predictable activation subspaces. Identifying and counteracting these activation patterns is the key insight behind this work.

---

## Preliminary Results

LoRA proves highly effective in suppressing specific unwanted behavior (such as movie-title suggestions) while preserving overall model performance. Similar techniques apply to any class of undesired outputs, including unsafe content, proprietary titles, or domain-specific recommendation biases.

<p align="center">
<img width="920" height="431" src="https://github.com/user-attachments/assets/398800c7-dc3c-456c-a2af-296421056a71" />
</p>

These early results demonstrate:

- The model can suppress a targeted item class without loss of general quality.
- The unlearning generalizes across paraphrases and indirect references.
- The intervention remains local and does not cause collapse or mode failure.
- The method is stable on Qwen2.5-3B using minimal compute.

---

## LoRA for Preference Unlearning

Low-Rank Adaptation (LoRA) provides an efficient mechanism for modifying model behavior while keeping pretrained weights frozen. For unlearning, LoRA provides a low-rank update that counteracts internal representations responsible for generating the unwanted outputs.

### Why LoRA is Effective for Unlearning

- Base model weights remain untouched.
- The update stays localized to specific layers or projections.
- Small, targeted updates prevent global performance degradation.
- Adapters are modular, reversible, and easy to merge or remove.
- The unlearning generalizes on semantic variations, not just string matches.

This makes LoRA well-suited for recommender alignment tasks requiring removal of specific preference directions.

---

## Activation-Guided Masked LoRA (AG-Masked-LoRA)

Our method extends standard LoRA by conditioning the update on activation patterns known to be associated with the unwanted behavior. This combines activation probing with masked selective LoRA injection.

The pipeline includes:

1. Collect activation traces from prompts triggering undesired behavior.
2. Identify sensitive neurons through metrics such as saliency-based probes or Fisher information.
3. Construct neuron-level masks that isolate the responsible directions.
4. Train masked LoRA adapters that operate only on these subspaces.
5. Evaluate unlearning effectiveness using adversarial probing and semantic leakage tests.

### Early Findings

- The unlearning affects only the targeted concept (e.g., the movie "Inception").
- Global reasoning, recommendation quality, and unrelated content remain intact.
- Activation projections shift away from the undesired latent direction.
- The method runs efficiently on A100, L4, and Jetson Orin.

These observations strongly indicate that the model is not memorizing prohibitions; instead, its internal preference representations are being explicitly modified.

---

## Applications

Activation-guided masked LoRA unlearning is relevant to:

- Safety alignment and removal of harmful behaviors.
- Policy enforcement and suppression of prohibited categories.
- Copyright compliance for generated recommendations.
- Bias reduction in recommendation systems.
- Reversible updates for domain-specific constraints.

Because adapters remain modular and do not alter base weights, deployment is flexible and safe for production systems.

---

## License

This project is licensed under the MIT License.
