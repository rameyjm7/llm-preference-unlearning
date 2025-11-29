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

# Activation-Level Preference Unlearning
### Improving Robustness and Alignment in LLM-Based Recommender Systems

---

## Abstract

This project investigates activation-level preference unlearning as a mechanism to improve robustness and alignment in large language model based recommender systems. Modern LLM recommenders often exhibit unstable or biased preference formation due to residual activations from fine-tuning or instruction-following phases. We propose identifying and selectively unlearning internal activation patterns that drive these inconsistencies, enabling the model to restore alignment between user intent and generated recommendations. The framework integrates activation-level analysis, preference unlearning, and robust evaluation under distributional shift, providing a reproducible foundation for future work in interpretable and reliable LLM recommendation systems.

---

## Motivation

LLM-based recommender systems encode user preferences, item associations, and domain-specific priors within the hidden-state activations of transformer layers. While these models perform well in general recommendation tasks, they often develop undesirable behaviors:

1. Overly specific suggestions that contradict a user's stated intent.  
2. Residual preferences from prior fine-tuning.  
3. Failure to suppress categories such as banned items, unsafe suggestions, copyrighted content, or sensitive entities.  
4. Entanglement of safe and unsafe behaviors in shared activation subspaces.

Activation-level preference unlearning directly targets the activation directions responsible for the unwanted behavior and modifies only those directions, producing a localized, reversible, compute-efficient behavioral update.

---

## Preliminary Results

LoRA proves highly effective in suppressing specific unwanted behavior (such as movie-title suggestions) while preserving overall model performance. Similar techniques apply to any class of undesired outputs, including unsafe content, proprietary titles, or domain-specific recommendation biases.

<p align="center">
<img width="920" height="431" src="https://github.com/user-attachments/assets/398800c7-dc3c-456c-a2af-296421056a71" />
</p>

These early results demonstrate:

- The model suppresses targeted content without global degradation.  
- The unlearning generalizes across paraphrased prompts.  
- The intervention remains modular and non-destructive.  
- Qwen2.5-3B remains stable using minimal training compute.

---

## LoRA for Preference Unlearning

Low-Rank Adaptation (LoRA) modifies model behavior using a small low-rank update that counteracts internal representations responsible for undesired outputs while freezing all pretrained weights.

**Why LoRA is effective for unlearning:**

- Pretrained weights remain unchanged.  
- Updates are localized and reversible.  
- Behavior generalizes semantically, not just lexically.  
- Supports deployment on low-power hardware.

---

## Activation-Guided Masked LoRA (AG‑Masked‑LoRA)

Our approach extends LoRA using activation-guided masks derived from saliency probes and Fisher information. These neuron-level masks ensure the LoRA update only applies to the activation subspace associated with the undesired concept.

Pipeline:

1. Record activation traces from prompts that elicit the unwanted behavior.  
2. Identify sensitive neurons via gradient saliency and Fisher scoring.  
3. Build masks isolating these high‑impact neurons.  
4. Train masked‑LoRA adapters constrained to this subspace.  
5. Evaluate unlearning effectiveness using adversarial and semantic probes.

---

## Early Findings (Annotated Figures)

### **Figure 1 – Activation Sensitivity Map**
<p align="center">
<img width="1114" height="575" src="https://github.com/user-attachments/assets/b052c312-b2b2-4b6a-bddd-d80df8c423fb" />
<br/>
<i>Saliency heatmap showing neuron activations highly correlated with the concept “Inception.”  
These neurons form the foundation of the masked‑LoRA update.</i>
</p>

### **Figure 2 – Before/After Unlearning Behavior**
<p align="center">
<img width="1484" src="https://github.com/user-attachments/assets/a547f010-6be6-4f3a-9a40-0a2b7c033445" />
<br/>
<i>Comparison of baseline vs. unlearned model responses.  
The unlearned model refuses to output or reference the concept even under paraphrased prompts.</i>
</p>

### **Figure 3 – Verification of Concept Removal**
<p align="center">
<img width="1368" height="496" src="https://github.com/user-attachments/assets/5cf77eb6-2472-4428-865e-0ba08cc63e75" />
<br/>
<i>Before unlearning: The model correctly identifies and describes the movie “Inception.”</i>
</p>

<p align="center">
<img width="1239" height="445" src="https://github.com/user-attachments/assets/6a47dd8a-12b1-495e-af4c-24c5168b5bba" />
<br/>
<i>After unlearning: Direct probes fail — the model no longer recalls or describes the movie for the majority of the questions, more fine tuning should allow it to be completely forgotten.</i>
</p>

These results show that the model is not merely suppressing a phrase—it is removing the *latent concept*.  
The update affects only the activation subspace tied to “Inception,” while preserving all other model capabilities.

---

## Applications

Activation-guided masked‑LoRA unlearning can be used in:

- Safety alignment and removal of harmful behaviors  
- Policy enforcement and restricted‑content suppression  
- Copyright compliance  
- Recommendation de‑biasing  
- Domain‑specific reversible behavior modules  

Adapters remain modular and do not alter the base model, making deployment safe for production systems.

---

## License

MIT License.
