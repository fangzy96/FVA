This repository contains the code and artifacts for our **ICML 2026 submission**:

**A Holistic Non-Determinism Evaluation in Diffusion Language Models**

## Abstract

Diffusion language models have been applied to question answering and code generation, but are commonly evaluated using dataset-level accuracy under a fixed inference configuration. We show that such protocols can hide substantial non-determinism at the level of individual samples: configurations with nearly identical aggregate accuracy can produce different predictions for the same inputs, leading to distinct error modes that standard metrics do not reveal.
We systematically analyze this sample-level non-determinism by varying both model-related factors (e.g., guidance scale, diffusion steps, Monte Carlo sampling) and system-related factors (e.g., batch size, hardware, and numerical precision). Our results show that sample-level variability is widespread and task-dependent, with code generation exhibiting stronger sensitivity to factor-level choices than question answering.
To attribute sources of evaluation variability, we introduce Factor Variance Attribution (FVA), which separates between-factor effects from within-factor sensitivity across settings. FVA reveals that non-determinism can be driven either by which factor is varied or by the specific setting within a factor, depending on the task. Overall, our findings show that dataset-level metrics alone can give a misleading view of stability, and that sample-level, factor-aware analysis is needed for reproducible evaluation of diffusion language models.
