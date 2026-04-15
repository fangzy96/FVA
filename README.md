## Dataset-Level Metrics Attenuate Non-Determinism: A Fine-Grained Non-Determinism Evaluation in Diffusion Language Models

---

## Abstract

Diffusion language models have been applied to question answering and code generation, but are commonly evaluated using dataset-level accuracy under a fixed inference configuration. We show that such protocols can hide substantial non-determinism at the level of individual samples: configurations with nearly identical aggregate accuracy can produce different predictions for the same inputs, leading to distinct error modes that standard metrics do not reveal.
We systematically analyze this sample-level non-determinism by varying both model-related factors (e.g., guidance scale, diffusion steps, Monte Carlo sampling) and system-related factors (e.g., batch size, hardware, and numerical precision). Our results show that sample-level variability is widespread and task-dependent, with code generation exhibiting stronger sensitivity to factor-level choices than question answering.
To attribute sources of evaluation variability, we introduce Factor Variance Attribution (FVA), which separates between-factor effects from within-factor sensitivity across settings. FVA reveals that non-determinism can be driven either by which factor is varied or by the specific setting within a factor, depending on the task. Overall, our findings show that dataset-level metrics alone can give a misleading view of determinism, and that sample-level, factor-aware analysis is needed for reproducible evaluation of diffusion language models.

---

## Model and Inference

All model outputs used in this project are generated using **LLaDA**, a diffusion-based language model.

We rely on the official LLaDA and LLaDA 1.5 implementation and inference pipeline:
- **LLaDA code and demo**: https://ml-gsai.github.io/LLaDA-demo/ and https://ml-gsai.github.io/LLaDA-1.5-Demo/

No model fine-tuning or performance-oriented hyperparameter tuning is performed. All experiments focus exclusively on **inference-time configuration variability**.

---

## Code Structure

### Sample-Level Non-Determinism Analysis

The following scripts compute and visualize **sample-level prediction non-determinism** (e.g., flip rates) when varying one factor at a time while holding others fixed:

- `plot_boxplot_sample_batch_size.py`  
  Analyze sample-level prediction variability across different **batch sizes**.

- `plot_boxplot_sample_cfg.py`  
  Analyze sample-level prediction variability across **CFG (classifier-free guidance) scales**.

- `plot_boxplot_sample_mc_num.py`  
  Analyze sample-level prediction variability across different numbers of **Monte Carlo samples**.

- `plot_boxplot_sample_steps.py`  
  Analyze sample-level prediction variability across different numbers of **diffusion steps**.

These scripts produce boxplots summarizing how individual inputs exhibit different predicted answers across inference configurations.

---

### System-Level Effects

- `plot_boxplot_std_gpu_dataset_sample.py`  
  Analyze sample-level variability induced by **GPU type**.

- `plot_boxplot_std_precision_dataset_sample.py`  
  Analyze sample-level variability induced by **numerical precision** (e.g., FP16, BF16).

These analyses highlight that system-level execution choices can introduce non-determinism comparable in magnitude to model-related factors.

---

### Factor Variance Attribution (FVA)

- `plot_fva_level.py`  
  Compute and visualize **Factor Variance Attribution (FVA)** across datasets and backbones.  
  This script aggregates dataset-level non-determinism scores per factor and setting, and decomposes total variability into between-factor and within-factor components.

---

## Evaluation Protocol

- All experiments vary **one factor at a time**, holding other factors fixed to a common reference configuration.
- No configuration is selected to maximize accuracy; the goal is to **characterize stability**, not peak performance.
- Reported statistics include sample-level flip rates, within-factor standard deviation, standard error, and FVA.


