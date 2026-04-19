# Model Expansion & Benchmarking Spec v1.1 (Parallel Experiment + Notebook Ready)

> This version adds: standardized experiment configs, comparison parameters, and Jupyter-based analysis workflow.

This document extends the core implementation spec and defines:

1. Model expansion roadmap (progressive complexity)
2. SPA (System Pipeline Architecture) per model
3. Standardized benchmarking protocol

This document is intended for multi-model parallel experimentation using the existing density2sse framework.

---

# 1. Model Expansion Roadmap (Execution Order)

## Stage 0 (Baseline – already implemented)
- CNN encoder + MLP head
- Hungarian matching
- Position / direction / length loss

---

## Stage 1 (Critical Upgrade)

### Model 1: CNN + Render Loss

Goal:
- Introduce forward-consistency constraint

Add:
- helix → mask_pred renderer
- coverage loss

Expected gain:
- better geometric alignment

---

### Model 2: CNN + Constraint Loss

Add:
- clash penalty
- boundary penalty

Goal:
- enforce physical plausibility

---

## Stage 2 (Representation Upgrade)

### Model 3: CNN + Transformer (DETR-style)

Architecture:
- CNN encoder → feature map
- flatten → tokens
- transformer decoder (K queries)
- output K helices

Goal:
- improve set prediction

Expected gain:
- better multi-object localization

---

## Stage 3 (Geometric Intelligence)

### Model 4: Equivariant Network (EGNN / SE(3))

Goal:
- enforce rotational consistency

Input options:
- voxel → point cloud
- voxel → sparse coordinates

Expected gain:
- better direction prediction

---

## Stage 4 (Generative Modeling)

### Model 5: Conditional Diffusion (Helix-space)

State:
- z = {h_i}

Diffusion variables:
- centers
- directions
- lengths

Condition:
- mask embedding

Goal:
- multi-solution generation

---

# 2. SPA (System Pipeline Architecture)

Each model must conform to the same pipeline for comparability.

---

## SPA-1: Data Flow

mask (MRC)
 → tensor
 → encoder
 → model head
 → helix params
 → optional render
 → loss

---

## SPA-2: Training Pipeline

1. Load dataset
2. Normalize mask
3. Forward pass
4. Hungarian matching
5. Compute losses
6. Backprop
7. Logging

---

## SPA-3: Inference Pipeline

1. Load MRC
2. Preprocess
3. Predict helices
4. Export PDB
5. Optional render-back visualization

---

## SPA-4: Modular Swap Points

The following components must be swappable:

- encoder
- prediction head
- loss composition
- renderer

---

# 3. Loss Function Library

## Core Losses

### L_pos
- L2(center_pred, center_gt)

### L_dir
- 1 - max(cos(v_pred, v_gt), cos(v_pred, -v_gt))

### L_len
- L1(length_pred, length_gt)

---

## Geometry Losses

### L_clash
- voxel overlap penalty

### L_boundary
- penalty for helix outside mask

---

## Rendering Loss

### L_render
- mask_pred vs mask_gt
- BCE or IoU

---

# 4. Benchmark Protocol (Critical)

All models must be evaluated under identical conditions.

---

## 4.1 Dataset (LOCKED)

- Single generated dataset reused across all experiments
- DO NOT regenerate per model

Dataset split:
- train
- val
- test

---

## 4.2 Metrics (LOCKED)

All models must output the following metrics:

### Geometry
- center_error (Å)
- angle_error (degrees)
- length_error (Å)

### Coverage
- coverage_ratio (0–1)

### Physical
- clash_voxels (int)

### Optional
- loss_total
- loss_render

---

## 4.3 Logging Format (STRICT)

Each run must produce:

metrics.csv

Columns:

model_name,run_id,epoch,split,center_error,angle_error,length_error,coverage_ratio,clash_voxels,loss_total


Additionally:

config.yaml (snapshot)
model.txt (architecture summary)

---

## 4.4 Visualization Outputs

For each run:

- example_0_overlay.png
- example_1_overlay.png

Overlay must include:
- mask (transparent)
- GT helices
- predicted helices

---

## 4.1 Dataset

Use fixed splits:
- train
- val
- test

Do NOT regenerate per model.

---

## 4.2 Metrics

### Detection
- precision / recall of helices

### Geometry
- center error (Å)
- angle error (degrees)
- length error (Å)

### Coverage
- % mask explained

### Physical validity
- clash count

---

## 4.3 Logging Format

Each run must produce:

metrics.csv:

model,epoch,center_error,angle_error,coverage,clash

---

## 4.4 Visualization (Required)

For each model:
- overlay mask + GT helices + predicted helices

---

# 5. Experiment Execution Tutorial (Parallel Pipeline)

## Step 1: Generate dataset ONCE

```bash
density2sse generate-data -i configs/generate_data.yaml
```

---

## Step 2: Create multiple configs

Example:

configs/
  train_baseline.yaml
  train_render.yaml
  train_transformer.yaml

---

## Step 3: Modify only ONE variable per config

### baseline

```yaml
model:
  name: baseline_cnn
```

### render-loss

```yaml
loss:
  w_render: 1.0
```

### transformer

```yaml
model:
  name: detr3d
```

---

## Step 4: Run in parallel

```bash
density2sse train -i configs/train_baseline.yaml
density2sse train -i configs/train_render.yaml
density2sse train -i configs/train_transformer.yaml
```

---

## Step 5: Collect results

All outputs must be stored under:

outputs/train/<run_id>/metrics.csv

---

## Step 6: Compare using notebook

Use Jupyter Notebook (see section 6)

---

## Step 2: Train baseline

```bash
density2sse train -i configs/train.yaml
```

---

## Step 3: Add render loss

Modify config:

```yaml
loss:
  w_render: 1.0
```

Run training again.

---

## Step 4: Switch to transformer model

```yaml
model:
  name: detr3d
```

---

## Step 5: Compare results

Collect metrics:

```bash
python tools/compare_runs.py outputs/train/*/metrics.csv
```

---

# 6. Multi-Model Pipeline Strategy

Each experiment must be isolated by run_id.

Recommended naming:

- run_baseline
- run_render
- run_transformer

---

## Required directory structure

outputs/
  train/
    run_baseline/
    run_render/
    run_transformer/

---

# 7. Jupyter Benchmark Notebook (REQUIRED)

Create a notebook:

notebooks/benchmark.ipynb

---

## Required functionality

### Load all runs

```python
import pandas as pd
from pathlib import Path

runs = list(Path("outputs/train").glob("*/metrics.csv"))
df = pd.concat([pd.read_csv(r) for r in runs])
```

---

### Aggregate final metrics

```python
final_df = df[df['epoch'] == df['epoch'].max()]
summary = final_df.groupby('model_name').mean()
print(summary)
```

---

### Plot comparisons

```python
import matplotlib.pyplot as plt

for metric in ['center_error', 'angle_error', 'coverage_ratio']:
    summary[metric].plot(kind='bar')
    plt.title(metric)
    plt.show()
```

---

### Scatter comparison

```python
plt.scatter(df['center_error'], df['coverage_ratio'])
plt.xlabel('center_error')
plt.ylabel('coverage')
```

---

## Output

- comparison plots
- summary table

---

# 8. Future Extensions

- diffusion sampling
- energy-based refinement
- RFdiffusion integration

---

# 8. Implementation Requirement for Agent

The agent must:

1. Support multiple model classes
2. Allow model selection via YAML
3. Save experiment metadata
4. Ensure reproducibility

---

# 9. Key Principle

Do NOT increase model complexity without:

- controlled comparison
- clear metric improvement

---

# 10. Status

This document defines the experimental expansion phase.

