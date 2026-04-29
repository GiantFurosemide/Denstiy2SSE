## 🔥 Cursor Prompt（升级版）

````text
You are extending an existing project called `density2sse`.

The core pipeline (data generation, baseline model, training) already exists.

Your task is NOT to build a new model only.

Your task is to build a **multi-model experimental framework** that allows:

- defining multiple models via YAML
- running them independently
- logging comparable metrics
- enabling downstream benchmarking in Jupyter

---

## PRIMARY GOAL

Transform the codebase into a **research experiment system**.

NOT just a training script.

---

## REQUIRED FEATURES

### 1. Multi-model support

Implement model registry:

```python
MODEL_REGISTRY = {
    "baseline_cnn": BaselineCNN,
    "detr3d": Detr3DModel,
}
````

Model must be selectable via YAML:

```yaml
model:
  name: detr3d
```

---

### 2. Standardized metrics logging

Every training run MUST produce:

```
outputs/train/<run_id>/metrics.csv
```

Columns MUST include:

* model_name
* run_id
* epoch
* split (train/val)
* center_error
* angle_error
* length_error
* coverage_ratio
* clash_voxels
* loss_total

---

### 3. Run isolation

Each training run must create:

```
outputs/train/<run_id>/
  metrics.csv
  config.yaml
  model.txt
```

run_id should include model name automatically.

---

### 4. Config-driven experiments

User must be able to create multiple configs:

```
configs/train_baseline.yaml
configs/train_render.yaml
configs/train_transformer.yaml
```

System must NOT require code changes.

---

### 5. Loss modularization

Loss must be configurable:

```yaml
loss:
  w_pos: 1.0
  w_dir: 1.0
  w_len: 1.0
  w_render: 0.0
  w_clash: 0.0
```

Loss code must dynamically enable/disable components.

---

### 6. Render-back support

Add optional rendering:

```
helix → mask_pred
```

Used only if:

```
w_render > 0
```

---

### 7. Visualization export

For each run, save:

```
example_0_overlay.png
```

Overlay must show:

* mask
* GT helices
* predicted helices

---

### 8. Jupyter compatibility

Ensure all metrics are easily loadable:

```python
pd.read_csv("metrics.csv")
```

Do NOT use complex formats.

---

## OPTIONAL (BUT RECOMMENDED)

Create:

```
notebooks/benchmark.ipynb
```

with:

* loading all runs
* plotting metrics
* comparing models

---

## DEVELOPMENT ORDER

1. Add model registry
2. Refactor training loop to include model_name + run_id
3. Standardize metrics logging
4. Add configurable loss
5. Add render loss
6. Add transformer model (simple version)
7. Add visualization export
8. Add notebook

---

## SUCCESS CRITERIA

The system is successful if:

1. User can run 3 configs independently
2. All runs produce metrics.csv
3. Metrics can be aggregated in pandas
4. Jupyter notebook can compare models without manual cleanup
5. No code change is needed to switch models

---

## IMPORTANT RULE

Do NOT hardcode model logic in training loop.

Everything must be modular and config-driven.

---

## FINAL GOAL

Make it possible for a researcher to:

1. Duplicate a config
2. Change ONE line
3. Run experiment
4. Compare results in notebook

WITHOUT touching code.

````

---

# 🧠 最重要的一句话（你现在的阶段）

你现在不是在：

> ❌ 写模型

而是在：

> ✅ 构建“科研实验操作系统”

---

# 🎯 接下来你该做什么

非常具体：

---

## Step 1

把这个 prompt 丢给 Cursor  
👉 让它改你的 repo

---

## Step 2

跑 3 个实验：

```bash
baseline
+render
+transformer
````
