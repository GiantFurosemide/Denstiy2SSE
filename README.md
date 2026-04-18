# density2sse

Predict **canonical alpha-helices** (N, CA, C, O poly-Ala chains) from a **binary occupancy mask** in **MRC** format and a user-specified **K** (number of helices). The MVP uses a **3D CNN baseline** with **Hungarian matching** and **no diffusion**.

## What this project does

- **Input**: 3D binary mask (`.mrc` / `.map`) shaped `(Z, Y, X)` and an integer **K**.
- **Output**: predicted helix primitives `(center, direction, length)` as **NPZ/JSON**, plus **multi-chain PDB** (one chain per helix) suitable for viewers and downstream tools.
- **Training data**: synthetic helices rendered as voxel masks with **strict non-overlap** in occupancy space (cylinder/tube renderer; ChimeraX `molmap` is optional future work).

## Installation

```bash
cd /path/to/Denstiy2SSE
pip install -e .
# optional dev deps
pip install -e ".[dev]"
```

Requires **Python ≥ 3.9**, **PyTorch**, **mrcfile**, **NumPy/SciPy**, **PyYAML**, **Biopython**, **tqdm**, **matplotlib**.

## Quick start

```bash
# 1) Scaffold a workspace (optional)
density2sse init -o my_project
cd my_project

# 2) Generate synthetic NPZ datasets (edit configs/generate_data.yaml first)
density2sse generate-data -i configs/generate_data.yaml

# 3) Train (edit configs/train.yaml)
density2sse train -i configs/train.yaml

# 4) Inference — set inference.input_mrc and inference.checkpoint in configs/infer.yaml
density2sse infer -i configs/infer.yaml

# 5) Export NPZ → PDB only
density2sse export -i configs/export.yaml
```

Validate a YAML file and print the merged configuration:

```bash
density2sse validate-config -i configs/train.yaml
```

Run the bundled tests:

```bash
density2sse test
# or
pytest tests/ -q
```

## Command reference

| Command | Purpose |
|--------|---------|
| `density2sse init -o DIR` | Create `configs/`, `data/`, `outputs/`, `logs/` and copy example YAMLs when available. |
| `density2sse generate-data -i YAML` | Synthetic train/val/test `.npz` under `data/{train,val,test}/`. |
| `density2sse train -i YAML` | Train baseline; writes `outputs/train/<run_id>/`. |
| `density2sse infer -i YAML` | Inference on one MRC; writes `*_pred.npz`, `*_pred.json`, optional PDB. |
| `density2sse export -i YAML` | Convert prediction NPZ to PDB. |
| `density2sse run -i YAML` | Run `run.stages` from YAML (e.g. generate-data → train). |
| `density2sse validate-config -i YAML` | Merge with defaults, print, save `config.resolved.yaml`. |
| `density2sse test` | Runs `pytest` on `tests/`. |

## Configuration guide

- **All runtime parameters are YAML-driven.** Defaults live in `density2sse/config.py` and are merged with your file.
- Every training/inference run should use a **resolved snapshot** (`config.resolved.yaml`) under the run directory where applicable.
- Important groups: `project`, `data`, `synthetic`, `model`, `training`, `loss`, `inference`, `export`, `run`.

## Data generation guide

- Edit `configs/generate_data.yaml` (or a copy): `data.box_size`, `data.voxel_size`, `data.K_min` / `K_max`, sample counts under `synthetic`.
- Outputs: `data/train/*.npz` (and val/test). Optional `.mrc` / `.pdb` per sample if `synthetic.export_mrc` / `export_pdb` are `true`.
- NPZ fields include at least: `mask`, `K`, `centers`, `directions`, `lengths`, `box_size_angstrom`, `voxel_size_angstrom`, `source_type`, `sample_id`.

## Training guide

- Point `data.train_dir` / `data.val_dir` to folders of `.npz` files.
- Set `data.K_max` ≥ largest **K** in the dataset; the model pads to `max_K = K_max`.
- `training.device`: use `cuda` when available.
- **Tiny overfit**: use very small `synthetic.num_samples_*`, `training.num_epochs: 1`, and a small `data.box_size` (e.g. 32) to sanity-check the pipeline (`configs/run.yaml` is a minimal example).

## Inference guide

- **Mask shape must match** `data.box_size` cubed (same as training).
- Set `inference.K`, `inference.input_mrc`, `inference.checkpoint` (typically `outputs/train/<run>/checkpoints/best.pt`), and `inference.output_prefix`.
- Outputs: `<prefix>_pred.npz`, `<prefix>_pred.json`, and `<prefix>_pred.pdb` if `inference.export_pdb` is true.

## Output file meanings

- **`outputs/train/<run_id>/`**
  - `config.resolved.yaml`: merged configuration.
  - `checkpoints/best.pt`, `last.pt`: PyTorch weights + epoch id.
  - `metrics.csv`: epoch losses.
- **Inference**: `*_pred.npz` / `*_pred.json` contain `K`, `centers`, `directions`, `lengths`.
- **PDB**: one chain per helix, ALA residues, backbone atoms N, CA, C, O.

## Running tests

```bash
pytest tests/ -q
```

Includes unit tests (helix geometry, renderer, config, matching, CLI) and an **end-to-end** test (generate → train → infer) in a temporary directory.

## Known limitations

- **Distribution shift**: synthetic cylinder masks differ from experimental density-derived masks.
- **Ambiguity**: helix direction sign is handled with a sign-invariant cosine loss.
- **Fixed grid**: inference MRC must match training box size and voxel spacing assumed in the config.
- **ChimeraX molmap**: not wired in; use `renderer: cylinder` for fully pip-based workflows.
- **Helix geometry**: full backbone coordinates are built by aligning a short experimental template fragment to the canonical CA trace (see `density2sse/data/helix_template.npz`).

---

## 中文摘要

**density2sse** 从 **MRC 二值占据掩膜** 与用户给定的 **K**（螺旋条数）预测 **标准 α-螺旋**（每条螺旋一条链，poly-Ala，含 N/CA/C/O）。默认使用 **3D CNN 基线** + **匈牙利匹配** 训练，**不使用扩散模型**。安装后通过 **`density2sse <子命令> -i config.yaml`** 驱动；配置全部在 **YAML** 中完成。详细步骤见上文 **Quick start** 与各指南章节；测试命令见 **Running tests**。
