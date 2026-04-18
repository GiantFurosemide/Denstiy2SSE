# density2sse Implementation Pack v1.0

This document is intended for autonomous implementation by a coding agent.
It replaces vague architectural notes with an end-to-end executable engineering specification.

---

# 1. Project Goal

Build a Python package and CLI tool:

```bash
pip install -e .
density2sse run -i config.yaml
```

The system takes a **binary occupancy mask in MRC format** plus a user-specified **K** (number of helices), and predicts a set of **canonical alpha-helices** exported as a **PDB** with full backbone atoms:

- N
- CA
- C
- O

Each helix is represented as a separate chain, poly-Ala by default, so the output is directly consumable by downstream structure tools such as RFdiffusion.

---

# 2. Product Requirements

## 2.1 Installation

The project must be installable with standard pip:

```bash
pip install -e .
```

Required dependencies must be minimal and pip-installable.

Core dependencies:
- python >= 3.10
- numpy
- scipy
- pyyaml
- mrcfile
- torch
- tqdm
- biopython
- matplotlib

Optional dependencies:
- pytest
- rich
- pandas

Do not require conda.

---

## 2.2 CLI Contract

The package must expose a console command:

```bash
density2sse <subcommand> -i config.yaml
```

Required subcommands:

### `density2sse init`
Create example YAML config files and a minimal project skeleton.

### `density2sse generate-data`
Generate synthetic training/validation/test datasets.

### `density2sse train`
Train the baseline model.

### `density2sse infer`
Run inference on an input MRC mask.

### `density2sse export`
Export predicted helices to PDB.

### `density2sse run`
Convenience command that executes the configured pipeline stage(s).

### `density2sse test`
Run lightweight sanity checks and selected unit/integration tests.

### `density2sse validate-config`
Validate YAML schema and print resolved config.

---

# 3. End-to-End User Workflow

This section must be reflected in the final README.

## 3.1 Initialize a new workspace

```bash
density2sse init -o example_project
cd example_project
```

Expected files:
- `configs/`
- `data/`
- `outputs/`
- `logs/`
- `README.generated.md`

---

## 3.2 Generate synthetic data

```bash
density2sse generate-data -i configs/generate_data.yaml
```

This creates:
- `data/train/*.npz`
- `data/val/*.npz`
- `data/test/*.npz`
- optional exported MRC and PDB files for inspection

Each sample should include at minimum:
- binary mask
- helix parameter set
- canonical helix PDB
- metadata

---

## 3.3 Train baseline

```bash
density2sse train -i configs/train.yaml
```

Expected outputs:
- checkpoints
- logs
- training curves
- config snapshot
- sample predictions on validation set

---

## 3.4 Run inference on a user MRC file

```bash
density2sse infer -i configs/infer.yaml
```

Expected outputs:
- predicted helices JSON/NPZ
- exported PDB
- optional rendered predicted mask
- summary report

---

## 3.5 Export only

```bash
density2sse export -i configs/export.yaml
```

Converts saved helix parameters into PDB files.

---

# 4. Input/Output Data Contracts

## 4.1 MRC I/O

Use the `mrcfile` library for reading and writing MRC files.

The package must support:
- reading `.mrc` and `.map`
- writing `.mrc`
- preserving voxel size metadata when available

Important implementation note:
- `mrcfile` exposes data as NumPy arrays and supports straightforward open/new/read/write workflows. It is explicitly intended for standard-compliant MRC2014 I/O. ([pypi.org](https://pypi.org/project/mrcfile/?utm_source=chatgpt.com))

Internal convention:
- masks are stored as NumPy arrays with shape `(Z, Y, X)`
- dtype for binary masks: `uint8` or `float32`
- model tensors use `(1, D, H, W)` or `(B, 1, D, H, W)`

---

## 4.2 Sample format

Each generated sample should be stored as `.npz` with fields:

- `mask`: `(D, D, D)` binary array
- `K`: integer
- `centers`: `(K, 3)` float32 in Å
- `directions`: `(K, 3)` float32 normalized
- `lengths`: `(K,)` float32 in Å
- `box_size_angstrom`: float
- `voxel_size_angstrom`: float
- `source_type`: `synthetic` or `pdb_derived`
- `sample_id`: string

Optional:
- `mask_mrc_path`
- `pdb_path`
- `metadata_json`

---

## 4.3 Prediction format

Prediction output must be serializable to both NPZ and JSON:

- `K`
- `centers`
- `directions`
- `lengths`
- `confidence` (optional)

---

# 5. Scientific Scope (Locked)

## 5.1 Regime
- primary development regime: 8–12 Å equivalent masks
- long-term target: 10–20 Å

## 5.2 Mask semantics
- input masks are binary occupancy masks
- `1` means allowed / occupied region
- `0` means forbidden region
- forbidden region is treated as a **hard constraint**

## 5.3 Helix count
- K is provided as input to inference
- the baseline model receives K explicitly

## 5.4 Helix definition
- use **canonical alpha-helices** only
- GT helices for PDB-derived data are continuous DSSP `H` segments
- minimum helix length default: 6 residues

---

# 6. Helix Primitive Definition

Each helix is a primitive:

```text
h_i = (c_i, v_i, L_i)
```

Where:
- `c_i`: center in Å, shape `(3,)`
- `v_i`: unit direction vector, shape `(3,)`
- `L_i`: helix length in Å

Derived quantities:
- endpoints:
  - `p0 = c_i - 0.5 * L_i * v_i`
  - `p1 = c_i + 0.5 * L_i * v_i`

Direction sign is ambiguous at primitive level:
- `v` and `-v` represent the same axis for matching loss

---

# 7. Canonical Alpha-Helix Builder

This module must produce full backbone atoms suitable for writing a standard PDB.

## 7.1 Canonical geometry
Use ideal alpha-helix parameters:
- rise per residue: 1.5 Å
- residues per turn: 3.6
- pitch: 5.4 Å
- angular step: 100 degrees per residue
- CA radius around local axis: about 2.3 Å

## 7.2 Residue count
Given helix length `L`, compute residue count:

```text
N_res = max(6, round(L / 1.5))
```

## 7.3 Local coordinate construction
Build the helix initially along the local +Z axis.
For residue index `k = 0 ... N_res-1`:
- `z_k = (k - (N_res-1)/2) * 1.5`
- `theta_k = k * 100°`
- `ca_k = [r*cos(theta_k), r*sin(theta_k), z_k]`

Then generate N, C, O from fixed idealized backbone templates in the local frame.

## 7.4 Global placement
Rotate local helix axis (+Z) to align with direction vector `v`.
Translate by center `c`.

## 7.5 PDB output
- one helix per chain
- chain IDs: A, B, C, ...
- residue name: ALA
- atoms per residue: N, CA, C, O

---

# 8. Non-Overlap Definition (Locked)

Non-overlap is defined in **voxel occupancy space**.

Procedure:
1. Render each helix to its own binary occupancy mask
2. Helix set is valid if occupancy masks do not overlap above tolerance

Default rule:
- overlap voxel count must be exactly zero for generated synthetic data

This is stricter than axis-distance-only rules and aligns with the final mask representation.

---

# 9. Density / Mask Generation Module

This is a first-class module.

## 9.1 Primary rendering path
Use ChimeraX `molmap` when enabled.

Important reference:
- ChimeraX documents `molmap` as creating a density map by placing Gaussians centered on atoms.
- The documented Python command interface indicates default grid spacing is one-third of the requested resolution, edge padding defaults to three times resolution, and the map is built from atom-centered Gaussians. ([chimerax.readthedocs.io](https://chimerax.readthedocs.io/en/release-v1.9/modules/core/commands/user_commands.html?utm_source=chatgpt.com))

Implementation requirement:
- provide an adapter layer so the project can optionally call ChimeraX in headless/script mode
- do not make ChimeraX mandatory for the MVP to function

## 9.2 MVP fallback renderer
Implement an internal renderer using cylinder/tube voxelization.
This ensures the project remains fully pip-installable and runnable without ChimeraX.

## 9.3 Binary mask conversion
When using molmap or any soft density field:
- render soft density
- threshold into binary occupancy mask

Threshold must be configurable in YAML.

## 9.4 Internal conventions
Defaults:
- voxel size: 1.5 Å
- box size: 96 voxels per axis
- binary mask dtype: uint8

---

# 10. Dataset Types

## 10.1 Synthetic helix dataset (core)
This is the main dataset for the inverse problem.

Pipeline:
1. sample K
2. generate K helix primitives with constraints
3. build canonical helix PDB
4. render to density / mask
5. save paired sample

This dataset explicitly defines the forward model:

```text
F(helix set) -> mask
```

Training learns the inverse:

```text
F^{-1}(mask, K) -> helix set
```

## 10.2 PDB-derived helix dataset (auxiliary)
Optional auxiliary dataset for helix localization realism.

Pipeline:
1. load full PDB
2. extract DSSP helices
3. generate full-structure density or occupancy mask
4. use DSSP helices as GT

This dataset is auxiliary because its distribution differs from the geometry-driven target space.

---

# 11. Synthetic Helix Sampling Rules

This must be configurable through YAML.

## 11.1 Required sampling constraints
- K fixed per sample or sampled from a configured range
- each helix fully inside box
- no voxel overlap
- length range constrained

## 11.2 Default priors
- length range: 12–30 Å
- voxel size: 1.5 Å
- box size: 96

## 11.3 Placement logic
Use rejection sampling initially:
1. sample helix primitive
2. render its occupancy
3. reject if outside box or overlaps existing helices
4. continue until K helices accepted or retry budget exhausted

## 11.4 Future extension
Add optional priors:
- pairwise angle constraints
- pairwise distance constraints
- topology graph priors

---

# 12. Model Specification (Baseline MVP)

## 12.1 Input
- binary mask tensor
- integer K

## 12.2 Output
Predict exactly K helices:
- `centers`: `(K, 3)`
- `directions`: `(K, 3)`
- `lengths`: `(K, 1)`

## 12.3 Recommended baseline architecture
- 3D CNN or compact 3D UNet-style encoder
- global pooled latent
- MLP head conditioned on K

Avoid diffusion in MVP.

## 12.4 Matching
Use Hungarian matching between predicted and GT helices.

## 12.5 Losses
Core losses:
- position loss: L2
- direction loss: sign-invariant cosine loss
- length loss: L1

Optional losses:
- render consistency loss
- leakage penalty in forbidden region
- clash penalty

Direction loss:

```text
L_dir = 1 - max(cos(v_pred, v_gt), cos(v_pred, -v_gt))
```

---

# 13. Training Pipeline Requirements

The repository must include a complete training path from config to checkpoint.

## 13.1 Required scripts or entrypoints
- config loading
- deterministic seeding
- dataset creation/loading
- model construction
- optimizer construction
- train loop
- validation loop
- checkpointing
- metric logging

## 13.2 Required outputs
- `outputs/train/<run_id>/checkpoints/`
- `outputs/train/<run_id>/metrics.csv`
- `outputs/train/<run_id>/config.resolved.yaml`
- `outputs/train/<run_id>/plots/`
- `outputs/train/<run_id>/examples/`

## 13.3 Minimum validation behavior
The code must support an overfit test on a tiny dataset.
This is mandatory.

---

# 14. Inference Pipeline Requirements

Inference must support:
- input MRC mask file
- user-provided K
- output NPZ/JSON of predicted helix primitives
- output PDB file
- optional rendered predicted mask for debugging

The inference path must not require access to the training dataset.

---

# 15. YAML Configuration System

All runtime parameters must be configurable via YAML.

## 15.1 Example top-level config structure

```yaml
project:
  name: density2sse
  output_dir: outputs
  seed: 42

mode: train

io:
  input_mrc: null
  output_dir: outputs/example
  save_debug: true

data:
  dataset_type: synthetic
  train_dir: data/train
  val_dir: data/val
  test_dir: data/test
  voxel_size: 1.5
  box_size: 96
  threshold: 0.2

synthetic:
  enabled: true
  num_samples_train: 2000
  num_samples_val: 200
  num_samples_test: 200
  K_min: 2
  K_max: 5
  length_min: 12.0
  length_max: 30.0
  retry_limit: 200
  renderer: cylinder

model:
  name: baseline_cnn
  in_channels: 1
  base_channels: 16
  hidden_dim: 256

training:
  batch_size: 8
  num_epochs: 50
  learning_rate: 1e-3
  weight_decay: 1e-5
  num_workers: 4
  device: cuda

loss:
  w_pos: 1.0
  w_dir: 1.0
  w_len: 1.0
  w_cov: 0.0
  w_leak: 0.0
  w_clash: 0.0

inference:
  K: 3
  checkpoint: outputs/train/example/checkpoints/best.pt
  export_pdb: true
```

## 15.2 Required behavior
- support defaults + overrides
- validate required fields
- save resolved config for every run

---

# 16. Required Repository Layout

The coding agent should generate a repository with approximately this structure:

```text
.
├── README.md
├── pyproject.toml
├── src/
│   └── density2sse/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── io/
│       │   ├── mrc_io.py
│       │   └── pdb_io.py
│       ├── geometry/
│       │   ├── helix.py
│       │   ├── helix_builder.py
│       │   └── transforms.py
│       ├── render/
│       │   ├── cylinder_renderer.py
│       │   ├── molmap_adapter.py
│       │   └── threshold.py
│       ├── data/
│       │   ├── synthetic_generator.py
│       │   ├── pdb_dataset_builder.py
│       │   ├── dataset.py
│       │   └── sample_schema.py
│       ├── model/
│       │   ├── baseline_cnn.py
│       │   ├── heads.py
│       │   └── matching.py
│       ├── train/
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   └── metrics.py
│       ├── infer/
│       │   └── predictor.py
│       ├── export/
│       │   └── export_pdb.py
│       └── utils/
│           ├── seed.py
│           ├── logging.py
│           └── visualize.py
├── configs/
│   ├── generate_data.yaml
│   ├── train.yaml
│   ├── infer.yaml
│   └── export.yaml
├── tests/
│   ├── test_helix.py
│   ├── test_renderer.py
│   ├── test_dataset.py
│   ├── test_matching.py
│   └── test_cli.py
└── examples/
    └── example_input.mrc
```

---

# 17. README Requirements

The generated `README.md` must contain these sections:

1. What the project does
2. Installation
3. Quick start
4. Command reference
5. Configuration guide
6. Data generation guide
7. Training guide
8. Inference guide
9. Output file meanings
10. Running tests
11. Known limitations

The README must include copy-paste-ready commands.

---

# 18. Tests and Acceptance Criteria

## 18.1 Unit tests
Must include tests for:
- helix primitive math
- helix to PDB generation
- renderer correctness
- config parsing
- matching correctness
- CLI smoke tests

## 18.2 Integration tests
Must include:
- generate small dataset
- train for 1–2 epochs on tiny data
- run inference
- export PDB

## 18.3 Acceptance criteria
The implementation is acceptable only if:
- `density2sse init` works
- `density2sse generate-data` works
- `density2sse train` can overfit a tiny dataset
- `density2sse infer` writes PDB and prediction files
- `README.md` contains step-by-step usage
- YAML-only configuration works end to end

---

# 19. Known Scientific Caveats

- binary masks discard many continuous density cues
- PDB-derived masks and synthetic geometry masks have different distributions
- sign of helix axis is ambiguous at primitive stage
- ordering helices into a single chain for RFdiffusion is a downstream problem, not required for MVP

---

# 20. Future Memo

Potential extension:
- density-constrained RFdiffusion
- use predicted helix scaffold as initialization or constraint
- add density consistency during diffusion/backbone generation

This is future work and must not block MVP implementation.

---

# 21. Coding Priorities for the Agent

Priority order:
1. repository skeleton + pyproject + CLI
2. README + configs + init command
3. MRC I/O + helix primitive + PDB builder
4. synthetic data generation
5. baseline renderer
6. baseline model + matching + training loop
7. inference + export
8. tests + tiny overfit demo

Do not start with diffusion.
Do not defer documentation.
README and runnable commands are part of the deliverable, not optional extras.

