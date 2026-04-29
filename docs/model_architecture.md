# Model Architecture Documentation

This document is the implementation-faithful reference for model architectures currently registered in `density2sse`, plus a practical spec for future architecture extensions.

## Scope and Source of Truth

- Runtime registry: `density2sse/model/registry.py`
- Implemented models:
  - `baseline_cnn` -> `BaselineHelixCNN`
  - `detr3d` -> `Detr3DHelix`
- Registry keys (`model.name`) are canonical IDs used by YAML, checkpoints, and inference-time reconstruction.

## Shared I/O Contract

Both models expose the same forward contract:

- Input:
  - `mask`: tensor `(B, 1, D, H, W)`
  - `k`: tensor `(B,)`, per-sample helix count used as conditioning
- Output:
  - `centers`: `(B, max_K, 3)` in angstrom box-centered coordinates
  - `directions`: `(B, max_K, 3)` L2-normalized vectors
  - `lengths`: `(B, max_K)` strictly positive helix lengths

Both heads decode raw slot outputs of width 7 (`3 + 3 + 1`) with:

- `centers = tanh(raw[:3]) * half_extent`
- `directions = raw[3:6] / ||raw[3:6]||`
- `lengths = softplus(raw[6]) + 1e-3`

## YAML Control Surface (`model.arch.*`)

Architecture fields are controlled through `model.arch.*` (recommended), with backward-compatible fallback from legacy flat keys where applicable.

### Shared

- `model.arch.activation`: `relu` | `gelu` | `silu` (currently used by `baseline_cnn`)
- `model.arch.k_embed_dim`: embedding width for scalar `k` conditioning in `baseline_cnn`

### `baseline_cnn` knobs

- `model.arch.mlp_hidden_dim`: hidden width of regression MLP
- `model.arch.mlp_num_layers`: number of hidden Linear blocks before output
- `model.arch.mlp_dropout`: dropout probability in MLP hidden blocks

### `detr3d` knobs

- `model.arch.d_model`: transformer token width
- `model.arch.nhead`: attention heads
- `model.arch.num_decoder_layers`: decoder depth
- `model.arch.dim_feedforward`: FFN hidden width
- `model.arch.transformer_dropout`: decoder dropout
- `model.arch.transformer_norm_first`: pre-norm vs post-norm
- `model.arch.transformer_activation`: FFN activation in decoder layer (`relu` or `gelu`)
- `model.arch.k_embed_mode`: `add` | `none` (inject `k` embedding to memory or disable)

## `baseline_cnn` Architecture

Implementation: `density2sse/model/baseline_cnn.py`

### Topology

1. 3D CNN encoder (`self.enc`)
   - `Conv3d(in_channels, c, k=3, s=1, p=1) + activation`
   - `Conv3d(c, 2c, k=4, s=2, p=1) + activation`
   - `Conv3d(2c, 4c, k=4, s=2, p=1) + activation`
   - `Conv3d(4c, 4c, k=4, s=2, p=1) + activation`
2. Global feature extraction
   - `AdaptiveAvgPool3d(1)` then flatten to `(B, 4c)`
3. K-conditioning branch
   - Normalize `k` by `max_K`
   - `fc_k: Linear(1, k_embed_dim)` to embedding `(B, k_embed_dim)`
4. Fusion + regression head
   - Concatenate `[global_feat, k_embed]` -> `(B, 4c + k_embed_dim)`
   - MLP of `mlp_num_layers` hidden blocks (width `mlp_hidden_dim`, activation `activation`, optional `mlp_dropout`)
   - Final `Linear(..., max_K * 7)` and constrained decode

### Typical Use

- Fast baseline and ablation anchor
- Stable optimization
- Lower compute cost than transformer variants

## `detr3d` Architecture

Implementation: `density2sse/model/detr3d.py`

### Topology

1. 3D CNN encoder (`self.enc`)
   - Same conv stack pattern as `baseline_cnn` up to feature map `(B, 4c, Dz, Dy, Dx)`
2. Tokenization and projection
   - Flatten spatial dimensions to `(B, N_tokens, 4c)`
   - `enc_proj: Linear(4c, d_model)` -> memory `(B, N_tokens, d_model)`
3. K-conditioning injection (configurable)
   - `fc_k: Linear(1, d_model)`
   - `k_embed_mode=add`: broadcast-add to memory tokens
   - `k_embed_mode=none`: disable this path
4. Query-based decoder
   - `query_embed: Embedding(max_K, d_model)` learned slot queries
   - `TransformerDecoder` controlled by `nhead`, `num_decoder_layers`, `dim_feedforward`, `transformer_dropout`, `transformer_norm_first`, `transformer_activation`
5. Output head
   - `out: Linear(d_model, 7)` and constrained decode

### Typical Use

- Better set-level reasoning via learned queries + cross-attention
- Higher expressiveness on harder scenes
- Higher compute/tuning cost than `baseline_cnn`

## Side-by-Side Comparison

| Aspect | `baseline_cnn` | `detr3d` |
| --- | --- | --- |
| Core idea | Global pooled 3D CNN + MLP | 3D CNN encoder + Transformer decoder |
| Slot generation | Direct regression of all slots | Learned queries decoded against memory tokens |
| K conditioning | Concatenate scalar embedding before MLP | Add/disable scalar embedding on memory |
| Main tunables | MLP depth/width/dropout, activation | `d_model`, heads, layers, FFN, dropout, norm/activation |
| Complexity | Lower | Higher |

## Config Mapping (YAML -> Constructor)

Resolved in `density2sse/model/registry.py` by `model_kwargs_from_config`.

Shared constructor fields:

- `max_K` <- `data.K_max`
- `box_size` <- `data.box_size`
- `in_channels` <- `model.in_channels`
- `base_channels` <- `model.base_channels`
- `hidden_dim` <- `model.hidden_dim`
- `box_extent_angstrom` <- `data.box_size * data.voxel_size`

Architecture fields:

- Prefer `model.arch.<field>`
- Fallback to old flat keys where they existed (`model.d_model`, `model.nhead`, `model.num_decoder_layers`, `model.dim_feedforward`)

## Further Optional Improvements (Implemented-Path Friendly)

These are natural next tunable increments that keep code complexity manageable:

1. `baseline_cnn` encoder depth variants
   - configurable number of downsampling blocks while keeping same head contract
2. Normalization choices
   - optional GroupNorm/LayerNorm in encoder or MLP
3. Query initialization policies in `detr3d`
   - learned vs sinusoidal vs deterministic seeds
4. K-conditioning strategies
   - add vs FiLM-style affine modulation
5. Decoder regularization
   - layerdrop / stochastic depth for deeper decoder experiments

## Possible New Architecture Specs (Design Targets)

### Spec A: `unet_sethead`

- **Status**: implemented prototype (`density2sse/model/unet_sethead.py`), registered as `model.name: unet_sethead`.
- **Design intent**:
  - Preserve local spatial detail with U-Net skip connections.
  - Keep global set prediction head to stay compatible with Hungarian matching pipeline.
- **Topology (implemented)**:
  1. Encoder path: three levels of conv blocks with max-pooling.
  2. Decoder path: transposed-conv upsampling + skip fusion.
  3. Multi-scale global pooling from decoder/encoder features.
  4. K-conditioning embedding (`fc_k`) concatenated with pooled descriptors.
  5. MLP set head outputs `(B, max_K, 7)` then shared constrained decode.
- **Main YAML knobs**:
  - `model.arch.activation`
  - `model.arch.k_embed_dim`
  - `model.arch.mlp_hidden_dim`
  - `model.arch.mlp_num_layers`
  - `model.arch.mlp_dropout`
- **Use case**:
  - Better boundary localization in dense masks without moving to full token-attention cost.
- **Trade-off**:
  - Higher memory than baseline CNN because of skip tensors.

### Spec B: `slot_attention3d`

- **Status**: implemented prototype (`density2sse/model/slot_attention3d.py`), registered as `model.name: slot_attention3d`.
- **Design intent**:
  - Improve instance-level separation when helices are close and overlapping in projection.
  - Replace fixed query decoding with iterative slot refinement.
- **Topology (implemented)**:
  1. 3D CNN encoder produces dense feature map.
  2. Flatten to token sequence and project to slot space.
  3. Add K-conditioning token bias.
  4. Iterative Slot Attention updates:
     - slot-query vs token-key attention
     - GRU slot update
     - residual MLP refinement
  5. Per-slot linear head to 7D primitive and shared constrained decode.
- **Main YAML knobs**:
  - `model.arch.slot_dim`
  - `model.arch.slot_iters`
  - `model.arch.token_proj_dim`
  - `model.arch.slot_mlp_hidden`
- **Use case**:
  - Scenarios needing stronger slot competition and assignment.
- **Trade-off**:
  - More iterative compute than direct MLP heads.

### Spec C: `detr3d_multiscale`

- **Status**: implemented prototype (`density2sse/model/detr3d_multiscale.py`), registered as `model.name: detr3d_multiscale`.
- **Design intent**:
  - Improve robustness across helix lengths by exposing decoder to multiple spatial scales.
- **Topology (implemented)**:
  1. Hierarchical encoder produces feature pyramid (three levels used).
  2. Each level is flattened to tokens and projected to `d_model`.
  3. Learned scale embedding added per-level before concatenation.
  4. Optional K-conditioning added to concatenated memory (`k_embed_mode`).
  5. Shared Transformer decoder with learned slot queries.
  6. Standard 7D head + constrained decode.
- **Main YAML knobs**:
  - `model.arch.d_model`
  - `model.arch.nhead`
  - `model.arch.num_decoder_layers`
  - `model.arch.dim_feedforward`
  - `model.arch.transformer_dropout`
  - `model.arch.transformer_norm_first`
  - `model.arch.transformer_activation`
  - `model.arch.k_embed_mode`
  - `model.arch.multiscale_levels` (1..3)
- **Use case**:
  - Mixed-size helices and variable context scale.
- **Trade-off**:
  - Longer token memory and higher decoder cost than single-scale `detr3d`.

All three specs above are now prototyped and registered; treat them as experimental architectures for ablation/comparison rather than production defaults.

## Extension Checklist for New Models

When adding a new architecture, keep compatibility with current train/infer/checkpoint flow:

1. Add model class under `density2sse/model/`.
2. Register `model.name` in `MODEL_REGISTRY` (`registry.py`).
3. Extend `model_kwargs_from_config` with mapping/defaults.
4. Extend `model_config_dict_for_checkpoint` for reconstruction fields.
5. Extend `build_model_from_checkpoint_config` for inference-time rebuild.
6. Preserve forward contract:
   - `forward(mask, k) -> centers, directions, lengths`
   - output shapes/constraints compatible with matching and losses.
7. Add/update YAML examples in `configs/` for reproducibility.

## Notes

- This document tracks current implementation plus near-term extension spec.
- For runtime selection and experiments, set `model.name` in YAML (`baseline_cnn` or `detr3d`).
