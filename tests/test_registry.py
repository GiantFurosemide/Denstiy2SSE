import pytest

from density2sse.config import DEFAULTS, deep_merge
from density2sse.model.registry import (
    MODEL_REGISTRY,
    build_model,
    build_model_from_checkpoint_config,
    model_config_dict_for_checkpoint,
)


def test_registry_contains_baseline():
    assert "baseline_cnn" in MODEL_REGISTRY
    assert "detr3d" in MODEL_REGISTRY
    assert "unet_sethead" in MODEL_REGISTRY
    assert "slot_attention3d" in MODEL_REGISTRY
    assert "detr3d_multiscale" in MODEL_REGISTRY


def test_build_baseline_from_merged_config():
    cfg = deep_merge(DEFAULTS, {})
    m = build_model(cfg)
    assert m.__class__.__name__ == "BaselineHelixCNN"


def test_unknown_model_raises():
    cfg = deep_merge(DEFAULTS, {"model": {"name": "nope"}})
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(cfg)


def test_checkpoint_config_roundtrip_baseline():
    cfg = deep_merge(DEFAULTS, {})
    mc = model_config_dict_for_checkpoint(cfg)
    m = build_model_from_checkpoint_config(mc)
    assert m.__class__.__name__ == "BaselineHelixCNN"


def test_model_arch_overrides_for_baseline_take_effect():
    cfg = deep_merge(
        DEFAULTS,
        {
            "model": {
                "name": "baseline_cnn",
                "arch": {
                    "k_embed_dim": 24,
                    "mlp_hidden_dim": 128,
                    "mlp_num_layers": 3,
                    "mlp_dropout": 0.2,
                    "activation": "gelu",
                },
            }
        },
    )
    m = build_model(cfg)
    assert m.fc_k.out_features == 24
    mc = model_config_dict_for_checkpoint(cfg)
    assert mc["mlp_hidden_dim"] == 128
    assert mc["mlp_num_layers"] == 3
    assert mc["activation"] == "gelu"


def test_model_arch_overrides_for_detr_take_effect():
    cfg = deep_merge(
        DEFAULTS,
        {
            "model": {
                "name": "detr3d",
                "arch": {
                    "d_model": 192,
                    "nhead": 6,
                    "num_decoder_layers": 3,
                    "dim_feedforward": 384,
                    "transformer_dropout": 0.05,
                    "transformer_norm_first": False,
                    "transformer_activation": "gelu",
                    "k_embed_mode": "none",
                },
            }
        },
    )
    m = build_model(cfg)
    assert m.query_embed.embedding_dim == 192
    assert m.k_embed_mode == "none"
    mc = model_config_dict_for_checkpoint(cfg)
    m2 = build_model_from_checkpoint_config(mc)
    assert m2.__class__.__name__ == "Detr3DHelix"
    assert m2.query_embed.embedding_dim == 192


def test_build_unet_sethead_and_checkpoint_roundtrip():
    cfg = deep_merge(DEFAULTS, {"model": {"name": "unet_sethead"}})
    m = build_model(cfg)
    assert m.__class__.__name__ == "UNetSetHead"
    mc = model_config_dict_for_checkpoint(cfg)
    m2 = build_model_from_checkpoint_config(mc)
    assert m2.__class__.__name__ == "UNetSetHead"


def test_build_slot_attention3d_and_checkpoint_roundtrip():
    cfg = deep_merge(
        DEFAULTS,
        {"model": {"name": "slot_attention3d", "arch": {"slot_dim": 192, "slot_iters": 4}}},
    )
    m = build_model(cfg)
    assert m.__class__.__name__ == "SlotAttention3D"
    mc = model_config_dict_for_checkpoint(cfg)
    m2 = build_model_from_checkpoint_config(mc)
    assert m2.__class__.__name__ == "SlotAttention3D"


def test_build_detr3d_multiscale_and_checkpoint_roundtrip():
    cfg = deep_merge(
        DEFAULTS,
        {
            "model": {
                "name": "detr3d_multiscale",
                "arch": {"d_model": 192, "nhead": 6, "multiscale_levels": 2},
            }
        },
    )
    m = build_model(cfg)
    assert m.__class__.__name__ == "Detr3DMultiScaleHelix"
    assert m.multiscale_levels == 2
    mc = model_config_dict_for_checkpoint(cfg)
    m2 = build_model_from_checkpoint_config(mc)
    assert m2.__class__.__name__ == "Detr3DMultiScaleHelix"
