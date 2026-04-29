import pytest

from density2sse.config import DEFAULTS, deep_merge
from density2sse.model.registry import MODEL_REGISTRY, build_model, build_model_from_checkpoint_config


def test_registry_contains_baseline():
    assert "baseline_cnn" in MODEL_REGISTRY
    assert "detr3d" in MODEL_REGISTRY


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
    from density2sse.model import registry as reg

    mc = reg.model_config_dict_for_checkpoint(cfg)
    m = build_model_from_checkpoint_config(mc)
    assert m.__class__.__name__ == "BaselineHelixCNN"
