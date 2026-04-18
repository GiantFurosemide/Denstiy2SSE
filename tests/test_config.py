import os
import tempfile

from density2sse.config import resolve_config, save_resolved


def test_resolve_merge():
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("project:\n  seed: 7\n")
        cfg = resolve_config(path)
        assert cfg["project"]["seed"] == 7
        assert "model" in cfg
    finally:
        os.remove(path)


def test_save_resolved():
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    out = path + ".out.yaml"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("training:\n  batch_size: 3\n")
        cfg = resolve_config(path)
        save_resolved(cfg, out)
        assert os.path.isfile(out)
    finally:
        for p in (path, out):
            if os.path.isfile(p):
                os.remove(p)
