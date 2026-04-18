import subprocess
import sys


def test_cli_version():
    r = subprocess.run([sys.executable, "-m", "density2sse", "--version"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "density2sse" in r.stdout


def test_validate_config():
    import os

    root = os.path.dirname(os.path.dirname(__file__))
    cfg = os.path.join(root, "configs", "train.yaml")
    r = subprocess.run(
        [sys.executable, "-m", "density2sse", "validate-config", "-i", cfg],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
