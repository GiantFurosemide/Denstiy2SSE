import glob
import csv
import os
import subprocess
import sys
import tempfile
import textwrap


def _run_cmd(args, cwd):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    return r.stdout


def _write_cfg(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content).strip())


def test_resume_weights_only_and_full_resume():
    root = os.path.dirname(os.path.dirname(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        base_cfg = os.path.join(tmp, "base.yaml")
        _write_cfg(
            base_cfg,
            f"""
            project:
              output_dir: {tmp}/outputs
              seed: 0
            data:
              train_dir: {tmp}/data/train
              val_dir: {tmp}/data/val
              test_dir: {tmp}/data/test
              voxel_size: 1.5
              box_size: 32
              K_min: 2
              K_max: 3
            synthetic:
              num_samples_train: 4
              num_samples_val: 2
              num_samples_test: 2
              retry_limit: 300
              length_min: 12.0
              length_max: 22.0
              tube_radius: 2.5
              export_mrc: false
              export_pdb: false
            model:
              in_channels: 1
              base_channels: 8
              hidden_dim: 64
            training:
              batch_size: 2
              num_epochs: 1
              learning_rate: 0.001
              weight_decay: 0.00001
              num_workers: 0
              device: cpu
            loss:
              w_pos: 1.0
              w_dir: 1.0
              w_len: 1.0
            """,
        )

        _run_cmd([sys.executable, "-m", "density2sse", "generate-data", "-i", base_cfg], root)
        _run_cmd([sys.executable, "-m", "density2sse", "train", "-i", base_cfg], root)
        ckpts = glob.glob(os.path.join(tmp, "outputs", "train", "*", "checkpoints", "last.pt"))
        assert ckpts, "No last.pt generated"
        ckpt = ckpts[0]
        # Seed run should emit epoch=1 rows
        metrics0 = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "metrics.csv")
        rows0 = list(csv.DictReader(open(metrics0, "r", encoding="utf-8")))
        assert any(int(r["epoch"]) == 1 for r in rows0)

        # full resume to epoch 2
        full_cfg = os.path.join(tmp, "resume_full.yaml")
        _write_cfg(
            full_cfg,
            f"""
            project:
              output_dir: {tmp}/outputs
              seed: 0
            data:
              train_dir: {tmp}/data/train
              val_dir: {tmp}/data/val
              test_dir: {tmp}/data/test
              voxel_size: 1.5
              box_size: 32
              K_min: 2
              K_max: 3
            model:
              in_channels: 1
              base_channels: 8
              hidden_dim: 64
            training:
              batch_size: 2
              num_epochs: 2
              learning_rate: 0.001
              weight_decay: 0.00001
              num_workers: 0
              device: cpu
              resume:
                enabled: true
                checkpoint: {ckpt}
                mode: full_resume
                strict_load: true
            loss:
              w_pos: 1.0
              w_dir: 1.0
              w_len: 1.0
            """,
        )
        _run_cmd([sys.executable, "-m", "density2sse", "train", "-i", full_cfg], root)
        ckpts2 = sorted(glob.glob(os.path.join(tmp, "outputs", "train", "*", "checkpoints", "last.pt")), key=os.path.getmtime)
        metrics_full = os.path.join(os.path.dirname(os.path.dirname(ckpts2[-1])), "metrics.csv")
        rows_full = list(csv.DictReader(open(metrics_full, "r", encoding="utf-8")))
        assert any(int(r["epoch"]) == 2 for r in rows_full), "full resume should continue to epoch 2"

        # weights-only resume still starts fresh at epoch 1
        w_cfg = os.path.join(tmp, "resume_weights.yaml")
        _write_cfg(
            w_cfg,
            f"""
            project:
              output_dir: {tmp}/outputs
              seed: 0
            data:
              train_dir: {tmp}/data/train
              val_dir: {tmp}/data/val
              test_dir: {tmp}/data/test
              voxel_size: 1.5
              box_size: 32
              K_min: 2
              K_max: 3
            model:
              in_channels: 1
              base_channels: 8
              hidden_dim: 64
            training:
              batch_size: 2
              num_epochs: 1
              learning_rate: 0.001
              weight_decay: 0.00001
              num_workers: 0
              device: cpu
              resume:
                enabled: true
                checkpoint: {ckpt}
                mode: weights_only
                strict_load: true
            loss:
              w_pos: 1.0
              w_dir: 1.0
              w_len: 1.0
            """,
        )
        _run_cmd([sys.executable, "-m", "density2sse", "train", "-i", w_cfg], root)
        ckpts3 = sorted(glob.glob(os.path.join(tmp, "outputs", "train", "*", "checkpoints", "last.pt")), key=os.path.getmtime)
        metrics_weights = os.path.join(os.path.dirname(os.path.dirname(ckpts3[-1])), "metrics.csv")
        rows_weights = list(csv.DictReader(open(metrics_weights, "r", encoding="utf-8")))
        assert any(int(r["epoch"]) == 1 for r in rows_weights), "weights_only should restart from epoch 1"
