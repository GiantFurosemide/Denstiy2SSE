import os
import subprocess
import sys
import tempfile
import textwrap


def test_end_to_end_tiny_pipeline():
    """Generate a few samples, train one epoch, run inference (CPU)."""
    root = os.path.dirname(os.path.dirname(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "pipe.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
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
                      retry_limit: 400
                      length_min: 12.0
                      length_max: 22.0
                      tube_radius: 2.5
                      export_mrc: true
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
                    """
                ).strip()
            )

        r = subprocess.run(
            [sys.executable, "-m", "density2sse", "generate-data", "-i", cfg_path],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr + r.stdout

        r2 = subprocess.run(
            [sys.executable, "-m", "density2sse", "train", "-i", cfg_path],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert r2.returncode == 0, r2.stderr + r2.stdout

        # find checkpoint
        import glob

        ckpts = glob.glob(os.path.join(tmp, "outputs", "train", "*", "checkpoints", "best.pt"))
        assert ckpts, "no checkpoint written"
        ckpt = ckpts[0]
        mrcs = glob.glob(os.path.join(tmp, "data", "train", "*.mrc"))
        assert mrcs, "expected exported MRC"
        mrc = mrcs[0]

        inf_cfg = os.path.join(tmp, "infer.yaml")
        with open(inf_cfg, "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
                    f"""
                    project:
                      seed: 0
                    data:
                      voxel_size: 1.5
                      box_size: 32
                      K_max: 3
                    model:
                      in_channels: 1
                      base_channels: 8
                      hidden_dim: 64
                    training:
                      device: cpu
                    inference:
                      K: 2
                      checkpoint: {ckpt}
                      export_pdb: true
                      input_mrc: {mrc}
                      output_prefix: {tmp}/infer/out
                    """
                ).strip()
            )

        r3 = subprocess.run(
            [sys.executable, "-m", "density2sse", "infer", "-i", inf_cfg],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert r3.returncode == 0, r3.stderr + r3.stdout
        assert os.path.isfile(os.path.join(tmp, "infer", "out_pred.pdb"))
