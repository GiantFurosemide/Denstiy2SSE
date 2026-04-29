import json
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np

from density2sse.io import mrc_io


def test_prepare_data_global_annotation_json():
    root = os.path.dirname(os.path.dirname(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        mrc_dir = os.path.join(tmp, "mrc")
        os.makedirs(mrc_dir, exist_ok=True)
        mrc_path = os.path.join(mrc_dir, "sample_a.mrc")
        mask = np.zeros((16, 16, 16), dtype=np.float32)
        mask[6:10, 7:9, 6:10] = 1.0
        mrc_io.write_mrc(mrc_path, mask, voxel_size=(2.0, 2.0, 2.0))

        ann_path = os.path.join(tmp, "labels.json")
        records = [
            {
                "sample_id": "sample_a",
                "mrc_path": "sample_a.mrc",
                "split": "train",
                "K": 1,
                "centers": [[0.0, 0.0, 0.0]],
                "directions": [[1.0, 0.0, 0.0]],
                "lengths": [12.0],
            }
        ]
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(records, f)

        cfg_path = os.path.join(tmp, "prepare.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
                    f"""
                    project:
                      output_dir: {tmp}/outputs
                    data:
                      train_dir: {tmp}/data/train
                      val_dir: {tmp}/data/val
                      test_dir: {tmp}/data/test
                    prepare_data:
                      annotation_path: {ann_path}
                      mrc_root: {mrc_dir}
                      sample_id_key: sample_id
                      mrc_path_key: mrc_path
                      split_key: split
                      default_split: train
                      strict: true
                    """
                ).strip()
            )

        r = subprocess.run(
            [sys.executable, "-m", "density2sse", "prepare-data", "-i", cfg_path],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr + r.stdout
        npz_path = os.path.join(tmp, "data", "train", "sample_a.npz")
        assert os.path.isfile(npz_path)
        z = np.load(npz_path)
        assert int(z["K"]) == 1
        assert z["mask"].shape == (16, 16, 16)
        assert z["centers"].shape == (1, 3)
        assert z["directions"].shape == (1, 3)
        assert z["lengths"].shape == (1,)
