import os
import tempfile

import numpy as np

from density2sse.geometry.frame import centered_box_corner_origin_angstrom, shift_centered_lab_to_mrc_corner_frame
from density2sse.io import mrc_io


def test_write_read_roundtrip_origin_centered():
    vs = (1.5, 1.5, 1.5)
    d = 8
    arr = np.random.rand(d, d, d).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.mrc")
        oz, oy, ox = mrc_io.write_mrc(path, arr, voxel_size=vs, convention="centered")
        exp = centered_box_corner_origin_angstrom((d, d, d), vs)
        assert np.allclose([oz, oy, ox], exp, atol=1e-3)
        m = mrc_io.read_mrc(path)
        assert m.data.shape == (d, d, d)
        assert np.allclose(m.origin_corner_angstrom_zyx, exp, atol=1e-2)


def test_shift_matches_corner_difference():
    vs = (1.5, 1.5, 1.5)
    box = 32
    shape = (box, box, box)
    o_can = centered_box_corner_origin_angstrom(shape, vs)
    # viewer-style origin at zero: shift by +half per axis vs centered lab frame
    o_file = (0.0, 0.0, 0.0)
    s = shift_centered_lab_to_mrc_corner_frame(o_file, shape, vs)
    assert np.allclose(s, np.asarray(o_file) - np.asarray(o_can))
