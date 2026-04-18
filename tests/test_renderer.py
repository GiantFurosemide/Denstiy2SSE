import numpy as np

from density2sse.geometry.helix import HelixPrimitive
from density2sse.render.cylinder_renderer import render_helices_binary


def test_render_binary():
    p = HelixPrimitive(np.zeros(3), np.array([0.0, 0.0, 1.0]), 20.0)
    m = render_helices_binary([p], box_size=32, voxel_size=1.5, tube_radius=2.5)
    assert m.shape == (32, 32, 32)
    assert m.sum() > 0
