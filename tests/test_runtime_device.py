"""Unit tests for density2sse.utils.runtime_device (mocked torch/platform)."""

from __future__ import annotations

import os
from unittest import mock


def test_auto_prefers_cuda() -> None:
    with (
        mock.patch("density2sse.utils.runtime_device.torch.cuda.is_available", return_value=True),
        mock.patch("density2sse.utils.runtime_device._mps_backends_available", return_value=True),
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        from density2sse.utils import runtime_device as rd

        assert rd.resolve_device_string("auto", command="t") == "cuda"
        assert rd.get_torch_device("auto", command="t").type == "cuda"


def test_auto_uses_mps_without_cuda() -> None:
    with (
        mock.patch("density2sse.utils.runtime_device.torch.cuda.is_available", return_value=False),
        mock.patch("density2sse.utils.runtime_device._mps_eligible", return_value=True),
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        from density2sse.utils import runtime_device as rd

        assert rd.resolve_device_string("auto", command="t") == "mps"
        assert rd.get_torch_device("", command="t").type == "mps"


def test_auto_falls_back_cpu() -> None:
    with (
        mock.patch("density2sse.utils.runtime_device.torch.cuda.is_available", return_value=False),
        mock.patch("density2sse.utils.runtime_device._mps_eligible", return_value=False),
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        from density2sse.utils import runtime_device as rd

        assert rd.resolve_device_string("auto", command="t") == "cpu"


def test_force_cpu() -> None:
    with (
        mock.patch("density2sse.utils.runtime_device.torch.cuda.is_available", return_value=False),
        mock.patch.dict(os.environ, {"DENSITY2SSE_FORCE_CPU": "1"}),
    ):
        from density2sse.utils import runtime_device as rd

        assert rd.resolve_device_string("auto", command="t") == "cpu"
        # Even if YAML says mps, FORCE_CPU makes MPS ineligible; fall back to auto -> cpu.
        assert rd.get_torch_device("mps", command="t").type == "cpu"


def test_explicit_cpu_overridden_to_cuda() -> None:
    with (
        mock.patch("density2sse.utils.runtime_device.torch.cuda.is_available", return_value=True),
        mock.patch("density2sse.utils.runtime_device._mps_eligible", return_value=True),
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        from density2sse.utils import runtime_device as rd

        assert rd.resolve_device_string("cpu", command="t") == "cuda"


def test_explicit_cpu_overridden_to_mps() -> None:
    with (
        mock.patch("density2sse.utils.runtime_device.torch.cuda.is_available", return_value=False),
        mock.patch("density2sse.utils.runtime_device._mps_eligible", return_value=True),
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        from density2sse.utils import runtime_device as rd

        assert rd.resolve_device_string("cpu", command="t") == "mps"


def test_explicit_cuda_preserved() -> None:
    with (
        mock.patch("density2sse.utils.runtime_device.torch.cuda.is_available", return_value=True),
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        from density2sse.utils import runtime_device as rd

        assert rd.resolve_device_string("cuda:1", command="t") == "cuda:1"
        assert str(rd.get_torch_device("cuda:1", command="t")) == "cuda:1"


def test_mps_unavailable_uses_auto_fallback() -> None:
    with (
        mock.patch("density2sse.utils.runtime_device.torch.cuda.is_available", return_value=False),
        mock.patch("density2sse.utils.runtime_device._mps_eligible", return_value=False),
        mock.patch("density2sse.utils.runtime_device.LOG.warning") as wlog,
    ):
        from density2sse.utils import runtime_device as rd

        s = rd.resolve_device_string("mps", command="t")
    assert s == "cpu"
    wlog.assert_called()
