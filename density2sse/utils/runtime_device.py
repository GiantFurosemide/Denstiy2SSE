"""Model-agnostic training/inference device resolution (CUDA / MPS / CPU)."""

from __future__ import annotations

import os
import sys
import torch

from density2sse.utils.logging_utils import setup_logging

LOG = setup_logging(name="density2sse.runtime_device")


def _env_force_cpu() -> bool:
    return os.environ.get("DENSITY2SSE_FORCE_CPU", "0") == "1"


def _env_disable_mps() -> bool:
    return os.environ.get("DENSITY2SSE_DISABLE_MPS", "0") == "1"


def _mps_backends_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def _mps_eligible() -> bool:
    if _env_force_cpu() or _env_disable_mps():
        return False
    if sys.platform != "darwin":
        return False
    return _mps_backends_available()


def _auto_device_string() -> str:
    """Priority: CUDA, then MPS (macOS), then CPU. Respects DENSITY2SSE_FORCE_CPU."""
    if _env_force_cpu():
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if _mps_eligible():
        return "mps"
    return "cpu"


def _log_choice(command: str, device_str: str, reason: str) -> None:
    LOG.info("%s: using device %s — %s", command, device_str, reason)


def resolve_device_string(training_device: str, *, command: str) -> str:
    """
    Return a string suitable for ``torch.device`` (e.g. ``cuda``, ``cuda:1``, ``mps``, ``cpu``).
    For ``auto``/empty: CUDA if available, else MPS on macOS when available, else CPU.
    """
    raw = str(training_device).strip()
    ds = raw.lower() if raw else "auto"
    if not ds:
        ds = "auto"

    s: str
    reason: str

    if ds in ("auto", ""):
        s = _auto_device_string()
        if s == "cuda":
            reason = "auto: CUDA available"
        elif s == "mps":
            reason = "auto: MPS (macOS)"
        else:
            reason = "auto: no accelerator (or FORCE_CPU) → CPU"
        _log_choice(command, s, reason)
        return s

    if ds == "cpu":
        if _env_force_cpu():
            s = "cpu"
            reason = "explicit CPU (DENSITY2SSE_FORCE_CPU=1)"
        elif torch.cuda.is_available():
            s = "cuda"
            reason = "config had CPU but CUDA available; override to CUDA (set DENSITY2SSE_FORCE_CPU=1 to force CPU)"
        elif _mps_eligible():
            s = "mps"
            reason = "config had CPU but MPS available; override to MPS (set DENSITY2SSE_FORCE_CPU=1 to force CPU)"
        else:
            s = "cpu"
            reason = "explicit CPU, no CUDA/MPS to promote to"
        _log_choice(command, s, reason)
        return s

    if ds.startswith("mps"):
        if _mps_eligible():
            s = "mps"
            _log_choice(command, s, f"requested {raw}")
        else:
            LOG.warning(
                "%s: %s not usable; falling back to auto pick",
                command,
                raw,
            )
            s = _auto_device_string()
            _log_choice(
                command,
                s,
                f"auto fallback (was {raw!r})",
            )
        return s

    if ds.startswith("cuda"):
        if torch.cuda.is_available():
            s = raw
            _log_choice(command, s, f"requested {raw}")
        else:
            LOG.warning(
                "%s: %s requested but CUDA unavailable; falling back to auto pick",
                command,
                raw,
            )
            s = _auto_device_string()
            _log_choice(command, s, f"auto fallback (was {raw!r})")
        return s

    LOG.warning("%s: unknown training.device %r; using auto pick", command, training_device)
    s = _auto_device_string()
    _log_choice(command, s, "unknown device, auto")
    return s


def get_torch_device(training_device: str, *, command: str) -> "torch.device":
    """``torch.device(resolve_device_string(...))``."""
    return torch.device(resolve_device_string(training_device, command=command))
