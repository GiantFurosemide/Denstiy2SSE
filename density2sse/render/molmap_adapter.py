"""Optional ChimeraX ``molmap`` adapter (not required for MVP)."""

from __future__ import annotations

from typing import Any, Optional


def render_with_molmap(pdb_path: str, resolution: float, grid_spacing: Optional[float] = None) -> Any:
    """
    Placeholder for future ChimeraX headless integration.

    Raises ``NotImplementedError`` until a concrete adapter is wired in.
    """
    raise NotImplementedError(
        "ChimeraX molmap adapter is not implemented; use renderer=cylinder in YAML."
    )
