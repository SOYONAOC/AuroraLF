"""Strict typed public API for AuroraLF v2."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .api import run_uvlf
    from .config import UVLFRunConfig
    from .results import UVLFRunResult


__all__ = ["UVLFRunConfig", "UVLFRunResult", "run_uvlf"]


def __getattr__(name: str) -> Any:
    if name == "UVLFRunConfig":
        from .config import UVLFRunConfig

        value = UVLFRunConfig
    elif name == "UVLFRunResult":
        from .results import UVLFRunResult

        value = UVLFRunResult
    elif name == "run_uvlf":
        from .api import run_uvlf

        value = run_uvlf
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
