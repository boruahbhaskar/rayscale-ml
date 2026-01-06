"""Configuration module for RayScale ML Platform."""

# Re-export the `settings` instance from the settings submodule so that
# `from src.config import settings` returns the configured Settings instance
# (not the submodule object).
from .settings import settings

__all__ = ["settings"]
