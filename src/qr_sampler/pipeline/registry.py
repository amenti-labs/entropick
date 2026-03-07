"""Stage registry with entry-point auto-discovery."""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any, ClassVar

logger = logging.getLogger("qr_sampler")


class StageRegistry:
    """Registry for pipeline stage classes.

    Stages are registered via the ``@StageRegistry.register("name")``
    decorator or discovered from the ``qr_sampler.pipeline_stages``
    entry-point group. Built-in registrations take precedence over
    entry points.
    """

    _stages: ClassVar[dict[str, type]] = {}
    _entry_points_loaded: ClassVar[bool] = False

    @classmethod
    def register(cls, name: str) -> Any:
        """Class decorator to register a pipeline stage.

        Args:
            name: Unique identifier for the stage.

        Returns:
            Decorator that registers the class and returns it unchanged.
        """

        def decorator(stage_cls: type) -> type:
            cls._stages[name] = stage_cls
            return stage_cls

        return decorator

    @classmethod
    def _load_entry_points(cls) -> None:
        """Lazily load stages from the ``qr_sampler.pipeline_stages`` entry-point group."""
        if cls._entry_points_loaded:
            return
        cls._entry_points_loaded = True

        try:
            eps = importlib.metadata.entry_points(group="qr_sampler.pipeline_stages")
        except Exception:
            logger.debug("Entry point discovery failed for qr_sampler.pipeline_stages")
            return

        for ep in eps:
            if ep.name not in cls._stages:
                try:
                    cls._stages[ep.name] = ep.load()
                except Exception:
                    logger.warning(
                        "Failed to load pipeline stage entry point: %s", ep.name
                    )

    @classmethod
    def get(cls, name: str) -> type:
        """Look up a stage class by name.

        Args:
            name: Registered stage name.

        Returns:
            The stage class.

        Raises:
            KeyError: If no stage is registered with that name.
        """
        cls._load_entry_points()
        if name not in cls._stages:
            available = ", ".join(sorted(cls._stages.keys()))
            raise KeyError(
                f"Unknown pipeline stage: {name!r}. Available: {available}"
            )
        return cls._stages[name]

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return sorted list of all registered stage names."""
        cls._load_entry_points()
        return sorted(cls._stages.keys())
