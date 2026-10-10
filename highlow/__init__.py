"""Composable high-level, low-level, and simulator bridge runtime.

The package deliberately does not import model frameworks at module import
time.  High- and low-level implementations are loaded through their own
backend interfaces so the shared contracts and bridge stay lightweight.
"""

from .contracts import (
    ActionEnvelope,
    EpisodeContext,
    EpisodeResult,
    Modality,
    ObservationEnvelope,
    RequestContext,
)

__all__ = [
    "ActionEnvelope",
    "EpisodeContext",
    "EpisodeResult",
    "Modality",
    "ObservationEnvelope",
    "RequestContext",
]
