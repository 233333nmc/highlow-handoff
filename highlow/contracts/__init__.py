"""Framework-free contracts shared by bridges and inference services."""

from .action import ActionEnvelope
from .common import RequestContext
from .episode import EpisodeContext, EpisodeResult
from .observation import Modality, ObservationEnvelope

__all__ = [
    "ActionEnvelope",
    "EpisodeContext",
    "EpisodeResult",
    "Modality",
    "ObservationEnvelope",
    "RequestContext",
]
