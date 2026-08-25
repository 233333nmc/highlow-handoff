"""Explicit deployment composition roots.

Composition modules may know both a simulator integration and a concrete
policy plugin.  Keeping that knowledge here prevents either core namespace
from depending on the other.
"""

__all__: list[str] = []
