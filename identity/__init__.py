"""Identity package — persistent behavioural identity for the evolutionary renderer.

Provides the :class:`~identity.state.IdentityGenome` (genetic blueprint for
identity-level traits) and :class:`~identity.state.VisualIdentity` (the
persistent identity that ages, accumulates preferences, and guides evolution).
"""

from __future__ import annotations

from .state import IdentityGenome, VisualIdentity

__all__ = ["IdentityGenome", "VisualIdentity"]
