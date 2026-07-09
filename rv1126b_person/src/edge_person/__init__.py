"""Lightweight person detection support code for RV1126B edge demos."""

from .behavior import BehaviorAnalyzer, BehaviorConfig
from .tracker import LightByteTracker

__all__ = [
    "BehaviorAnalyzer",
    "BehaviorConfig",
    "LightByteTracker",
]
