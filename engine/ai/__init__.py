"""
ASCEND V9 - AI Module

Autonomous system hooks for evolution, repair, and live adaptation.
"""

from engine.ai.evolution import EvolutionEngine, Genome, mutate_ir, score_ir
from engine.ai.repair import RepairEngine

__all__ = [
    "EvolutionEngine",
    "Genome",
    "RepairEngine",
    "mutate_ir",
    "score_ir",
]
