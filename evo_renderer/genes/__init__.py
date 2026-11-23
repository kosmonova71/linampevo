"""gene package

Typed sub-genome modules that replace the flat ``VisualGenome.params`` dict.

Each module holds a specific domain of genes (camera, field, colour, particle,
feedback, lighting, behaviour). This makes crossover and mutation much more
coherent than operating on one large flat parameter dictionary.
"""

from __future__ import annotations

from .behaviour_genes import BehaviourGenes
from .camera_genes import CameraGenes
from .colour_genes import ColourGenes
from .feedback_genes import FeedbackGenes
from .field_genes import FieldGenes
from .lighting_genes import LightingGenes
from .particle_genes import ParticleGenes

__all__ = [
    "BehaviourGenes",
    "CameraGenes",
    "ColourGenes",
    "FeedbackGenes",
    "FieldGenes",
    "LightingGenes",
    "ParticleGenes",
]
