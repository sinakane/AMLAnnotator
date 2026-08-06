"""
AMLAnnotator — Hierarchical annotation of AML single-cell data.

Three-level classification pipeline:
  Level 1: Malignant vs Normal
  Level 2: Blast group (Primitive, GMP-like, Erythroid-like, Mature myeloid-like, High LSC score shared)
  Level 3: LSC type (P-LSC, M-LSC, E-LSC, Patient Specific)
"""

from .annotator import AMLAnnotator

__version__ = "0.1.0"
__all__ = ["AMLAnnotator"]
