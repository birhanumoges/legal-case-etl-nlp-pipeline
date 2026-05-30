"""
modeling/__init__.py
---------------------
Public API for the modeling package.

Three independent models, each with its own class:
  CaseClassifier    – predicts top-level Case_Type
  SubtypeClassifier – hierarchical Sub_Type (one model per Case_Type)
  VerdictPredictor  – predicts simplified verdict group

Shared infrastructure:
  Trainer           – wraps sklearn estimators with fit/predict/save/load
"""

from .trainer            import Trainer
from .case_classifier    import CaseClassifier
from .subtype_classifier import SubtypeClassifier
from .verdict_predictor  import VerdictPredictor, map_verdict_to_group

__all__ = [
    "Trainer",
    "CaseClassifier",
    "SubtypeClassifier",
    "VerdictPredictor",
    "map_verdict_to_group",
]