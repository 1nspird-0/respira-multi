"""Interpretability modules for RESPIRA-MULTI."""

from respiramulti.interpretability.prototypes import PrototypeBank, PrototypeRetrieval
from respiramulti.interpretability.explanations import ExplanationGenerator, GradCAMExplainer

__all__ = [
    "PrototypeBank",
    "PrototypeRetrieval",
    "ExplanationGenerator",
    "GradCAMExplainer",
]

