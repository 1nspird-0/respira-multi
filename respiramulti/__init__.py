"""
RESPIRA-MULTI: Multimodal Respiratory Disease Screening System

A cutting-edge on-device AI system for respiratory disease screening using
smartphone-capturable inputs with teacher-student distillation.
"""

__version__ = "1.0.0"
__author__ = "RESPIRA-MULTI Team"

from respiramulti.models.full_model import RespiraMultiStudent, RespiraMultiTeacher
from respiramulti.datasets.unified_loader import UnifiedDataLoader
from respiramulti.datasets.schema import SessionSchema, DISEASES, CONCEPTS

__all__ = [
    "RespiraMultiStudent",
    "RespiraMultiTeacher",
    "UnifiedDataLoader",
    "SessionSchema",
    "DISEASES",
    "CONCEPTS",
]

