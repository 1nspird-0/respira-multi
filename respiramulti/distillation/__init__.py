"""Distillation losses and training utilities."""

from respiramulti.distillation.losses import (
    DistillationLoss,
    LogitDistillationLoss,
    FeatureDistillationLoss,
    AttentionDistillationLoss,
    CombinedDistillationLoss,
)
from respiramulti.distillation.trainer import DistillationTrainer

__all__ = [
    "DistillationLoss",
    "LogitDistillationLoss",
    "FeatureDistillationLoss",
    "AttentionDistillationLoss",
    "CombinedDistillationLoss",
    "DistillationTrainer",
]

