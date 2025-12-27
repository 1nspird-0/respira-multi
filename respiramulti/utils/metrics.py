"""
Metrics computation for RESPIRA-MULTI.

Implements AUROC, AUPRC, sensitivity at specificity, and more.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from dataclasses import dataclass


@dataclass
class DiseaseMetrics:
    """Metrics for a single disease."""
    auroc: float
    auprc: float
    sensitivity_at_90_specificity: float
    sensitivity_at_95_specificity: float
    threshold_at_90_specificity: float


def compute_auroc(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute AUROC for a single class."""
    if mask is not None:
        probs = probs[mask > 0]
        labels = labels[mask > 0]
    
    if len(np.unique(labels)) < 2:
        return 0.5  # Undefined
    
    try:
        return roc_auc_score(labels, probs)
    except ValueError:
        return 0.5


def compute_auprc(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Average Precision (AUPRC)."""
    if mask is not None:
        probs = probs[mask > 0]
        labels = labels[mask > 0]
    
    if len(np.unique(labels)) < 2:
        return 0.0
    
    try:
        return average_precision_score(labels, probs)
    except ValueError:
        return 0.0


def sensitivity_at_specificity(
    probs: np.ndarray,
    labels: np.ndarray,
    target_specificity: float = 0.90,
) -> Tuple[float, float]:
    """
    Compute sensitivity at a given specificity level.
    
    Returns (sensitivity, threshold)
    """
    if len(np.unique(labels)) < 2:
        return 0.0, 0.5
    
    fpr, tpr, thresholds = roc_curve(labels, probs)
    specificity = 1 - fpr
    
    # Find threshold that achieves target specificity
    idx = np.argmin(np.abs(specificity - target_specificity))
    
    return tpr[idx], thresholds[idx]


def compute_per_disease_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    disease_names: Optional[List[str]] = None,
) -> Dict[str, DiseaseMetrics]:
    """
    Compute metrics for each disease.
    
    Args:
        probs: [N, num_diseases] predicted probabilities
        labels: [N, num_diseases] ground truth labels
        mask: [N, num_diseases] mask for valid labels
    """
    probs_np = probs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    mask_np = mask.cpu().numpy() if mask is not None else None
    
    num_diseases = probs_np.shape[1]
    
    if disease_names is None:
        disease_names = [f"disease_{i}" for i in range(num_diseases)]
    
    metrics = {}
    
    for i, name in enumerate(disease_names):
        m = mask_np[:, i] if mask_np is not None else None
        
        auroc = compute_auroc(probs_np[:, i], labels_np[:, i], m)
        auprc = compute_auprc(probs_np[:, i], labels_np[:, i], m)
        sens_90, thresh_90 = sensitivity_at_specificity(
            probs_np[:, i][m > 0] if m is not None else probs_np[:, i],
            labels_np[:, i][m > 0] if m is not None else labels_np[:, i],
            0.90
        )
        sens_95, _ = sensitivity_at_specificity(
            probs_np[:, i][m > 0] if m is not None else probs_np[:, i],
            labels_np[:, i][m > 0] if m is not None else labels_np[:, i],
            0.95
        )
        
        metrics[name] = DiseaseMetrics(
            auroc=auroc,
            auprc=auprc,
            sensitivity_at_90_specificity=sens_90,
            sensitivity_at_95_specificity=sens_95,
            threshold_at_90_specificity=thresh_90,
        )
    
    return metrics


def compute_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    disease_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute aggregate metrics.
    
    Returns dict with macro-averaged metrics and per-disease metrics.
    """
    per_disease = compute_per_disease_metrics(probs, labels, mask, disease_names)
    
    # Macro averages
    aurocs = [m.auroc for m in per_disease.values()]
    auprcs = [m.auprc for m in per_disease.values()]
    sens_90s = [m.sensitivity_at_90_specificity for m in per_disease.values()]
    
    result = {
        'auroc': np.mean(aurocs),
        'auprc': np.mean(auprcs),
        'sensitivity_at_90_spec': np.mean(sens_90s),
    }
    
    # Per-disease metrics
    for name, metrics in per_disease.items():
        result[f'{name}_auroc'] = metrics.auroc
        result[f'{name}_auprc'] = metrics.auprc
    
    return result


class MetricsLogger:
    """
    Logger for tracking metrics during training.
    """
    
    def __init__(self, disease_names: List[str]):
        self.disease_names = disease_names
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auroc': [],
            'val_auprc': [],
        }
        
        for name in disease_names:
            self.history[f'{name}_auroc'] = []
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: Dict[str, float],
    ):
        """Log metrics for an epoch."""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_auroc'].append(val_metrics.get('auroc', 0))
        self.history['val_auprc'].append(val_metrics.get('auprc', 0))
        
        for name in self.disease_names:
            key = f'{name}_auroc'
            self.history[key].append(val_metrics.get(key, 0))
    
    def get_best_epoch(self) -> int:
        """Get epoch with best validation AUROC."""
        if not self.history['val_auroc']:
            return 0
        return np.argmax(self.history['val_auroc'])
    
    def get_summary(self) -> Dict:
        """Get summary of best metrics."""
        best_epoch = self.get_best_epoch()
        
        return {
            'best_epoch': best_epoch,
            'best_auroc': self.history['val_auroc'][best_epoch],
            'best_auprc': self.history['val_auprc'][best_epoch],
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
        }


def compute_confusion_matrix(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> np.ndarray:
    """Compute confusion matrix for binary classification."""
    preds = (probs >= threshold).float()
    
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    
    return np.array([[tn, fp], [fn, tp]])


def compute_calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data.
    
    Returns (mean_predicted_probs, fraction_positives, counts)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    mean_probs = []
    fraction_pos = []
    counts = []
    
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_probs.append(probs[mask].mean())
            fraction_pos.append(labels[mask].mean())
            counts.append(mask.sum())
        else:
            mean_probs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            fraction_pos.append(0)
            counts.append(0)
    
    return np.array(mean_probs), np.array(fraction_pos), np.array(counts)

