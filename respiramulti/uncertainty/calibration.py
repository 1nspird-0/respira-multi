"""
Calibration methods for probability estimates.

Implements temperature scaling and calibration metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    reliability_data: Dict  # For reliability diagrams
    temperatures: Optional[List[float]] = None


class CalibrationMetrics:
    """
    Compute calibration metrics for model predictions.
    """
    
    @staticmethod
    def expected_calibration_error(
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
    ) -> Tuple[float, Dict]:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            probs: [N, num_classes] predicted probabilities
            labels: [N, num_classes] ground truth (binary)
            n_bins: Number of bins for calibration
            
        Returns:
            ECE value and reliability diagram data
        """
        # Flatten for binary case
        probs_flat = probs.flatten()
        labels_flat = labels.flatten()
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        bin_data = {
            'accuracies': [],
            'confidences': [],
            'counts': [],
        }
        
        for i in range(n_bins):
            in_bin = (probs_flat > bin_boundaries[i]) & (probs_flat <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels_flat[in_bin].float().mean()
                avg_confidence_in_bin = probs_flat[in_bin].mean()
                
                ece += prop_in_bin * torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                
                bin_data['accuracies'].append(accuracy_in_bin.item())
                bin_data['confidences'].append(avg_confidence_in_bin.item())
                bin_data['counts'].append(in_bin.sum().item())
            else:
                bin_data['accuracies'].append(0)
                bin_data['confidences'].append((bin_boundaries[i] + bin_boundaries[i+1]).item() / 2)
                bin_data['counts'].append(0)
        
        return ece.item(), bin_data
    
    @staticmethod
    def maximum_calibration_error(
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
    ) -> float:
        """Compute Maximum Calibration Error (MCE)."""
        probs_flat = probs.flatten()
        labels_flat = labels.flatten()
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
        mce = 0.0
        
        for i in range(n_bins):
            in_bin = (probs_flat > bin_boundaries[i]) & (probs_flat <= bin_boundaries[i + 1])
            
            if in_bin.sum() > 0:
                accuracy_in_bin = labels_flat[in_bin].float().mean()
                avg_confidence_in_bin = probs_flat[in_bin].mean()
                
                calibration_error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                mce = max(mce, calibration_error.item())
        
        return mce
    
    @staticmethod
    def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute Brier score (mean squared error of probabilities)."""
        return F.mse_loss(probs, labels.float()).item()
    
    @staticmethod
    def compute_all_metrics(
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
    ) -> CalibrationResult:
        """Compute all calibration metrics."""
        ece, reliability_data = CalibrationMetrics.expected_calibration_error(
            probs, labels, n_bins
        )
        mce = CalibrationMetrics.maximum_calibration_error(probs, labels, n_bins)
        brier = CalibrationMetrics.brier_score(probs, labels)
        
        return CalibrationResult(
            ece=ece,
            mce=mce,
            brier_score=brier,
            reliability_data=reliability_data,
        )


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibration.
    
    Learns a single temperature parameter (or per-class) to scale logits.
    """
    
    def __init__(
        self,
        num_classes: int,
        per_class: bool = True,
        init_temperature: float = 1.0,
    ):
        super().__init__()
        
        self.per_class = per_class
        
        if per_class:
            self.temperature = nn.Parameter(
                torch.ones(num_classes) * init_temperature
            )
        else:
            self.temperature = nn.Parameter(torch.tensor(init_temperature))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature.clamp(min=0.01)
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> Dict:
        """
        Fit temperature(s) using validation data.
        
        Args:
            logits: [N, num_classes] validation logits
            labels: [N, num_classes] validation labels
            
        Returns:
            Dict with final temperatures and metrics
        """
        if self.per_class:
            return self._fit_per_class(logits, labels, max_iter, lr)
        else:
            return self._fit_single(logits, labels, max_iter, lr)
    
    def _fit_single(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int,
        lr: float,
    ) -> Dict:
        """Fit single temperature."""
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature.clamp(min=0.01)
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels.float())
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        # Compute final metrics
        with torch.no_grad():
            probs_before = torch.sigmoid(logits)
            probs_after = torch.sigmoid(self.forward(logits))
            
            ece_before = CalibrationMetrics.expected_calibration_error(probs_before, labels)[0]
            ece_after = CalibrationMetrics.expected_calibration_error(probs_after, labels)[0]
        
        return {
            'temperature': self.temperature.item(),
            'ece_before': ece_before,
            'ece_after': ece_after,
        }
    
    def _fit_per_class(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int,
        lr: float,
    ) -> Dict:
        """Fit temperature per class."""
        num_classes = logits.shape[1]
        results = {'temperatures': [], 'ece_before': [], 'ece_after': []}
        
        for c in range(num_classes):
            # Fit for this class
            temp_c = nn.Parameter(torch.tensor(1.0))
            optimizer = optim.LBFGS([temp_c], lr=lr, max_iter=max_iter)
            
            def closure():
                optimizer.zero_grad()
                scaled_logits = logits[:, c] / temp_c.clamp(min=0.01)
                loss = F.binary_cross_entropy_with_logits(
                    scaled_logits, labels[:, c].float()
                )
                loss.backward()
                return loss
            
            optimizer.step(closure)
            
            self.temperature.data[c] = temp_c.data
            
            # Metrics for this class
            with torch.no_grad():
                probs_before = torch.sigmoid(logits[:, c])
                probs_after = torch.sigmoid(logits[:, c] / temp_c)
                
                ece_b = CalibrationMetrics.expected_calibration_error(
                    probs_before.unsqueeze(1), labels[:, c].unsqueeze(1)
                )[0]
                ece_a = CalibrationMetrics.expected_calibration_error(
                    probs_after.unsqueeze(1), labels[:, c].unsqueeze(1)
                )[0]
            
            results['temperatures'].append(temp_c.item())
            results['ece_before'].append(ece_b)
            results['ece_after'].append(ece_a)
        
        return results


class IsotonicCalibration:
    """
    Isotonic regression for calibration.
    
    Non-parametric method that can correct any monotonic miscalibration.
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.calibrators = []
    
    def fit(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Fit isotonic regression per class."""
        from sklearn.isotonic import IsotonicRegression
        
        probs_np = probs.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        self.calibrators = []
        for c in range(self.num_classes):
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(probs_np[:, c], labels_np[:, c])
            self.calibrators.append(ir)
    
    def transform(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply calibration."""
        probs_np = probs.cpu().numpy()
        calibrated = np.zeros_like(probs_np)
        
        for c, ir in enumerate(self.calibrators):
            calibrated[:, c] = ir.transform(probs_np[:, c])
        
        return torch.tensor(calibrated, dtype=probs.dtype, device=probs.device)

