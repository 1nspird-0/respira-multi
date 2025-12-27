"""
Conformal Prediction for RESPIRA-MULTI.

Provides prediction sets with statistical coverage guarantees.
Enables "abstain" behavior when confidence is insufficient.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class PredictionSet:
    """A conformal prediction set with metadata."""
    # Predicted diseases (set with coverage guarantee)
    diseases: Set[str]
    disease_indices: Set[int]
    
    # Confidence metrics
    max_prob: float
    set_size: int
    
    # Coverage level used
    coverage_level: float
    
    # Whether to abstain (insufficient confidence)
    abstain: bool
    abstain_reason: Optional[str] = None
    
    # Raw probabilities for reference
    probabilities: Optional[Dict[str, float]] = None


class ConformalPredictor:
    """
    Conformal prediction for multi-label classification.
    
    Uses split conformal prediction with Bonferroni correction for
    multiple labels.
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        coverage_levels: List[float] = [0.80, 0.90, 0.95],
        abstain_threshold: float = 0.3,
        max_set_size: int = 5,
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.coverage_levels = coverage_levels
        self.abstain_threshold = abstain_threshold
        self.max_set_size = max_set_size
        
        # Calibration thresholds (per class, per coverage level)
        self.thresholds = {}
    
    def calibrate(
        self,
        cal_probs: torch.Tensor,
        cal_labels: torch.Tensor,
    ):
        """
        Calibrate thresholds using calibration set.
        
        Args:
            cal_probs: [N, num_classes] calibration probabilities
            cal_labels: [N, num_classes] calibration labels (binary)
        """
        cal_probs = cal_probs.cpu().numpy()
        cal_labels = cal_labels.cpu().numpy()
        
        n_cal = len(cal_probs)
        
        for coverage in self.coverage_levels:
            self.thresholds[coverage] = []
            
            for c in range(self.num_classes):
                # Non-conformity scores: 1 - probability when label is 1
                # For label = 0, we use probability as score
                positive_mask = cal_labels[:, c] == 1
                negative_mask = cal_labels[:, c] == 0
                
                # Combine scores
                scores = np.zeros(n_cal)
                scores[positive_mask] = 1 - cal_probs[positive_mask, c]
                scores[negative_mask] = cal_probs[negative_mask, c]
                
                # Quantile for coverage guarantee
                # With Bonferroni: adjust for multiple testing
                adjusted_coverage = 1 - (1 - coverage) / self.num_classes
                quantile = np.ceil((n_cal + 1) * adjusted_coverage) / n_cal
                threshold = np.quantile(scores, min(quantile, 1.0))
                
                self.thresholds[coverage].append(threshold)
    
    def predict_set(
        self,
        probs: torch.Tensor,
        coverage: float = 0.90,
    ) -> List[PredictionSet]:
        """
        Generate prediction sets for given probabilities.
        
        Args:
            probs: [batch, num_classes] or [num_classes] probabilities
            coverage: Target coverage level
            
        Returns:
            List of PredictionSet objects
        """
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        
        probs = probs.cpu().numpy()
        batch_size = len(probs)
        
        if coverage not in self.thresholds:
            # Use nearest available coverage
            coverage = min(self.coverage_levels, key=lambda x: abs(x - coverage))
        
        thresholds = self.thresholds[coverage]
        
        results = []
        
        for i in range(batch_size):
            sample_probs = probs[i]
            
            # Include in set if prob > (1 - threshold) for positive
            # or prob < threshold for negative
            # Simplified: include if probability is high enough
            predicted_set = set()
            predicted_indices = set()
            
            for c in range(self.num_classes):
                # For positive prediction: prob should be > 1 - threshold
                if sample_probs[c] > 1 - thresholds[c]:
                    predicted_set.add(self.class_names[c])
                    predicted_indices.add(c)
            
            # Compute metrics
            max_prob = float(sample_probs.max())
            set_size = len(predicted_set)
            
            # Abstain logic
            abstain = False
            abstain_reason = None
            
            if max_prob < self.abstain_threshold:
                abstain = True
                abstain_reason = "Confidence too low. Please re-record or seek medical evaluation."
            elif set_size > self.max_set_size:
                abstain = True
                abstain_reason = "Prediction set too large. Results uncertain."
            elif set_size == 0 and max_prob < 0.5:
                abstain = True
                abstain_reason = "No clear prediction. Consider re-recording."
            
            # Probabilities dict
            prob_dict = {
                self.class_names[c]: float(sample_probs[c])
                for c in range(self.num_classes)
            }
            
            results.append(PredictionSet(
                diseases=predicted_set,
                disease_indices=predicted_indices,
                max_prob=max_prob,
                set_size=set_size,
                coverage_level=coverage,
                abstain=abstain,
                abstain_reason=abstain_reason,
                probabilities=prob_dict,
            ))
        
        return results
    
    def compute_empirical_coverage(
        self,
        test_probs: torch.Tensor,
        test_labels: torch.Tensor,
        coverage: float = 0.90,
    ) -> Dict[str, float]:
        """
        Compute empirical coverage on test set.
        
        Returns coverage per class and overall.
        """
        prediction_sets = self.predict_set(test_probs, coverage)
        test_labels = test_labels.cpu().numpy()
        
        coverages = {name: [] for name in self.class_names}
        
        for i, pred_set in enumerate(prediction_sets):
            for c, name in enumerate(self.class_names):
                # True label is in prediction set?
                if test_labels[i, c] == 1:
                    coverages[name].append(c in pred_set.disease_indices)
        
        # Average coverage per class
        result = {}
        for name in self.class_names:
            if coverages[name]:
                result[name] = np.mean(coverages[name])
            else:
                result[name] = None
        
        # Overall coverage
        all_coverages = []
        for name in self.class_names:
            all_coverages.extend(coverages[name])
        result['overall'] = np.mean(all_coverages) if all_coverages else 0.0
        
        return result
    
    def get_abstain_statistics(
        self,
        test_probs: torch.Tensor,
        coverage: float = 0.90,
    ) -> Dict[str, float]:
        """Compute abstain rate and reasons."""
        prediction_sets = self.predict_set(test_probs, coverage)
        
        total = len(prediction_sets)
        abstain_count = sum(1 for ps in prediction_sets if ps.abstain)
        
        reasons = {}
        for ps in prediction_sets:
            if ps.abstain and ps.abstain_reason:
                reason = ps.abstain_reason.split('.')[0]
                reasons[reason] = reasons.get(reason, 0) + 1
        
        return {
            'abstain_rate': abstain_count / total if total > 0 else 0,
            'abstain_count': abstain_count,
            'total': total,
            'reasons': reasons,
        }


class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Adaptive conformal prediction that adjusts based on input quality.
    
    Uses signal quality scores to adjust coverage requirements.
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        quality_aware: bool = True,
        **kwargs,
    ):
        super().__init__(num_classes, class_names, **kwargs)
        self.quality_aware = quality_aware
    
    def predict_set_adaptive(
        self,
        probs: torch.Tensor,
        quality_score: float,
        base_coverage: float = 0.90,
    ) -> PredictionSet:
        """
        Adaptive prediction based on input quality.
        
        Lower quality -> stricter thresholds (smaller sets with more abstention)
        """
        if not self.quality_aware:
            return self.predict_set(probs, base_coverage)[0]
        
        # Adjust coverage based on quality
        # Low quality -> higher coverage requirement -> more abstention
        if quality_score < 0.3:
            adjusted_coverage = 0.99  # Very strict
            self.abstain_threshold = 0.5  # Higher threshold for abstention
        elif quality_score < 0.6:
            adjusted_coverage = min(base_coverage + 0.05, 0.95)
            self.abstain_threshold = 0.4
        else:
            adjusted_coverage = base_coverage
            self.abstain_threshold = 0.3
        
        # Use closest available coverage level
        closest_coverage = min(
            self.coverage_levels,
            key=lambda x: abs(x - adjusted_coverage)
        )
        
        result = self.predict_set(probs, closest_coverage)[0]
        
        # Additional abstain for very low quality
        if quality_score < 0.2 and not result.abstain:
            result.abstain = True
            result.abstain_reason = "Signal quality too low. Please re-record."
        
        return result

