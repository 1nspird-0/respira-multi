"""
Test-Time Adaptation (TTA) for noisy environments.

Implements guarded TTA that only updates normalization layers
and has rollback capability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import copy


class TestTimeAdaptation:
    """
    Test-time adaptation for handling distribution shift.
    
    Only updates BatchNorm/LayerNorm statistics, never classifier weights.
    """
    
    def __init__(
        self,
        model: nn.Module,
        update_only_norms: bool = True,
        lr: float = 1e-3,
        max_steps: int = 3,
        entropy_threshold: float = 0.7,
    ):
        self.model = model
        self.update_only_norms = update_only_norms
        self.lr = lr
        self.max_steps = max_steps
        self.entropy_threshold = entropy_threshold
        
        # Store original state for rollback
        self.original_state = None
        
        # Identify norm layers
        self.norm_layers = self._get_norm_layers()
    
    def _get_norm_layers(self):
        """Get all normalization layers."""
        norm_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                norm_layers.append((name, module))
        return norm_layers
    
    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute prediction entropy."""
        probs = torch.sigmoid(logits)
        # Binary entropy
        entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
        return entropy.mean()
    
    def _get_adaptable_params(self):
        """Get parameters that should be adapted."""
        params = []
        for name, module in self.norm_layers:
            if hasattr(module, 'weight') and module.weight is not None:
                params.append(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                params.append(module.bias)
        return params
    
    def save_state(self):
        """Save current model state for rollback."""
        self.original_state = {
            name: copy.deepcopy(module.state_dict())
            for name, module in self.norm_layers
        }
    
    def restore_state(self):
        """Restore original model state."""
        if self.original_state is not None:
            for name, module in self.norm_layers:
                if name in self.original_state:
                    module.load_state_dict(self.original_state[name])
    
    def adapt(
        self,
        audio_tokens: torch.Tensor,
        segment_types: torch.Tensor,
        vitals: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        vitals_mask: Optional[torch.Tensor] = None,
        demographics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform test-time adaptation and return predictions.
        
        Uses entropy minimization on normalization layers only.
        """
        # Save state for potential rollback
        self.save_state()
        
        # Set model to eval but enable grad for norms
        self.model.eval()
        
        # Enable gradients only for norm parameters
        adaptable_params = self._get_adaptable_params()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in adaptable_params:
            param.requires_grad = True
        
        # Create optimizer
        optimizer = torch.optim.Adam(adaptable_params, lr=self.lr)
        
        # Initial prediction
        with torch.no_grad():
            initial_output = self.model(
                audio_tokens, segment_types, vitals,
                audio_mask, vitals_mask, demographics,
            )
            initial_entropy = self._compute_entropy(initial_output.disease_logits)
        
        # TTA steps
        best_entropy = initial_entropy
        best_output = initial_output
        
        for step in range(self.max_steps):
            optimizer.zero_grad()
            
            output = self.model(
                audio_tokens, segment_types, vitals,
                audio_mask, vitals_mask, demographics,
            )
            
            # Entropy minimization loss
            entropy = self._compute_entropy(output.disease_logits)
            
            # Check if we should continue
            if entropy > self.entropy_threshold * 1.5:
                # Entropy too high, adaptation may be diverging
                break
            
            if entropy < best_entropy:
                best_entropy = entropy
                best_output = output
            
            # Backward and update
            entropy.backward()
            optimizer.step()
        
        # Check if adaptation improved things
        final_entropy = self._compute_entropy(best_output.disease_logits)
        
        if final_entropy > initial_entropy * 1.2:
            # Adaptation made things worse, rollback
            self.restore_state()
            with torch.no_grad():
                best_output = self.model(
                    audio_tokens, segment_types, vitals,
                    audio_mask, vitals_mask, demographics,
                )
        
        # Reset requires_grad
        for param in self.model.parameters():
            param.requires_grad = True
        
        return {
            'disease_logits': best_output.disease_logits,
            'concept_logits': best_output.concept_logits,
            'adapted': final_entropy < initial_entropy,
            'entropy_before': initial_entropy.item(),
            'entropy_after': final_entropy.item(),
        }


class GuardedTTA(TestTimeAdaptation):
    """
    Guarded TTA with additional safety measures.
    
    Includes:
    - Confidence collapse detection
    - Automatic rollback
    - Signal quality-based adaptation
    """
    
    def __init__(
        self,
        model: nn.Module,
        confidence_collapse_threshold: float = 0.1,
        min_quality_for_tta: float = 0.3,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.confidence_collapse_threshold = confidence_collapse_threshold
        self.min_quality_for_tta = min_quality_for_tta
    
    def _detect_confidence_collapse(
        self,
        initial_probs: torch.Tensor,
        final_probs: torch.Tensor,
    ) -> bool:
        """Detect if confidence has collapsed."""
        initial_max = initial_probs.max().item()
        final_max = final_probs.max().item()
        
        # Collapse if max confidence dropped significantly
        if final_max < self.confidence_collapse_threshold:
            return True
        
        # Collapse if relative drop is too large
        if final_max < initial_max * 0.5:
            return True
        
        return False
    
    def adapt_with_quality(
        self,
        audio_tokens: torch.Tensor,
        segment_types: torch.Tensor,
        vitals: torch.Tensor,
        signal_quality: float,
        audio_mask: Optional[torch.Tensor] = None,
        vitals_mask: Optional[torch.Tensor] = None,
        demographics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Quality-aware TTA.
        
        Only adapts if signal quality is sufficient.
        """
        # Skip TTA for very low quality signals
        if signal_quality < self.min_quality_for_tta:
            with torch.no_grad():
                output = self.model(
                    audio_tokens, segment_types, vitals,
                    audio_mask, vitals_mask, demographics,
                )
            return {
                'disease_logits': output.disease_logits,
                'concept_logits': output.concept_logits,
                'adapted': False,
                'skip_reason': 'low_signal_quality',
            }
        
        # Get initial predictions
        with torch.no_grad():
            initial_output = self.model(
                audio_tokens, segment_types, vitals,
                audio_mask, vitals_mask, demographics,
            )
            initial_probs = torch.sigmoid(initial_output.disease_logits)
        
        # Perform TTA
        result = self.adapt(
            audio_tokens, segment_types, vitals,
            audio_mask, vitals_mask, demographics,
        )
        
        # Check for collapse
        final_probs = torch.sigmoid(result['disease_logits'])
        
        if self._detect_confidence_collapse(initial_probs, final_probs):
            # Rollback
            self.restore_state()
            result['disease_logits'] = initial_output.disease_logits
            result['concept_logits'] = initial_output.concept_logits
            result['adapted'] = False
            result['rollback_reason'] = 'confidence_collapse'
        
        return result

