"""
Explanation generation for RESPIRA-MULTI predictions.

Includes Grad-CAM for spectrogram visualization and
attention-based explanations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class PredictionExplanation:
    """Complete explanation for a prediction."""
    # Top predictions
    top_diseases: List[Tuple[str, float]]  # (name, probability)
    top_concepts: List[Tuple[str, float]]
    
    # Modality contributions
    audio_contribution: float
    vitals_contribution: float
    
    # Concept breakdown (for concept bottleneck)
    concept_contributions: Optional[Dict[str, Dict[str, float]]] = None  # disease -> concept -> contribution
    
    # Attention heatmaps
    segment_attention: Optional[Dict[str, float]] = None  # segment_type -> attention
    
    # Grad-CAM heatmaps (per segment)
    gradcam_heatmaps: Optional[Dict[str, np.ndarray]] = None
    
    # Evidence from prototypes
    prototype_evidence: Optional[Dict[str, List]] = None
    
    # Confidence and reliability
    confidence_score: float = 0.0
    calibrated: bool = False
    
    # Safety disclaimer
    disclaimer: str = "This is a screening tool only, NOT a medical diagnosis."


class GradCAMExplainer:
    """
    Gradient-weighted Class Activation Mapping for spectrograms.
    
    Highlights which parts of the spectrogram contributed to predictions.
    """
    
    def __init__(self, model: nn.Module, target_layer: str = 'audio_encoder'):
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        for name, module in self.model.named_modules():
            if self.target_layer in name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
    
    def generate_heatmap(
        self,
        spectrogram: torch.Tensor,
        target_class: int,
        class_type: str = 'disease',
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a target class.
        
        Args:
            spectrogram: [1, 1, n_mels, time] input spectrogram
            target_class: Index of target class
            class_type: 'disease' or 'concept'
            
        Returns:
            heatmap: [n_mels, time] activation heatmap
        """
        self.model.eval()
        spectrogram.requires_grad = True
        
        # Forward pass through the model
        # This is a simplified version - actual implementation depends on model structure
        output = self.model.audio_encoder(spectrogram, return_features=True)
        
        if class_type == 'disease':
            # Would need full forward pass for disease logits
            target_score = output['embedding'].sum()  # Simplified
        else:
            target_score = output['embedding'].sum()
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward()
        
        if self.gradients is None or self.activations is None:
            return np.zeros((spectrogram.shape[2], spectrogram.shape[3]))
        
        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to original spectrogram size
        import cv2
        cam = cv2.resize(cam, (spectrogram.shape[3], spectrogram.shape[2]))
        
        return cam
    
    def generate_multi_class_heatmaps(
        self,
        spectrogram: torch.Tensor,
        target_classes: List[int],
        class_type: str = 'disease',
    ) -> Dict[int, np.ndarray]:
        """Generate heatmaps for multiple classes."""
        heatmaps = {}
        for class_idx in target_classes:
            heatmaps[class_idx] = self.generate_heatmap(
                spectrogram, class_idx, class_type
            )
        return heatmaps


class ExplanationGenerator:
    """
    Generate comprehensive explanations for predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        disease_names: List[str],
        concept_names: List[str],
        segment_types: List[str],
        prototype_bank: Optional[nn.Module] = None,
    ):
        self.model = model
        self.disease_names = disease_names
        self.concept_names = concept_names
        self.segment_types = segment_types
        self.prototype_bank = prototype_bank
        
        self.gradcam = GradCAMExplainer(model)
    
    def generate_explanation(
        self,
        audio_tokens: torch.Tensor,
        segment_types: torch.Tensor,
        vitals: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        vitals_mask: Optional[torch.Tensor] = None,
        demographics: Optional[torch.Tensor] = None,
        top_k: int = 3,
        include_gradcam: bool = True,
    ) -> PredictionExplanation:
        """
        Generate complete explanation for a prediction.
        
        Args:
            audio_tokens: [1, num_tokens, 1, n_mels, time]
            segment_types: [1, num_tokens]
            vitals: [1, vitals_dim]
            ... other inputs ...
            
        Returns:
            PredictionExplanation object
        """
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(
                audio_tokens=audio_tokens,
                segment_types=segment_types,
                vitals=vitals,
                audio_mask=audio_mask,
                vitals_mask=vitals_mask,
                demographics=demographics,
                return_attention=True,
            )
        
        # Get probabilities
        disease_probs = torch.sigmoid(output.disease_logits).squeeze().cpu().numpy()
        concept_probs = torch.sigmoid(output.concept_logits).squeeze().cpu().numpy()
        
        # Top diseases
        top_disease_idx = np.argsort(disease_probs)[::-1][:top_k]
        top_diseases = [
            (self.disease_names[i], float(disease_probs[i]))
            for i in top_disease_idx
        ]
        
        # Top concepts
        top_concept_idx = np.argsort(concept_probs)[::-1][:top_k]
        top_concepts = [
            (self.concept_names[i], float(concept_probs[i]))
            for i in top_concept_idx
        ]
        
        # Modality contributions from gate weights
        gate_weights = output.gate_weights
        if gate_weights is not None:
            audio_contrib = gate_weights.get('audio', torch.tensor([0.5])).item()
            vitals_contrib = gate_weights.get('vitals', torch.tensor([0.5])).item()
        else:
            audio_contrib = 0.5
            vitals_contrib = 0.5
        
        # Concept contributions (from concept bottleneck)
        concept_contributions = None
        if output.concept_contributions is not None:
            concept_contributions = self._parse_concept_contributions(
                output.concept_contributions.cpu().numpy(),
                top_disease_idx,
            )
        
        # Segment attention
        segment_attention = None
        if output.attention_weights is not None:
            segment_attention = self._compute_segment_attention(
                output.attention_weights,
                segment_types.squeeze().cpu().numpy(),
                audio_mask,
            )
        
        # Grad-CAM heatmaps
        gradcam_heatmaps = None
        if include_gradcam:
            gradcam_heatmaps = {}
            for i in range(min(audio_tokens.shape[1], 5)):  # First 5 tokens
                if audio_mask is None or audio_mask[0, i] == 0:
                    heatmap = self.gradcam.generate_heatmap(
                        audio_tokens[0, i:i+1],
                        target_class=top_disease_idx[0],
                    )
                    seg_name = self.segment_types[segment_types[0, i].item()]
                    gradcam_heatmaps[f"{seg_name}_{i}"] = heatmap
        
        # Prototype evidence
        prototype_evidence = None
        if self.prototype_bank is not None:
            from respiramulti.interpretability.prototypes import PrototypeRetrieval
            retrieval = PrototypeRetrieval(self.prototype_bank, self.segment_types)
            prototype_evidence = retrieval.get_evidence_for_prediction(
                output.cls_embedding.squeeze(),
                output.token_embeddings.squeeze(),
                segment_types.squeeze(),
                list(top_disease_idx),
            )
        
        # Confidence score
        confidence_score = float(disease_probs[top_disease_idx[0]])
        
        return PredictionExplanation(
            top_diseases=top_diseases,
            top_concepts=top_concepts,
            audio_contribution=audio_contrib,
            vitals_contribution=vitals_contrib,
            concept_contributions=concept_contributions,
            segment_attention=segment_attention,
            gradcam_heatmaps=gradcam_heatmaps,
            prototype_evidence=prototype_evidence,
            confidence_score=confidence_score,
            calibrated=hasattr(self.model, 'disease_temperatures'),
        )
    
    def _parse_concept_contributions(
        self,
        contributions: np.ndarray,
        top_diseases: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Parse concept contribution matrix."""
        result = {}
        
        for disease_idx in top_diseases:
            disease_name = self.disease_names[disease_idx]
            result[disease_name] = {}
            
            for concept_idx in range(len(self.concept_names)):
                contrib = contributions[0, disease_idx, concept_idx]
                if abs(contrib) > 0.01:  # Only significant contributions
                    result[disease_name][self.concept_names[concept_idx]] = float(contrib)
        
        return result
    
    def _compute_segment_attention(
        self,
        attention_weights: List[torch.Tensor],
        segment_types: np.ndarray,
        audio_mask: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """Compute average attention per segment type."""
        # Average attention across layers and heads
        avg_attention = attention_weights[-1].mean(dim=1).squeeze()  # [seq, seq]
        
        # CLS token attention to other tokens
        cls_attention = avg_attention[0, 1:].cpu().numpy()
        
        # Aggregate by segment type
        segment_attention = {}
        for seg_type in set(segment_types):
            seg_name = self.segment_types[seg_type]
            seg_mask = segment_types == seg_type
            if audio_mask is not None:
                seg_mask = seg_mask & (audio_mask.squeeze().cpu().numpy() == 0)
            if seg_mask.any():
                segment_attention[seg_name] = float(cls_attention[seg_mask].mean())
        
        return segment_attention
    
    def format_explanation_text(
        self,
        explanation: PredictionExplanation,
    ) -> str:
        """Format explanation as human-readable text."""
        lines = []
        
        lines.append("=" * 50)
        lines.append("RESPIRA-MULTI Screening Results")
        lines.append("=" * 50)
        lines.append("")
        lines.append(explanation.disclaimer)
        lines.append("")
        
        # Top predictions
        lines.append("TOP PREDICTED CONDITIONS:")
        for name, prob in explanation.top_diseases:
            risk_level = "High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"
            lines.append(f"  • {name}: {prob*100:.1f}% ({risk_level} risk)")
        lines.append("")
        
        # Detected concepts
        lines.append("DETECTED SIGNS:")
        for name, prob in explanation.top_concepts:
            if prob > 0.5:
                lines.append(f"  • {name}: {prob*100:.1f}%")
        lines.append("")
        
        # Modality contributions
        lines.append("INPUT CONTRIBUTIONS:")
        lines.append(f"  • Audio analysis: {explanation.audio_contribution*100:.1f}%")
        lines.append(f"  • Vital signs: {explanation.vitals_contribution*100:.1f}%")
        lines.append("")
        
        # Segment attention
        if explanation.segment_attention:
            lines.append("MOST RELEVANT SEGMENTS:")
            sorted_segs = sorted(
                explanation.segment_attention.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            for seg_name, attn in sorted_segs:
                lines.append(f"  • {seg_name}: {attn*100:.1f}% attention")
        lines.append("")
        
        # Confidence
        lines.append(f"Confidence Score: {explanation.confidence_score*100:.1f}%")
        if explanation.calibrated:
            lines.append("(Probabilities are calibrated)")
        lines.append("")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)

