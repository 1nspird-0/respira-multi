"""
Distillation losses for teacher-student training.

Implements logit, feature, and attention distillation losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple


class LogitDistillationLoss(nn.Module):
    """
    KL divergence loss on logits for knowledge distillation.
    
    For multi-label classification, uses binary cross-entropy style.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        reduction: str = 'batchmean',
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence between student and teacher logits.
        
        Args:
            student_logits: [batch, num_classes]
            teacher_logits: [batch, num_classes]
            mask: [batch, num_classes] optional mask for valid labels
        """
        # Scale by temperature
        student_soft = torch.sigmoid(student_logits / self.temperature)
        teacher_soft = torch.sigmoid(teacher_logits / self.temperature)
        
        # Binary cross-entropy as KL approximation for multi-label
        loss = F.binary_cross_entropy(
            student_soft, teacher_soft, reduction='none'
        )
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        # Scale by temperature squared (standard practice)
        loss = loss * (self.temperature ** 2)
        
        return loss


class FeatureDistillationLoss(nn.Module):
    """
    L2 loss on intermediate feature embeddings.
    
    Aligns student representations with teacher representations.
    """
    
    def __init__(
        self,
        normalize: bool = True,
        projector_dim: Optional[int] = None,
        student_dim: int = 256,
        teacher_dim: int = 768,
    ):
        super().__init__()
        self.normalize = normalize
        
        # Projector to align dimensions if needed
        if projector_dim is not None or student_dim != teacher_dim:
            proj_dim = projector_dim or teacher_dim
            self.student_projector = nn.Sequential(
                nn.Linear(student_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, teacher_dim),
            )
        else:
            self.student_projector = nn.Identity()
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute L2 distance between features.
        
        Args:
            student_features: [batch, dim] or [batch, seq, dim]
            teacher_features: [batch, dim] or [batch, seq, dim]
            mask: [batch] or [batch, seq] for valid positions
        """
        # Project student features
        student_features = self.student_projector(student_features)
        
        # Normalize if requested
        if self.normalize:
            student_features = F.normalize(student_features, p=2, dim=-1)
            teacher_features = F.normalize(teacher_features, p=2, dim=-1)
        
        # Compute MSE
        loss = F.mse_loss(student_features, teacher_features, reduction='none')
        
        # Mean over feature dimension
        loss = loss.mean(dim=-1)
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return loss


class AttentionDistillationLoss(nn.Module):
    """
    MSE loss on attention maps.
    
    Aligns student attention patterns with teacher attention patterns.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        student_attention: List[torch.Tensor],
        teacher_attention: List[torch.Tensor],
        layer_weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Compute attention distillation loss.
        
        Args:
            student_attention: List of [batch, heads, seq, seq] tensors
            teacher_attention: List of [batch, heads, seq, seq] tensors
            layer_weights: Optional per-layer weights
        """
        if not student_attention or not teacher_attention:
            return torch.tensor(0.0)
        
        num_layers = min(len(student_attention), len(teacher_attention))
        
        if layer_weights is None:
            layer_weights = [1.0] * num_layers
        
        total_loss = 0.0
        
        for i in range(num_layers):
            student_attn = student_attention[i]
            teacher_attn = teacher_attention[i]
            
            # Handle head dimension mismatch by averaging
            if student_attn.shape[1] != teacher_attn.shape[1]:
                # Average over heads
                student_attn = student_attn.mean(dim=1, keepdim=True)
                teacher_attn = teacher_attn.mean(dim=1, keepdim=True)
            
            # Handle sequence length mismatch
            min_seq = min(student_attn.shape[-1], teacher_attn.shape[-1])
            student_attn = student_attn[:, :, :min_seq, :min_seq]
            teacher_attn = teacher_attn[:, :, :min_seq, :min_seq]
            
            layer_loss = F.mse_loss(student_attn, teacher_attn)
            total_loss += layer_weights[i] * layer_loss
        
        return total_loss / num_layers


class HierarchyLoss(nn.Module):
    """
    Loss to enforce disease hierarchy constraints.
    
    E.g., P(LRTI) >= P(pneumonia), P(LRTI) >= P(bronchitis), etc.
    """
    
    def __init__(
        self,
        hierarchy: Dict[str, List[str]],
        disease_names: List[str],
        margin: float = 0.0,
    ):
        super().__init__()
        self.margin = margin
        
        # Build parent-child index pairs
        self.constraints = []
        for parent, children in hierarchy.items():
            if parent in disease_names:
                parent_idx = disease_names.index(parent)
                for child in children:
                    if child in disease_names:
                        child_idx = disease_names.index(child)
                        self.constraints.append((parent_idx, child_idx))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute hierarchy constraint loss.
        
        Args:
            logits: [batch, num_diseases]
        """
        if not self.constraints:
            return torch.tensor(0.0, device=logits.device)
        
        probs = torch.sigmoid(logits)
        
        total_loss = 0.0
        for parent_idx, child_idx in self.constraints:
            parent_prob = probs[:, parent_idx]
            child_prob = probs[:, child_idx]
            
            # Parent should be >= child + margin
            violation = F.relu(child_prob - parent_prob + self.margin)
            total_loss += violation.mean()
        
        return total_loss / len(self.constraints)


class CombinedDistillationLoss(nn.Module):
    """
    Combined distillation loss with all components.
    
    Combines:
    - Logit KL divergence
    - Feature L2 matching
    - Attention matching (optional)
    - Hard label loss
    - Hierarchy constraints
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        logit_weight: float = 1.0,
        feature_weight: float = 0.5,
        attention_weight: float = 0.0,
        hard_label_weight: float = 0.0,
        concept_weight: float = 0.5,
        hierarchy_weight: float = 0.1,
        gate_entropy_weight: float = 0.05,
        student_dim: int = 256,
        teacher_dim: int = 768,
        hierarchy: Optional[Dict[str, List[str]]] = None,
        disease_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.logit_weight = logit_weight
        self.feature_weight = feature_weight
        self.attention_weight = attention_weight
        self.hard_label_weight = hard_label_weight
        self.concept_weight = concept_weight
        self.hierarchy_weight = hierarchy_weight
        self.gate_entropy_weight = gate_entropy_weight
        
        # Component losses
        self.logit_loss = LogitDistillationLoss(temperature=temperature)
        self.feature_loss = FeatureDistillationLoss(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
        )
        self.attention_loss = AttentionDistillationLoss()
        
        if hierarchy and disease_names:
            self.hierarchy_loss = HierarchyLoss(hierarchy, disease_names)
        else:
            self.hierarchy_loss = None
    
    def forward(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Dict[str, torch.Tensor],
        disease_labels: Optional[torch.Tensor] = None,
        disease_mask: Optional[torch.Tensor] = None,
        concept_labels: Optional[torch.Tensor] = None,
        concept_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation loss.
        
        Returns dict with individual losses and total.
        """
        losses = {}
        device = student_output['disease_logits'].device
        
        # Logit distillation (disease)
        losses['kl_disease'] = self.logit_loss(
            student_output['disease_logits'],
            teacher_output['disease_logits'],
        )
        
        # Logit distillation (concepts)
        losses['kl_concept'] = self.logit_loss(
            student_output['concept_logits'],
            teacher_output['concept_logits'],
        )
        
        # Feature distillation (CLS)
        losses['feature_cls'] = self.feature_loss(
            student_output['cls_embedding'],
            teacher_output['cls_embedding'],
        )
        
        # Feature distillation (tokens)
        if 'token_embeddings' in student_output and 'token_embeddings' in teacher_output:
            student_tokens = student_output['token_embeddings']
            teacher_tokens = teacher_output['token_embeddings']
            
            # Handle sequence length mismatch
            min_len = min(student_tokens.shape[1], teacher_tokens.shape[1])
            losses['feature_tokens'] = self.feature_loss(
                student_tokens[:, :min_len],
                teacher_tokens[:, :min_len],
            )
        else:
            losses['feature_tokens'] = torch.tensor(0.0, device=device)
        
        # Attention distillation (optional)
        if self.attention_weight > 0:
            student_attn = student_output.get('attention_weights', [])
            teacher_attn = teacher_output.get('attention_weights', [])
            losses['attention'] = self.attention_loss(student_attn, teacher_attn)
        else:
            losses['attention'] = torch.tensor(0.0, device=device)
        
        # Hard label loss (disease)
        if self.hard_label_weight > 0 and disease_labels is not None:
            hard_disease_loss = F.binary_cross_entropy_with_logits(
                student_output['disease_logits'],
                disease_labels,
                reduction='none',
            )
            if disease_mask is not None:
                hard_disease_loss = (hard_disease_loss * disease_mask).sum() / disease_mask.sum().clamp(min=1)
            else:
                hard_disease_loss = hard_disease_loss.mean()
            losses['hard_disease'] = hard_disease_loss
        else:
            losses['hard_disease'] = torch.tensor(0.0, device=device)
        
        # Hard label loss (concepts)
        if self.hard_label_weight > 0 and concept_labels is not None:
            hard_concept_loss = F.binary_cross_entropy_with_logits(
                student_output['concept_logits'],
                concept_labels,
                reduction='none',
            )
            if concept_mask is not None:
                hard_concept_loss = (hard_concept_loss * concept_mask).sum() / concept_mask.sum().clamp(min=1)
            else:
                hard_concept_loss = hard_concept_loss.mean()
            losses['hard_concept'] = hard_concept_loss
        else:
            losses['hard_concept'] = torch.tensor(0.0, device=device)
        
        # Hierarchy constraint
        if self.hierarchy_loss is not None:
            losses['hierarchy'] = self.hierarchy_loss(student_output['disease_logits'])
        else:
            losses['hierarchy'] = torch.tensor(0.0, device=device)
        
        # Gate entropy penalty
        losses['gate_entropy'] = student_output.get(
            'gate_entropy_penalty', torch.tensor(0.0, device=device)
        )
        
        # Total weighted loss
        total = (
            self.logit_weight * (losses['kl_disease'] + losses['kl_concept']) +
            self.feature_weight * (losses['feature_cls'] + losses['feature_tokens']) +
            self.attention_weight * losses['attention'] +
            self.hard_label_weight * (losses['hard_disease'] + self.concept_weight * losses['hard_concept']) +
            self.hierarchy_weight * losses['hierarchy'] +
            self.gate_entropy_weight * losses['gate_entropy']
        )
        
        losses['total'] = total
        
        return losses


class DistillationLoss(CombinedDistillationLoss):
    """Alias for CombinedDistillationLoss for backward compatibility."""
    pass

