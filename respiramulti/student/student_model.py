"""
Complete Student Model for RESPIRA-MULTI.

Combines audio encoder, conformer, vitals encoder, and fusion transformer
into a unified model optimized for mobile deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from respiramulti.student.audio_encoder import MobileAudioEncoder, AudioTokenizer
from respiramulti.student.conformer import LightweightConformer
from respiramulti.student.fusion_transformer import FusionTransformer
from respiramulti.student.vitals_encoder import VitalsEncoder
from respiramulti.datasets.schema import DISEASES, CONCEPTS, BINARY_CONCEPTS


@dataclass
class StudentOutput:
    """Output from student model."""
    # Predictions
    disease_logits: torch.Tensor  # [batch, num_diseases]
    concept_logits: torch.Tensor  # [batch, num_concepts]
    
    # Embeddings (for distillation)
    cls_embedding: torch.Tensor  # [batch, d_model]
    token_embeddings: torch.Tensor  # [batch, num_tokens, d_model]
    
    # Interpretability
    gate_weights: Optional[Dict[str, torch.Tensor]] = None
    attention_weights: Optional[List[torch.Tensor]] = None
    concept_contributions: Optional[torch.Tensor] = None
    
    # Losses
    gate_entropy_penalty: torch.Tensor = None


class ConceptBottleneck(nn.Module):
    """
    Concept bottleneck layer for interpretable predictions.
    
    Forces disease predictions to go through concept predictions,
    with an optional residual path.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_concepts: int = 17,
        num_diseases: int = 12,
        concept_dim: int = 128,
        residual_dim: int = 128,
        residual_weight: float = 0.3,
    ):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.num_diseases = num_diseases
        self.residual_weight = residual_weight
        
        # Concept prediction
        self.concept_head = nn.Sequential(
            nn.Linear(d_model, concept_dim),
            nn.GELU(),
            nn.Linear(concept_dim, num_concepts),
        )
        
        # Concept-to-disease mapping
        self.concept_to_disease = nn.Linear(num_concepts, num_diseases)
        
        # Residual path (for information not captured by concepts)
        self.residual_path = nn.Sequential(
            nn.Linear(d_model, residual_dim),
            nn.GELU(),
            nn.Linear(residual_dim, num_diseases),
        )
        
        # Learnable balance between concept and residual paths
        self.balance = nn.Parameter(torch.tensor(residual_weight))
    
    def forward(
        self,
        embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            embedding: [batch, d_model]
            
        Returns:
            disease_logits: [batch, num_diseases]
            concept_logits: [batch, num_concepts]
            concept_contributions: [batch, num_concepts] per-concept contribution to disease pred
        """
        # Predict concepts
        concept_logits = self.concept_head(embedding)
        
        # Get concept activations (probabilities for binary, values for continuous)
        concept_probs = torch.sigmoid(concept_logits)
        
        # Disease prediction from concepts
        disease_from_concepts = self.concept_to_disease(concept_probs)
        
        # Residual disease prediction
        disease_residual = self.residual_path(embedding)
        
        # Combine with learned balance
        balance = torch.sigmoid(self.balance)
        disease_logits = (1 - balance) * disease_from_concepts + balance * disease_residual
        
        # Compute concept contributions (for interpretability)
        # How much each concept contributes to disease predictions
        concept_weights = self.concept_to_disease.weight  # [num_diseases, num_concepts]
        concept_contributions = concept_probs.unsqueeze(1) * concept_weights.unsqueeze(0)  # [batch, diseases, concepts]
        
        return disease_logits, concept_logits, concept_contributions


class DiseaseHead(nn.Module):
    """
    Disease classification head with hierarchical constraints.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_diseases: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_diseases),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ConceptHead(nn.Module):
    """
    Concept prediction head for interpretable features.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_binary_concepts: int = 5,
        num_continuous_concepts: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_binary = num_binary_concepts
        self.num_continuous = num_continuous_concepts
        
        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Binary concept head (sigmoid activation)
        self.binary_head = nn.Linear(hidden_dim, num_binary_concepts)
        
        # Continuous concept head (no activation, or softplus for positive)
        self.continuous_head = nn.Linear(hidden_dim, num_continuous_concepts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared(x)
        
        binary_logits = self.binary_head(features)
        continuous_pred = self.continuous_head(features)
        
        # Concatenate (binary first, then continuous)
        return torch.cat([binary_logits, continuous_pred], dim=-1)


class RespiraMultiStudent(nn.Module):
    """
    Complete student model for RESPIRA-MULTI.
    
    Architecture:
    1. Audio Encoder (MobileNetV3) processes each spectrogram segment
    2. Conformer refines audio token sequences
    3. Vitals Encoder processes vital signs with missingness handling
    4. Fusion Transformer combines all modalities
    5. Concept Bottleneck for interpretable disease predictions
    """
    
    def __init__(
        self,
        # Audio encoder
        audio_embedding_dim: int = 256,
        audio_width_mult: float = 1.0,
        
        # Conformer
        use_conformer: bool = True,
        conformer_layers: int = 2,
        conformer_kernel_size: int = 15,
        
        # Vitals encoder
        vitals_input_dim: int = 15,
        vitals_hidden_dims: list = [128, 128],
        vitals_output_dim: int = 128,
        
        # Fusion transformer
        d_model: int = 256,
        fusion_layers: int = 4,
        fusion_heads: int = 4,
        fusion_ff_dim: int = 512,
        use_gated_fusion: bool = True,
        gate_entropy_penalty: float = 0.05,
        
        # Heads
        num_diseases: int = 12,
        num_concepts: int = 17,
        use_concept_bottleneck: bool = True,
        concept_residual_weight: float = 0.3,
        
        # General
        dropout: float = 0.1,
        max_audio_tokens: int = 20,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_diseases = num_diseases
        self.num_concepts = num_concepts
        self.use_concept_bottleneck = use_concept_bottleneck
        
        # Audio encoder
        self.audio_encoder = MobileAudioEncoder(
            in_channels=1,
            embedding_dim=audio_embedding_dim,
            width_mult=audio_width_mult,
        )
        
        # Audio tokenizer
        self.audio_tokenizer = AudioTokenizer(
            embedding_dim=audio_embedding_dim,
            num_segment_types=6,
        )
        
        # Optional conformer
        self.use_conformer = use_conformer
        if use_conformer:
            self.conformer = LightweightConformer(
                input_dim=audio_embedding_dim,
                d_model=d_model,
                num_heads=fusion_heads,
                num_layers=conformer_layers,
                conv_kernel_size=conformer_kernel_size,
                dropout=dropout,
            )
            audio_out_dim = d_model
        else:
            audio_out_dim = audio_embedding_dim
        
        # Project audio to d_model if needed
        self.audio_proj = nn.Linear(audio_out_dim, d_model) if audio_out_dim != d_model else nn.Identity()
        
        # Vitals encoder
        self.vitals_encoder = VitalsEncoder(
            input_dim=vitals_input_dim,
            hidden_dims=vitals_hidden_dims,
            output_dim=vitals_output_dim,
            dropout=dropout,
        )
        
        # Project vitals to d_model
        self.vitals_proj = nn.Linear(vitals_output_dim, d_model)
        
        # Fusion transformer
        self.fusion = FusionTransformer(
            d_model=d_model,
            num_layers=fusion_layers,
            num_heads=fusion_heads,
            ff_dim=fusion_ff_dim,
            dropout=dropout,
            use_gated_fusion=use_gated_fusion,
            gate_entropy_penalty=gate_entropy_penalty,
            max_audio_tokens=max_audio_tokens,
        )
        
        # Prediction heads
        if use_concept_bottleneck:
            self.concept_bottleneck = ConceptBottleneck(
                d_model=d_model,
                num_concepts=num_concepts,
                num_diseases=num_diseases,
                residual_weight=concept_residual_weight,
            )
        else:
            self.disease_head = DiseaseHead(
                d_model=d_model,
                num_diseases=num_diseases,
                dropout=dropout,
            )
            self.concept_head = ConceptHead(
                d_model=d_model,
                num_binary_concepts=len(BINARY_CONCEPTS),
                num_continuous_concepts=num_concepts - len(BINARY_CONCEPTS),
                dropout=dropout,
            )
    
    def encode_audio_tokens(
        self,
        audio_tokens: torch.Tensor,
        segment_types: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio spectrograms to token embeddings.
        
        Args:
            audio_tokens: [batch, num_tokens, 1, n_mels, time]
            segment_types: [batch, num_tokens]
            audio_mask: [batch, num_tokens]
        """
        batch_size, num_tokens = audio_tokens.shape[:2]
        
        # Flatten batch and tokens for efficient processing
        flat_tokens = audio_tokens.view(-1, *audio_tokens.shape[2:])  # [batch*tokens, 1, mels, time]
        
        # Encode each token
        encoder_output = self.audio_encoder(flat_tokens)
        embeddings = encoder_output['embedding']  # [batch*tokens, embed_dim]
        
        # Reshape back
        embeddings = embeddings.view(batch_size, num_tokens, -1)  # [batch, tokens, embed_dim]
        
        # Add segment type embeddings
        embeddings = self.audio_tokenizer(embeddings, segment_types)
        
        # Apply conformer if enabled
        if self.use_conformer:
            conformer_mask = audio_mask if audio_mask is not None else None
            conformer_out = self.conformer(embeddings, mask=conformer_mask)
            embeddings = conformer_out['output']
        
        # Project to d_model
        embeddings = self.audio_proj(embeddings)
        
        return embeddings
    
    def encode_vitals(
        self,
        vitals: torch.Tensor,
        vitals_mask: Optional[torch.Tensor] = None,
        demographics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode vitals to embedding.
        
        Args:
            vitals: [batch, vitals_dim]
            vitals_mask: [batch, vitals_dim]
            demographics: [batch, demo_dim]
        """
        vitals_output = self.vitals_encoder(vitals, vitals_mask, demographics)
        vitals_embedding = vitals_output['embedding']
        
        # Project to d_model
        vitals_embedding = self.vitals_proj(vitals_embedding)
        
        return vitals_embedding
    
    def forward(
        self,
        audio_tokens: torch.Tensor,
        segment_types: torch.Tensor,
        vitals: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        vitals_mask: Optional[torch.Tensor] = None,
        demographics: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> StudentOutput:
        """
        Full forward pass.
        
        Args:
            audio_tokens: [batch, num_tokens, 1, n_mels, time]
            segment_types: [batch, num_tokens]
            vitals: [batch, vitals_dim]
            audio_mask: [batch, num_tokens] - True for padded
            vitals_mask: [batch, vitals_dim] - 1 for present, 0 for missing
            demographics: [batch, demo_dim]
        """
        # Encode audio
        audio_embeddings = self.encode_audio_tokens(
            audio_tokens, segment_types, audio_mask
        )
        
        # Encode vitals
        vitals_embedding = self.encode_vitals(vitals, vitals_mask, demographics)
        
        # Fuse modalities
        fusion_output = self.fusion(
            audio_tokens=audio_embeddings,
            vitals_token=vitals_embedding,
            audio_mask=audio_mask,
            return_attention=return_attention,
        )
        
        cls_embedding = fusion_output['cls_embedding']
        gate_entropy_penalty = fusion_output['gate_entropy_penalty']
        
        # Predict diseases and concepts
        if self.use_concept_bottleneck:
            disease_logits, concept_logits, concept_contributions = self.concept_bottleneck(
                cls_embedding
            )
        else:
            disease_logits = self.disease_head(cls_embedding)
            concept_logits = self.concept_head(cls_embedding)
            concept_contributions = None
        
        return StudentOutput(
            disease_logits=disease_logits,
            concept_logits=concept_logits,
            cls_embedding=cls_embedding,
            token_embeddings=fusion_output['audio_embeddings'],
            gate_weights=fusion_output.get('gate_weights'),
            attention_weights=fusion_output.get('attention_weights'),
            concept_contributions=concept_contributions,
            gate_entropy_penalty=gate_entropy_penalty,
        )
    
    def get_embeddings(
        self,
        audio_tokens: torch.Tensor,
        segment_types: torch.Tensor,
        vitals: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        vitals_mask: Optional[torch.Tensor] = None,
        demographics: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get embeddings for distillation loss computation.
        """
        output = self.forward(
            audio_tokens, segment_types, vitals,
            audio_mask, vitals_mask, demographics,
        )
        return output.cls_embedding, output.token_embeddings
    
    @torch.no_grad()
    def predict(
        self,
        audio_tokens: torch.Tensor,
        segment_types: torch.Tensor,
        vitals: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        vitals_mask: Optional[torch.Tensor] = None,
        demographics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-time prediction with probabilities.
        """
        output = self.forward(
            audio_tokens, segment_types, vitals,
            audio_mask, vitals_mask, demographics,
        )
        
        disease_probs = torch.sigmoid(output.disease_logits)
        concept_probs = torch.sigmoid(output.concept_logits)
        
        return {
            'disease_probs': disease_probs,
            'concept_probs': concept_probs,
            'disease_names': DISEASES,
            'concept_names': CONCEPTS,
            'gate_weights': output.gate_weights,
        }


def create_student_model(config: Dict) -> RespiraMultiStudent:
    """Create student model from configuration."""
    model_config = config.get('student', config.get('model', {}))
    
    return RespiraMultiStudent(
        audio_embedding_dim=model_config.get('audio_encoder', {}).get('embedding_dim', 256),
        audio_width_mult=model_config.get('audio_encoder', {}).get('width_mult', 1.0),
        use_conformer=model_config.get('conformer', {}).get('enabled', True),
        conformer_layers=model_config.get('conformer', {}).get('num_layers', 2),
        conformer_kernel_size=model_config.get('conformer', {}).get('conv_kernel_size', 15),
        vitals_input_dim=model_config.get('vitals_encoder', {}).get('input_dim', 15),
        vitals_hidden_dims=model_config.get('vitals_encoder', {}).get('hidden_dims', [128, 128]),
        vitals_output_dim=model_config.get('vitals_encoder', {}).get('output_dim', 128),
        d_model=model_config.get('fusion', {}).get('d_model', 256),
        fusion_layers=model_config.get('fusion', {}).get('num_layers', 4),
        fusion_heads=model_config.get('fusion', {}).get('num_heads', 4),
        fusion_ff_dim=model_config.get('fusion', {}).get('ff_dim', 512),
        use_gated_fusion=model_config.get('fusion', {}).get('use_gated_fusion', True),
        gate_entropy_penalty=model_config.get('fusion', {}).get('gate_entropy_penalty', 0.05),
        num_diseases=len(DISEASES),
        num_concepts=len(CONCEPTS),
        use_concept_bottleneck=model_config.get('concept_bottleneck', {}).get('enabled', True),
        concept_residual_weight=model_config.get('concept_bottleneck', {}).get('residual_weight', 0.3),
        dropout=model_config.get('dropout', 0.1),
    )

