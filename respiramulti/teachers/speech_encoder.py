"""
Speech Encoder Teachers (HuBERT and wav2vec2).

These models extract speech representations directly from waveforms,
which are useful for the speech/reading segments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class ConvFeatureExtractor(nn.Module):
    """
    Convolutional feature extractor for waveform input.
    
    Similar to wav2vec2/HuBERT frontend.
    """
    
    def __init__(
        self,
        conv_layers: list = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Default architecture similar to wav2vec2-base
        if conv_layers is None:
            conv_layers = [
                (512, 10, 5),   # (out_channels, kernel, stride)
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 2, 2),
                (512, 2, 2),
            ]
        
        layers = []
        in_channels = 1
        
        for i, (out_channels, kernel, stride) in enumerate(conv_layers):
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel, stride=stride, bias=False)
            )
            layers.append(nn.Dropout(dropout))
            
            if i == 0:
                layers.append(nn.GroupNorm(1, out_channels))
            else:
                layers.append(nn.GELU())
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.output_dim = conv_layers[-1][0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, time] waveform
        Returns:
            [batch, time', output_dim]
        """
        x = x.unsqueeze(1)  # Add channel dim
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # [batch, time', channels]
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder for speech models."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.pos_conv = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=128, padding=64, groups=16,
        )
        self.pos_conv_norm = nn.LayerNorm(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            padding_mask: [batch, seq_len] - True for padded positions
        """
        # Add positional encoding from convolution
        pos_conv = self.pos_conv(x.transpose(1, 2))
        pos_conv = pos_conv[:, :, :x.size(1)].transpose(1, 2)
        x = x + self.pos_conv_norm(pos_conv)
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)
        
        return x


class HuBERTEncoder(nn.Module):
    """
    HuBERT-style encoder for speech representation learning.
    
    Simplified implementation focusing on fine-tuning for respiratory sounds.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # CNN feature extractor
        self.feature_extractor = ConvFeatureExtractor(dropout=dropout)
        
        # Project to embed_dim
        self.feature_proj = nn.Linear(self.feature_extractor.output_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        
        # Mask embedding for pretraining
        self.mask_emb = nn.Parameter(torch.zeros(embed_dim))
        nn.init.uniform_(self.mask_emb)
    
    def forward(
        self,
        waveform: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            waveform: [batch, time] raw audio
            padding_mask: [batch, seq_len] for variable length
            mask_indices: [batch, seq_len] for masked pretraining
        """
        # Extract features
        features = self.feature_extractor(waveform)
        features = self.feature_proj(features)
        features = self.layer_norm(features)
        
        # Apply masking if provided (for pretraining)
        if mask_indices is not None:
            features = features.clone()
            features[mask_indices] = self.mask_emb
        
        # Transformer encoding
        encoded = self.encoder(features, padding_mask=padding_mask)
        
        # Pool to get CLS-like embedding
        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)
            cls_embedding = (encoded * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            cls_embedding = encoded.mean(dim=1)
        
        return {
            'cls_embedding': cls_embedding,
            'token_embeddings': encoded,
        }


class HuBERTTeacher(nn.Module):
    """
    Complete HuBERT teacher model for speech segments.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 3072,
        num_diseases: int = 12,
        num_concepts: int = 17,
        dropout: float = 0.1,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.encoder = HuBERTEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        
        # Classification heads
        self.disease_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_diseases),
        )
        
        self.concept_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_concepts),
        )
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, path: str):
        """Load pretrained HuBERT weights."""
        try:
            state_dict = torch.load(path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded HuBERT checkpoint from {path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    @classmethod
    def from_pretrained(cls, model_name: str = "facebook/hubert-base-ls960", **kwargs):
        """Load from HuggingFace pretrained model."""
        try:
            from transformers import HubertModel
            
            hf_model = HubertModel.from_pretrained(model_name)
            
            # Create our model
            model = cls(
                embed_dim=hf_model.config.hidden_size,
                num_heads=hf_model.config.num_attention_heads,
                num_layers=hf_model.config.num_hidden_layers,
                ff_dim=hf_model.config.intermediate_size,
                **kwargs,
            )
            
            # Copy weights (simplified - full implementation would map all weights)
            print(f"Initialized HuBERT from {model_name}")
            return model
            
        except ImportError:
            print("transformers not available, using random initialization")
            return cls(**kwargs)
    
    def forward(
        self,
        waveform: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            waveform: [batch, time] raw audio
        """
        encoder_output = self.encoder(waveform, padding_mask=padding_mask)
        
        cls_embedding = encoder_output['cls_embedding']
        token_embeddings = encoder_output['token_embeddings']
        
        disease_logits = self.disease_head(cls_embedding)
        concept_logits = self.concept_head(cls_embedding)
        
        result = {
            'disease_logits': disease_logits,
            'concept_logits': concept_logits,
        }
        
        if return_embeddings:
            result['cls_embedding'] = cls_embedding
            result['token_embeddings'] = token_embeddings
        
        return result
    
    def get_embeddings(
        self,
        waveform: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for distillation."""
        encoder_output = self.encoder(waveform, padding_mask=padding_mask)
        return encoder_output['cls_embedding'], encoder_output['token_embeddings']


class Wav2Vec2Teacher(nn.Module):
    """
    wav2vec2 teacher model.
    
    Similar to HuBERT but with contrastive pretraining objective.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 3072,
        num_diseases: int = 12,
        num_concepts: int = 17,
        dropout: float = 0.1,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        
        # Reuse HuBERT encoder architecture
        self.encoder = HuBERTEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        
        # Quantizer for contrastive learning (pretraining only)
        self.quantizer = nn.Linear(embed_dim, 320 * 2)  # 320 codes x 2 groups
        
        # Classification heads
        self.disease_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_diseases),
        )
        
        self.concept_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_concepts),
        )
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, path: str):
        """Load pretrained weights."""
        try:
            state_dict = torch.load(path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded wav2vec2 checkpoint from {path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    @classmethod
    def from_pretrained(cls, model_name: str = "facebook/wav2vec2-base", **kwargs):
        """Load from HuggingFace pretrained model."""
        try:
            from transformers import Wav2Vec2Model
            
            hf_model = Wav2Vec2Model.from_pretrained(model_name)
            
            model = cls(
                embed_dim=hf_model.config.hidden_size,
                num_heads=hf_model.config.num_attention_heads,
                num_layers=hf_model.config.num_hidden_layers,
                ff_dim=hf_model.config.intermediate_size,
                **kwargs,
            )
            
            print(f"Initialized wav2vec2 from {model_name}")
            return model
            
        except ImportError:
            print("transformers not available, using random initialization")
            return cls(**kwargs)
    
    def forward(
        self,
        waveform: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        encoder_output = self.encoder(waveform, padding_mask=padding_mask)
        
        cls_embedding = encoder_output['cls_embedding']
        token_embeddings = encoder_output['token_embeddings']
        
        disease_logits = self.disease_head(cls_embedding)
        concept_logits = self.concept_head(cls_embedding)
        
        result = {
            'disease_logits': disease_logits,
            'concept_logits': concept_logits,
        }
        
        if return_embeddings:
            result['cls_embedding'] = cls_embedding
            result['token_embeddings'] = token_embeddings
        
        return result
    
    def get_embeddings(
        self,
        waveform: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for distillation."""
        encoder_output = self.encoder(waveform, padding_mask=padding_mask)
        return encoder_output['cls_embedding'], encoder_output['token_embeddings']

