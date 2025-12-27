"""
BEATs (Bidirectional Encoder representation from Audio Transformers) Teacher.

BEATs is a self-supervised audio representation learning model that uses
iterative audio tokenization for acoustic unit discovery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class MultiheadAttention(nn.Module):
    """Multi-head attention with relative positional encoding."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: [batch, seq_len, embed_dim]
            key: [batch, seq_len, embed_dim]
            value: [batch, seq_len, embed_dim]
            key_padding_mask: [batch, seq_len] - True for padded positions
        """
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            return attn_output, attn_weights
        return attn_output, None


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            src: [batch, seq_len, d_model]
        """
        # Pre-norm self-attention
        x = self.norm1(src)
        x, attn_weights = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask, need_weights=need_weights)
        src = src + self.dropout1(x)
        
        # Pre-norm feedforward
        x = self.norm2(src)
        x = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        src = src + self.dropout3(x)
        
        return src, attn_weights


class BEATsEncoder(nn.Module):
    """
    BEATs-style audio encoder.
    
    Uses a CNN frontend followed by transformer layers.
    """
    
    def __init__(
        self,
        input_dim: int = 64,  # n_mels
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        conv_bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Convolutional feature extractor (similar to wav2vec2)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, 512, kernel_size=10, stride=5, bias=conv_bias),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=conv_bias),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=conv_bias),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=conv_bias),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=2, stride=2, bias=conv_bias),
                nn.GELU(),
            ),
        ])
        
        # Project to embed_dim
        self.post_extract_proj = nn.Linear(512, embed_dim)
        
        # Positional encoding
        self.pos_conv = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=128, padding=64, groups=16,
        )
        self.pos_conv_norm = nn.LayerNorm(embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, channels, n_mels, time] or [batch, n_mels, time]
            padding_mask: [batch, time] - True for padded positions
            
        Returns:
            Dictionary with:
            - 'cls_embedding': [batch, embed_dim]
            - 'token_embeddings': [batch, seq_len, embed_dim]
            - 'attention_weights': list of [batch, heads, seq, seq] (if requested)
            - 'all_layers': list of layer outputs (if return_all_layers)
        """
        # Handle 4D input
        if x.dim() == 4:
            x = x.squeeze(1)  # [batch, n_mels, time]
        
        # CNN feature extraction
        for conv in self.conv_layers:
            x = conv(x)
        
        # Project and transpose
        x = x.transpose(1, 2)  # [batch, seq_len, 512]
        x = self.post_extract_proj(x)  # [batch, seq_len, embed_dim]
        
        # Add positional encoding
        pos_conv = self.pos_conv(x.transpose(1, 2))
        pos_conv = pos_conv[:, :, :x.size(1)].transpose(1, 2)
        x = x + self.pos_conv_norm(pos_conv)
        
        # Add CLS token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Update padding mask for CLS token
        if padding_mask is not None:
            padding_mask = F.pad(padding_mask, (1, 0), value=False)
        
        # Transformer layers
        all_layers = []
        attention_weights = []
        
        for layer in self.layers:
            x, attn = layer(x, src_key_padding_mask=padding_mask, need_weights=return_all_layers)
            if return_all_layers:
                all_layers.append(x)
                if attn is not None:
                    attention_weights.append(attn)
        
        x = self.layer_norm(x)
        
        result = {
            'cls_embedding': x[:, 0],  # CLS token
            'token_embeddings': x[:, 1:],  # Exclude CLS
        }
        
        if return_all_layers:
            result['all_layers'] = all_layers
            result['attention_weights'] = attention_weights
        
        return result


class BEATsTeacher(nn.Module):
    """
    Complete BEATs teacher model with classification heads.
    
    Used for generating soft labels for distillation.
    """
    
    def __init__(
        self,
        num_diseases: int = 12,
        num_concepts: int = 17,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        
        # Audio encoder
        self.encoder = BEATsEncoder(
            input_dim=64,
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
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, path: str):
        """Load pretrained weights."""
        try:
            state_dict = torch.load(path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded BEATs checkpoint from {path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = True,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for teacher inference.
        
        Args:
            x: [batch, 1, n_mels, time] mel spectrogram
            padding_mask: [batch, time]
            return_embeddings: Whether to return intermediate embeddings
            return_attention: Whether to return attention weights
        """
        # Encode
        encoder_output = self.encoder(
            x, 
            padding_mask=padding_mask,
            return_all_layers=return_attention,
        )
        
        cls_embedding = encoder_output['cls_embedding']
        token_embeddings = encoder_output['token_embeddings']
        
        # Predict
        disease_logits = self.disease_head(cls_embedding)
        concept_logits = self.concept_head(cls_embedding)
        
        result = {
            'disease_logits': disease_logits,
            'concept_logits': concept_logits,
        }
        
        if return_embeddings:
            result['cls_embedding'] = cls_embedding
            result['token_embeddings'] = token_embeddings
        
        if return_attention and 'attention_weights' in encoder_output:
            result['attention_weights'] = encoder_output['attention_weights']
        
        return result
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get CLS and token embeddings for distillation."""
        encoder_output = self.encoder(x, padding_mask=padding_mask)
        return encoder_output['cls_embedding'], encoder_output['token_embeddings']

