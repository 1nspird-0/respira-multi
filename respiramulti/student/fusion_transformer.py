"""
Fusion Transformer for multimodal integration.

Fuses audio tokens and vitals using cross-modal attention and gated fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) for better position modeling."""
    
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the dimensions."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class FusionAttention(nn.Module):
    """Multi-head attention with optional rotary embeddings."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rotary: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.use_rotary = use_rotary
        if use_rotary:
            self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if self.use_rotary:
            cos, sin = self.rotary_emb(query)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn_weights
        return out, None


class FusionTransformerLayer(nn.Module):
    """Single transformer layer for fusion."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        use_rotary: bool = True,
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = FusionAttention(d_model, num_heads, dropout, use_rotary)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(x, x, x, mask, return_attention)
        x = residual + self.dropout(attn_out)
        
        # Pre-norm feed-forward
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        
        return x, attn_weights


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for interpretable modality weighting.
    
    Learns to weight audio vs vitals contributions.
    """
    
    def __init__(
        self,
        d_model: int,
        num_modalities: int = 2,
        entropy_penalty_weight: float = 0.05,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.entropy_penalty_weight = entropy_penalty_weight
        
        # Gate computation from fused representation
        self.gate_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_modalities),
        )
        
        # Temperature for gate sharpness
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        fused_embedding: torch.Tensor,
        modality_embeddings: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply gated fusion.
        
        Args:
            fused_embedding: [batch, d_model] from transformer
            modality_embeddings: Dict of modality -> [batch, d_model] embeddings
            
        Returns:
            gated_output: [batch, d_model]
            gate_weights: Dict of modality -> [batch, 1] weights
            entropy_penalty: scalar penalty for regularization
        """
        # Compute gates
        gate_logits = self.gate_network(fused_embedding)  # [batch, num_modalities]
        gate_weights = F.softmax(gate_logits / self.temperature.clamp(min=0.1), dim=-1)
        
        # Apply gates to modality embeddings
        modality_list = list(modality_embeddings.keys())
        weighted_sum = torch.zeros_like(fused_embedding)
        gate_dict = {}
        
        for i, modality in enumerate(modality_list):
            if i < gate_weights.shape[1]:
                weight = gate_weights[:, i:i+1]
                weighted_sum = weighted_sum + weight * modality_embeddings[modality]
                gate_dict[modality] = weight
        
        # Combine with residual
        output = fused_embedding + weighted_sum
        
        # Entropy penalty (encourage decisive gates)
        entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=-1).mean()
        entropy_penalty = self.entropy_penalty_weight * entropy
        
        return output, gate_dict, entropy_penalty


class FusionTransformer(nn.Module):
    """
    Fusion Transformer that combines audio tokens and vitals.
    
    Uses [CLS] token to aggregate information across modalities.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        use_rotary: bool = True,
        use_gated_fusion: bool = True,
        gate_entropy_penalty: float = 0.05,
        max_audio_tokens: int = 20,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Modality type embeddings
        self.modality_embeddings = nn.Embedding(3, d_model)  # audio, vitals, cls
        
        # Transformer layers
        self.layers = nn.ModuleList([
            FusionTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                use_rotary=use_rotary,
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # Gated fusion
        self.use_gated_fusion = use_gated_fusion
        if use_gated_fusion:
            self.gated_fusion = GatedFusion(
                d_model=d_model,
                num_modalities=2,
                entropy_penalty_weight=gate_entropy_penalty,
            )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        audio_tokens: torch.Tensor,
        vitals_token: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            audio_tokens: [batch, num_tokens, d_model]
            vitals_token: [batch, d_model]
            audio_mask: [batch, num_tokens] - True for padded positions
        """
        batch_size = audio_tokens.shape[0]
        
        # Prepare CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Prepare vitals token
        vitals_tokens = vitals_token.unsqueeze(1)  # [batch, 1, d_model]
        
        # Add modality embeddings
        # 0 = CLS, 1 = audio, 2 = vitals
        audio_mod_emb = self.modality_embeddings(
            torch.ones(batch_size, audio_tokens.shape[1], dtype=torch.long, device=audio_tokens.device)
        )
        vitals_mod_emb = self.modality_embeddings(
            torch.full((batch_size, 1), 2, dtype=torch.long, device=audio_tokens.device)
        )
        cls_mod_emb = self.modality_embeddings(
            torch.zeros(batch_size, 1, dtype=torch.long, device=audio_tokens.device)
        )
        
        audio_tokens = audio_tokens + audio_mod_emb
        vitals_tokens = vitals_tokens + vitals_mod_emb
        cls_tokens = cls_tokens + cls_mod_emb
        
        # Concatenate: [CLS, audio_tokens, vitals]
        x = torch.cat([cls_tokens, audio_tokens, vitals_tokens], dim=1)
        
        # Build attention mask
        if audio_mask is not None:
            # Extend mask for CLS and vitals (both should attend)
            cls_mask = torch.zeros(batch_size, 1, device=audio_mask.device, dtype=torch.bool)
            vitals_mask = torch.zeros(batch_size, 1, device=audio_mask.device, dtype=torch.bool)
            full_mask = torch.cat([cls_mask, audio_mask, vitals_mask], dim=1)
        else:
            full_mask = None
        
        # Transformer layers
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, mask=full_mask, return_attention=return_attention)
            if return_attention and attn is not None:
                attention_weights.append(attn)
        
        x = self.final_norm(x)
        
        # Extract outputs
        cls_output = x[:, 0]  # [batch, d_model]
        audio_output = x[:, 1:-1]  # [batch, num_audio_tokens, d_model]
        vitals_output = x[:, -1]  # [batch, d_model]
        
        # Apply gated fusion
        gate_weights = None
        entropy_penalty = torch.tensor(0.0, device=x.device)
        
        if self.use_gated_fusion:
            modality_embeddings = {
                'audio': audio_output.mean(dim=1),
                'vitals': vitals_output,
            }
            cls_output, gate_weights, entropy_penalty = self.gated_fusion(
                cls_output, modality_embeddings
            )
        
        # Final projection
        cls_output = self.output_proj(cls_output)
        
        result = {
            'cls_embedding': cls_output,
            'audio_embeddings': audio_output,
            'vitals_embedding': vitals_output,
            'gate_entropy_penalty': entropy_penalty,
        }
        
        if gate_weights is not None:
            result['gate_weights'] = gate_weights
        
        if return_attention:
            result['attention_weights'] = attention_weights
        
        return result

