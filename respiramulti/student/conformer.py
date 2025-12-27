"""
Lightweight Conformer blocks for the student model.

Conformer combines convolution and self-attention for better local+global modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for Conformer attention."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create sinusoidal encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        return x + self.pe[:x.size(1)]


class ConvolutionModule(nn.Module):
    """
    Conformer convolution module.
    
    Applies depthwise separable convolution with gating.
    """
    
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
        expansion_factor: int = 2,
    ):
        super().__init__()
        
        inner_dim = d_model * expansion_factor
        
        # Pointwise conv + gated linear unit
        self.pointwise_conv1 = nn.Linear(d_model, inner_dim * 2)
        
        # Depthwise conv
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            inner_dim, inner_dim, kernel_size,
            padding=padding, groups=inner_dim, bias=False
        )
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        
        # Pointwise conv back
        self.pointwise_conv2 = nn.Linear(inner_dim, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        # Pointwise + GLU
        x = self.pointwise_conv1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)
        
        # Depthwise conv
        x = x.transpose(1, 2)  # [batch, channels, seq_len]
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = x.transpose(1, 2)  # [batch, seq_len, channels]
        
        # Pointwise back
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
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
        
        # Relative position bias
        self.pos_bias = nn.Linear(d_model, num_heads, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            pos_emb: [seq_len, d_model] positional encoding
            mask: [batch, seq_len] padding mask
        """
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias if provided
        if pos_emb is not None:
            pos_bias = self.pos_bias(pos_emb)  # [seq_len, num_heads]
            attn = attn + pos_bias.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out, attn


class FeedForwardModule(nn.Module):
    """Feed-forward module with SwiGLU-like activation."""
    
    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        inner_dim = d_model * expansion_factor
        
        self.linear1 = nn.Linear(d_model, inner_dim * 2)
        self.linear2 = nn.Linear(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.silu(gate)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    """
    Single Conformer block.
    
    Structure: FFN/2 -> MHSA -> Conv -> FFN/2
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        conv_kernel_size: int = 31,
        ff_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # First FFN (half)
        self.ff1 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.ff1_norm = nn.LayerNorm(d_model)
        self.ff1_scale = 0.5
        
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Convolution module
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout, conv_expansion_factor)
        self.conv_norm = nn.LayerNorm(d_model)
        
        # Second FFN (half)
        self.ff2 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.ff2_norm = nn.LayerNorm(d_model)
        self.ff2_scale = 0.5
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            pos_emb: [seq_len, d_model]
            mask: [batch, seq_len]
        """
        # First FFN (half-step)
        residual = x
        x = self.ff1_norm(x)
        x = residual + self.ff1_scale * self.ff1(x)
        
        # Self-attention
        residual = x
        x = self.attn_norm(x)
        attn_out, attn_weights = self.attn(x, pos_emb, mask)
        x = residual + attn_out
        
        # Convolution
        residual = x
        x = self.conv_norm(x)
        x = residual + self.conv(x)
        
        # Second FFN (half-step)
        residual = x
        x = self.ff2_norm(x)
        x = residual + self.ff2_scale * self.ff2(x)
        
        # Final norm
        x = self.final_norm(x)
        
        return x, attn_weights


class LightweightConformer(nn.Module):
    """
    Lightweight Conformer encoder for mobile deployment.
    
    Uses fewer layers and smaller dimensions than standard Conformer.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        conv_kernel_size: int = 15,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()
        
        # Input projection if needed
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        
        # Positional encoding
        self.pos_encoding = RelativePositionalEncoding(d_model, max_len)
        
        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                ff_expansion_factor=ff_expansion_factor,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.d_model = d_model
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> dict:
        """
        Args:
            x: [batch, seq_len, input_dim]
            mask: [batch, seq_len] padding mask (True for padded)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        pos_emb = self.pos_encoding.pe[:x.size(1)]
        x = x + pos_emb
        
        # Conformer layers
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, pos_emb, mask)
            if return_attention:
                attention_weights.append(attn)
        
        result = {
            'output': x,
            'cls_embedding': x.mean(dim=1),  # Global average pooling
        }
        
        if return_attention:
            result['attention_weights'] = attention_weights
        
        return result

