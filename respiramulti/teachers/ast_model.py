"""
AST (Audio Spectrogram Transformer) Teacher.

AST is a pure attention model for audio classification that treats spectrograms
as sequences of patches, similar to ViT for images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class ASTAttention(nn.Module):
    """Multi-head self-attention for AST."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self, 
        x: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, dim = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if need_weights:
            return x, attn
        return x, None


class ASTMlp(nn.Module):
    """MLP block for AST."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ASTBlock(nn.Module):
    """Transformer block for AST."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ASTAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ASTMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.norm1(x), need_weights=need_weights)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class ASTPatchEmbed(nn.Module):
    """Patch embedding for AST with overlap."""
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (128, 1024),
        patch_size: Tuple[int, int] = (16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
        stride: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride or patch_size
        
        # Calculate number of patches with stride
        self.grid_size = (
            (img_size[0] - patch_size[0]) // self.stride[0] + 1,
            (img_size[1] - patch_size[1]) // self.stride[1] + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=self.stride,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [batch, embed_dim, h, w]
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        return x


class ASTEncoder(nn.Module):
    """
    Audio Spectrogram Transformer encoder.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 201),
        patch_size: Tuple[int, int] = (16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        
        # Patch embedding (with overlap for better representation)
        stride = (patch_size[0] // 2, patch_size[1] // 2)
        self.patch_embed = ASTPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            stride=stride,
        )
        self.num_patches = self.patch_embed.num_patches
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Distillation token (from DeiT)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ASTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def interpolate_pos_encoding(
        self, 
        x: torch.Tensor, 
        h: int, 
        w: int,
    ) -> torch.Tensor:
        """Interpolate positional encoding for different input sizes."""
        num_patches = x.shape[1] - 2  # Exclude CLS and dist tokens
        
        if num_patches == self.num_patches:
            return self.pos_embed
        
        # Separate class/dist tokens and patch embeddings
        class_pos_embed = self.pos_embed[:, :2]
        patch_pos_embed = self.pos_embed[:, 2:]
        
        # Reshape and interpolate
        dim = x.shape[-1]
        orig_h = self.patch_embed.grid_size[0]
        orig_w = self.patch_embed.grid_size[1]
        
        patch_pos_embed = patch_pos_embed.reshape(1, orig_h, orig_w, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(h, w), mode='bilinear', align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        
        return torch.cat([class_pos_embed, patch_pos_embed], dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_layers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, channels, height, width] spectrogram
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS and distillation tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        dist_tokens = self.dist_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        all_layers = []
        attention_weights = []
        
        for block in self.blocks:
            x, attn = block(x, need_weights=return_all_layers)
            if return_all_layers:
                all_layers.append(x)
                if attn is not None:
                    attention_weights.append(attn)
        
        x = self.norm(x)
        
        result = {
            'cls_embedding': x[:, 0],
            'dist_embedding': x[:, 1],
            'token_embeddings': x[:, 2:],
        }
        
        if return_all_layers:
            result['all_layers'] = all_layers
            result['attention_weights'] = attention_weights
        
        return result


class ASTTeacher(nn.Module):
    """
    Complete AST teacher model with classification heads.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 201),
        patch_size: Tuple[int, int] = (16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_diseases: int = 12,
        num_concepts: int = 17,
        drop_rate: float = 0.0,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        
        # Encoder
        self.encoder = ASTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )
        
        # Classification heads
        # Use both CLS and dist tokens for prediction
        self.disease_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, num_diseases),
        )
        
        self.concept_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
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
            print(f"Loaded AST checkpoint from {path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = True,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [batch, 1, n_mels, time] mel spectrogram
        """
        # Encode
        encoder_output = self.encoder(x, return_all_layers=return_attention)
        
        cls_embedding = encoder_output['cls_embedding']
        dist_embedding = encoder_output['dist_embedding']
        token_embeddings = encoder_output['token_embeddings']
        
        # Combine CLS and dist for classification
        combined = torch.cat([cls_embedding, dist_embedding], dim=-1)
        
        # Predict
        disease_logits = self.disease_head(combined)
        concept_logits = self.concept_head(combined)
        
        result = {
            'disease_logits': disease_logits,
            'concept_logits': concept_logits,
        }
        
        if return_embeddings:
            result['cls_embedding'] = cls_embedding
            result['dist_embedding'] = dist_embedding
            result['token_embeddings'] = token_embeddings
        
        if return_attention and 'attention_weights' in encoder_output:
            result['attention_weights'] = encoder_output['attention_weights']
        
        return result
    
    def get_embeddings(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for distillation."""
        encoder_output = self.encoder(x)
        # Average CLS and dist tokens
        cls_combined = (encoder_output['cls_embedding'] + encoder_output['dist_embedding']) / 2
        return cls_combined, encoder_output['token_embeddings']

