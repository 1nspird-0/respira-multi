"""
Audio-MAE (Masked Autoencoder for Audio) Teacher.

Audio-MAE uses masked autoencoding on spectrogram patches for self-supervised
audio representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class PatchEmbed(nn.Module):
    """Convert spectrogram to patch embeddings."""
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 201),  # (n_mels, time_frames)
        patch_size: Tuple[int, int] = (16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            [batch, num_patches, embed_dim]
        """
        x = self.proj(x)  # [batch, embed_dim, h, w]
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        return x


class AudioMAEBlock(nn.Module):
    """Transformer block for Audio-MAE."""
    
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
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True, bias=qkv_bias
        )
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=need_weights)
        x = x + attn_out
        
        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


class AudioMAEEncoder(nn.Module):
    """
    Audio-MAE encoder that processes visible (unmasked) patches.
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
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        
        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            AudioMAEBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches.
        
        Args:
            x: [batch, num_patches, embed_dim]
            mask_ratio: Ratio of patches to mask
            
        Returns:
            x_masked: [batch, num_visible, embed_dim]
            mask: [batch, num_patches] binary mask (1 = masked)
            ids_restore: indices for restoring original order
        """
        batch_size, num_patches, dim = x.shape
        num_keep = int(num_patches * (1 - mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(batch_size, num_patches, device=x.device)
        
        # Sort noise and get indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first num_keep patches
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))
        
        # Generate binary mask (1 = masked)
        mask = torch.ones(batch_size, num_patches, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0,
        return_all_layers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, channels, height, width] spectrogram
            mask_ratio: 0 for inference, 0.8 for pretraining
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional embeddings (excluding CLS position)
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking for pretraining
        mask = None
        ids_restore = None
        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Add CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
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
            'token_embeddings': x[:, 1:],
        }
        
        if mask is not None:
            result['mask'] = mask
            result['ids_restore'] = ids_restore
        
        if return_all_layers:
            result['all_layers'] = all_layers
            result['attention_weights'] = attention_weights
        
        return result


class AudioMAEDecoder(nn.Module):
    """
    Audio-MAE decoder for reconstruction (used in pretraining).
    """
    
    def __init__(
        self,
        num_patches: int,
        patch_size: Tuple[int, int] = (16, 16),
        in_chans: int = 1,
        encoder_embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
    ):
        super().__init__()
        
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            AudioMAEBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
            )
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size[0] * patch_size[1] * in_chans,
        )
        
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, num_visible + 1, encoder_embed_dim] (includes CLS)
            ids_restore: [batch, num_patches]
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)
        
        # Append mask tokens
        batch_size = x.shape[0]
        num_patches = ids_restore.shape[1]
        num_visible = x.shape[1] - 1  # Exclude CLS
        
        mask_tokens = self.mask_token.expand(batch_size, num_patches - num_visible, -1)
        
        # Combine visible and mask tokens (excluding CLS)
        x_no_cls = x[:, 1:, :]
        x_combined = torch.cat([x_no_cls, mask_tokens], dim=1)
        
        # Unshuffle to original positions
        x_combined = torch.gather(
            x_combined,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_combined.shape[-1])
        )
        
        # Add back CLS token
        x = torch.cat([x[:, :1, :], x_combined], dim=1)
        
        # Add positional embeddings
        x = x + self.decoder_pos_embed
        
        # Decoder blocks
        for block in self.decoder_blocks:
            x, _ = block(x)
        
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        
        # Remove CLS token
        x = x[:, 1:, :]
        
        return x


class AudioMAETeacher(nn.Module):
    """
    Complete Audio-MAE teacher model.
    
    Can be used for:
    1. Pretraining with masked reconstruction
    2. Fine-tuning on disease/concept classification
    3. Generating embeddings for distillation
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 201),
        patch_size: Tuple[int, int] = (16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        num_diseases: int = 12,
        num_concepts: int = 17,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        
        # Encoder
        self.encoder = AudioMAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )
        
        # Decoder (for pretraining)
        self.decoder = AudioMAEDecoder(
            num_patches=self.encoder.num_patches,
            patch_size=patch_size,
            in_chans=in_chans,
            encoder_embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
        )
        
        # Classification heads (for fine-tuning)
        self.disease_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, num_diseases),
        )
        
        self.concept_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, num_concepts),
        )
        
        self.patch_size = patch_size
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, path: str):
        """Load pretrained weights."""
        try:
            state_dict = torch.load(path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded Audio-MAE checkpoint from {path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.
        
        Args:
            imgs: [batch, channels, height, width]
        Returns:
            [batch, num_patches, patch_size*patch_size*channels]
        """
        p_h, p_w = self.patch_size
        batch, c, h, w = imgs.shape
        
        assert h % p_h == 0 and w % p_w == 0
        
        h_patches = h // p_h
        w_patches = w // p_w
        
        x = imgs.reshape(batch, c, h_patches, p_h, w_patches, p_w)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [batch, h_patches, w_patches, p_h, p_w, c]
        x = x.reshape(batch, h_patches * w_patches, p_h * p_w * c)
        
        return x
    
    def forward_pretrain(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.8,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pretraining with masked reconstruction.
        """
        # Encode
        encoder_output = self.encoder(x, mask_ratio=mask_ratio)
        
        # Get latent with CLS token
        latent = torch.cat([
            encoder_output['cls_embedding'].unsqueeze(1),
            encoder_output['token_embeddings'],
        ], dim=1)
        
        # Decode
        pred = self.decoder(latent, encoder_output['ids_restore'])
        
        # Compute loss on masked patches
        target = self.patchify(x)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over patch pixels
        
        # Only compute loss on masked patches
        mask = encoder_output['mask']
        loss = (loss * mask).sum() / mask.sum()
        
        return {
            'loss': loss,
            'pred': pred,
            'mask': mask,
        }
    
    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = True,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for fine-tuning/inference.
        """
        # Encode (no masking)
        encoder_output = self.encoder(
            x, 
            mask_ratio=0.0,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for distillation."""
        encoder_output = self.encoder(x, mask_ratio=0.0)
        return encoder_output['cls_embedding'], encoder_output['token_embeddings']

