"""
Mobile-optimized audio encoders for the student model.

Implements MobileNetV3 and EfficientNet-Lite based encoders for spectrograms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(
        self,
        in_channels: int,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(x)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class InvertedResidual(nn.Module):
    """MobileNetV3 inverted residual block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: float = 4.0,
        se_ratio: float = 0.25,
        activation: str = "relu",
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish() if activation == "hardswish" else nn.ReLU(inplace=True),
            ])
        
        # Depthwise
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish() if activation == "hardswish" else nn.ReLU(inplace=True),
        ])
        
        # SE
        if se_ratio > 0:
            layers.append(SqueezeExcitation(hidden_dim, se_ratio))
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out


class MobileNetV3Small(nn.Module):
    """
    MobileNetV3-Small adapted for spectrogram input.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        width_mult: float = 1.0,
        output_dim: int = 256,
    ):
        super().__init__()
        
        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        # First conv
        first_channels = _make_divisible(16 * width_mult)
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, first_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(first_channels),
            nn.Hardswish(),
        )
        
        # MobileNetV3-Small configuration
        # (kernel, exp_ratio, out_channels, se, activation, stride)
        config = [
            (3, 1, 16, True, "relu", 2),
            (3, 4.5, 24, False, "relu", 2),
            (3, 3.67, 24, False, "relu", 1),
            (5, 4, 40, True, "hardswish", 2),
            (5, 6, 40, True, "hardswish", 1),
            (5, 6, 40, True, "hardswish", 1),
            (5, 3, 48, True, "hardswish", 1),
            (5, 3, 48, True, "hardswish", 1),
            (5, 6, 96, True, "hardswish", 2),
            (5, 6, 96, True, "hardswish", 1),
            (5, 6, 96, True, "hardswish", 1),
        ]
        
        layers = []
        in_ch = first_channels
        
        for kernel, exp, out, se, act, stride in config:
            out_ch = _make_divisible(out * width_mult)
            se_ratio = 0.25 if se else 0
            layers.append(InvertedResidual(
                in_ch, out_ch, kernel, stride, exp, se_ratio, act
            ))
            in_ch = out_ch
        
        self.layers = nn.Sequential(*layers)
        
        # Final layers
        last_channels = _make_divisible(576 * width_mult)
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_ch, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
            nn.Hardswish(),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_channels, output_dim),
            nn.Hardswish(),
        )
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width] spectrogram
        Returns:
            [batch, output_dim] embedding
        """
        x = self.first_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features and embedding.
        
        Returns:
            Tuple of (feature_map, embedding)
        """
        x = self.first_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        
        feature_map = x  # [batch, channels, h, w]
        
        x = self.pool(x)
        x = x.flatten(1)
        embedding = self.classifier(x)
        
        return feature_map, embedding


class MobileAudioEncoder(nn.Module):
    """
    Mobile-optimized audio encoder using MobileNetV3.
    
    Processes mel spectrograms and outputs embeddings suitable for fusion.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 256,
        width_mult: float = 1.0,
        pretrained: bool = False,
    ):
        super().__init__()
        
        self.backbone = MobileNetV3Small(
            in_channels=in_channels,
            width_mult=width_mult,
            output_dim=embedding_dim,
        )
        
        self.embedding_dim = embedding_dim
        
        if pretrained:
            self._load_pretrained()
    
    def _load_pretrained(self):
        """Load pretrained ImageNet weights and adapt."""
        try:
            import torchvision.models as models
            pretrained = models.mobilenet_v3_small(pretrained=True)
            
            # Copy compatible weights
            state_dict = self.backbone.state_dict()
            pretrained_dict = pretrained.state_dict()
            
            # Filter and adapt (skip first conv due to channel mismatch)
            for name, param in pretrained_dict.items():
                if name in state_dict and param.shape == state_dict[name].shape:
                    state_dict[name] = param
            
            self.backbone.load_state_dict(state_dict, strict=False)
            print("Loaded pretrained MobileNetV3 weights")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [batch, 1, n_mels, time] mel spectrogram
            return_features: Whether to return intermediate features
        """
        if return_features:
            features, embedding = self.backbone.forward_features(x)
            return {
                'embedding': embedding,
                'features': features,
            }
        else:
            embedding = self.backbone(x)
            return {'embedding': embedding}


class EfficientNetAudioEncoder(nn.Module):
    """
    EfficientNet-Lite based audio encoder.
    
    Alternative to MobileNetV3 with potentially better accuracy.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 256,
        model_variant: str = "lite0",  # lite0, lite1, lite2
    ):
        super().__init__()
        
        # EfficientNet-Lite0 configuration
        # (expand_ratio, channels, layers, stride, kernel)
        if model_variant == "lite0":
            config = [
                (1, 16, 1, 1, 3),
                (6, 24, 2, 2, 3),
                (6, 40, 2, 2, 5),
                (6, 80, 3, 2, 3),
                (6, 112, 3, 1, 5),
                (6, 192, 4, 2, 5),
                (6, 320, 1, 1, 3),
            ]
            stem_channels = 32
            head_channels = 1280
        else:
            # Use lite0 as default
            config = [
                (1, 16, 1, 1, 3),
                (6, 24, 2, 2, 3),
                (6, 40, 2, 2, 5),
                (6, 80, 3, 2, 3),
                (6, 112, 3, 1, 5),
                (6, 192, 4, 2, 5),
                (6, 320, 1, 1, 3),
            ]
            stem_channels = 32
            head_channels = 1280
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU6(inplace=True),
        )
        
        # Build blocks
        layers = []
        in_ch = stem_channels
        
        for expand, out_ch, num_layers, stride, kernel in config:
            for i in range(num_layers):
                s = stride if i == 0 else 1
                layers.append(InvertedResidual(
                    in_ch, out_ch, kernel, s, expand, 
                    se_ratio=0.25, activation="relu"
                ))
                in_ch = out_ch
        
        self.blocks = nn.Sequential(*layers)
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU6(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(head_channels, embedding_dim)
        
        self.embedding_dim = embedding_dim
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [batch, 1, n_mels, time] mel spectrogram
        """
        x = self.stem(x)
        x = self.blocks(x)
        features = self.head(x)
        
        x = self.pool(features)
        x = x.flatten(1)
        embedding = self.fc(x)
        
        result = {'embedding': embedding}
        
        if return_features:
            result['features'] = features
        
        return result


class AudioTokenizer(nn.Module):
    """
    Converts audio embeddings to sequence of tokens for transformer fusion.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        num_segment_types: int = 6,  # cough_shallow, cough_deep, breath_normal, etc.
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Segment type embeddings
        self.segment_embeddings = nn.Embedding(num_segment_types, embedding_dim)
        
        # Position embedding (for multiple tokens from same segment type)
        self.max_positions = 20
        self.position_embeddings = nn.Embedding(self.max_positions, embedding_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        segment_types: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add segment and position information to embeddings.
        
        Args:
            embeddings: [batch, num_tokens, embedding_dim]
            segment_types: [batch, num_tokens] indices
            positions: [batch, num_tokens] position indices (optional)
        """
        # Add segment type embeddings
        segment_emb = self.segment_embeddings(segment_types)
        
        # Add position embeddings
        if positions is None:
            positions = torch.arange(
                embeddings.shape[1], device=embeddings.device
            ).unsqueeze(0).expand(embeddings.shape[0], -1)
            positions = positions.clamp(max=self.max_positions - 1)
        
        pos_emb = self.position_embeddings(positions)
        
        # Combine
        output = embeddings + segment_emb + pos_emb
        output = self.layer_norm(output)
        
        return output

