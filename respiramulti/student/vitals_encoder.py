"""
Vitals encoder for processing heart rate, HRV, respiratory rate, and SpO2.

Handles missing values with learned missingness embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class VitalsEncoder(nn.Module):
    """
    Encodes vital signs and demographic features into embeddings.
    
    Features handled:
    - HR (mean, std)
    - HRV (RMSSD, SDNN)
    - RR (respiratory rate)
    - SpO2 (optional)
    - Quality scores
    - Demographics (age, sex, smoker status, etc.)
    
    Uses learned embeddings for missing values.
    """
    
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: list = [128, 128],
        output_dim: int = 128,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_missingness_embedding: bool = True,
        missingness_dim: int = 16,
        num_vitals_features: int = 10,  # Number of vital sign features
        num_demo_features: int = 15,  # Number of demographic features
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_missingness_embedding = use_missingness_embedding
        self.num_vitals_features = num_vitals_features
        
        # Activation function
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        else:
            act_fn = nn.SiLU()
        
        # Missingness embeddings (one per feature that can be missing)
        if use_missingness_embedding:
            self.missingness_embeddings = nn.Embedding(num_vitals_features, missingness_dim)
            # Projection to combine with input
            self.miss_proj = nn.Linear(missingness_dim * num_vitals_features, hidden_dims[0])
            actual_input_dim = input_dim + hidden_dims[0]
        else:
            actual_input_dim = input_dim
        
        # Feature normalization per feature type
        self.feature_norms = nn.ModuleDict({
            'hr': nn.LayerNorm(2),  # hr_mean, hr_std
            'hrv': nn.LayerNorm(2),  # rmssd, sdnn
            'rr': nn.LayerNorm(1),
            'spo2': nn.LayerNorm(1),
            'quality': nn.LayerNorm(3),  # quality scores
            'demographics': nn.LayerNorm(num_demo_features),
        })
        
        # MLP layers
        layers = []
        in_dim = actual_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn,
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Present/absent indicator
        self.present_embedding = nn.Parameter(torch.zeros(1, output_dim))
        self.absent_embedding = nn.Parameter(torch.zeros(1, output_dim))
        nn.init.normal_(self.present_embedding, std=0.02)
        nn.init.normal_(self.absent_embedding, std=0.02)
    
    def forward(
        self,
        vitals: torch.Tensor,
        vitals_mask: Optional[torch.Tensor] = None,
        demographics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            vitals: [batch, num_vitals_features] vital sign values
            vitals_mask: [batch, num_vitals_features] 1 if present, 0 if missing
            demographics: [batch, num_demo_features] demographic features
            
        Returns:
            Dict with 'embedding' and 'present_ratio'
        """
        batch_size = vitals.shape[0]
        
        # Handle missing values
        if vitals_mask is None:
            vitals_mask = (vitals != 0).float()
        
        # Replace missing values with zeros (will be handled by missingness embeddings)
        vitals = vitals * vitals_mask
        
        # Compute missingness embedding
        if self.use_missingness_embedding:
            # Get indices of missing features
            missing_indicator = (1 - vitals_mask)  # [batch, num_features]
            
            # Embed each feature's missingness
            feature_indices = torch.arange(self.num_vitals_features, device=vitals.device)
            miss_emb = self.missingness_embeddings(feature_indices)  # [num_features, miss_dim]
            miss_emb = miss_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_features, miss_dim]
            
            # Weight by missingness
            miss_emb = miss_emb * missing_indicator[:, :self.num_vitals_features].unsqueeze(-1)
            miss_emb = miss_emb.flatten(1)  # [batch, num_features * miss_dim]
            miss_emb = self.miss_proj(miss_emb)  # [batch, hidden_dim]
        
        # Combine vitals and demographics
        if demographics is not None:
            features = torch.cat([vitals, demographics], dim=-1)
        else:
            features = vitals
        
        # Add missingness embedding
        if self.use_missingness_embedding:
            # Pad features to match expected input
            if features.shape[-1] < self.input_dim:
                features = F.pad(features, (0, self.input_dim - features.shape[-1]))
            features = torch.cat([features[:, :self.input_dim], miss_emb], dim=-1)
        
        # MLP encoding
        embedding = self.mlp(features)
        
        # Add present/absent indicator based on data availability
        present_ratio = vitals_mask.mean(dim=-1, keepdim=True)  # [batch, 1]
        presence_emb = present_ratio * self.present_embedding + (1 - present_ratio) * self.absent_embedding
        embedding = embedding + presence_emb
        
        return {
            'embedding': embedding,
            'present_ratio': present_ratio.squeeze(-1),
        }


class VitalsAttention(nn.Module):
    """
    Attention mechanism for vitals to attend to audio tokens.
    
    Enables vitals to gather relevant audio information.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        vitals_query: torch.Tensor,
        audio_keys: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention from vitals to audio.
        
        Args:
            vitals_query: [batch, d_model]
            audio_keys: [batch, num_tokens, d_model]
            audio_mask: [batch, num_tokens]
        """
        # Expand vitals query
        query = vitals_query.unsqueeze(1)  # [batch, 1, d_model]
        
        # Cross-attention
        attn_out, _ = self.attention(
            query, audio_keys, audio_keys,
            key_padding_mask=audio_mask,
        )
        
        # Residual and norm
        vitals_enhanced = self.norm(vitals_query + self.dropout(attn_out.squeeze(1)))
        
        return vitals_enhanced


class DemographicsEncoder(nn.Module):
    """
    Separate encoder for demographic features.
    
    Handles categorical and continuous demographic variables.
    """
    
    def __init__(
        self,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Embeddings for categorical features
        self.sex_embedding = nn.Embedding(3, 8)  # male, female, other
        self.smoker_embedding = nn.Embedding(4, 8)  # never, former, current, unknown
        
        # Continuous feature projections
        self.age_proj = nn.Linear(1, 8)
        self.bmi_proj = nn.Linear(1, 8)  # computed from height/weight
        
        # Symptom embeddings (multi-hot)
        self.symptom_proj = nn.Linear(6, 16)  # 6 symptom flags
        
        # Condition embeddings (multi-hot)
        self.condition_proj = nn.Linear(4, 16)  # 4 condition flags
        
        # Combined projection
        self.output_proj = nn.Sequential(
            nn.Linear(8 + 8 + 8 + 8 + 16 + 16, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        age: torch.Tensor,  # [batch]
        sex: torch.Tensor,  # [batch] categorical index
        height: torch.Tensor,  # [batch]
        weight: torch.Tensor,  # [batch]
        smoker: torch.Tensor,  # [batch] categorical index
        symptoms: torch.Tensor,  # [batch, 6] multi-hot
        conditions: torch.Tensor,  # [batch, 4] multi-hot
    ) -> torch.Tensor:
        """Encode demographics."""
        # Categorical embeddings
        sex_emb = self.sex_embedding(sex)
        smoker_emb = self.smoker_embedding(smoker)
        
        # Continuous features
        age_emb = self.age_proj(age.unsqueeze(-1))
        
        # Compute BMI
        bmi = weight / (height / 100) ** 2
        bmi = bmi.clamp(10, 60)  # Reasonable BMI range
        bmi_emb = self.bmi_proj(bmi.unsqueeze(-1))
        
        # Symptom and condition embeddings
        symptom_emb = self.symptom_proj(symptoms.float())
        condition_emb = self.condition_proj(conditions.float())
        
        # Combine
        combined = torch.cat([
            sex_emb, smoker_emb, age_emb, bmi_emb,
            symptom_emb, condition_emb
        ], dim=-1)
        
        output = self.output_proj(combined)
        
        return output

