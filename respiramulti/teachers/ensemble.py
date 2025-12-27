"""
Teacher Ensemble for Knowledge Distillation.

Combines multiple teacher models (BEATs, Audio-MAE, AST, HuBERT) to generate
soft labels and embeddings for distilling into the student model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from respiramulti.teachers.beats import BEATsTeacher
from respiramulti.teachers.audio_mae import AudioMAETeacher
from respiramulti.teachers.ast_model import ASTTeacher
from respiramulti.teachers.speech_encoder import HuBERTTeacher, Wav2Vec2Teacher


@dataclass
class TeacherOutput:
    """Output from teacher ensemble."""
    # Averaged logits
    disease_logits: torch.Tensor  # [batch, num_diseases]
    concept_logits: torch.Tensor  # [batch, num_concepts]
    
    # Averaged embeddings
    cls_embedding: torch.Tensor  # [batch, embed_dim]
    token_embeddings: torch.Tensor  # [batch, seq_len, embed_dim]
    
    # Per-teacher outputs (for analysis)
    teacher_outputs: Dict[str, Dict[str, torch.Tensor]]
    
    # Attention weights (if available)
    attention_weights: Optional[Dict[str, List[torch.Tensor]]] = None


class TeacherEnsemble(nn.Module):
    """
    Ensemble of teacher models for knowledge distillation.
    
    Combines BEATs, Audio-MAE, AST for spectrogram inputs,
    and HuBERT/wav2vec2 for speech inputs.
    """
    
    def __init__(
        self,
        num_diseases: int = 12,
        num_concepts: int = 17,
        embed_dim: int = 768,
        enable_beats: bool = True,
        enable_audio_mae: bool = True,
        enable_ast: bool = True,
        enable_speech: bool = True,
        speech_encoder_type: str = "hubert",  # "hubert" or "wav2vec2"
        weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
        config: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.config = config or {}
        
        # Default weights
        self.weights = weights or {
            'beats': 1.0,
            'audio_mae': 1.0,
            'ast': 1.0,
            'speech': 0.5,
        }
        
        # Initialize enabled teachers
        self.teachers = nn.ModuleDict()
        
        if enable_beats:
            beats_config = self.config.get('beats', {})
            self.teachers['beats'] = BEATsTeacher(
                num_diseases=num_diseases,
                num_concepts=num_concepts,
                embed_dim=embed_dim,
                checkpoint_path=beats_config.get('checkpoint'),
            )
        
        if enable_audio_mae:
            mae_config = self.config.get('audio_mae', {})
            self.teachers['audio_mae'] = AudioMAETeacher(
                num_diseases=num_diseases,
                num_concepts=num_concepts,
                embed_dim=embed_dim,
                checkpoint_path=mae_config.get('checkpoint'),
            )
        
        if enable_ast:
            ast_config = self.config.get('ast', {})
            self.teachers['ast'] = ASTTeacher(
                num_diseases=num_diseases,
                num_concepts=num_concepts,
                embed_dim=embed_dim,
                checkpoint_path=ast_config.get('checkpoint'),
            )
        
        if enable_speech:
            speech_config = self.config.get('speech_encoder', {})
            if speech_encoder_type == "hubert":
                self.teachers['speech'] = HuBERTTeacher(
                    num_diseases=num_diseases,
                    num_concepts=num_concepts,
                    embed_dim=embed_dim,
                    checkpoint_path=speech_config.get('checkpoint'),
                )
            else:
                self.teachers['speech'] = Wav2Vec2Teacher(
                    num_diseases=num_diseases,
                    num_concepts=num_concepts,
                    embed_dim=embed_dim,
                    checkpoint_path=speech_config.get('checkpoint'),
                )
        
        # Projection layers to align embedding dimensions
        self.embed_projections = nn.ModuleDict()
        for name in self.teachers:
            # Identity if same dimension, otherwise project
            self.embed_projections[name] = nn.Identity()
        
        # Compute normalization factor
        self.weight_sum = sum(
            self.weights.get(name, 1.0) 
            for name in self.teachers
        )
    
    def freeze_teachers(self):
        """Freeze all teacher parameters."""
        for teacher in self.teachers.values():
            for param in teacher.parameters():
                param.requires_grad = False
    
    def unfreeze_teachers(self):
        """Unfreeze all teacher parameters."""
        for teacher in self.teachers.values():
            for param in teacher.parameters():
                param.requires_grad = True
    
    def forward(
        self,
        spectrogram: Optional[torch.Tensor] = None,
        waveform: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> TeacherOutput:
        """
        Forward pass through all teachers.
        
        Args:
            spectrogram: [batch, 1, n_mels, time] mel spectrogram
            waveform: [batch, time] raw audio (for speech encoder)
            padding_mask: [batch, seq_len]
            return_attention: Whether to return attention weights
        """
        teacher_outputs = {}
        
        # Run spectrogram-based teachers
        spec_teachers = ['beats', 'audio_mae', 'ast']
        if spectrogram is not None:
            for name in spec_teachers:
                if name in self.teachers:
                    output = self.teachers[name](
                        spectrogram,
                        return_embeddings=True,
                        return_attention=return_attention,
                    )
                    teacher_outputs[name] = output
        
        # Run speech encoder
        if waveform is not None and 'speech' in self.teachers:
            output = self.teachers['speech'](
                waveform,
                padding_mask=padding_mask,
                return_embeddings=True,
            )
            teacher_outputs['speech'] = output
        
        # Aggregate outputs
        return self._aggregate_outputs(teacher_outputs, return_attention)
    
    def _aggregate_outputs(
        self,
        teacher_outputs: Dict[str, Dict[str, torch.Tensor]],
        return_attention: bool = False,
    ) -> TeacherOutput:
        """Aggregate outputs from all teachers."""
        
        # Weighted average of logits
        disease_logits_list = []
        concept_logits_list = []
        cls_embeddings = []
        token_embeddings_list = []
        weights_list = []
        
        for name, output in teacher_outputs.items():
            weight = self.weights.get(name, 1.0)
            weights_list.append(weight)
            
            # Logits with temperature
            disease_logits_list.append(
                output['disease_logits'] / self.temperature * weight
            )
            concept_logits_list.append(
                output['concept_logits'] / self.temperature * weight
            )
            
            # Embeddings
            cls_emb = self.embed_projections[name](output['cls_embedding'])
            cls_embeddings.append(cls_emb * weight)
            
            if 'token_embeddings' in output:
                token_emb = self.embed_projections[name](output['token_embeddings'])
                token_embeddings_list.append(token_emb * weight)
        
        # Normalize by weight sum
        total_weight = sum(weights_list)
        
        avg_disease_logits = sum(disease_logits_list) / total_weight
        avg_concept_logits = sum(concept_logits_list) / total_weight
        avg_cls_embedding = sum(cls_embeddings) / total_weight
        
        # Average token embeddings (need to handle different sequence lengths)
        if token_embeddings_list:
            # For simplicity, use first non-None or pad to max length
            max_len = max(t.shape[1] for t in token_embeddings_list)
            padded_tokens = []
            for t in token_embeddings_list:
                if t.shape[1] < max_len:
                    pad = torch.zeros(
                        t.shape[0], max_len - t.shape[1], t.shape[2],
                        device=t.device, dtype=t.dtype
                    )
                    t = torch.cat([t, pad], dim=1)
                padded_tokens.append(t)
            avg_token_embeddings = sum(padded_tokens) / total_weight
        else:
            avg_token_embeddings = avg_cls_embedding.unsqueeze(1)
        
        # Collect attention weights if requested
        attention_weights = None
        if return_attention:
            attention_weights = {}
            for name, output in teacher_outputs.items():
                if 'attention_weights' in output:
                    attention_weights[name] = output['attention_weights']
        
        return TeacherOutput(
            disease_logits=avg_disease_logits,
            concept_logits=avg_concept_logits,
            cls_embedding=avg_cls_embedding,
            token_embeddings=avg_token_embeddings,
            teacher_outputs=teacher_outputs,
            attention_weights=attention_weights,
        )
    
    def get_soft_labels(
        self,
        spectrogram: Optional[torch.Tensor] = None,
        waveform: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get soft labels (probabilities) for distillation.
        
        Args:
            spectrogram: [batch, 1, n_mels, time]
            waveform: [batch, time]
            temperature: Override default temperature
            
        Returns:
            Tuple of (disease_probs, concept_probs)
        """
        temp = temperature or self.temperature
        
        with torch.no_grad():
            output = self.forward(spectrogram=spectrogram, waveform=waveform)
            
            disease_probs = torch.sigmoid(output.disease_logits / temp)
            concept_probs = torch.sigmoid(output.concept_logits / temp)
            
            return disease_probs, concept_probs
    
    def get_embeddings_for_distillation(
        self,
        spectrogram: Optional[torch.Tensor] = None,
        waveform: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get embeddings for feature distillation.
        
        Returns:
            Tuple of (cls_embedding, token_embeddings)
        """
        with torch.no_grad():
            output = self.forward(spectrogram=spectrogram, waveform=waveform)
            return output.cls_embedding, output.token_embeddings


class DistillationLoss(nn.Module):
    """
    Combined distillation loss for training student model.
    
    Includes:
    - KL divergence on logits
    - L2 loss on embeddings
    - Optional attention matching
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        logit_weight: float = 1.0,
        feature_weight: float = 0.5,
        attention_weight: float = 0.1,
        use_attention_distillation: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.logit_weight = logit_weight
        self.feature_weight = feature_weight
        self.attention_weight = attention_weight
        self.use_attention_distillation = use_attention_distillation
    
    def forward(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: TeacherOutput,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.
        
        Args:
            student_output: Dict with 'disease_logits', 'concept_logits', 
                          'cls_embedding', 'token_embeddings'
            teacher_output: TeacherOutput from ensemble
            
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        
        # KL divergence on disease logits
        student_disease_log_probs = F.log_softmax(
            student_output['disease_logits'] / self.temperature, dim=-1
        )
        teacher_disease_probs = F.softmax(
            teacher_output.disease_logits / self.temperature, dim=-1
        )
        kl_disease = F.kl_div(
            student_disease_log_probs, teacher_disease_probs, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        losses['kl_disease'] = kl_disease
        
        # KL divergence on concept logits (binary, use BCE-style)
        student_concept_probs = torch.sigmoid(
            student_output['concept_logits'] / self.temperature
        )
        teacher_concept_probs = torch.sigmoid(
            teacher_output.concept_logits / self.temperature
        )
        kl_concept = F.binary_cross_entropy(
            student_concept_probs, teacher_concept_probs, reduction='mean'
        ) * (self.temperature ** 2)
        losses['kl_concept'] = kl_concept
        
        # Feature distillation (L2 on embeddings)
        cls_loss = F.mse_loss(
            student_output['cls_embedding'], 
            teacher_output.cls_embedding
        )
        losses['cls_mse'] = cls_loss
        
        # Token embedding loss (if dimensions match)
        if student_output.get('token_embeddings') is not None:
            student_tokens = student_output['token_embeddings']
            teacher_tokens = teacher_output.token_embeddings
            
            # Handle dimension mismatch
            min_len = min(student_tokens.shape[1], teacher_tokens.shape[1])
            student_tokens = student_tokens[:, :min_len]
            teacher_tokens = teacher_tokens[:, :min_len]
            
            if student_tokens.shape[-1] != teacher_tokens.shape[-1]:
                # Project if dimensions differ
                teacher_tokens = F.linear(
                    teacher_tokens,
                    torch.eye(student_tokens.shape[-1], teacher_tokens.shape[-1],
                             device=teacher_tokens.device)
                )
            
            token_loss = F.mse_loss(student_tokens, teacher_tokens)
            losses['token_mse'] = token_loss
        else:
            losses['token_mse'] = torch.tensor(0.0, device=student_output['cls_embedding'].device)
        
        # Attention distillation (optional)
        if self.use_attention_distillation and teacher_output.attention_weights:
            attn_loss = torch.tensor(0.0, device=student_output['cls_embedding'].device)
            # Would need student attention weights here
            losses['attention_mse'] = attn_loss
        
        # Total loss
        total_loss = (
            self.logit_weight * (losses['kl_disease'] + losses['kl_concept']) +
            self.feature_weight * (losses['cls_mse'] + losses['token_mse'])
        )
        
        if self.use_attention_distillation:
            total_loss += self.attention_weight * losses.get('attention_mse', 0)
        
        losses['total'] = total_loss
        
        return losses

