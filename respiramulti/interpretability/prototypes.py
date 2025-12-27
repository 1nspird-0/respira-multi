"""
Prototype-based evidence retrieval for interpretability.

Maintains a bank of prototypical embeddings for each disease and concept,
enabling case-based reasoning and auditability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from dataclasses import dataclass


@dataclass
class PrototypeMatch:
    """A matched prototype with metadata."""
    prototype_id: int
    disease_or_concept: str
    similarity: float
    segment_type: str
    timestamp_range: Optional[Tuple[float, float]] = None
    training_session_id: Optional[str] = None


class PrototypeBank(nn.Module):
    """
    Bank of prototype embeddings for diseases and concepts.
    
    Stores representative token-level embeddings that can be used
    for case-based reasoning and evidence retrieval.
    """
    
    def __init__(
        self,
        num_diseases: int = 12,
        num_concepts: int = 17,
        num_prototypes_per_class: int = 10,
        embedding_dim: int = 256,
        temperature: float = 0.1,
    ):
        super().__init__()
        
        self.num_diseases = num_diseases
        self.num_concepts = num_concepts
        self.num_prototypes = num_prototypes_per_class
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # Prototype embeddings (learnable)
        # [num_classes, num_prototypes, embedding_dim]
        self.disease_prototypes = nn.Parameter(
            torch.randn(num_diseases, num_prototypes_per_class, embedding_dim) * 0.1
        )
        self.concept_prototypes = nn.Parameter(
            torch.randn(num_concepts, num_prototypes_per_class, embedding_dim) * 0.1
        )
        
        # Metadata storage (non-differentiable)
        self.prototype_metadata: Dict[str, List[Dict]] = {
            'diseases': [[] for _ in range(num_diseases)],
            'concepts': [[] for _ in range(num_concepts)],
        }
    
    def compute_similarity(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings and prototypes.
        
        Args:
            embeddings: [batch, dim] or [batch, seq, dim]
            prototypes: [num_classes, num_prototypes, dim]
            
        Returns:
            similarities: [batch, num_classes, num_prototypes] or 
                         [batch, seq, num_classes, num_prototypes]
        """
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        
        # Reshape for batch matmul
        if embeddings.dim() == 2:
            # [batch, dim] -> [batch, 1, dim]
            embeddings = embeddings.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        
        # [num_classes, num_prototypes, dim] -> [1, num_classes * num_prototypes, dim]
        flat_prototypes = prototypes.view(-1, self.embedding_dim).unsqueeze(0)
        
        # Compute similarity
        # [batch, seq, dim] @ [1, num_classes * num_prototypes, dim].T -> [batch, seq, num_classes * num_prototypes]
        similarity = torch.bmm(
            embeddings,
            flat_prototypes.expand(embeddings.size(0), -1, -1).transpose(1, 2)
        )
        
        # Reshape to [batch, seq, num_classes, num_prototypes]
        num_classes = prototypes.size(0)
        similarity = similarity.view(embeddings.size(0), -1, num_classes, self.num_prototypes)
        
        if squeeze:
            similarity = similarity.squeeze(1)
        
        return similarity / self.temperature
    
    def get_prototype_scores(
        self,
        cls_embedding: torch.Tensor,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get prototype similarity scores for classification.
        
        Returns max similarity per class as additional features.
        """
        # Disease prototype similarities
        disease_sim = self.compute_similarity(cls_embedding, self.disease_prototypes)
        disease_scores = disease_sim.max(dim=-1).values  # [batch, num_diseases]
        
        # Concept prototype similarities
        concept_sim = self.compute_similarity(cls_embedding, self.concept_prototypes)
        concept_scores = concept_sim.max(dim=-1).values  # [batch, num_concepts]
        
        result = {
            'disease_prototype_scores': disease_scores,
            'concept_prototype_scores': concept_scores,
            'disease_similarities': disease_sim,
            'concept_similarities': concept_sim,
        }
        
        # Token-level similarities if provided
        if token_embeddings is not None:
            token_disease_sim = self.compute_similarity(token_embeddings, self.disease_prototypes)
            token_concept_sim = self.compute_similarity(token_embeddings, self.concept_prototypes)
            result['token_disease_similarities'] = token_disease_sim
            result['token_concept_similarities'] = token_concept_sim
        
        return result
    
    def retrieve_top_k(
        self,
        embedding: torch.Tensor,
        class_type: str = 'diseases',
        class_idx: int = 0,
        k: int = 3,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k most similar prototypes for a class.
        
        Returns list of (prototype_idx, similarity) tuples.
        """
        if class_type == 'diseases':
            prototypes = self.disease_prototypes[class_idx]
        else:
            prototypes = self.concept_prototypes[class_idx]
        
        # Compute similarities
        embedding = F.normalize(embedding, p=2, dim=-1)
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        
        similarities = torch.matmul(embedding, prototypes.T)  # [num_prototypes]
        
        # Get top-k
        topk_sim, topk_idx = similarities.topk(k)
        
        return [(idx.item(), sim.item()) for idx, sim in zip(topk_idx, topk_sim)]
    
    def update_prototypes_from_data(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_type: str = 'diseases',
    ):
        """
        Update prototypes using k-means clustering on labeled embeddings.
        
        Should be called after training to create meaningful prototypes.
        """
        embeddings_np = embeddings.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        if class_type == 'diseases':
            prototypes = self.disease_prototypes
            num_classes = self.num_diseases
        else:
            prototypes = self.concept_prototypes
            num_classes = self.num_concepts
        
        with torch.no_grad():
            for class_idx in range(num_classes):
                # Get embeddings for this class
                class_mask = labels_np[:, class_idx] > 0.5
                if class_mask.sum() < self.num_prototypes:
                    continue
                
                class_embeddings = embeddings_np[class_mask]
                
                # K-means clustering
                kmeans = KMeans(n_clusters=self.num_prototypes, random_state=42)
                kmeans.fit(class_embeddings)
                
                # Update prototypes
                prototypes.data[class_idx] = torch.tensor(
                    kmeans.cluster_centers_,
                    dtype=prototypes.dtype,
                    device=prototypes.device,
                )


class PrototypeRetrieval:
    """
    Retrieve evidence from prototype bank for explanations.
    """
    
    def __init__(
        self,
        prototype_bank: PrototypeBank,
        segment_types: List[str],
    ):
        self.prototype_bank = prototype_bank
        self.segment_types = segment_types
    
    def get_evidence_for_prediction(
        self,
        embedding: torch.Tensor,
        token_embeddings: torch.Tensor,
        segment_type_indices: torch.Tensor,
        predicted_diseases: List[int],
        k: int = 3,
    ) -> Dict[str, List[PrototypeMatch]]:
        """
        Get evidence for each predicted disease.
        
        Returns dict mapping disease names to list of matching prototypes.
        """
        evidence = {}
        
        for disease_idx in predicted_diseases:
            matches = []
            
            # Get top-k prototypes for this disease
            top_prototypes = self.prototype_bank.retrieve_top_k(
                embedding, 'diseases', disease_idx, k=k
            )
            
            # Find most relevant tokens
            token_sims = self.prototype_bank.compute_similarity(
                token_embeddings.unsqueeze(0),
                self.prototype_bank.disease_prototypes[disease_idx:disease_idx+1]
            ).squeeze(0)  # [seq, 1, num_prototypes]
            
            # Get max similarity per token
            max_sim_per_token = token_sims.max(dim=-1).values.max(dim=-1).values
            top_token_idx = max_sim_per_token.argmax().item()
            
            for proto_idx, similarity in top_prototypes:
                match = PrototypeMatch(
                    prototype_id=proto_idx,
                    disease_or_concept=f"disease_{disease_idx}",
                    similarity=similarity,
                    segment_type=self.segment_types[segment_type_indices[top_token_idx].item()],
                )
                matches.append(match)
            
            evidence[f"disease_{disease_idx}"] = matches
        
        return evidence
    
    def get_concept_evidence(
        self,
        token_embeddings: torch.Tensor,
        segment_type_indices: torch.Tensor,
        predicted_concepts: List[int],
        k: int = 3,
    ) -> Dict[str, List[PrototypeMatch]]:
        """
        Get evidence for predicted concepts (wheeze, crackle, etc.).
        """
        evidence = {}
        
        for concept_idx in predicted_concepts:
            matches = []
            
            # Token-level similarities
            token_sims = self.prototype_bank.compute_similarity(
                token_embeddings.unsqueeze(0),
                self.prototype_bank.concept_prototypes[concept_idx:concept_idx+1]
            ).squeeze(0)  # [seq, 1, num_prototypes]
            
            # Find tokens with highest similarity
            max_sim_per_token = token_sims.max(dim=-1).values.squeeze(-1)  # [seq]
            top_k_tokens = max_sim_per_token.topk(min(k, len(max_sim_per_token)))
            
            for token_idx, sim in zip(top_k_tokens.indices, top_k_tokens.values):
                match = PrototypeMatch(
                    prototype_id=token_idx.item(),
                    disease_or_concept=f"concept_{concept_idx}",
                    similarity=sim.item(),
                    segment_type=self.segment_types[segment_type_indices[token_idx].item()],
                )
                matches.append(match)
            
            evidence[f"concept_{concept_idx}"] = matches
        
        return evidence

