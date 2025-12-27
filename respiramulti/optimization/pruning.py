"""
Structured pruning for model compression.

Implements magnitude-based and gradual pruning strategies.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Tuple
import numpy as np


class StructuredPruner:
    """
    Structured pruning manager.
    
    Supports channel-wise pruning for efficient mobile inference.
    """
    
    def __init__(
        self,
        target_sparsity: float = 0.3,
        pruning_method: str = "magnitude",  # "magnitude", "l1", "random"
        structured: bool = True,
        exclude_layers: Optional[List[str]] = None,
    ):
        self.target_sparsity = target_sparsity
        self.pruning_method = pruning_method
        self.structured = structured
        self.exclude_layers = exclude_layers or []
    
    def get_prunable_layers(
        self,
        model: nn.Module,
    ) -> List[Tuple[nn.Module, str]]:
        """Get list of layers to prune."""
        layers = []
        
        for name, module in model.named_modules():
            # Skip excluded layers
            if any(excl in name for excl in self.exclude_layers):
                continue
            
            # Prunable layer types
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers.append((module, name))
        
        return layers
    
    def compute_importance_scores(
        self,
        module: nn.Module,
    ) -> torch.Tensor:
        """Compute importance scores for weights."""
        weight = module.weight.data
        
        if self.pruning_method == "magnitude":
            # L2 norm of weights
            if self.structured:
                # Channel-wise importance (for Conv2d: output channels)
                importance = weight.view(weight.size(0), -1).norm(p=2, dim=1)
            else:
                importance = weight.abs()
        
        elif self.pruning_method == "l1":
            if self.structured:
                importance = weight.view(weight.size(0), -1).norm(p=1, dim=1)
            else:
                importance = weight.abs()
        
        elif self.pruning_method == "random":
            importance = torch.rand(weight.size(0), device=weight.device)
        
        else:
            raise ValueError(f"Unknown pruning method: {self.pruning_method}")
        
        return importance
    
    def prune_layer(
        self,
        module: nn.Module,
        sparsity: float,
    ):
        """Apply pruning to a single layer."""
        if self.structured:
            # Structured (channel) pruning
            importance = self.compute_importance_scores(module)
            num_channels = len(importance)
            num_to_prune = int(num_channels * sparsity)
            
            # Get threshold
            threshold = torch.kthvalue(importance, num_to_prune).values
            
            # Create mask
            mask = importance > threshold
            
            # Apply mask (simplified - full implementation would handle dependent layers)
            with torch.no_grad():
                module.weight.data *= mask.view(-1, 1, 1, 1) if module.weight.dim() == 4 else mask.view(-1, 1)
        else:
            # Unstructured pruning using PyTorch's prune module
            prune.l1_unstructured(module, name='weight', amount=sparsity)
    
    def prune_model(
        self,
        model: nn.Module,
        sparsity: Optional[float] = None,
    ) -> nn.Module:
        """Apply pruning to entire model."""
        sparsity = sparsity or self.target_sparsity
        
        layers = self.get_prunable_layers(model)
        
        for module, name in layers:
            self.prune_layer(module, sparsity)
            print(f"Pruned layer {name} with sparsity {sparsity:.1%}")
        
        return model
    
    def compute_model_sparsity(self, model: nn.Module) -> float:
        """Compute actual sparsity of model."""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0


class GradualPruning:
    """
    Gradual magnitude pruning during training.
    
    Slowly increases sparsity over training for better accuracy.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.3,
        start_epoch: int = 10,
        end_epoch: int = 40,
        pruning_frequency: int = 100,  # steps
    ):
        self.model = model
        self.target_sparsity = target_sparsity
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.pruning_frequency = pruning_frequency
        
        self.pruner = StructuredPruner(target_sparsity=target_sparsity)
        self.current_sparsity = 0.0
        self.step_count = 0
    
    def compute_current_sparsity(self, epoch: int) -> float:
        """Compute target sparsity for current epoch (cubic schedule)."""
        if epoch < self.start_epoch:
            return 0.0
        if epoch >= self.end_epoch:
            return self.target_sparsity
        
        # Cubic schedule
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        sparsity = self.target_sparsity * (1 - (1 - progress) ** 3)
        
        return sparsity
    
    def step(self, epoch: int):
        """Called each training step."""
        self.step_count += 1
        
        if self.step_count % self.pruning_frequency != 0:
            return
        
        target_sparsity = self.compute_current_sparsity(epoch)
        
        if target_sparsity > self.current_sparsity:
            # Apply pruning
            self.pruner.prune_model(self.model, target_sparsity)
            self.current_sparsity = target_sparsity
    
    def finalize(self):
        """Apply final pruning and make masks permanent."""
        # Remove pruning reparameterization to make masks permanent
        for module, name in self.pruner.get_prunable_layers(self.model):
            if prune.is_pruned(module):
                prune.remove(module, 'weight')


def magnitude_pruning(
    model: nn.Module,
    sparsity: float,
    structured: bool = True,
) -> nn.Module:
    """Convenience function for one-shot magnitude pruning."""
    pruner = StructuredPruner(
        target_sparsity=sparsity,
        pruning_method="magnitude",
        structured=structured,
    )
    return pruner.prune_model(model)


class FilterPruning(nn.Module):
    """
    Filter (channel) pruning with reconstruction.
    
    More sophisticated structured pruning that considers inter-layer dependencies.
    """
    
    def __init__(
        self,
        model: nn.Module,
        prune_ratio: float = 0.3,
    ):
        super().__init__()
        self.model = model
        self.prune_ratio = prune_ratio
        
        # Build dependency graph
        self.dependencies = self._build_dependencies()
    
    def _build_dependencies(self) -> Dict[str, List[str]]:
        """Build dependency graph between layers."""
        # Simplified version - full implementation would trace computation graph
        dependencies = {}
        
        prev_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if prev_layer is not None:
                    dependencies[name] = [prev_layer]
                prev_layer = name
        
        return dependencies
    
    def prune(self) -> nn.Module:
        """Apply filter pruning with dependency handling."""
        # This is a simplified implementation
        # Full implementation would handle skip connections, batch norm fusion, etc.
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Compute importance
                weight = module.weight.data
                importance = weight.view(weight.size(0), -1).norm(p=2, dim=1)
                
                # Get channels to keep
                num_keep = int(len(importance) * (1 - self.prune_ratio))
                keep_indices = torch.topk(importance, num_keep).indices
                
                # Prune output channels
                new_weight = weight[keep_indices]
                
                # Update module
                new_conv = nn.Conv2d(
                    module.in_channels,
                    num_keep,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    bias=module.bias is not None,
                )
                new_conv.weight.data = new_weight
                if module.bias is not None:
                    new_conv.bias.data = module.bias.data[keep_indices]
                
                # Replace in model (would need proper module replacement)
        
        return self.model

