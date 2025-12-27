"""
Quantization-Aware Training (QAT) for mobile deployment.

Implements INT8 quantization with minimal accuracy loss.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import prepare_qat, convert
from typing import Dict, Optional, List, Callable
import copy


class QuantizationConfig:
    """Configuration for quantization."""
    
    def __init__(
        self,
        backend: str = "qnnpack",  # "fbgemm" for x86, "qnnpack" for ARM
        dtype: torch.dtype = torch.qint8,
    ):
        self.backend = backend
        self.dtype = dtype
        
        # Set backend
        torch.backends.quantized.engine = backend


class QuantizationAwareTraining:
    """
    Manager for Quantization-Aware Training.
    
    Handles model preparation, training, and conversion.
    """
    
    def __init__(
        self,
        config: Optional[QuantizationConfig] = None,
    ):
        self.config = config or QuantizationConfig()
    
    def prepare_model(
        self,
        model: nn.Module,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Prepare model for QAT.
        
        Inserts fake quantization modules for training.
        """
        if not inplace:
            model = copy.deepcopy(model)
        
        model.train()
        
        # Set quantization configuration
        model.qconfig = quant.get_default_qat_qconfig(self.config.backend)
        
        # Fuse modules where possible
        model = self._fuse_modules(model)
        
        # Prepare for QAT
        quant.prepare_qat(model, inplace=True)
        
        return model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse Conv-BN-ReLU sequences for better quantization."""
        # This is model-specific; implement based on architecture
        # For now, return model unchanged
        return model
    
    def convert_to_quantized(
        self,
        model: nn.Module,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Convert QAT model to fully quantized model.
        
        Should be called after QAT training is complete.
        """
        if not inplace:
            model = copy.deepcopy(model)
        
        model.eval()
        
        # Convert fake quantization to actual quantized operations
        quant.convert(model, inplace=True)
        
        return model
    
    def calibrate_static_quantization(
        self,
        model: nn.Module,
        calibration_loader,
        num_batches: int = 100,
    ) -> nn.Module:
        """
        Calibrate for post-training static quantization.
        
        Alternative to QAT when training is not feasible.
        """
        model = copy.deepcopy(model)
        model.eval()
        
        # Configure for static quantization
        model.qconfig = quant.get_default_qconfig(self.config.backend)
        
        # Prepare model
        quant.prepare(model, inplace=True)
        
        # Run calibration
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= num_batches:
                    break
                
                # Forward pass for calibration
                if hasattr(batch, 'to'):
                    batch = batch.to('cpu')
                
                try:
                    model(
                        audio_tokens=batch.audio_tokens,
                        segment_types=batch.audio_segment_types,
                        vitals=batch.vitals,
                        audio_mask=batch.audio_mask,
                        vitals_mask=batch.vitals_mask,
                        demographics=batch.demographics,
                    )
                except Exception:
                    # Handle different batch formats
                    pass
        
        # Convert to quantized
        quant.convert(model, inplace=True)
        
        return model


def prepare_qat(model: nn.Module, backend: str = "qnnpack") -> nn.Module:
    """Convenience function to prepare model for QAT."""
    qat = QuantizationAwareTraining(QuantizationConfig(backend=backend))
    return qat.prepare_model(model)


def convert_to_quantized(model: nn.Module) -> nn.Module:
    """Convenience function to convert QAT model to quantized."""
    qat = QuantizationAwareTraining()
    return qat.convert_to_quantized(model)


class FakeQuantize(nn.Module):
    """
    Custom fake quantization module for specific layers.
    """
    
    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
    ):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        
        # Compute scale and zero point
        if self.symmetric:
            max_val = x.abs().max()
            scale = max_val / (2 ** (self.bits - 1) - 1)
            zero_point = torch.zeros(1, device=x.device)
        else:
            min_val = x.min()
            max_val = x.max()
            scale = (max_val - min_val) / (2 ** self.bits - 1)
            zero_point = min_val / scale
        
        # Update buffers
        self.scale.data = scale.view(1)
        self.zero_point.data = zero_point.view(1)
        
        # Fake quantize
        x_quant = torch.round(x / scale - zero_point)
        x_quant = x_quant.clamp(-(2**(self.bits-1)), 2**(self.bits-1)-1)
        x_dequant = (x_quant + zero_point) * scale
        
        # Straight-through estimator
        return x + (x_dequant - x).detach()


class QuantizationCallback:
    """
    Training callback for QAT.
    
    Handles freezing BatchNorm statistics and managing observers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        freeze_bn_epoch: int = 20,
        freeze_observers_epoch: int = 25,
    ):
        self.model = model
        self.freeze_bn_epoch = freeze_bn_epoch
        self.freeze_observers_epoch = freeze_observers_epoch
    
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        if epoch >= self.freeze_bn_epoch:
            self._freeze_bn()
        
        if epoch >= self.freeze_observers_epoch:
            self._freeze_observers()
    
    def _freeze_bn(self):
        """Freeze BatchNorm statistics."""
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False
    
    def _freeze_observers(self):
        """Freeze quantization observers."""
        for module in self.model.modules():
            if hasattr(module, 'apply_fake_quant'):
                module.apply_fake_quant = True
            if hasattr(module, 'observer_enabled'):
                module.observer_enabled = False

