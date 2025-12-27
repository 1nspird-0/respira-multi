"""
Model export for mobile deployment.

Exports to ONNX, TorchScript, and TFLite formats.
"""

import torch
import torch.nn as nn
import torch.onnx
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json


class ModelExporter:
    """
    Export trained model for mobile deployment.
    """
    
    def __init__(
        self,
        model: nn.Module,
        output_dir: str = "exports",
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_onnx(
        self,
        sample_inputs: Dict[str, torch.Tensor],
        output_name: str = "model.onnx",
        opset_version: int = 13,
        dynamic_axes: Optional[Dict] = None,
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            sample_inputs: Dict of sample input tensors
            output_name: Output filename
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
        """
        output_path = self.output_dir / output_name
        
        self.model.eval()
        
        # Prepare inputs
        inputs = (
            sample_inputs['audio_tokens'],
            sample_inputs['segment_types'],
            sample_inputs['vitals'],
            sample_inputs.get('audio_mask'),
            sample_inputs.get('vitals_mask'),
            sample_inputs.get('demographics'),
        )
        
        input_names = [
            'audio_tokens',
            'segment_types',
            'vitals',
            'audio_mask',
            'vitals_mask',
            'demographics',
        ]
        
        output_names = [
            'disease_logits',
            'concept_logits',
        ]
        
        # Default dynamic axes (for variable sequence lengths)
        if dynamic_axes is None:
            dynamic_axes = {
                'audio_tokens': {0: 'batch', 1: 'num_tokens'},
                'segment_types': {0: 'batch', 1: 'num_tokens'},
                'vitals': {0: 'batch'},
                'disease_logits': {0: 'batch'},
                'concept_logits': {0: 'batch'},
            }
        
        # Export
        torch.onnx.export(
            self.model,
            inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )
        
        print(f"Exported ONNX model to {output_path}")
        
        # Verify export
        try:
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification passed")
        except ImportError:
            print("Install onnx to verify exported model")
        except Exception as e:
            print(f"ONNX verification failed: {e}")
        
        return str(output_path)
    
    def export_torchscript(
        self,
        sample_inputs: Dict[str, torch.Tensor],
        output_name: str = "model.pt",
        optimize_for_mobile: bool = True,
    ) -> str:
        """
        Export model to TorchScript format.
        """
        output_path = self.output_dir / output_name
        
        self.model.eval()
        
        # Trace model
        with torch.no_grad():
            traced = torch.jit.trace(
                self.model,
                example_inputs=(
                    sample_inputs['audio_tokens'],
                    sample_inputs['segment_types'],
                    sample_inputs['vitals'],
                    sample_inputs.get('audio_mask'),
                    sample_inputs.get('vitals_mask'),
                    sample_inputs.get('demographics'),
                ),
            )
        
        # Optimize for mobile if requested
        if optimize_for_mobile:
            try:
                from torch.utils.mobile_optimizer import optimize_for_mobile
                traced = optimize_for_mobile(traced)
                print("Applied mobile optimizations")
            except ImportError:
                print("Mobile optimizer not available")
        
        # Save
        traced.save(str(output_path))
        print(f"Exported TorchScript model to {output_path}")
        
        return str(output_path)
    
    def export_tflite(
        self,
        sample_inputs: Dict[str, torch.Tensor],
        output_name: str = "model.tflite",
        quantize: bool = True,
    ) -> str:
        """
        Export model to TFLite format via ONNX.
        
        Requires onnx-tf package.
        """
        output_path = self.output_dir / output_name
        
        # First export to ONNX
        onnx_path = self.export_onnx(
            sample_inputs,
            output_name="temp_model.onnx",
        )
        
        try:
            import onnx
            from onnx_tf.backend import prepare
            import tensorflow as tf
            
            # Load ONNX and convert to TF
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            
            # Export to SavedModel
            saved_model_path = self.output_dir / "temp_saved_model"
            tf_rep.export_graph(str(saved_model_path))
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]
            
            tflite_model = converter.convert()
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"Exported TFLite model to {output_path}")
            
            # Cleanup temp files
            import shutil
            shutil.rmtree(saved_model_path, ignore_errors=True)
            Path(onnx_path).unlink(missing_ok=True)
            
        except ImportError as e:
            print(f"TFLite export requires additional packages: {e}")
            print("Install with: pip install onnx-tf tensorflow")
            return ""
        except Exception as e:
            print(f"TFLite export failed: {e}")
            return ""
        
        return str(output_path)
    
    def export_all(
        self,
        sample_inputs: Dict[str, torch.Tensor],
        model_name: str = "respiramulti",
        include_tflite: bool = False,
    ) -> Dict[str, str]:
        """Export to all supported formats."""
        exports = {}
        
        # ONNX
        exports['onnx'] = self.export_onnx(
            sample_inputs,
            output_name=f"{model_name}.onnx",
        )
        
        # TorchScript
        exports['torchscript'] = self.export_torchscript(
            sample_inputs,
            output_name=f"{model_name}.pt",
        )
        
        # TFLite (optional)
        if include_tflite:
            exports['tflite'] = self.export_tflite(
                sample_inputs,
                output_name=f"{model_name}.tflite",
            )
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'exports': exports,
            'input_shapes': {
                k: list(v.shape) for k, v in sample_inputs.items()
            },
        }
        
        with open(self.output_dir / "export_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return exports
    
    def benchmark_inference(
        self,
        sample_inputs: Dict[str, torch.Tensor],
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark inference latency.
        """
        import time
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in sample_inputs.items()}
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                self.model(
                    audio_tokens=inputs['audio_tokens'],
                    segment_types=inputs['segment_types'],
                    vitals=inputs['vitals'],
                    audio_mask=inputs.get('audio_mask'),
                    vitals_mask=inputs.get('vitals_mask'),
                    demographics=inputs.get('demographics'),
                )
        
        # Synchronize if CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                self.model(
                    audio_tokens=inputs['audio_tokens'],
                    segment_types=inputs['segment_types'],
                    vitals=inputs['vitals'],
                    audio_mask=inputs.get('audio_mask'),
                    vitals_mask=inputs.get('vitals_mask'),
                    demographics=inputs.get('demographics'),
                )
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / num_runs) * 1000
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Model size
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        
        return {
            'avg_latency_ms': avg_time_ms,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
        }


def create_sample_inputs(
    batch_size: int = 1,
    num_tokens: int = 10,
    n_mels: int = 64,
    time_frames: int = 201,
    vitals_dim: int = 15,
    demographics_dim: int = 15,
    device: str = 'cpu',
) -> Dict[str, torch.Tensor]:
    """Create sample inputs for export."""
    return {
        'audio_tokens': torch.randn(batch_size, num_tokens, 1, n_mels, time_frames, device=device),
        'segment_types': torch.zeros(batch_size, num_tokens, dtype=torch.long, device=device),
        'vitals': torch.randn(batch_size, vitals_dim, device=device),
        'audio_mask': torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device),
        'vitals_mask': torch.ones(batch_size, vitals_dim, device=device),
        'demographics': torch.randn(batch_size, demographics_dim, device=device),
    }

