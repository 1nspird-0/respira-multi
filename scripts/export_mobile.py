#!/usr/bin/env python3
"""
Export trained model for mobile deployment.

Usage:
    python scripts/export_mobile.py --checkpoint outputs/final_calibrated.pt --config configs/mobile_int8.yaml
"""

import argparse
import yaml
from pathlib import Path
import torch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from respiramulti.student.student_model import create_student_model
from respiramulti.optimization.quantization import QuantizationAwareTraining
from respiramulti.optimization.pruning import magnitude_pruning
from respiramulti.optimization.export import ModelExporter, create_sample_inputs


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Export model for mobile deployment')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/mobile_int8.yaml',
                        help='Path to export configuration')
    parser.add_argument('--output_dir', type=str, default='exports',
                        help='Output directory')
    parser.add_argument('--quantize', action='store_true',
                        help='Apply quantization')
    parser.add_argument('--prune', action='store_true',
                        help='Apply pruning')
    parser.add_argument('--prune_ratio', type=float, default=0.3,
                        help='Pruning ratio')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create model
    print("Loading model...")
    model = create_student_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Apply pruning if requested
    if args.prune:
        print(f"Applying pruning with ratio {args.prune_ratio}...")
        model = magnitude_pruning(model, args.prune_ratio, structured=True)
    
    # Apply quantization if requested
    if args.quantize:
        print("Applying static quantization...")
        qat = QuantizationAwareTraining()
        model = qat.convert_to_quantized(model)
    
    # Create sample inputs
    print("Creating sample inputs...")
    sample_inputs = create_sample_inputs(
        batch_size=1,
        num_tokens=10,
        n_mels=64,
        time_frames=201,
    )
    
    # Export
    print("Exporting model...")
    exporter = ModelExporter(model, output_dir=args.output_dir)
    
    exports = exporter.export_all(
        sample_inputs=sample_inputs,
        model_name="respiramulti_mobile",
        include_tflite=config.get('export', {}).get('formats', {}).get('tflite', {}).get('enabled', False),
    )
    
    # Benchmark
    print("\nBenchmarking inference...")
    benchmark = exporter.benchmark_inference(sample_inputs)
    
    print("\n" + "=" * 50)
    print("EXPORT COMPLETE")
    print("=" * 50)
    print(f"\nExported models:")
    for fmt, path in exports.items():
        if path:
            print(f"  {fmt}: {path}")
    
    print(f"\nModel statistics:")
    print(f"  Total parameters: {benchmark['total_params']:,}")
    print(f"  Model size: {benchmark['model_size_mb']:.2f} MB")
    print(f"  Avg latency (CPU): {benchmark['avg_latency_ms']:.2f} ms")
    
    # Check against targets
    target_latency = config.get('performance', {}).get('target_latency_ms', 150)
    target_size = config.get('performance', {}).get('target_model_size_mb', 15)
    
    print(f"\nPerformance targets:")
    latency_ok = benchmark['avg_latency_ms'] < target_latency
    size_ok = benchmark['model_size_mb'] < target_size
    
    print(f"  Latency: {benchmark['avg_latency_ms']:.2f}ms / {target_latency}ms "
          f"{'✓' if latency_ok else '✗'}")
    print(f"  Size: {benchmark['model_size_mb']:.2f}MB / {target_size}MB "
          f"{'✓' if size_ok else '✗'}")
    
    print("=" * 50)


if __name__ == '__main__':
    main()

