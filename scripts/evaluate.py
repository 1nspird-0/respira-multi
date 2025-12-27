#!/usr/bin/env python3
"""
Evaluate trained model.

Usage:
    python scripts/evaluate.py --checkpoint outputs/final_calibrated.pt --config configs/student.yaml
"""

import argparse
import yaml
from pathlib import Path
import torch
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from respiramulti.student.student_model import create_student_model
from respiramulti.datasets.unified_loader import UnifiedDataLoader
from respiramulti.datasets.schema import DISEASES, CONCEPTS
from respiramulti.utils.metrics import compute_metrics, compute_per_disease_metrics
from respiramulti.uncertainty.calibration import CalibrationMetrics, TemperatureScaling
from respiramulti.uncertainty.conformal import ConformalPredictor


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Evaluate RESPIRA-MULTI model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/student.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--split', type=str, default='test',
                        help='Data split to evaluate on')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Create model
    print("Loading model...")
    model = create_student_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load temperature if available
    if 'temperatures' in checkpoint:
        model.disease_temperatures = torch.tensor(
            checkpoint['temperatures'], device=device
        )
    
    model = model.to(device)
    model.eval()
    
    # Create data loader
    print(f"Loading {args.split} data...")
    data_loader = UnifiedDataLoader(
        config=config,
        index_dir=config.get('data', {}).get('index_dir', 'data/indices'),
        processed_dir=config.get('data', {}).get('processed_dir', 'data/processed'),
    )
    test_loader = data_loader.get_dataloader(args.split, shuffle=False)
    
    print(f"Evaluating on {len(test_loader.dataset)} samples...")
    
    # Collect predictions
    all_disease_logits = []
    all_concept_logits = []
    all_disease_labels = []
    all_concept_labels = []
    all_disease_masks = []
    all_concept_masks = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            output = model(
                audio_tokens=batch.audio_tokens,
                segment_types=batch.audio_segment_types,
                vitals=batch.vitals,
                audio_mask=batch.audio_mask.bool(),
                vitals_mask=batch.vitals_mask,
                demographics=batch.demographics,
            )
            
            all_disease_logits.append(output.disease_logits.cpu())
            all_concept_logits.append(output.concept_logits.cpu())
            all_disease_labels.append(batch.disease_labels.cpu())
            all_concept_labels.append(batch.concept_labels.cpu())
            all_disease_masks.append(batch.disease_mask.cpu())
            all_concept_masks.append(batch.concept_mask.cpu())
    
    # Concatenate
    disease_logits = torch.cat(all_disease_logits, dim=0)
    concept_logits = torch.cat(all_concept_logits, dim=0)
    disease_labels = torch.cat(all_disease_labels, dim=0)
    concept_labels = torch.cat(all_concept_labels, dim=0)
    disease_masks = torch.cat(all_disease_masks, dim=0)
    concept_masks = torch.cat(all_concept_masks, dim=0)
    
    # Apply temperature scaling if available
    if hasattr(model, 'disease_temperatures'):
        disease_logits = disease_logits / model.disease_temperatures.cpu()
    
    disease_probs = torch.sigmoid(disease_logits)
    concept_probs = torch.sigmoid(concept_logits)
    
    # Compute metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Per-disease metrics
    print("\nPer-Disease Metrics:")
    print("-" * 50)
    
    per_disease_metrics = compute_per_disease_metrics(
        disease_probs, disease_labels, disease_masks, DISEASES
    )
    
    for name, metrics in per_disease_metrics.items():
        print(f"  {name}:")
        print(f"    AUROC: {metrics.auroc:.4f}")
        print(f"    AUPRC: {metrics.auprc:.4f}")
        print(f"    Sens@90Spec: {metrics.sensitivity_at_90_specificity:.4f}")
    
    # Aggregate metrics
    aggregate_metrics = compute_metrics(
        disease_probs, disease_labels, disease_masks, DISEASES
    )
    
    print("\nAggregate Metrics:")
    print("-" * 50)
    print(f"  Macro AUROC: {aggregate_metrics['auroc']:.4f}")
    print(f"  Macro AUPRC: {aggregate_metrics['auprc']:.4f}")
    print(f"  Sens@90Spec: {aggregate_metrics['sensitivity_at_90_spec']:.4f}")
    
    # Calibration metrics
    print("\nCalibration Metrics:")
    print("-" * 50)
    
    calibration_result = CalibrationMetrics.compute_all_metrics(
        disease_probs, disease_labels
    )
    print(f"  ECE: {calibration_result.ece:.4f}")
    print(f"  MCE: {calibration_result.mce:.4f}")
    print(f"  Brier Score: {calibration_result.brier_score:.4f}")
    
    # Conformal prediction
    print("\nConformal Prediction:")
    print("-" * 50)
    
    # Split data for calibration
    n_cal = len(disease_probs) // 2
    cal_probs = disease_probs[:n_cal]
    cal_labels = disease_labels[:n_cal]
    test_probs = disease_probs[n_cal:]
    test_labels = disease_labels[n_cal:]
    
    conformal = ConformalPredictor(
        num_classes=len(DISEASES),
        class_names=DISEASES,
    )
    conformal.calibrate(cal_probs, cal_labels)
    
    for coverage in [0.80, 0.90, 0.95]:
        emp_coverage = conformal.compute_empirical_coverage(
            test_probs, test_labels, coverage
        )
        abstain_stats = conformal.get_abstain_statistics(test_probs, coverage)
        
        print(f"  Coverage level {coverage*100:.0f}%:")
        print(f"    Empirical coverage: {emp_coverage['overall']:.4f}")
        print(f"    Abstain rate: {abstain_stats['abstain_rate']:.4f}")
    
    # Save results
    results = {
        'per_disease_metrics': {
            name: {
                'auroc': m.auroc,
                'auprc': m.auprc,
                'sensitivity_at_90_specificity': m.sensitivity_at_90_specificity,
            }
            for name, m in per_disease_metrics.items()
        },
        'aggregate_metrics': aggregate_metrics,
        'calibration': {
            'ece': calibration_result.ece,
            'mce': calibration_result.mce,
            'brier_score': calibration_result.brier_score,
        },
        'config': args.config,
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_samples': len(disease_probs),
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()

