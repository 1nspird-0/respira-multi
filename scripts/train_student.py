#!/usr/bin/env python3
"""
Train student model with teacher distillation.

Usage:
    python scripts/train_student.py --config configs/student.yaml
"""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from respiramulti.student.student_model import create_student_model
from respiramulti.teachers.ensemble import TeacherEnsemble
from respiramulti.distillation.trainer import DistillationTrainer
from respiramulti.datasets.unified_loader import UnifiedDataLoader
from respiramulti.utils.logging import TrainingLogger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train RESPIRA-MULTI student model')
    parser.add_argument('--config', type=str, default='configs/student.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize logger
    logger = TrainingLogger(
        log_dir=args.output_dir,
        use_wandb=config.get('logging', {}).get('use_wandb', False),
    )
    logger.log_config(config)
    
    # Create data loaders
    print("Loading data...")
    data_loader = UnifiedDataLoader(
        config=config,
        index_dir=config.get('data', {}).get('index_dir', 'data/indices'),
        processed_dir=config.get('data', {}).get('processed_dir', 'data/processed'),
    )
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create student model
    print("Creating student model...")
    student = create_student_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create teacher ensemble
    print("Creating teacher ensemble...")
    teachers = TeacherEnsemble(
        num_diseases=config.get('labels', {}).get('num_diseases', 12),
        num_concepts=config.get('labels', {}).get('num_concepts', 17),
        enable_beats=config.get('teachers', {}).get('beats', {}).get('enabled', True),
        enable_audio_mae=config.get('teachers', {}).get('audio_mae', {}).get('enabled', True),
        enable_ast=config.get('teachers', {}).get('ast', {}).get('enabled', True),
        enable_speech=config.get('teachers', {}).get('speech_encoder', {}).get('enabled', True),
        config=config.get('teachers', {}),
    )
    
    # Create trainer
    trainer = DistillationTrainer(
        student=student,
        teachers=teachers,
        config=config,
        device=device,
        output_dir=args.output_dir,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Training stages
    distill_config = config.get('distillation_training', {})
    
    # Stage 1: Pure distillation
    print("\n" + "=" * 60)
    print("STAGE 1: Pure Distillation")
    print("=" * 60)
    
    stage1_config = distill_config.get('stage1', {})
    stage1_history = trainer.train_stage1(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=stage1_config.get('epochs', 30),
        lr=stage1_config.get('lr', 5e-4),
    )
    
    # Stage 2: Mixed training
    print("\n" + "=" * 60)
    print("STAGE 2: Mixed Training (Hard Labels + Distillation)")
    print("=" * 60)
    
    stage2_config = distill_config.get('stage2', {})
    stage2_history = trainer.train_stage2(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=stage2_config.get('epochs', 40),
        lr=stage2_config.get('lr', 1e-4),
        freeze_backbone_epochs=stage2_config.get('freeze_backbone_epochs', 5),
    )
    
    # Stage 3: Calibration
    print("\n" + "=" * 60)
    print("STAGE 3: Calibration")
    print("=" * 60)
    
    calibration_results = trainer.train_stage3_calibration(val_loader)
    
    print(f"\nCalibration complete:")
    print(f"  ECE before: {calibration_results['ece_before']:.4f}")
    print(f"  ECE after: {calibration_results['ece_after']:.4f}")
    
    # Finish logging
    logger.finish()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model saved to: {args.output_dir}/best_stage2.pt")
    print(f"Final calibrated model: {args.output_dir}/final_calibrated.pt")
    print("=" * 60)


if __name__ == '__main__':
    main()

