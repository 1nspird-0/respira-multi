"""
Distillation trainer for RESPIRA-MULTI.

Implements the 3-stage training pipeline:
S1: Pure distillation (no hard labels)
S2: Mixed training (hard labels + distillation)
S3: Calibration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, List, Callable
from pathlib import Path
import json
from tqdm import tqdm
import wandb

from respiramulti.distillation.losses import CombinedDistillationLoss
from respiramulti.student.student_model import RespiraMultiStudent
from respiramulti.teachers.ensemble import TeacherEnsemble
from respiramulti.datasets.unified_loader import BatchedSession
from respiramulti.utils.metrics import compute_metrics


class DistillationTrainer:
    """
    Training manager for teacher-student distillation.
    
    Handles the complete 3-stage training pipeline.
    """
    
    def __init__(
        self,
        student: RespiraMultiStudent,
        teachers: TeacherEnsemble,
        config: Dict,
        device: str = 'cuda',
        output_dir: str = 'outputs',
    ):
        self.student = student.to(device)
        self.teachers = teachers.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Freeze teachers
        self.teachers.freeze_teachers()
        self.teachers.eval()
        
        # Training config
        train_config = config.get('training', {})
        self.use_amp = train_config.get('use_amp', True)
        self.max_grad_norm = train_config.get('max_grad_norm', 1.0)
        
        # Initialize scaler for mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging
        self.use_wandb = config.get('logging', {}).get('use_wandb', False)
        if self.use_wandb:
            wandb.init(project='respira-multi', config=config)
        
        # Best metrics tracking
        self.best_auroc = 0.0
        self.best_epoch = 0
    
    def create_optimizer(
        self,
        lr: float,
        weight_decay: float,
        freeze_backbone_epochs: int = 0,
    ) -> optim.Optimizer:
        """Create optimizer with optional backbone freezing."""
        if freeze_backbone_epochs > 0:
            # Freeze audio backbone initially
            for param in self.student.audio_encoder.parameters():
                param.requires_grad = False
        
        params = filter(lambda p: p.requires_grad, self.student.parameters())
        
        optimizer = optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        
        return optimizer
    
    def create_scheduler(
        self,
        optimizer: optim.Optimizer,
        num_epochs: int,
        warmup_epochs: int = 5,
    ):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_stage1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        lr: float = 5e-4,
    ) -> Dict:
        """
        Stage 1: Pure distillation training.
        
        No hard labels, only distillation from teachers.
        Can use unlabeled data.
        """
        print("=" * 50)
        print("Stage 1: Pure Distillation Training")
        print("=" * 50)
        
        # Loss function (no hard labels)
        criterion = CombinedDistillationLoss(
            temperature=self.config.get('distillation', {}).get('temperature', 4.0),
            logit_weight=1.0,
            feature_weight=0.5,
            hard_label_weight=0.0,  # No hard labels
            student_dim=self.student.d_model,
            teacher_dim=self.teachers.embed_dim,
        )
        
        optimizer = self.create_optimizer(lr, weight_decay=1e-4)
        scheduler = self.create_scheduler(optimizer, epochs)
        
        history = {'train_loss': [], 'val_loss': [], 'val_auroc': []}
        
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader, criterion, optimizer, epoch)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self._validate(val_loader, criterion)
            history['val_loss'].append(val_metrics['loss'])
            history['val_auroc'].append(val_metrics.get('auroc', 0))
            
            # Update scheduler
            scheduler.step()
            
            # Log
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val AUROC: {val_metrics.get('auroc', 0):.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'stage1/train_loss': train_loss,
                    'stage1/val_loss': val_metrics['loss'],
                    'stage1/val_auroc': val_metrics.get('auroc', 0),
                    'stage1/lr': scheduler.get_last_lr()[0],
                })
            
            # Save best model
            if val_metrics.get('auroc', 0) > self.best_auroc:
                self.best_auroc = val_metrics['auroc']
                self.best_epoch = epoch
                self.save_checkpoint('best_stage1.pt', epoch, val_metrics)
        
        # Save final stage 1 model
        self.save_checkpoint('final_stage1.pt', epochs, val_metrics)
        
        return history
    
    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 40,
        lr: float = 1e-4,
        freeze_backbone_epochs: int = 5,
    ) -> Dict:
        """
        Stage 2: Mixed training with hard labels and distillation.
        """
        print("=" * 50)
        print("Stage 2: Mixed Training (Hard Labels + Distillation)")
        print("=" * 50)
        
        distill_config = self.config.get('distillation_training', {}).get('stage2', {})
        
        # Loss function with hard labels
        criterion = CombinedDistillationLoss(
            temperature=self.config.get('distillation', {}).get('temperature', 4.0),
            logit_weight=distill_config.get('loss_weights', {}).get('logit_kl', 0.3),
            feature_weight=distill_config.get('loss_weights', {}).get('feature_l2', 0.2),
            hard_label_weight=distill_config.get('loss_weights', {}).get('hard_label', 0.7),
            concept_weight=distill_config.get('loss_weights', {}).get('concept', 0.5),
            hierarchy_weight=distill_config.get('loss_weights', {}).get('hierarchy', 0.1),
            student_dim=self.student.d_model,
            teacher_dim=self.teachers.embed_dim,
        )
        
        optimizer = self.create_optimizer(
            lr, weight_decay=1e-4, 
            freeze_backbone_epochs=freeze_backbone_epochs
        )
        scheduler = self.create_scheduler(optimizer, epochs)
        
        history = {'train_loss': [], 'val_loss': [], 'val_auroc': []}
        
        for epoch in range(epochs):
            # Unfreeze backbone after initial epochs
            if epoch == freeze_backbone_epochs:
                print("Unfreezing audio backbone...")
                for param in self.student.audio_encoder.parameters():
                    param.requires_grad = True
                # Rebuild optimizer with all parameters
                optimizer = self.create_optimizer(lr * 0.1, weight_decay=1e-4)
            
            # Train
            train_loss = self._train_epoch(
                train_loader, criterion, optimizer, epoch,
                use_hard_labels=True
            )
            history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self._validate(val_loader, criterion)
            history['val_loss'].append(val_metrics['loss'])
            history['val_auroc'].append(val_metrics.get('auroc', 0))
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val AUROC: {val_metrics.get('auroc', 0):.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'stage2/train_loss': train_loss,
                    'stage2/val_loss': val_metrics['loss'],
                    'stage2/val_auroc': val_metrics.get('auroc', 0),
                })
            
            if val_metrics.get('auroc', 0) > self.best_auroc:
                self.best_auroc = val_metrics['auroc']
                self.best_epoch = epoch
                self.save_checkpoint('best_stage2.pt', epoch, val_metrics)
        
        self.save_checkpoint('final_stage2.pt', epochs, val_metrics)
        
        return history
    
    def train_stage3_calibration(
        self,
        val_loader: DataLoader,
    ) -> Dict:
        """
        Stage 3: Temperature scaling calibration.
        """
        print("=" * 50)
        print("Stage 3: Calibration (Temperature Scaling)")
        print("=" * 50)
        
        self.student.eval()
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                output = self.student(
                    audio_tokens=batch.audio_tokens,
                    segment_types=batch.audio_segment_types,
                    vitals=batch.vitals,
                    audio_mask=batch.audio_mask.bool(),
                    vitals_mask=batch.vitals_mask,
                    demographics=batch.demographics,
                )
                
                all_logits.append(output.disease_logits.cpu())
                all_labels.append(batch.disease_labels.cpu())
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Learn temperature per disease
        temperatures = self._learn_temperature(all_logits, all_labels)
        
        # Store temperatures
        self.student.disease_temperatures = nn.Parameter(
            torch.tensor(temperatures, device=self.device),
            requires_grad=False,
        )
        
        # Compute ECE before and after
        ece_before = self._compute_ece(all_logits, all_labels, temperature=1.0)
        ece_after = self._compute_ece(all_logits, all_labels, temperature=temperatures)
        
        print(f"ECE before calibration: {ece_before:.4f}")
        print(f"ECE after calibration: {ece_after:.4f}")
        
        self.save_checkpoint('final_calibrated.pt', -1, {'ece': ece_after})
        
        return {
            'temperatures': temperatures,
            'ece_before': ece_before,
            'ece_after': ece_after,
        }
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        use_hard_labels: bool = False,
    ) -> float:
        """Run one training epoch."""
        self.student.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                # Get teacher outputs
                with torch.no_grad():
                    teacher_output = self.teachers(
                        spectrogram=batch.audio_tokens[:, 0],  # Use first token for teacher
                        return_attention=False,
                    )
                
                # Get student outputs
                student_output = self.student(
                    audio_tokens=batch.audio_tokens,
                    segment_types=batch.audio_segment_types,
                    vitals=batch.vitals,
                    audio_mask=batch.audio_mask.bool(),
                    vitals_mask=batch.vitals_mask,
                    demographics=batch.demographics,
                )
                
                # Prepare outputs dict
                student_dict = {
                    'disease_logits': student_output.disease_logits,
                    'concept_logits': student_output.concept_logits,
                    'cls_embedding': student_output.cls_embedding,
                    'token_embeddings': student_output.token_embeddings,
                    'gate_entropy_penalty': student_output.gate_entropy_penalty,
                }
                
                teacher_dict = {
                    'disease_logits': teacher_output.disease_logits,
                    'concept_logits': teacher_output.concept_logits,
                    'cls_embedding': teacher_output.cls_embedding,
                    'token_embeddings': teacher_output.token_embeddings,
                }
                
                # Compute loss
                if use_hard_labels:
                    losses = criterion(
                        student_dict, teacher_dict,
                        disease_labels=batch.disease_labels,
                        disease_mask=batch.disease_mask,
                        concept_labels=batch.concept_labels,
                        concept_mask=batch.concept_mask,
                    )
                else:
                    losses = criterion(student_dict, teacher_dict)
                
                loss = losses['total']
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict:
        """Run validation."""
        self.student.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_masks = []
        
        for batch in val_loader:
            batch = batch.to(self.device)
            
            # Teacher forward
            teacher_output = self.teachers(
                spectrogram=batch.audio_tokens[:, 0],
            )
            
            # Student forward
            student_output = self.student(
                audio_tokens=batch.audio_tokens,
                segment_types=batch.audio_segment_types,
                vitals=batch.vitals,
                audio_mask=batch.audio_mask.bool(),
                vitals_mask=batch.vitals_mask,
                demographics=batch.demographics,
            )
            
            # Compute loss
            student_dict = {
                'disease_logits': student_output.disease_logits,
                'concept_logits': student_output.concept_logits,
                'cls_embedding': student_output.cls_embedding,
                'token_embeddings': student_output.token_embeddings,
                'gate_entropy_penalty': student_output.gate_entropy_penalty,
            }
            
            teacher_dict = {
                'disease_logits': teacher_output.disease_logits,
                'concept_logits': teacher_output.concept_logits,
                'cls_embedding': teacher_output.cls_embedding,
                'token_embeddings': teacher_output.token_embeddings,
            }
            
            losses = criterion(
                student_dict, teacher_dict,
                disease_labels=batch.disease_labels,
                disease_mask=batch.disease_mask,
            )
            
            total_loss += losses['total'].item()
            
            all_preds.append(student_output.disease_logits.cpu())
            all_labels.append(batch.disease_labels.cpu())
            all_masks.append(batch.disease_mask.cpu())
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        probs = torch.sigmoid(all_preds)
        
        metrics = compute_metrics(probs, all_labels, all_masks)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def _learn_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        num_iterations: int = 50,
    ) -> List[float]:
        """Learn temperature scaling for each disease."""
        num_diseases = logits.shape[1]
        temperatures = []
        
        for d in range(num_diseases):
            temp = nn.Parameter(torch.ones(1))
            optimizer = optim.LBFGS([temp], lr=0.01, max_iter=num_iterations)
            
            def closure():
                optimizer.zero_grad()
                scaled_logits = logits[:, d] / temp
                loss = F.binary_cross_entropy_with_logits(
                    scaled_logits, labels[:, d]
                )
                loss.backward()
                return loss
            
            optimizer.step(closure)
            temperatures.append(temp.item())
        
        return temperatures
    
    def _compute_ece(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0,
        num_bins: int = 15,
    ) -> float:
        """Compute Expected Calibration Error."""
        if isinstance(temperature, list):
            probs = torch.sigmoid(logits / torch.tensor(temperature))
        else:
            probs = torch.sigmoid(logits / temperature)
        
        # Flatten
        probs_flat = probs.flatten()
        labels_flat = labels.flatten()
        
        # Bin boundaries
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        
        ece = 0.0
        for i in range(num_bins):
            in_bin = (probs_flat > bin_boundaries[i]) & (probs_flat <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_acc = labels_flat[in_bin].float().mean()
                bin_conf = probs_flat[in_bin].mean()
                ece += (in_bin.float().mean() * abs(bin_acc - bin_conf)).item()
        
        return ece
    
    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }
        
        if hasattr(self.student, 'disease_temperatures'):
            checkpoint['temperatures'] = self.student.disease_temperatures.cpu().tolist()
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint['model_state_dict'])
        
        if 'temperatures' in checkpoint:
            self.student.disease_temperatures = nn.Parameter(
                torch.tensor(checkpoint['temperatures'], device=self.device),
                requires_grad=False,
            )
        
        print(f"Loaded checkpoint from {path}")
        return checkpoint

