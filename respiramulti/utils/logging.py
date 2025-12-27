"""
Logging utilities for training.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    experiment_name: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for training.
    
    Creates console and file handlers.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logger
    logger = logging.getLogger("respiramulti")
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    log_file = log_dir / f"{experiment_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")
    
    return logger


class TrainingLogger:
    """
    Training logger with metrics tracking and checkpoint management.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = "respira-multi",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Python logging
        self.logger = setup_logging(
            str(self.experiment_dir),
            experiment_name=experiment_name,
        )
        
        # Wandb integration
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    dir=str(self.log_dir),
                )
                self.wandb = wandb
            except ImportError:
                self.logger.warning("wandb not installed, disabling")
                self.use_wandb = False
        
        # Metrics history
        self.history: Dict[str, list] = {}
        self.best_metrics: Dict[str, float] = {}
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Config saved to {config_path}")
        
        if self.use_wandb:
            self.wandb.config.update(config)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "",
    ):
        """Log metrics."""
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            
            # Add to history
            if full_key not in self.history:
                self.history[full_key] = []
            self.history[full_key].append(value)
            
            # Track best
            if 'auroc' in key.lower() or 'auprc' in key.lower():
                if full_key not in self.best_metrics or value > self.best_metrics[full_key]:
                    self.best_metrics[full_key] = value
        
        # Log to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        if step is not None:
            self.logger.info(f"Step {step} - {prefix} - {metrics_str}")
        else:
            self.logger.info(f"{prefix} - {metrics_str}")
        
        # Log to wandb
        if self.use_wandb:
            log_dict = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
            if step is not None:
                log_dict['step'] = step
            self.wandb.log(log_dict)
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Log epoch metrics."""
        self.log_metrics(train_metrics, step=epoch, prefix="train")
        self.log_metrics(val_metrics, step=epoch, prefix="val")
    
    def save_history(self):
        """Save metrics history to JSON."""
        history_path = self.experiment_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info(f"History saved to {history_path}")
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics achieved during training."""
        return self.best_metrics
    
    def finish(self):
        """Finish logging, save history, close wandb."""
        self.save_history()
        
        # Log best metrics
        self.logger.info("=" * 50)
        self.logger.info("Best metrics:")
        for key, value in self.best_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        self.logger.info("=" * 50)
        
        if self.use_wandb:
            self.wandb.finish()

