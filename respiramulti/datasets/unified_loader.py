"""
Unified data loader for RESPIRA-MULTI.

Handles loading from multiple datasets and normalizing to SessionSchema.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import json
import jsonlines
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import random

from respiramulti.datasets.schema import (
    SessionSchema,
    DISEASES,
    CONCEPTS,
    BINARY_CONCEPTS,
    CONTINUOUS_CONCEPTS,
    AUDIO_SEGMENT_TYPES,
)
from respiramulti.datasets.audio_transforms import AudioTransforms, ModalityDropout
from respiramulti.features.spectrogram import SpectrogramExtractor


@dataclass
class BatchedSession:
    """Batched session data for model input."""
    # Audio features
    audio_tokens: torch.Tensor  # [batch, num_tokens, channels, freq, time]
    audio_mask: torch.Tensor  # [batch, num_tokens] - which tokens are valid
    audio_segment_types: torch.Tensor  # [batch, num_tokens] - segment type indices
    
    # Vitals features
    vitals: torch.Tensor  # [batch, vitals_dim]
    vitals_mask: torch.Tensor  # [batch, vitals_dim] - which features are valid
    
    # Demographics
    demographics: torch.Tensor  # [batch, demo_dim]
    
    # Labels
    disease_labels: torch.Tensor  # [batch, num_diseases]
    disease_mask: torch.Tensor  # [batch, num_diseases] - which labels are valid
    concept_labels: torch.Tensor  # [batch, num_concepts]
    concept_mask: torch.Tensor  # [batch, num_concepts]
    label_confidence: torch.Tensor  # [batch]
    
    # Metadata
    session_ids: List[str]
    subject_ids: List[str]
    
    def to(self, device: torch.device) -> "BatchedSession":
        """Move all tensors to device."""
        return BatchedSession(
            audio_tokens=self.audio_tokens.to(device),
            audio_mask=self.audio_mask.to(device),
            audio_segment_types=self.audio_segment_types.to(device),
            vitals=self.vitals.to(device),
            vitals_mask=self.vitals_mask.to(device),
            demographics=self.demographics.to(device),
            disease_labels=self.disease_labels.to(device),
            disease_mask=self.disease_mask.to(device),
            concept_labels=self.concept_labels.to(device),
            concept_mask=self.concept_mask.to(device),
            label_confidence=self.label_confidence.to(device),
            session_ids=self.session_ids,
            subject_ids=self.subject_ids,
        )


class UnifiedDataset(Dataset):
    """
    Unified dataset that loads from multiple sources.
    
    Normalizes all datasets to SessionSchema format and provides
    consistent preprocessing and augmentation.
    """
    
    def __init__(
        self,
        index_path: Union[str, Path],
        processed_dir: Union[str, Path],
        config: Dict[str, Any],
        split: str = "train",
        augment: bool = True,
        max_audio_tokens: int = 20,
        segment_length_sec: float = 2.0,
        segment_overlap: float = 0.5,
    ):
        """
        Initialize unified dataset.
        
        Args:
            index_path: Path to JSONL index file
            processed_dir: Directory with preprocessed features
            config: Configuration dictionary
            split: One of "train", "val", "test"
            augment: Whether to apply augmentations
            max_audio_tokens: Maximum number of audio tokens per session
            segment_length_sec: Length of each audio segment in seconds
            segment_overlap: Overlap ratio between segments
        """
        self.index_path = Path(index_path)
        self.processed_dir = Path(processed_dir)
        self.config = config
        self.split = split
        self.augment = augment and split == "train"
        self.max_audio_tokens = max_audio_tokens
        self.segment_length_sec = segment_length_sec
        self.segment_overlap = segment_overlap
        
        # Load index
        self.sessions = self._load_index()
        
        # Initialize spectrogram extractor
        audio_config = config.get("audio", {})
        self.spec_extractor = SpectrogramExtractor(
            sample_rate=audio_config.get("sample_rate", 16000),
            n_mels=audio_config.get("n_mels", 64),
            n_fft=audio_config.get("n_fft", 400),
            hop_length=audio_config.get("hop_length", 160),
            fmin=audio_config.get("fmin", 50),
            fmax=audio_config.get("fmax", 8000),
        )
        
        # Initialize augmentations
        if self.augment:
            aug_config = config.get("augmentation", {}).get("audio", {})
            self.audio_transforms = AudioTransforms(
                sample_rate=audio_config.get("sample_rate", 16000),
                config=aug_config,
            )
            self.audio_transforms.train()
            
            modality_config = config.get("augmentation", {}).get("modality_dropout", {})
            self.modality_dropout = ModalityDropout(
                audio_prob=modality_config.get("audio_prob", 0.15),
                vitals_prob=modality_config.get("vitals_prob", 0.2),
                per_segment_prob=modality_config.get("per_segment_prob", 0.1),
            ) if modality_config.get("enabled", True) else None
        else:
            self.audio_transforms = None
            self.modality_dropout = None
        
        # Segment type to index mapping
        self.segment_type_to_idx = {
            seg: i for i, seg in enumerate(AUDIO_SEGMENT_TYPES)
        }
        
        # Sample rate
        self.sample_rate = audio_config.get("sample_rate", 16000)
        self.segment_samples = int(segment_length_sec * self.sample_rate)
    
    def _load_index(self) -> List[SessionSchema]:
        """Load session index from JSONL file."""
        sessions = []
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        with jsonlines.open(self.index_path, 'r') as reader:
            for item in reader:
                try:
                    session = SessionSchema.from_dict(item)
                    sessions.append(session)
                except Exception as e:
                    print(f"Warning: Failed to parse session: {e}")
                    continue
        
        print(f"Loaded {len(sessions)} sessions from {self.index_path}")
        return sessions
    
    def __len__(self) -> int:
        return len(self.sessions)
    
    def _load_audio(self, path: str) -> Optional[torch.Tensor]:
        """Load and preprocess audio file."""
        try:
            path = Path(path)
            if not path.exists():
                return None
            
            waveform, sr = torchaudio.load(path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)
            
            # Resample if needed
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sr, self.sample_rate
                )
            
            return waveform
            
        except Exception as e:
            print(f"Warning: Failed to load audio {path}: {e}")
            return None
    
    def _segment_audio(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Segment long audio into windows."""
        segments = []
        step = int(self.segment_samples * (1 - self.segment_overlap))
        
        for start in range(0, len(waveform) - self.segment_samples + 1, step):
            segment = waveform[start:start + self.segment_samples]
            segments.append(segment)
        
        # Handle short audio
        if not segments:
            # Pad if too short
            if len(waveform) < self.segment_samples:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.segment_samples - len(waveform))
                )
            segments.append(waveform[:self.segment_samples])
        
        return segments
    
    def _process_audio_segment(
        self, 
        waveform: torch.Tensor,
        segment_type: str,
    ) -> Tuple[torch.Tensor, int]:
        """Process a single audio segment to spectrogram."""
        # Apply augmentations
        if self.audio_transforms and self.augment:
            waveform = self.audio_transforms.augment_waveform(waveform)
        
        # Extract spectrogram
        spec = self.spec_extractor(waveform)
        
        # Apply spectrogram augmentations
        if self.audio_transforms and self.augment:
            spec = self.audio_transforms.augment_spectrogram(spec.unsqueeze(0)).squeeze(0)
        
        segment_idx = self.segment_type_to_idx.get(segment_type, 0)
        
        return spec, segment_idx
    
    def _get_vitals_vector(self, session: SessionSchema) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract vitals vector and mask from session."""
        vitals = session.vitals
        
        # Build feature vector
        features = [
            vitals.hr_mean,
            vitals.hr_std,
            vitals.hrv_rmssd,
            vitals.hrv_sdnn,
            vitals.rr_est,
            vitals.spo2_est,
            vitals.perfusion_quality,
            vitals.hr_quality,
            vitals.rr_quality,
            vitals.spo2_quality,
        ]
        
        # Add demographics
        demo_vector = session.demographics.to_vector()
        features.extend(demo_vector[:5])  # age, sex, height, weight, smoker
        
        # Create tensor and mask
        mask = torch.tensor([f is not None for f in features], dtype=torch.float32)
        features = torch.tensor(
            [f if f is not None else 0.0 for f in features],
            dtype=torch.float32
        )
        
        # Normalize features (simple min-max for common ranges)
        # HR: 40-200 bpm
        features[0] = (features[0] - 80) / 40 if mask[0] else 0
        features[1] = features[1] / 20 if mask[1] else 0
        # HRV: 0-200ms
        features[2] = features[2] / 100 if mask[2] else 0
        features[3] = features[3] / 100 if mask[3] else 0
        # RR: 8-40 bpm
        features[4] = (features[4] - 15) / 10 if mask[4] else 0
        # SpO2: 80-100%
        features[5] = (features[5] - 95) / 5 if mask[5] else 0
        
        return features, mask
    
    def _get_labels(
        self, 
        session: SessionSchema
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Extract disease and concept labels."""
        # Disease labels
        disease_labels = []
        disease_mask = []
        for disease in DISEASES:
            label = session.labels.diseases.get(disease)
            disease_labels.append(float(label) if label is not None else 0.0)
            disease_mask.append(float(label is not None))
        
        # Concept labels
        concept_labels = []
        concept_mask = []
        for concept in CONCEPTS:
            label = session.labels.concepts.get(concept)
            concept_labels.append(float(label) if label is not None else 0.0)
            concept_mask.append(float(label is not None))
        
        return (
            torch.tensor(disease_labels, dtype=torch.float32),
            torch.tensor(disease_mask, dtype=torch.float32),
            torch.tensor(concept_labels, dtype=torch.float32),
            torch.tensor(concept_mask, dtype=torch.float32),
            session.labels.label_confidence,
        )
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single session."""
        session = self.sessions[idx]
        
        # Process audio segments
        audio_tokens = []
        segment_types = []
        
        for seg_type in AUDIO_SEGMENT_TYPES:
            if seg_type in session.audio_segments:
                segment = session.audio_segments[seg_type]
                if segment.is_valid():
                    waveform = self._load_audio(segment.file_path)
                    if waveform is not None:
                        # Segment breath recordings into windows
                        if seg_type in ["breath_normal", "breath_deep"]:
                            windows = self._segment_audio(waveform)
                            for window in windows[:5]:  # Limit windows per breath type
                                spec, seg_idx = self._process_audio_segment(window, seg_type)
                                audio_tokens.append(spec)
                                segment_types.append(seg_idx)
                        else:
                            # Single token for coughs, vowels, reading
                            # Pad/crop to fixed length
                            if len(waveform) < self.segment_samples:
                                waveform = torch.nn.functional.pad(
                                    waveform, (0, self.segment_samples - len(waveform))
                                )
                            else:
                                waveform = waveform[:self.segment_samples]
                            
                            spec, seg_idx = self._process_audio_segment(waveform, seg_type)
                            audio_tokens.append(spec)
                            segment_types.append(seg_idx)
        
        # Pad or truncate to max_audio_tokens
        num_tokens = len(audio_tokens)
        if num_tokens == 0:
            # Create dummy token if no audio
            dummy_spec = torch.zeros(1, 64, 201)  # [channels, freq, time]
            audio_tokens = [dummy_spec]
            segment_types = [0]
            num_tokens = 1
        
        # Stack and pad
        if num_tokens < self.max_audio_tokens:
            # Pad with zeros
            pad_size = self.max_audio_tokens - num_tokens
            audio_tokens.extend([torch.zeros_like(audio_tokens[0])] * pad_size)
            segment_types.extend([0] * pad_size)
        else:
            audio_tokens = audio_tokens[:self.max_audio_tokens]
            segment_types = segment_types[:self.max_audio_tokens]
            num_tokens = self.max_audio_tokens
        
        audio_tokens = torch.stack(audio_tokens)  # [num_tokens, channels, freq, time]
        segment_types = torch.tensor(segment_types, dtype=torch.long)
        audio_mask = torch.zeros(self.max_audio_tokens)
        audio_mask[:num_tokens] = 1.0
        
        # Get vitals
        vitals, vitals_mask = self._get_vitals_vector(session)
        
        # Get demographics
        demographics = torch.tensor(session.demographics.to_vector(), dtype=torch.float32)
        
        # Get labels
        disease_labels, disease_mask, concept_labels, concept_mask, label_conf = self._get_labels(session)
        
        return {
            "audio_tokens": audio_tokens,
            "audio_mask": audio_mask,
            "segment_types": segment_types,
            "vitals": vitals,
            "vitals_mask": vitals_mask,
            "demographics": demographics,
            "disease_labels": disease_labels,
            "disease_mask": disease_mask,
            "concept_labels": concept_labels,
            "concept_mask": concept_mask,
            "label_confidence": label_conf,
            "session_id": session.session_id,
            "subject_id": session.subject_id,
        }


def collate_sessions(batch: List[Dict[str, Any]]) -> BatchedSession:
    """Collate function for DataLoader."""
    return BatchedSession(
        audio_tokens=torch.stack([b["audio_tokens"] for b in batch]),
        audio_mask=torch.stack([b["audio_mask"] for b in batch]),
        audio_segment_types=torch.stack([b["segment_types"] for b in batch]),
        vitals=torch.stack([b["vitals"] for b in batch]),
        vitals_mask=torch.stack([b["vitals_mask"] for b in batch]),
        demographics=torch.stack([b["demographics"] for b in batch]),
        disease_labels=torch.stack([b["disease_labels"] for b in batch]),
        disease_mask=torch.stack([b["disease_mask"] for b in batch]),
        concept_labels=torch.stack([b["concept_labels"] for b in batch]),
        concept_mask=torch.stack([b["concept_mask"] for b in batch]),
        label_confidence=torch.tensor([b["label_confidence"] for b in batch]),
        session_ids=[b["session_id"] for b in batch],
        subject_ids=[b["subject_id"] for b in batch],
    )


class UnifiedDataLoader:
    """
    Unified data loader manager.
    
    Creates train/val/test data loaders with proper configuration.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        index_dir: Union[str, Path] = "data/indices",
        processed_dir: Union[str, Path] = "data/processed",
    ):
        self.config = config
        self.index_dir = Path(index_dir)
        self.processed_dir = Path(processed_dir)
        
        # Data config
        self.batch_size = config.get("training", {}).get("batch_size", 16)
        self.num_workers = config.get("training", {}).get("num_workers", 4)
    
    def get_dataloader(
        self,
        split: str,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
    ) -> DataLoader:
        """
        Get DataLoader for specified split.
        
        Args:
            split: One of "train", "val", "test"
            batch_size: Override batch size
            shuffle: Override shuffle setting
            
        Returns:
            DataLoader instance
        """
        index_path = self.index_dir / f"{split}.jsonl"
        
        dataset = UnifiedDataset(
            index_path=index_path,
            processed_dir=self.processed_dir,
            config=self.config.get("data", {}),
            split=split,
            augment=(split == "train"),
        )
        
        if batch_size is None:
            batch_size = self.batch_size
        
        if shuffle is None:
            shuffle = (split == "train")
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_sessions,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    
    def get_train_loader(self) -> DataLoader:
        """Get training DataLoader."""
        return self.get_dataloader("train", shuffle=True)
    
    def get_val_loader(self) -> DataLoader:
        """Get validation DataLoader."""
        return self.get_dataloader("val", shuffle=False)
    
    def get_test_loader(self) -> DataLoader:
        """Get test DataLoader."""
        return self.get_dataloader("test", shuffle=False)

