"""
Spectrogram and audio feature extraction.

Implements mel spectrogram and MFCC extraction for audio processing.
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple
import numpy as np


class SpectrogramExtractor(nn.Module):
    """
    Extract log-mel spectrograms from audio waveforms.
    
    Optimized for respiratory sound analysis with configurable parameters.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 400,  # 25ms at 16kHz
        hop_length: int = 160,  # 10ms at 16kHz
        fmin: float = 50.0,
        fmax: float = 8000.0,
        power: float = 2.0,
        normalized: bool = True,
        center: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.power = power
        self.normalized = normalized
        
        # Mel spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=power,
            center=center,
            norm="slaney",
            mel_scale="slaney",
        )
        
        # Amplitude to dB
        self.amplitude_to_db = T.AmplitudeToDB(
            stype="power" if power == 2.0 else "magnitude",
            top_db=80,
        )
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract log-mel spectrogram from waveform.
        
        Args:
            waveform: Audio waveform [samples] or [batch, samples]
            
        Returns:
            Log-mel spectrogram [1, n_mels, time] or [batch, 1, n_mels, time]
        """
        # Handle dimensions
        squeeze_batch = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_batch = True
        
        # Extract mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to log scale (dB)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        
        # Normalize if requested
        if self.normalized:
            # Per-sample normalization
            mean = log_mel_spec.mean(dim=(-2, -1), keepdim=True)
            std = log_mel_spec.std(dim=(-2, -1), keepdim=True)
            log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)
        
        # Add channel dimension
        log_mel_spec = log_mel_spec.unsqueeze(1)
        
        if squeeze_batch:
            log_mel_spec = log_mel_spec.squeeze(0)
        
        return log_mel_spec
    
    def get_output_length(self, input_length: int) -> int:
        """Calculate output time dimension given input samples."""
        return (input_length - self.n_fft) // self.hop_length + 1


class MFCCExtractor(nn.Module):
    """
    Extract MFCC features from audio waveforms.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_mels: int = 128,
        n_fft: int = 400,
        hop_length: int = 160,
        fmin: float = 0.0,
        fmax: float = 8000.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "f_min": fmin,
                "f_max": fmax,
            },
        )
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC features from waveform.
        
        Args:
            waveform: Audio waveform [samples] or [batch, samples]
            
        Returns:
            MFCC features [n_mfcc, time] or [batch, n_mfcc, time]
        """
        return self.mfcc_transform(waveform)


class AudioPreprocessor(nn.Module):
    """
    Complete audio preprocessing pipeline.
    
    Handles resampling, normalization, and feature extraction.
    """
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_lufs: float = -20.0,
        n_mels: int = 64,
        n_fft: int = 400,
        hop_length: int = 160,
        fmin: float = 50.0,
        fmax: float = 8000.0,
        silence_threshold_db: float = -40.0,
    ):
        super().__init__()
        self.target_sample_rate = target_sample_rate
        self.target_lufs = target_lufs
        self.silence_threshold_db = silence_threshold_db
        
        # Spectrogram extractor
        self.spec_extractor = SpectrogramExtractor(
            sample_rate=target_sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
        )
        
        # MFCC extractor
        self.mfcc_extractor = MFCCExtractor(
            sample_rate=target_sample_rate,
            n_mfcc=40,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    
    def resample(
        self, 
        waveform: torch.Tensor, 
        orig_sample_rate: int
    ) -> torch.Tensor:
        """Resample audio to target sample rate."""
        if orig_sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_sample_rate, self.target_sample_rate
            )
        return waveform
    
    def normalize_loudness(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio loudness using RMS normalization.
        
        For proper LUFS normalization, use pyloudnorm in preprocessing scripts.
        """
        # Simple RMS normalization as approximation
        target_rms = 10 ** (self.target_lufs / 20)
        current_rms = torch.sqrt(torch.mean(waveform ** 2))
        
        if current_rms > 1e-8:
            waveform = waveform * (target_rms / current_rms)
        
        return waveform
    
    def trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """Trim leading and trailing silence."""
        # Convert threshold to linear
        threshold = 10 ** (self.silence_threshold_db / 20)
        
        # Find non-silent regions
        energy = torch.abs(waveform)
        
        # Smooth energy with small window
        kernel_size = 160  # 10ms at 16kHz
        if energy.dim() == 1:
            energy = energy.unsqueeze(0).unsqueeze(0)
        else:
            energy = energy.unsqueeze(1)
        
        smoothed = torch.nn.functional.avg_pool1d(
            energy, kernel_size, stride=1, padding=kernel_size // 2
        )
        smoothed = smoothed.squeeze()
        
        # Find first and last above threshold
        above_threshold = smoothed > threshold
        if not above_threshold.any():
            return waveform
        
        indices = torch.where(above_threshold)[0]
        start = max(0, indices[0].item() - kernel_size)
        end = min(len(waveform), indices[-1].item() + kernel_size)
        
        return waveform[start:end]
    
    def forward(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
        return_all: bool = False,
    ) -> dict:
        """
        Preprocess audio and extract features.
        
        Args:
            waveform: Raw audio waveform
            sample_rate: Original sample rate (if resampling needed)
            return_all: Whether to return all features or just mel spectrogram
            
        Returns:
            Dictionary with extracted features
        """
        # Resample if needed
        if sample_rate is not None:
            waveform = self.resample(waveform, sample_rate)
        
        # Normalize loudness
        waveform = self.normalize_loudness(waveform)
        
        # Trim silence
        waveform = self.trim_silence(waveform)
        
        # Extract mel spectrogram
        mel_spec = self.spec_extractor(waveform)
        
        result = {
            "mel_spectrogram": mel_spec,
            "waveform": waveform,
        }
        
        if return_all:
            result["mfcc"] = self.mfcc_extractor(waveform)
        
        return result


class DeltaFeatures(nn.Module):
    """
    Compute delta and delta-delta features from spectrograms.
    """
    
    def __init__(self, order: int = 2, width: int = 2):
        super().__init__()
        self.order = order
        self.width = width
        
        # Compute delta transform
        self.compute_deltas = T.ComputeDeltas(win_length=2 * width + 1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute delta features.
        
        Args:
            features: Input features [..., freq, time]
            
        Returns:
            Features with deltas [..., freq * (order + 1), time]
        """
        all_features = [features]
        
        current = features
        for _ in range(self.order):
            delta = self.compute_deltas(current)
            all_features.append(delta)
            current = delta
        
        return torch.cat(all_features, dim=-2)


class PatchEmbedding(nn.Module):
    """
    Convert spectrogram to patch embeddings (for Audio-MAE style models).
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int] = (16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert spectrogram to patches.
        
        Args:
            x: Spectrogram [batch, channels, freq, time]
            
        Returns:
            Patch embeddings [batch, num_patches, embed_dim]
        """
        # Project to patches
        x = self.proj(x)  # [batch, embed_dim, h, w]
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        
        # Normalize
        x = self.norm(x)
        
        return x
    
    def get_num_patches(self, input_shape: Tuple[int, int]) -> int:
        """Calculate number of patches for given input shape."""
        h, w = input_shape
        return (h // self.patch_size[0]) * (w // self.patch_size[1])

