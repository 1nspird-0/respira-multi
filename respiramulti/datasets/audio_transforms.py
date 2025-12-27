"""
Audio augmentation and transformation utilities.

Implements SpecAugment, additive noise, reverberation, and other augmentations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
import random


class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for ASR.
    
    Applies time and frequency masking to spectrograms.
    """
    
    def __init__(
        self,
        time_mask_param: int = 40,
        freq_mask_param: int = 8,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        replace_with_zero: bool = False,
    ):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.replace_with_zero = replace_with_zero
        
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spec: Spectrogram tensor of shape [batch, freq, time] or [freq, time]
            
        Returns:
            Augmented spectrogram
        """
        # Ensure 3D tensor
        squeeze = False
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
            squeeze = True
        
        # Apply time masks
        for _ in range(self.num_time_masks):
            spec = self.time_masking(spec)
        
        # Apply frequency masks
        for _ in range(self.num_freq_masks):
            spec = self.freq_masking(spec)
        
        if squeeze:
            spec = spec.squeeze(0)
        
        return spec


class AdditiveNoise(nn.Module):
    """
    Add noise to audio at random SNR levels.
    """
    
    def __init__(
        self,
        snr_range: Tuple[float, float] = (0, 25),
        noise_dir: Optional[str] = None,
        noise_types: List[str] = None,
    ):
        super().__init__()
        self.snr_range = snr_range
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.noise_types = noise_types or ["white"]
        self.noise_cache = {}
        
        # Load noise files if directory provided
        if self.noise_dir and self.noise_dir.exists():
            self._load_noise_files()
    
    def _load_noise_files(self):
        """Load noise files from directory."""
        for noise_type in self.noise_types:
            noise_files = list(self.noise_dir.glob(f"{noise_type}*.wav"))
            if noise_files:
                self.noise_cache[noise_type] = noise_files
    
    def _generate_noise(self, length: int, noise_type: str = "white") -> torch.Tensor:
        """Generate synthetic noise."""
        if noise_type == "white":
            return torch.randn(length)
        elif noise_type == "pink":
            # Simple pink noise approximation
            white = torch.randn(length)
            # Apply lowpass filter approximation
            kernel = torch.ones(16) / 16
            pink = F.conv1d(white.unsqueeze(0).unsqueeze(0), 
                           kernel.unsqueeze(0).unsqueeze(0), padding=8)
            return pink.squeeze()[:length]
        else:
            return torch.randn(length)
    
    def forward(
        self, 
        waveform: torch.Tensor,
        snr_db: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Add noise to waveform.
        
        Args:
            waveform: Audio waveform [samples] or [batch, samples]
            snr_db: Target SNR in dB. If None, randomly sampled from snr_range.
            
        Returns:
            Noisy waveform
        """
        if snr_db is None:
            snr_db = random.uniform(*self.snr_range)
        
        # Get or generate noise
        noise_type = random.choice(self.noise_types)
        
        if noise_type in self.noise_cache and self.noise_cache[noise_type]:
            # Load random noise file
            noise_file = random.choice(self.noise_cache[noise_type])
            noise, sr = torchaudio.load(noise_file)
            noise = noise.mean(dim=0)  # Convert to mono
            
            # Repeat or trim to match waveform length
            if noise.shape[-1] < waveform.shape[-1]:
                repeats = (waveform.shape[-1] // noise.shape[-1]) + 1
                noise = noise.repeat(repeats)
            noise = noise[:waveform.shape[-1]]
        else:
            noise = self._generate_noise(waveform.shape[-1], noise_type)
        
        # Calculate scaling factor for target SNR
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-8))
        
        # Add scaled noise
        noisy = waveform + noise * scale
        
        return noisy


class ReverbAugment(nn.Module):
    """
    Apply reverberation using room impulse responses (RIR).
    """
    
    def __init__(
        self,
        rir_dir: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.rir_dir = Path(rir_dir) if rir_dir else None
        self.sample_rate = sample_rate
        self.rir_cache = []
        
        if self.rir_dir and self.rir_dir.exists():
            self._load_rir_files()
    
    def _load_rir_files(self):
        """Load RIR files from directory."""
        rir_files = list(self.rir_dir.glob("*.wav"))
        for rir_file in rir_files[:100]:  # Limit cache size
            try:
                rir, sr = torchaudio.load(rir_file)
                if sr != self.sample_rate:
                    rir = torchaudio.functional.resample(rir, sr, self.sample_rate)
                rir = rir.mean(dim=0)  # Mono
                self.rir_cache.append(rir)
            except Exception:
                continue
    
    def _generate_synthetic_rir(self, length: int = 16000) -> torch.Tensor:
        """Generate a simple synthetic RIR."""
        # Simple exponential decay with random reflections
        t = torch.linspace(0, 1, length)
        decay = torch.exp(-5 * t)
        
        # Add random reflections
        rir = torch.randn(length) * decay
        rir[0] = 1.0  # Direct sound
        
        # Add a few discrete reflections
        num_reflections = random.randint(3, 8)
        for _ in range(num_reflections):
            pos = random.randint(100, length - 1)
            rir[pos] += random.uniform(0.1, 0.5) * (1 - pos / length)
        
        return rir / rir.abs().max()
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply reverberation to waveform.
        
        Args:
            waveform: Audio waveform [samples] or [batch, samples]
            
        Returns:
            Reverberant waveform
        """
        if self.rir_cache:
            rir = random.choice(self.rir_cache)
        else:
            rir = self._generate_synthetic_rir()
        
        # Convolve with RIR
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            squeeze = True
        else:
            waveform = waveform.unsqueeze(1)
            squeeze = False
        
        rir = rir.unsqueeze(0).unsqueeze(0)
        
        # Pad for 'same' convolution
        pad_len = rir.shape[-1] - 1
        waveform_padded = F.pad(waveform, (pad_len, 0))
        
        reverberant = F.conv1d(waveform_padded, rir)
        
        if squeeze:
            reverberant = reverberant.squeeze()
        else:
            reverberant = reverberant.squeeze(1)
        
        # Normalize to prevent clipping
        reverberant = reverberant / (reverberant.abs().max() + 1e-8)
        
        return reverberant


class MicResponseAugment(nn.Module):
    """
    Simulate different microphone frequency responses.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        eq_range: Tuple[float, float] = (-6, 6),
        lowpass_range: Tuple[int, int] = (4000, 8000),
        highpass_range: Tuple[int, int] = (50, 200),
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.eq_range = eq_range
        self.lowpass_range = lowpass_range
        self.highpass_range = highpass_range
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random mic response simulation.
        
        Args:
            waveform: Audio waveform [samples]
            
        Returns:
            Processed waveform
        """
        # Random highpass filter
        highpass_cutoff = random.randint(*self.highpass_range)
        
        # Random lowpass filter  
        lowpass_cutoff = random.randint(*self.lowpass_range)
        
        # Apply filters using torchaudio
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Highpass
        waveform = torchaudio.functional.highpass_biquad(
            waveform, self.sample_rate, highpass_cutoff
        )
        
        # Lowpass
        waveform = torchaudio.functional.lowpass_biquad(
            waveform, self.sample_rate, lowpass_cutoff
        )
        
        return waveform.squeeze(0)


class TimeAugment(nn.Module):
    """
    Time-domain augmentations: shift, stretch, crop.
    """
    
    def __init__(
        self,
        shift_ms: int = 200,
        stretch_range: Tuple[float, float] = (0.9, 1.1),
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.shift_samples = int(shift_ms * sample_rate / 1000)
        self.stretch_range = stretch_range
        self.sample_rate = sample_rate
    
    def time_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random time shift."""
        shift = random.randint(-self.shift_samples, self.shift_samples)
        
        if shift > 0:
            waveform = F.pad(waveform, (shift, 0))[..., :waveform.shape[-1]]
        elif shift < 0:
            waveform = F.pad(waveform, (0, -shift))[..., -shift:]
        
        return waveform
    
    def time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random time stretch."""
        stretch_factor = random.uniform(*self.stretch_range)
        
        original_length = waveform.shape[-1]
        new_length = int(original_length * stretch_factor)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            squeeze = True
        else:
            waveform = waveform.unsqueeze(1)
            squeeze = False
        
        stretched = F.interpolate(waveform, size=new_length, mode='linear', align_corners=False)
        
        if squeeze:
            stretched = stretched.squeeze()
        else:
            stretched = stretched.squeeze(1)
        
        # Pad or crop back to original length
        if new_length > original_length:
            stretched = stretched[..., :original_length]
        else:
            padding = original_length - new_length
            stretched = F.pad(stretched, (0, padding))
        
        return stretched
    
    def forward(
        self, 
        waveform: torch.Tensor,
        apply_shift: bool = True,
        apply_stretch: bool = True,
    ) -> torch.Tensor:
        """Apply time augmentations."""
        if apply_shift:
            waveform = self.time_shift(waveform)
        if apply_stretch:
            waveform = self.time_stretch(waveform)
        return waveform


class MixStyle(nn.Module):
    """
    MixStyle: Domain generalization via feature statistics perturbation.
    
    Mixes feature statistics between samples in a batch.
    """
    
    def __init__(self, prob: float = 0.5, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.prob = prob
        self.alpha = alpha
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MixStyle to feature maps.
        
        Args:
            x: Feature tensor [batch, channels, ...]
            
        Returns:
            Mixed features
        """
        if not self.training or random.random() > self.prob:
            return x
        
        batch_size = x.size(0)
        if batch_size < 2:
            return x
        
        # Compute mean and std along spatial dimensions
        dims = tuple(range(2, x.dim()))
        mu = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True)
        sigma = (var + self.eps).sqrt()
        
        # Normalize
        x_normed = (x - mu) / sigma
        
        # Random shuffle for mixing
        perm = torch.randperm(batch_size)
        mu_perm = mu[perm]
        sigma_perm = sigma[perm]
        
        # Sample mixing weight
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_size, 1, 1, 1))
        lmda = lmda.to(x.device)
        
        # Mix statistics
        mu_mix = lmda * mu + (1 - lmda) * mu_perm
        sigma_mix = lmda * sigma + (1 - lmda) * sigma_perm
        
        # Denormalize with mixed statistics
        return x_normed * sigma_mix + mu_mix


class AudioTransforms(nn.Module):
    """
    Complete audio augmentation pipeline.
    
    Combines all augmentations for training.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        config: Optional[Dict] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        config = config or {}
        
        # SpecAugment
        spec_config = config.get("spec_augment", {})
        self.spec_augment = SpecAugment(
            time_mask_param=spec_config.get("time_mask_param", 40),
            freq_mask_param=spec_config.get("freq_mask_param", 8),
            num_time_masks=spec_config.get("num_time_masks", 2),
            num_freq_masks=spec_config.get("num_freq_masks", 2),
        ) if spec_config.get("enabled", True) else None
        
        # Additive noise
        noise_config = config.get("additive_noise", {})
        self.additive_noise = AdditiveNoise(
            snr_range=tuple(noise_config.get("snr_range", [5, 20])),
            noise_dir=noise_config.get("noise_dir"),
            noise_types=noise_config.get("noise_types", ["white"]),
        ) if noise_config.get("enabled", True) else None
        
        # Reverb
        reverb_config = config.get("reverb", {})
        self.reverb = ReverbAugment(
            rir_dir=reverb_config.get("rir_dir"),
            sample_rate=sample_rate,
        ) if reverb_config.get("enabled", False) else None
        
        # Mic response
        mic_config = config.get("mic_response", {})
        self.mic_response = MicResponseAugment(
            sample_rate=sample_rate,
            eq_range=tuple(mic_config.get("eq_range", [-6, 6])),
            lowpass_range=tuple(mic_config.get("lowpass_range", [4000, 8000])),
            highpass_range=tuple(mic_config.get("highpass_range", [50, 200])),
        ) if mic_config.get("enabled", False) else None
        
        # Time augment
        self.time_augment = TimeAugment(
            shift_ms=config.get("time_shift_ms", 200),
            stretch_range=tuple(config.get("time_stretch_range", [0.9, 1.1])),
            sample_rate=sample_rate,
        )
        
        # MixStyle
        mixstyle_config = config.get("mixstyle", {})
        self.mixstyle = MixStyle(
            prob=mixstyle_config.get("prob", 0.5),
            alpha=mixstyle_config.get("alpha", 0.3),
        ) if mixstyle_config.get("enabled", False) else None
        
        # Augmentation probabilities
        self.p_noise = config.get("p_noise", 0.5)
        self.p_reverb = config.get("p_reverb", 0.3)
        self.p_mic = config.get("p_mic", 0.3)
        self.p_time = config.get("p_time", 0.5)
    
    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply waveform-level augmentations."""
        if self.training:
            # Time augmentations
            if random.random() < self.p_time:
                waveform = self.time_augment(waveform)
            
            # Additive noise
            if self.additive_noise and random.random() < self.p_noise:
                waveform = self.additive_noise(waveform)
            
            # Reverb
            if self.reverb and random.random() < self.p_reverb:
                waveform = self.reverb(waveform)
            
            # Mic response
            if self.mic_response and random.random() < self.p_mic:
                waveform = self.mic_response(waveform)
        
        return waveform
    
    def augment_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply spectrogram-level augmentations."""
        if self.training:
            # SpecAugment
            if self.spec_augment:
                spec = self.spec_augment(spec)
            
            # MixStyle (for batch of features)
            if self.mixstyle and spec.dim() >= 3:
                spec = self.mixstyle(spec)
        
        return spec
    
    def forward(
        self,
        waveform: Optional[torch.Tensor] = None,
        spectrogram: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply full augmentation pipeline.
        
        Args:
            waveform: Raw audio waveform
            spectrogram: Mel spectrogram
            
        Returns:
            Tuple of (augmented_waveform, augmented_spectrogram)
        """
        if waveform is not None:
            waveform = self.augment_waveform(waveform)
        
        if spectrogram is not None:
            spectrogram = self.augment_spectrogram(spectrogram)
        
        return waveform, spectrogram


class ModalityDropout(nn.Module):
    """
    Randomly drop modalities during training for robustness.
    """
    
    def __init__(
        self,
        audio_prob: float = 0.15,
        vitals_prob: float = 0.2,
        per_segment_prob: float = 0.1,
    ):
        super().__init__()
        self.audio_prob = audio_prob
        self.vitals_prob = vitals_prob
        self.per_segment_prob = per_segment_prob
    
    def forward(
        self,
        audio_tokens: Optional[torch.Tensor],
        vitals_token: Optional[torch.Tensor],
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Apply modality dropout.
        
        Args:
            audio_tokens: [batch, num_tokens, dim] audio embeddings
            vitals_token: [batch, dim] vitals embedding
            audio_mask: [batch, num_tokens] mask for valid audio tokens
            
        Returns:
            Tuple of (audio_tokens, vitals_token, updated_mask)
        """
        if not self.training:
            mask = audio_mask if audio_mask is not None else torch.ones(
                audio_tokens.shape[:2], device=audio_tokens.device
            )
            return audio_tokens, vitals_token, mask
        
        batch_size = audio_tokens.shape[0] if audio_tokens is not None else vitals_token.shape[0]
        device = audio_tokens.device if audio_tokens is not None else vitals_token.device
        
        # Initialize mask
        if audio_mask is not None:
            mask = audio_mask.clone()
        elif audio_tokens is not None:
            mask = torch.ones(audio_tokens.shape[:2], device=device)
        else:
            mask = torch.ones(batch_size, 1, device=device)
        
        # Drop entire audio modality
        if audio_tokens is not None and random.random() < self.audio_prob:
            audio_tokens = torch.zeros_like(audio_tokens)
            mask = torch.zeros_like(mask)
        elif audio_tokens is not None:
            # Drop individual segments
            for b in range(batch_size):
                for t in range(mask.shape[1]):
                    if random.random() < self.per_segment_prob:
                        mask[b, t] = 0
                        audio_tokens[b, t] = 0
        
        # Drop vitals modality
        if vitals_token is not None and random.random() < self.vitals_prob:
            vitals_token = torch.zeros_like(vitals_token)
        
        return audio_tokens, vitals_token, mask

