"""
Respiratory Rate (RR) estimation from audio and video.

Implements multiple methods for RR estimation from breathing audio and face video.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2


@dataclass
class RRFeatures:
    """Extracted respiratory rate features."""
    rr_mean: float  # Mean respiratory rate in breaths per minute
    rr_std: float  # RR variability
    breath_duration_mean: float  # Mean breath cycle duration in seconds
    breath_duration_std: float  # Variability in breath duration
    signal_quality: float  # 0-1 quality score
    breath_phases: Optional[np.ndarray] = None  # Detected breath phase boundaries


class RREstimator:
    """
    Estimate respiratory rate from audio or video.
    
    Supports multiple estimation methods:
    - Audio envelope analysis
    - Video-based chest motion
    - Face video respiratory motion
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        rr_range: Tuple[float, float] = (6, 40),  # Breaths per minute
        window_sec: float = 10.0,  # Analysis window
    ):
        self.sample_rate = sample_rate
        self.rr_range = rr_range
        self.window_sec = window_sec
        
        # Convert RR range to frequency range
        self.freq_range = (rr_range[0] / 60, rr_range[1] / 60)  # Hz
    
    def estimate_from_audio(
        self,
        waveform: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> RRFeatures:
        """
        Estimate RR from breathing audio.
        
        Uses envelope analysis to detect breath cycles.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        if len(waveform) < sample_rate * 5:  # Need at least 5 seconds
            return RRFeatures(
                rr_mean=0.0,
                rr_std=0.0,
                breath_duration_mean=0.0,
                breath_duration_std=0.0,
                signal_quality=0.0,
            )
        
        # Extract envelope
        envelope = self._extract_envelope(waveform, sample_rate)
        
        # Bandpass filter for breathing frequencies
        envelope_filtered = self._bandpass_filter(
            envelope, 
            sample_rate=100,  # Envelope is downsampled
            low_freq=self.freq_range[0],
            high_freq=self.freq_range[1],
        )
        
        # Detect breath cycles
        breath_starts = self._detect_breath_cycles(envelope_filtered, sample_rate=100)
        
        if len(breath_starts) < 2:
            return RRFeatures(
                rr_mean=0.0,
                rr_std=0.0,
                breath_duration_mean=0.0,
                breath_duration_std=0.0,
                signal_quality=0.0,
            )
        
        # Compute RR metrics
        breath_intervals = np.diff(breath_starts) / 100  # seconds
        
        # Filter outliers
        valid_intervals = breath_intervals[
            (breath_intervals > 60 / self.rr_range[1]) & 
            (breath_intervals < 60 / self.rr_range[0])
        ]
        
        if len(valid_intervals) < 1:
            return RRFeatures(
                rr_mean=0.0,
                rr_std=0.0,
                breath_duration_mean=0.0,
                breath_duration_std=0.0,
                signal_quality=0.0,
            )
        
        rr_values = 60 / valid_intervals  # Convert to BPM
        rr_mean = np.mean(rr_values)
        rr_std = np.std(rr_values) if len(rr_values) > 1 else 0.0
        
        breath_duration_mean = np.mean(valid_intervals)
        breath_duration_std = np.std(valid_intervals) if len(valid_intervals) > 1 else 0.0
        
        # Quality score
        quality = self._compute_quality(
            envelope_filtered, breath_starts, sample_rate=100
        )
        
        return RRFeatures(
            rr_mean=rr_mean,
            rr_std=rr_std,
            breath_duration_mean=breath_duration_mean,
            breath_duration_std=breath_duration_std,
            signal_quality=quality,
            breath_phases=breath_starts,
        )
    
    def _extract_envelope(
        self, 
        waveform: np.ndarray, 
        sample_rate: int,
        target_rate: int = 100,
    ) -> np.ndarray:
        """Extract amplitude envelope from waveform."""
        # Rectify
        rectified = np.abs(waveform)
        
        # Lowpass filter
        nyquist = sample_rate / 2
        b, a = signal.butter(2, 10 / nyquist, btype='low')
        smoothed = signal.filtfilt(b, a, rectified)
        
        # Downsample to target rate
        downsample_factor = sample_rate // target_rate
        envelope = signal.decimate(smoothed, downsample_factor, ftype='fir')
        
        return envelope
    
    def _bandpass_filter(
        self,
        signal_data: np.ndarray,
        sample_rate: int,
        low_freq: float,
        high_freq: float,
    ) -> np.ndarray:
        """Apply bandpass filter."""
        nyquist = sample_rate / 2
        low = max(low_freq / nyquist, 0.01)
        high = min(high_freq / nyquist, 0.99)
        
        b, a = signal.butter(2, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data)
        
        return filtered
    
    def _detect_breath_cycles(
        self,
        envelope: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Detect breath cycle start points."""
        # Minimum distance between breaths (based on max RR)
        min_distance = int(sample_rate * 60 / self.rr_range[1])
        
        # Find peaks (inspiration starts)
        peaks, _ = signal.find_peaks(
            envelope,
            distance=min_distance,
            prominence=0.1 * np.std(envelope),
        )
        
        return peaks
    
    def _compute_quality(
        self,
        envelope: np.ndarray,
        breath_starts: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Compute signal quality score."""
        scores = []
        
        # Periodicity score
        if len(breath_starts) >= 2:
            intervals = np.diff(breath_starts) / sample_rate
            cv = np.std(intervals) / np.mean(intervals)
            periodicity = max(0, 1 - cv)
            scores.append(periodicity)
        
        # SNR-like score
        if len(envelope) > 0:
            peak_power = np.mean(envelope[breath_starts] ** 2) if len(breath_starts) > 0 else 0
            noise_power = np.var(envelope)
            if noise_power > 0:
                snr = min(1.0, peak_power / (10 * noise_power))
                scores.append(snr)
        
        return np.mean(scores) if scores else 0.0
    
    def estimate_from_video(
        self,
        video_path: str,
        method: str = "motion",
        max_frames: int = 1800,
    ) -> RRFeatures:
        """
        Estimate RR from face video.
        
        Uses subtle motion analysis in chest/face region.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return RRFeatures(
                rr_mean=0.0,
                rr_std=0.0,
                breath_duration_mean=0.0,
                breath_duration_std=0.0,
                signal_quality=0.0,
            )
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Extract motion signal
        motion_signal = self._extract_motion_signal(cap, max_frames)
        cap.release()
        
        if len(motion_signal) < fps * 5:
            return RRFeatures(
                rr_mean=0.0,
                rr_std=0.0,
                breath_duration_mean=0.0,
                breath_duration_std=0.0,
                signal_quality=0.0,
            )
        
        # Convert to numpy and process
        # Bandpass filter for breathing frequencies
        filtered = self._bandpass_filter(
            motion_signal,
            sample_rate=int(fps),
            low_freq=self.freq_range[0],
            high_freq=self.freq_range[1],
        )
        
        # Detect breath cycles
        breath_starts = self._detect_breath_cycles(filtered, sample_rate=int(fps))
        
        if len(breath_starts) < 2:
            return RRFeatures(
                rr_mean=0.0,
                rr_std=0.0,
                breath_duration_mean=0.0,
                breath_duration_std=0.0,
                signal_quality=0.0,
            )
        
        # Compute metrics
        breath_intervals = np.diff(breath_starts) / fps
        valid_intervals = breath_intervals[
            (breath_intervals > 60 / self.rr_range[1]) & 
            (breath_intervals < 60 / self.rr_range[0])
        ]
        
        if len(valid_intervals) < 1:
            return RRFeatures(
                rr_mean=0.0,
                rr_std=0.0,
                breath_duration_mean=0.0,
                breath_duration_std=0.0,
                signal_quality=0.0,
            )
        
        rr_values = 60 / valid_intervals
        
        quality = self._compute_quality(filtered, breath_starts, sample_rate=int(fps))
        
        return RRFeatures(
            rr_mean=np.mean(rr_values),
            rr_std=np.std(rr_values) if len(rr_values) > 1 else 0.0,
            breath_duration_mean=np.mean(valid_intervals),
            breath_duration_std=np.std(valid_intervals) if len(valid_intervals) > 1 else 0.0,
            signal_quality=quality,
            breath_phases=breath_starts,
        )
    
    def _extract_motion_signal(
        self,
        cap: cv2.VideoCapture,
        max_frames: int,
    ) -> np.ndarray:
        """Extract breathing motion signal from video."""
        motion_values = []
        prev_frame = None
        
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use lower portion of frame (chest area if visible)
            h, w = gray.shape
            roi = gray[h//2:, w//4:3*w//4]
            
            if prev_frame is not None:
                # Compute frame difference
                diff = cv2.absdiff(roi, prev_frame)
                motion = np.mean(diff)
                motion_values.append(motion)
            
            prev_frame = roi.copy()
            frame_count += 1
        
        return np.array(motion_values)


class BreathPhaseDetector:
    """
    Detect inspiration and expiration phases in breathing audio.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def detect_phases(
        self,
        waveform: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Detect inspiration and expiration phases.
        
        Returns:
            Tuple of (inspiration_ranges, expiration_ranges)
            Each range is (start_sample, end_sample)
        """
        # Extract envelope
        envelope = np.abs(waveform)
        
        # Smooth
        kernel_size = self.sample_rate // 20  # 50ms window
        envelope = uniform_filter1d(envelope, kernel_size)
        
        # Find peaks and troughs
        peaks, _ = signal.find_peaks(envelope, distance=self.sample_rate // 2)
        troughs, _ = signal.find_peaks(-envelope, distance=self.sample_rate // 2)
        
        # Combine and sort
        events = [(p, 'peak') for p in peaks] + [(t, 'trough') for t in troughs]
        events.sort(key=lambda x: x[0])
        
        inspirations = []
        expirations = []
        
        for i in range(len(events) - 1):
            start_pos, start_type = events[i]
            end_pos, end_type = events[i + 1]
            
            if start_type == 'trough' and end_type == 'peak':
                # Inspiration: trough to peak
                inspirations.append((start_pos, end_pos))
            elif start_type == 'peak' and end_type == 'trough':
                # Expiration: peak to trough
                expirations.append((start_pos, end_pos))
        
        return inspirations, expirations
    
    def compute_phase_irregularity(
        self,
        waveform: np.ndarray,
    ) -> float:
        """
        Compute breath phase irregularity score.
        
        Higher values indicate more irregular breathing.
        """
        inspirations, expirations = self.detect_phases(waveform)
        
        if len(inspirations) < 2 or len(expirations) < 2:
            return 0.5  # Unknown
        
        # Compute I/E ratios
        ie_ratios = []
        for i in range(min(len(inspirations), len(expirations))):
            insp_duration = inspirations[i][1] - inspirations[i][0]
            exp_duration = expirations[i][1] - expirations[i][0]
            if exp_duration > 0:
                ie_ratios.append(insp_duration / exp_duration)
        
        if len(ie_ratios) < 2:
            return 0.5
        
        # Irregularity = coefficient of variation of I/E ratios
        ie_cv = np.std(ie_ratios) / np.mean(ie_ratios)
        
        # Normalize to 0-1 range
        irregularity = min(1.0, ie_cv)
        
        return irregularity

