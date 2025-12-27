"""
PPG (Photoplethysmography) feature extraction from finger video.

Extracts HR, HRV, and signal quality metrics from smartphone camera PPG recordings.
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
class PPGFeatures:
    """Extracted PPG features."""
    hr_mean: float  # Mean heart rate in BPM
    hr_std: float  # Heart rate variability (std of instantaneous HR)
    hrv_rmssd: float  # Root mean square of successive RR differences
    hrv_sdnn: float  # Standard deviation of RR intervals
    signal_quality: float  # 0-1 quality score
    rr_intervals: Optional[np.ndarray] = None  # Raw RR intervals in ms
    hr_trace: Optional[np.ndarray] = None  # Instantaneous HR over time


class PPGExtractor:
    """
    Extract PPG signal and heart rate from finger video.
    
    Uses the red channel from rear camera with flash for best signal.
    """
    
    def __init__(
        self,
        fps: int = 30,
        bandpass_low: float = 0.7,  # 42 BPM
        bandpass_high: float = 4.0,  # 240 BPM
        detrend_cutoff: float = 0.5,  # Hz
        roi_fraction: float = 0.5,  # Central ROI fraction
    ):
        self.fps = fps
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.detrend_cutoff = detrend_cutoff
        self.roi_fraction = roi_fraction
    
    def extract_signal_from_video(
        self, 
        video_path: str,
        max_frames: int = 1800,  # 60 seconds at 30fps
    ) -> Tuple[np.ndarray, float]:
        """
        Extract raw PPG signal from video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            
        Returns:
            Tuple of (ppg_signal, actual_fps)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate ROI
        roi_w = int(width * self.roi_fraction)
        roi_h = int(height * self.roi_fraction)
        x1 = (width - roi_w) // 2
        y1 = (height - roi_h) // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        
        # Extract red channel means
        red_means = []
        frame_count = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            
            # Get red channel mean
            red_channel = roi[:, :, 2]  # BGR format
            red_mean = np.mean(red_channel)
            red_means.append(red_mean)
            
            frame_count += 1
        
        cap.release()
        
        return np.array(red_means), fps
    
    def preprocess_signal(self, ppg_signal: np.ndarray, fps: float) -> np.ndarray:
        """
        Preprocess raw PPG signal.
        
        Applies detrending and bandpass filtering.
        """
        # Detrend using high-pass filter
        nyquist = fps / 2
        b, a = signal.butter(2, self.detrend_cutoff / nyquist, btype='high')
        detrended = signal.filtfilt(b, a, ppg_signal)
        
        # Bandpass filter
        low = self.bandpass_low / nyquist
        high = min(self.bandpass_high / nyquist, 0.99)
        b, a = signal.butter(2, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, detrended)
        
        return filtered
    
    def detect_peaks(
        self, 
        ppg_signal: np.ndarray, 
        fps: float,
        min_distance_sec: float = 0.3,  # 200 BPM max
    ) -> np.ndarray:
        """
        Detect peaks in PPG signal.
        
        Returns array of peak indices.
        """
        min_distance = int(min_distance_sec * fps)
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            ppg_signal,
            distance=min_distance,
            prominence=0.1 * np.std(ppg_signal),
        )
        
        return peaks
    
    def compute_rr_intervals(
        self, 
        peaks: np.ndarray, 
        fps: float
    ) -> np.ndarray:
        """Compute RR intervals in milliseconds from peak indices."""
        if len(peaks) < 2:
            return np.array([])
        
        rr_intervals = np.diff(peaks) / fps * 1000  # Convert to ms
        return rr_intervals
    
    def compute_hrv_metrics(
        self, 
        rr_intervals: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Compute HRV metrics from RR intervals.
        
        Returns:
            Tuple of (hr_mean, hr_std, rmssd, sdnn)
        """
        if len(rr_intervals) < 2:
            return 0.0, 0.0, 0.0, 0.0
        
        # Filter outliers (physiologically implausible)
        valid_rr = rr_intervals[(rr_intervals > 250) & (rr_intervals < 1500)]
        
        if len(valid_rr) < 2:
            return 0.0, 0.0, 0.0, 0.0
        
        # HR from RR intervals
        hr_values = 60000 / valid_rr  # BPM
        hr_mean = np.mean(hr_values)
        hr_std = np.std(hr_values)
        
        # RMSSD: Root mean square of successive differences
        rr_diff = np.diff(valid_rr)
        rmssd = np.sqrt(np.mean(rr_diff ** 2))
        
        # SDNN: Standard deviation of RR intervals
        sdnn = np.std(valid_rr)
        
        return hr_mean, hr_std, rmssd, sdnn
    
    def compute_signal_quality(
        self,
        ppg_signal: np.ndarray,
        peaks: np.ndarray,
        fps: float,
    ) -> float:
        """
        Compute signal quality score.
        
        Based on:
        - Peak periodicity
        - Signal-to-noise ratio
        - Consistent amplitude
        """
        if len(peaks) < 3:
            return 0.0
        
        scores = []
        
        # 1. Peak periodicity score
        rr_intervals = np.diff(peaks) / fps * 1000
        if len(rr_intervals) > 1:
            rr_cv = np.std(rr_intervals) / np.mean(rr_intervals)
            periodicity_score = max(0, 1 - rr_cv)
            scores.append(periodicity_score)
        
        # 2. SNR-like score (ratio of peak amplitude to baseline noise)
        peak_amplitudes = ppg_signal[peaks]
        baseline_std = np.std(ppg_signal)
        if baseline_std > 0:
            snr_score = min(1.0, np.mean(np.abs(peak_amplitudes)) / (3 * baseline_std))
            scores.append(snr_score)
        
        # 3. Amplitude consistency
        if len(peak_amplitudes) > 1:
            amp_cv = np.std(peak_amplitudes) / (np.mean(np.abs(peak_amplitudes)) + 1e-8)
            amp_score = max(0, 1 - amp_cv)
            scores.append(amp_score)
        
        # 4. Expected HR range score
        if len(rr_intervals) > 0:
            hr = 60000 / np.mean(rr_intervals)
            if 40 <= hr <= 200:
                hr_score = 1.0
            elif 30 <= hr <= 220:
                hr_score = 0.5
            else:
                hr_score = 0.0
            scores.append(hr_score)
        
        return np.mean(scores) if scores else 0.0
    
    def extract_features(
        self,
        video_path: str,
    ) -> PPGFeatures:
        """
        Extract complete PPG features from video.
        
        Args:
            video_path: Path to finger PPG video
            
        Returns:
            PPGFeatures dataclass with all metrics
        """
        try:
            # Extract raw signal
            raw_signal, fps = self.extract_signal_from_video(video_path)
            
            if len(raw_signal) < fps * 5:  # Need at least 5 seconds
                return PPGFeatures(
                    hr_mean=0.0,
                    hr_std=0.0,
                    hrv_rmssd=0.0,
                    hrv_sdnn=0.0,
                    signal_quality=0.0,
                )
            
            # Preprocess
            filtered = self.preprocess_signal(raw_signal, fps)
            
            # Detect peaks
            peaks = self.detect_peaks(filtered, fps)
            
            # Compute RR intervals
            rr_intervals = self.compute_rr_intervals(peaks, fps)
            
            # Compute HRV metrics
            hr_mean, hr_std, rmssd, sdnn = self.compute_hrv_metrics(rr_intervals)
            
            # Compute quality
            quality = self.compute_signal_quality(filtered, peaks, fps)
            
            # Compute instantaneous HR trace
            if len(rr_intervals) > 0:
                hr_trace = 60000 / rr_intervals
            else:
                hr_trace = None
            
            return PPGFeatures(
                hr_mean=hr_mean,
                hr_std=hr_std,
                hrv_rmssd=rmssd,
                hrv_sdnn=sdnn,
                signal_quality=quality,
                rr_intervals=rr_intervals,
                hr_trace=hr_trace,
            )
            
        except Exception as e:
            print(f"PPG extraction failed: {e}")
            return PPGFeatures(
                hr_mean=0.0,
                hr_std=0.0,
                hrv_rmssd=0.0,
                hrv_sdnn=0.0,
                signal_quality=0.0,
            )
    
    def extract_features_from_signal(
        self,
        ppg_signal: np.ndarray,
        fps: float,
    ) -> PPGFeatures:
        """
        Extract features from pre-extracted PPG signal.
        
        Args:
            ppg_signal: Raw PPG signal array
            fps: Sampling rate
            
        Returns:
            PPGFeatures dataclass
        """
        if len(ppg_signal) < fps * 5:
            return PPGFeatures(
                hr_mean=0.0,
                hr_std=0.0,
                hrv_rmssd=0.0,
                hrv_sdnn=0.0,
                signal_quality=0.0,
            )
        
        # Preprocess
        filtered = self.preprocess_signal(ppg_signal, fps)
        
        # Detect peaks
        peaks = self.detect_peaks(filtered, fps)
        
        # Compute metrics
        rr_intervals = self.compute_rr_intervals(peaks, fps)
        hr_mean, hr_std, rmssd, sdnn = self.compute_hrv_metrics(rr_intervals)
        quality = self.compute_signal_quality(filtered, peaks, fps)
        
        return PPGFeatures(
            hr_mean=hr_mean,
            hr_std=hr_std,
            hrv_rmssd=rmssd,
            hrv_sdnn=sdnn,
            signal_quality=quality,
            rr_intervals=rr_intervals,
        )


class RemotePPGExtractor:
    """
    Extract PPG from face video using remote photoplethysmography (rPPG).
    
    Uses color changes in facial skin to estimate heart rate.
    """
    
    def __init__(
        self,
        fps: int = 30,
        method: str = "green",  # "green", "chrom", or "pos"
    ):
        self.fps = fps
        self.method = method
    
    def extract_roi_from_face(
        self,
        frame: np.ndarray,
        face_detector: Optional[object] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract forehead/cheek ROI from face frame.
        
        Uses face detection to find skin regions.
        """
        # Simple implementation: use central region
        # In production, use face detection + landmark estimation
        h, w = frame.shape[:2]
        
        # Central upper region (approximate forehead)
        roi = frame[h//4:h//2, w//4:3*w//4]
        
        return roi
    
    def extract_signal_green(
        self,
        video_path: str,
        max_frames: int = 1800,
    ) -> Tuple[np.ndarray, float]:
        """
        Extract PPG using green channel method.
        
        The green channel has highest pulsatile component for skin.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        green_means = []
        
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            roi = self.extract_roi_from_face(frame)
            if roi is not None:
                green_channel = roi[:, :, 1]  # BGR format
                green_means.append(np.mean(green_channel))
            
            frame_count += 1
        
        cap.release()
        
        return np.array(green_means), fps
    
    def extract_signal_chrom(
        self,
        video_path: str,
        max_frames: int = 1800,
    ) -> Tuple[np.ndarray, float]:
        """
        Extract PPG using CHROM (Chrominance-based) method.
        
        More robust to motion artifacts than green channel.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        rgb_signals = []
        
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            roi = self.extract_roi_from_face(frame)
            if roi is not None:
                # Get mean RGB
                rgb_mean = np.mean(roi, axis=(0, 1))[::-1]  # BGR to RGB
                rgb_signals.append(rgb_mean)
            
            frame_count += 1
        
        cap.release()
        
        if len(rgb_signals) < 10:
            return np.array([]), fps
        
        rgb_signals = np.array(rgb_signals)
        
        # Normalize
        rgb_norm = rgb_signals / np.mean(rgb_signals, axis=0)
        
        # CHROM calculation
        Xs = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
        Ys = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
        
        # Combine with alpha
        std_xs = np.std(Xs)
        std_ys = np.std(Ys)
        alpha = std_xs / (std_ys + 1e-8)
        
        ppg_signal = Xs - alpha * Ys
        
        return ppg_signal, fps
    
    def extract_features(self, video_path: str) -> PPGFeatures:
        """Extract PPG features from face video."""
        ppg_extractor = PPGExtractor(fps=self.fps)
        
        if self.method == "green":
            signal, fps = self.extract_signal_green(video_path)
        elif self.method == "chrom":
            signal, fps = self.extract_signal_chrom(video_path)
        else:
            signal, fps = self.extract_signal_green(video_path)
        
        if len(signal) < 10:
            return PPGFeatures(
                hr_mean=0.0,
                hr_std=0.0,
                hrv_rmssd=0.0,
                hrv_sdnn=0.0,
                signal_quality=0.0,
            )
        
        return ppg_extractor.extract_features_from_signal(signal, fps)

