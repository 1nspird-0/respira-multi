"""
Data schema definitions for RESPIRA-MULTI.

Defines the structure of sessions, labels, and all data types used in the system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import json
from pathlib import Path


# =============================================================================
# LABEL DEFINITIONS
# =============================================================================

DISEASES = [
    "healthy",
    "urti",
    "lrti", 
    "asthma",
    "copd",
    "pneumonia",
    "bronchitis",
    "bronchiolitis",
    "bronchiectasis",
    "tb",
    "covid19",
    "heart_failure_pulmonary",
]

BINARY_CONCEPTS = [
    "wheeze_presence",
    "crackle_presence",
    "rhonchi_presence",
    "stridor_presence",
    "cough_detected",
]

CONTINUOUS_CONCEPTS = [
    "cough_rate_est",
    "cough_wetness_proxy",
    "breath_phase_irregularity",
    "speech_breathiness",
    "speech_phrase_read_quality",
    "hr_mean",
    "hr_std",
    "hrv_rmssd",
    "hrv_sdnn",
    "rr_est",
    "spo2_est",
    "perfusion_quality_score",
]

CONCEPTS = BINARY_CONCEPTS + CONTINUOUS_CONCEPTS

# Hierarchy constraints: parent -> children
DISEASE_HIERARCHY = {
    "lrti": ["pneumonia", "bronchitis", "bronchiolitis", "bronchiectasis"],
}

# Audio segment types
AUDIO_SEGMENT_TYPES = [
    "cough_shallow",
    "cough_deep",
    "breath_normal",
    "breath_deep",
    "vowel_a",
    "reading",
]

# Reading prompts for speech recording
READING_PROMPTS = {
    "prompt_01": "The rainbow appears after the rain stops falling.",
    "prompt_02": "Please call Stella and ask her to bring the documents.",
    "prompt_03": "The quick brown fox jumps over the lazy dog.",
}


class LabelSource(Enum):
    """Source of ground truth labels."""
    CLINICIAN_DX = "clinician_dx"
    LAB_TEST = "lab_test"
    SELF_REPORT = "self_report"
    UNKNOWN = "unknown"


class Sex(Enum):
    """Biological sex."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class SmokerStatus(Enum):
    """Smoking status."""
    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"
    UNKNOWN = "unknown"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AudioSegment:
    """Represents a single audio segment in a session."""
    segment_type: str  # One of AUDIO_SEGMENT_TYPES
    file_path: Optional[str] = None
    duration_sec: Optional[float] = None
    sample_rate: int = 16000
    reading_prompt_id: Optional[str] = None  # For reading segments
    
    # Preprocessed features (filled during preprocessing)
    mel_spectrogram_path: Optional[str] = None
    mfcc_path: Optional[str] = None
    waveform_path: Optional[str] = None
    
    # Quality metrics
    signal_quality: Optional[float] = None
    snr_db: Optional[float] = None
    
    def is_valid(self) -> bool:
        """Check if segment has valid audio file."""
        return self.file_path is not None and Path(self.file_path).exists()


@dataclass
class VideoSegment:
    """Represents a video segment for PPG/RR extraction."""
    segment_type: str  # "finger_ppg" or "face_video"
    file_path: Optional[str] = None
    duration_sec: Optional[float] = None
    fps: int = 30
    
    # Extracted features
    ppg_signal_path: Optional[str] = None
    extracted_hr: Optional[float] = None
    extracted_rr: Optional[float] = None
    signal_quality: Optional[float] = None
    
    def is_valid(self) -> bool:
        """Check if segment has valid video file."""
        return self.file_path is not None and Path(self.file_path).exists()


@dataclass
class IMUSegment:
    """Represents IMU data from phone sensors."""
    file_path: Optional[str] = None
    duration_sec: Optional[float] = None
    sample_rate: int = 100  # Hz
    
    # Extracted features
    breath_motion_features_path: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if segment has valid IMU file."""
        return self.file_path is not None and Path(self.file_path).exists()


@dataclass
class VitalsFeatures:
    """Extracted vital signs features."""
    hr_mean: Optional[float] = None
    hr_std: Optional[float] = None
    hrv_rmssd: Optional[float] = None
    hrv_sdnn: Optional[float] = None
    rr_est: Optional[float] = None
    spo2_est: Optional[float] = None
    perfusion_quality: Optional[float] = None
    
    # Quality scores for each modality
    hr_quality: Optional[float] = None
    rr_quality: Optional[float] = None
    spo2_quality: Optional[float] = None
    
    def to_vector(self) -> List[Optional[float]]:
        """Convert to feature vector for model input."""
        return [
            self.hr_mean,
            self.hr_std,
            self.hrv_rmssd,
            self.hrv_sdnn,
            self.rr_est,
            self.spo2_est,
            self.perfusion_quality,
            self.hr_quality,
            self.rr_quality,
            self.spo2_quality,
        ]
    
    def get_missingness_mask(self) -> List[bool]:
        """Return mask indicating which features are missing."""
        vector = self.to_vector()
        return [v is None for v in vector]


@dataclass
class Demographics:
    """Demographic information about the patient."""
    age: Optional[int] = None
    sex: Sex = Sex.UNKNOWN
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    smoker_status: SmokerStatus = SmokerStatus.UNKNOWN
    
    # Known conditions (multi-hot)
    known_asthma: bool = False
    known_copd: bool = False
    known_other_respiratory: bool = False
    known_heart_condition: bool = False
    
    # Current symptoms (multi-hot)
    symptom_fever: bool = False
    symptom_sore_throat: bool = False
    symptom_dyspnea: bool = False
    symptom_sputum: bool = False
    symptom_chest_pain: bool = False
    symptom_cough: bool = False
    
    def to_vector(self) -> List[float]:
        """Convert to feature vector for model input."""
        sex_encoding = {Sex.MALE: 0, Sex.FEMALE: 1, Sex.OTHER: 0.5, Sex.UNKNOWN: 0.5}
        smoker_encoding = {
            SmokerStatus.NEVER: 0, 
            SmokerStatus.FORMER: 0.5, 
            SmokerStatus.CURRENT: 1,
            SmokerStatus.UNKNOWN: 0.5
        }
        
        return [
            self.age / 100.0 if self.age else 0.5,  # Normalized age
            sex_encoding[self.sex],
            self.height_cm / 200.0 if self.height_cm else 0.5,
            self.weight_kg / 150.0 if self.weight_kg else 0.5,
            smoker_encoding[self.smoker_status],
            float(self.known_asthma),
            float(self.known_copd),
            float(self.known_other_respiratory),
            float(self.known_heart_condition),
            float(self.symptom_fever),
            float(self.symptom_sore_throat),
            float(self.symptom_dyspnea),
            float(self.symptom_sputum),
            float(self.symptom_chest_pain),
            float(self.symptom_cough),
        ]


@dataclass
class Labels:
    """Ground truth labels for a session."""
    # Disease labels (multi-label)
    diseases: Dict[str, Optional[int]] = field(default_factory=dict)
    
    # Concept labels
    concepts: Dict[str, Optional[float]] = field(default_factory=dict)
    
    # Metadata
    label_source: LabelSource = LabelSource.UNKNOWN
    label_confidence: float = 0.0  # 0-1, how confident we are in labels
    
    def get_disease_vector(self) -> List[Optional[int]]:
        """Get disease labels as vector in standard order."""
        return [self.diseases.get(d) for d in DISEASES]
    
    def get_concept_vector(self) -> List[Optional[float]]:
        """Get concept labels as vector in standard order."""
        return [self.concepts.get(c) for c in CONCEPTS]
    
    def get_disease_mask(self) -> List[bool]:
        """Return mask for which disease labels are available."""
        return [self.diseases.get(d) is not None for d in DISEASES]
    
    def get_concept_mask(self) -> List[bool]:
        """Return mask for which concept labels are available."""
        return [self.concepts.get(c) is not None for c in CONCEPTS]


@dataclass
class SessionSchema:
    """
    Complete session schema representing one recording session.
    
    This is the unified data structure that all datasets are normalized to.
    """
    # Identifiers
    session_id: str
    subject_id: str
    dataset_source: str  # Which dataset this came from
    recording_date: Optional[str] = None
    device_model: Optional[str] = None
    
    # Audio segments
    audio_segments: Dict[str, AudioSegment] = field(default_factory=dict)
    
    # Video segments
    video_segments: Dict[str, VideoSegment] = field(default_factory=dict)
    
    # IMU data
    imu_segment: Optional[IMUSegment] = None
    
    # Extracted features
    vitals: VitalsFeatures = field(default_factory=VitalsFeatures)
    
    # Labels
    labels: Labels = field(default_factory=Labels)
    
    # Demographics
    demographics: Demographics = field(default_factory=Demographics)
    
    # Paths to preprocessed features (for fast loading)
    preprocessed_dir: Optional[str] = None
    
    def get_available_audio_segments(self) -> List[str]:
        """Return list of available audio segment types."""
        return [k for k, v in self.audio_segments.items() if v.is_valid()]
    
    def get_available_video_segments(self) -> List[str]:
        """Return list of available video segment types."""
        return [k for k, v in self.video_segments.items() if v.is_valid()]
    
    def has_vitals(self) -> bool:
        """Check if any vitals are available."""
        return any(v is not None for v in self.vitals.to_vector())
    
    def has_labels(self) -> bool:
        """Check if any disease labels are available."""
        return any(self.labels.get_disease_mask())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "dataset_source": self.dataset_source,
            "recording_date": self.recording_date,
            "device_model": self.device_model,
            "audio": {
                k: {
                    "file_path": v.file_path,
                    "segment_type": v.segment_type,
                    "duration_sec": v.duration_sec,
                    "reading_prompt_id": v.reading_prompt_id,
                    "mel_spectrogram_path": v.mel_spectrogram_path,
                }
                for k, v in self.audio_segments.items()
            },
            "video": {
                k: {
                    "file_path": v.file_path,
                    "segment_type": v.segment_type,
                    "duration_sec": v.duration_sec,
                }
                for k, v in self.video_segments.items()
            },
            "imu": {
                "file_path": self.imu_segment.file_path if self.imu_segment else None,
            },
            "features": {
                "hr_bpm": self.vitals.hr_mean,
                "rr_bpm": self.vitals.rr_est,
                "spo2_pct": self.vitals.spo2_est,
                "signal_quality": self.vitals.perfusion_quality,
            },
            "labels": {
                "diseases": self.labels.diseases,
                "concepts": self.labels.concepts,
                "label_source": self.labels.label_source.value,
                "label_confidence": self.labels.label_confidence,
            },
            "demographics": {
                "age": self.demographics.age,
                "sex": self.demographics.sex.value,
                "smoker": self.demographics.smoker_status.value,
                "height_cm": self.demographics.height_cm,
                "weight_kg": self.demographics.weight_kg,
            },
            "preprocessed_dir": self.preprocessed_dir,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionSchema":
        """Create SessionSchema from dictionary."""
        session = cls(
            session_id=data["session_id"],
            subject_id=data["subject_id"],
            dataset_source=data.get("dataset_source", "unknown"),
            recording_date=data.get("recording_date"),
            device_model=data.get("device_model"),
        )
        
        # Parse audio segments
        if "audio" in data:
            for seg_type, seg_data in data["audio"].items():
                if seg_data and seg_data.get("file_path"):
                    session.audio_segments[seg_type] = AudioSegment(
                        segment_type=seg_type,
                        file_path=seg_data.get("file_path"),
                        duration_sec=seg_data.get("duration_sec"),
                        reading_prompt_id=seg_data.get("reading_prompt_id"),
                        mel_spectrogram_path=seg_data.get("mel_spectrogram_path"),
                    )
        
        # Parse video segments
        if "video" in data:
            for seg_type, seg_data in data["video"].items():
                if seg_data and seg_data.get("file_path"):
                    session.video_segments[seg_type] = VideoSegment(
                        segment_type=seg_type,
                        file_path=seg_data.get("file_path"),
                        duration_sec=seg_data.get("duration_sec"),
                    )
        
        # Parse IMU
        if "imu" in data and data["imu"].get("file_path"):
            session.imu_segment = IMUSegment(file_path=data["imu"]["file_path"])
        
        # Parse vitals
        if "features" in data:
            features = data["features"]
            session.vitals = VitalsFeatures(
                hr_mean=features.get("hr_bpm"),
                rr_est=features.get("rr_bpm"),
                spo2_est=features.get("spo2_pct"),
                perfusion_quality=features.get("signal_quality"),
            )
        
        # Parse labels
        if "labels" in data:
            labels_data = data["labels"]
            session.labels = Labels(
                diseases=labels_data.get("diseases", {}),
                concepts=labels_data.get("concepts", {}),
                label_source=LabelSource(labels_data.get("label_source", "unknown")),
                label_confidence=labels_data.get("label_confidence", 0.0),
            )
        
        # Parse demographics
        if "demographics" in data:
            demo = data["demographics"]
            session.demographics = Demographics(
                age=demo.get("age"),
                sex=Sex(demo.get("sex", "unknown")),
                height_cm=demo.get("height_cm"),
                weight_kg=demo.get("weight_kg"),
                smoker_status=SmokerStatus(demo.get("smoker", "unknown")),
            )
        
        session.preprocessed_dir = data.get("preprocessed_dir")
        
        return session
    
    def save_json(self, path: Union[str, Path]) -> None:
        """Save session to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "SessionSchema":
        """Load session from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def validate_session(session: SessionSchema) -> List[str]:
    """
    Validate a session and return list of issues.
    
    Returns empty list if session is valid.
    """
    issues = []
    
    # Check required fields
    if not session.session_id:
        issues.append("Missing session_id")
    if not session.subject_id:
        issues.append("Missing subject_id")
    
    # Check audio segments
    if not session.audio_segments:
        issues.append("No audio segments")
    else:
        for seg_type, segment in session.audio_segments.items():
            if seg_type not in AUDIO_SEGMENT_TYPES:
                issues.append(f"Unknown audio segment type: {seg_type}")
            if not segment.is_valid():
                issues.append(f"Invalid audio file for {seg_type}")
    
    # Check labels if present
    for disease in session.labels.diseases:
        if disease not in DISEASES:
            issues.append(f"Unknown disease label: {disease}")
    
    for concept in session.labels.concepts:
        if concept not in CONCEPTS:
            issues.append(f"Unknown concept label: {concept}")
    
    return issues

