#!/usr/bin/env python3
"""
Build unified index from multiple datasets.

Usage:
    python scripts/build_index.py --raw_dir data/raw --output_dir data/indices
"""

import argparse
import json
import jsonlines
from pathlib import Path
from typing import Dict, List, Optional
import random
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from respiramulti.datasets.schema import (
    SessionSchema,
    AudioSegment,
    Labels,
    Demographics,
    LabelSource,
    Sex,
    SmokerStatus,
    DISEASES,
    CONCEPTS,
)


def parse_coswara(raw_dir: Path) -> List[SessionSchema]:
    """Parse Coswara dataset."""
    sessions = []
    coswara_dir = raw_dir / "coswara"
    
    if not coswara_dir.exists():
        print(f"Coswara directory not found: {coswara_dir}")
        return sessions
    
    # Look for subject directories
    for subject_dir in coswara_dir.glob("*"):
        if not subject_dir.is_dir():
            continue
        
        subject_id = subject_dir.name
        
        # Load metadata if exists
        metadata_file = subject_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        
        # Create session
        session = SessionSchema(
            session_id=f"coswara_{subject_id}",
            subject_id=subject_id,
            dataset_source="coswara",
        )
        
        # Find audio files
        audio_mappings = {
            "cough-shallow": "cough_shallow",
            "cough-heavy": "cough_deep",
            "breathing-shallow": "breath_normal",
            "breathing-deep": "breath_deep",
            "vowel-a": "vowel_a",
            "counting-normal": "reading",
        }
        
        for file_prefix, segment_type in audio_mappings.items():
            audio_file = subject_dir / f"{file_prefix}.wav"
            if audio_file.exists():
                session.audio_segments[segment_type] = AudioSegment(
                    segment_type=segment_type,
                    file_path=str(audio_file),
                )
        
        # Parse labels from metadata
        if metadata:
            session.labels = Labels(
                diseases={d: 0 for d in DISEASES},
                label_source=LabelSource.SELF_REPORT,
                label_confidence=0.3,
            )
            
            # Map COVID status
            if metadata.get("covid_status") == "positive":
                session.labels.diseases["covid19"] = 1
            
            if metadata.get("asthma") == True:
                session.labels.diseases["asthma"] = 1
            
            # Demographics
            session.demographics = Demographics(
                age=metadata.get("age"),
                sex=Sex.MALE if metadata.get("gender") == "male" else Sex.FEMALE,
            )
        
        if session.audio_segments:
            sessions.append(session)
    
    print(f"Parsed {len(sessions)} sessions from Coswara")
    return sessions


def parse_coughvid(raw_dir: Path) -> List[SessionSchema]:
    """Parse COUGHVID dataset."""
    sessions = []
    coughvid_dir = raw_dir / "coughvid"
    
    if not coughvid_dir.exists():
        print(f"COUGHVID directory not found: {coughvid_dir}")
        return sessions
    
    # Look for metadata CSV
    metadata_file = coughvid_dir / "metadata_compiled.csv"
    if not metadata_file.exists():
        # Try to find individual audio files
        for audio_file in coughvid_dir.glob("*.wav"):
            session_id = audio_file.stem
            
            session = SessionSchema(
                session_id=f"coughvid_{session_id}",
                subject_id=session_id,
                dataset_source="coughvid",
            )
            
            session.audio_segments["cough_deep"] = AudioSegment(
                segment_type="cough_deep",
                file_path=str(audio_file),
            )
            
            sessions.append(session)
    else:
        import csv
        with open(metadata_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                uuid = row.get("uuid", "")
                audio_file = coughvid_dir / f"{uuid}.wav"
                
                if not audio_file.exists():
                    continue
                
                session = SessionSchema(
                    session_id=f"coughvid_{uuid}",
                    subject_id=uuid,
                    dataset_source="coughvid",
                )
                
                session.audio_segments["cough_deep"] = AudioSegment(
                    segment_type="cough_deep",
                    file_path=str(audio_file),
                )
                
                # Parse labels
                session.labels = Labels(
                    diseases={d: 0 for d in DISEASES},
                    label_source=LabelSource.SELF_REPORT,
                    label_confidence=0.2,
                )
                
                if row.get("status") == "COVID-19":
                    session.labels.diseases["covid19"] = 1
                
                # Demographics
                age = row.get("age")
                session.demographics = Demographics(
                    age=int(float(age)) if age else None,
                    sex=Sex.MALE if row.get("gender") == "male" else Sex.FEMALE if row.get("gender") == "female" else Sex.UNKNOWN,
                )
                
                sessions.append(session)
    
    print(f"Parsed {len(sessions)} sessions from COUGHVID")
    return sessions


def parse_icbhi(raw_dir: Path) -> List[SessionSchema]:
    """Parse ICBHI respiratory sounds dataset."""
    sessions = []
    icbhi_dir = raw_dir / "icbhi"
    
    if not icbhi_dir.exists():
        print(f"ICBHI directory not found: {icbhi_dir}")
        return sessions
    
    # Look for audio files
    for audio_file in icbhi_dir.glob("*.wav"):
        filename = audio_file.stem
        parts = filename.split("_")
        
        if len(parts) >= 1:
            subject_id = parts[0]
        else:
            subject_id = filename
        
        session = SessionSchema(
            session_id=f"icbhi_{filename}",
            subject_id=subject_id,
            dataset_source="icbhi",
        )
        
        session.audio_segments["breath_normal"] = AudioSegment(
            segment_type="breath_normal",
            file_path=str(audio_file),
        )
        
        # Parse annotations if available
        annotation_file = audio_file.with_suffix(".txt")
        if annotation_file.exists():
            with open(annotation_file) as f:
                lines = f.readlines()
            
            session.labels = Labels(
                diseases={d: 0 for d in DISEASES},
                concepts={},
                label_source=LabelSource.CLINICIAN_DX,
                label_confidence=0.8,
            )
            
            # Parse wheeze/crackle annotations
            has_wheeze = False
            has_crackle = False
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    crackle = int(parts[2])
                    wheeze = int(parts[3])
                    has_crackle = has_crackle or crackle == 1
                    has_wheeze = has_wheeze or wheeze == 1
            
            session.labels.concepts["wheeze_presence"] = 1 if has_wheeze else 0
            session.labels.concepts["crackle_presence"] = 1 if has_crackle else 0
        
        sessions.append(session)
    
    # Load diagnosis file if available
    diagnosis_file = icbhi_dir / "patient_diagnosis.csv"
    if diagnosis_file.exists():
        import csv
        diagnoses = {}
        with open(diagnosis_file) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    patient_id = row[0]
                    diagnosis = row[1].lower()
                    diagnoses[patient_id] = diagnosis
        
        # Update sessions with diagnosis
        for session in sessions:
            if session.subject_id in diagnoses:
                diagnosis = diagnoses[session.subject_id]
                if "copd" in diagnosis:
                    session.labels.diseases["copd"] = 1
                if "asthma" in diagnosis:
                    session.labels.diseases["asthma"] = 1
                if "pneumonia" in diagnosis:
                    session.labels.diseases["pneumonia"] = 1
                if "bronchi" in diagnosis:
                    session.labels.diseases["bronchitis"] = 1
                if "lrti" in diagnosis or "lower" in diagnosis:
                    session.labels.diseases["lrti"] = 1
                if "urti" in diagnosis or "upper" in diagnosis:
                    session.labels.diseases["urti"] = 1
                if "healthy" in diagnosis or "normal" in diagnosis:
                    session.labels.diseases["healthy"] = 1
    
    print(f"Parsed {len(sessions)} sessions from ICBHI")
    return sessions


def split_by_subject(
    sessions: List[SessionSchema],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[SessionSchema]]:
    """Split sessions by subject ID to prevent leakage."""
    random.seed(seed)
    
    # Group by subject
    by_subject = defaultdict(list)
    for session in sessions:
        by_subject[session.subject_id].append(session)
    
    subjects = list(by_subject.keys())
    random.shuffle(subjects)
    
    n_train = int(len(subjects) * train_ratio)
    n_val = int(len(subjects) * val_ratio)
    
    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train + n_val])
    test_subjects = set(subjects[n_train + n_val:])
    
    splits = {
        'train': [],
        'val': [],
        'test': [],
    }
    
    for session in sessions:
        if session.subject_id in train_subjects:
            splits['train'].append(session)
        elif session.subject_id in val_subjects:
            splits['val'].append(session)
        else:
            splits['test'].append(session)
    
    return splits


def main():
    parser = argparse.ArgumentParser(description='Build unified dataset index')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='Directory with raw datasets')
    parser.add_argument('--output_dir', type=str, default='data/indices',
                        help='Output directory for indices')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse all datasets
    print("Parsing datasets...")
    all_sessions = []
    
    all_sessions.extend(parse_coswara(raw_dir))
    all_sessions.extend(parse_coughvid(raw_dir))
    all_sessions.extend(parse_icbhi(raw_dir))
    
    print(f"\nTotal sessions: {len(all_sessions)}")
    
    # Split by subject
    print("\nSplitting by subject...")
    splits = split_by_subject(
        all_sessions,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    
    # Write indices
    for split_name, sessions in splits.items():
        output_path = output_dir / f"{split_name}.jsonl"
        
        with jsonlines.open(output_path, 'w') as writer:
            for session in sessions:
                writer.write(session.to_dict())
        
        print(f"Wrote {len(sessions)} sessions to {output_path}")
    
    # Write statistics
    stats = {
        'total_sessions': len(all_sessions),
        'train_sessions': len(splits['train']),
        'val_sessions': len(splits['val']),
        'test_sessions': len(splits['test']),
        'datasets': {
            'coswara': sum(1 for s in all_sessions if s.dataset_source == 'coswara'),
            'coughvid': sum(1 for s in all_sessions if s.dataset_source == 'coughvid'),
            'icbhi': sum(1 for s in all_sessions if s.dataset_source == 'icbhi'),
        },
    }
    
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to {output_dir / 'stats.json'}")
    print("\nDone!")


if __name__ == '__main__':
    main()

