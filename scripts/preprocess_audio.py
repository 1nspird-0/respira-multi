#!/usr/bin/env python3
"""
Preprocess audio files and extract features.

Usage:
    python scripts/preprocess_audio.py --index data/indices/train.jsonl --output data/processed
"""

import argparse
import json
import jsonlines
from pathlib import Path
from typing import Dict
import torch
import torchaudio
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from respiramulti.features.spectrogram import AudioPreprocessor
from respiramulti.datasets.schema import SessionSchema


def preprocess_session(
    session: SessionSchema,
    preprocessor: AudioPreprocessor,
    output_dir: Path,
) -> Dict:
    """Preprocess all audio in a session."""
    session_dir = output_dir / session.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    processed_paths = {}
    
    for seg_type, segment in session.audio_segments.items():
        if not segment.file_path or not Path(segment.file_path).exists():
            continue
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(segment.file_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
            
            # Preprocess
            result = preprocessor(waveform, sample_rate=sr, return_all=True)
            
            # Save features
            mel_path = session_dir / f"{seg_type}_mel.pt"
            mfcc_path = session_dir / f"{seg_type}_mfcc.pt"
            wave_path = session_dir / f"{seg_type}_wave.pt"
            
            torch.save(result['mel_spectrogram'], mel_path)
            torch.save(result['mfcc'], mfcc_path)
            torch.save(result['waveform'], wave_path)
            
            processed_paths[seg_type] = {
                'mel': str(mel_path),
                'mfcc': str(mfcc_path),
                'wave': str(wave_path),
            }
            
        except Exception as e:
            print(f"Error processing {segment.file_path}: {e}")
            continue
    
    return processed_paths


def main():
    parser = argparse.ArgumentParser(description='Preprocess audio files')
    parser.add_argument('--index', type=str, required=True,
                        help='Path to session index JSONL')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory for processed features')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--n_mels', type=int, default=64)
    parser.add_argument('--n_fft', type=int, default=400)
    parser.add_argument('--hop_length', type=int, default=160)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create preprocessor
    preprocessor = AudioPreprocessor(
        target_sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    
    # Load sessions
    print(f"Loading sessions from {args.index}...")
    sessions = []
    with jsonlines.open(args.index, 'r') as reader:
        for item in reader:
            sessions.append(SessionSchema.from_dict(item))
    
    print(f"Processing {len(sessions)} sessions...")
    
    # Process each session
    processed_index = []
    
    for session in tqdm(sessions):
        processed_paths = preprocess_session(session, preprocessor, output_dir)
        
        if processed_paths:
            session.preprocessed_dir = str(output_dir / session.session_id)
            
            # Update segment paths
            for seg_type, paths in processed_paths.items():
                if seg_type in session.audio_segments:
                    session.audio_segments[seg_type].mel_spectrogram_path = paths['mel']
            
            processed_index.append(session)
    
    # Save updated index
    output_index = output_dir / "processed_index.jsonl"
    with jsonlines.open(output_index, 'w') as writer:
        for session in processed_index:
            writer.write(session.to_dict())
    
    print(f"\nProcessed {len(processed_index)} sessions")
    print(f"Updated index saved to {output_index}")


if __name__ == '__main__':
    main()

