# RESPIRA-MULTI v1.0

**Cutting-Edge Multimodal Respiratory Disease Screening System**

An on-device (mobile) multimodal AI that screens for multiple respiratory diseases using smartphone-capturable inputs with state-of-the-art teacher-student distillation.

## 🎯 Overview

RESPIRA-MULTI uses a sophisticated teacher→student pipeline:
- **Teachers**: BEATs, Audio-MAE, AST, HuBERT/wav2vec2 for maximum accuracy
- **Student**: Lightweight MobileNetV3 + Conformer for on-device deployment
- **Distillation**: Logits + features + attention transfer

## 🏥 Supported Diseases

Multi-label classification for:
- Healthy/No concern
- URTI (Upper Respiratory Tract Infection)
- LRTI (Lower Respiratory Tract Infection)
- Asthma
- COPD
- Pneumonia
- Bronchitis
- Bronchiolitis
- Bronchiectasis
- Tuberculosis (screening/triage)
- COVID-19

## 📊 Interpretable Concepts

Audio concepts:
- Wheeze, Crackle, Rhonchi, Stridor detection
- Cough rate and wetness estimation
- Breath phase irregularity
- Speech breathiness

Vitals concepts:
- HR (mean, std), HRV (RMSSD, SDNN)
- Respiratory rate
- SpO2 (optional, requires calibration)

## 🎤 Input Protocol

### Audio (16kHz mono WAV)
1. `cough_shallow`: 5 shallow coughs
2. `cough_deep`: 5 deep coughs
3. `breath_normal`: 20 seconds normal breathing
4. `breath_deep`: 15 seconds deep breathing
5. `vowel_a`: Sustain "aaaa" for 6 seconds
6. `reading`: Read fixed phrase ~10 seconds

### Camera
7. `finger_ppg`: 30 seconds fingertip on rear camera + flash
8. `face_video_rr`: 30 seconds front camera (optional)

### IMU (optional)
9. `imu_chest`: 20 seconds phone on chest

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TEACHER ENSEMBLE                          │
│  ┌─────────┐ ┌───────────┐ ┌─────┐ ┌─────────────────────┐  │
│  │  BEATs  │ │ Audio-MAE │ │ AST │ │ HuBERT/wav2vec2    │  │
│  └────┬────┘ └─────┬─────┘ └──┬──┘ └──────────┬──────────┘  │
│       └────────────┼──────────┼───────────────┘              │
│                    ▼                                         │
│              Ensemble Logits + Embeddings                    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼ Distillation
┌─────────────────────────────────────────────────────────────┐
│                    STUDENT MODEL (Mobile)                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            Audio Encoder (MobileNetV3 + Conformer)     │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Vitals MLP + Missingness Embeddings       │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         Fusion Transformer (4 layers, d=256)           │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐  │
│  │  Concept Head   │ │  Disease Head   │ │ Prototype Bank│  │
│  │ (Bottleneck)    │ │ (Hierarchical)  │ │  (Evidence)   │  │
│  └─────────────────┘ └─────────────────┘ └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Download Datasets

```bash
# Download public datasets
python scripts/download_coswara.py
python scripts/download_coughvid.py
python scripts/download_icbhi.py

# Build unified index
python scripts/build_index.py
```

### Training

```bash
# Stage 1: Train teachers (offline, requires GPU)
python scripts/train_teachers.py --config configs/teachers.yaml

# Stage 2: Distillation (train student)
python scripts/train_student.py --config configs/student.yaml

# Stage 3: Multimodal fine-tuning
python scripts/train_multimodal.py --config configs/multimodal.yaml
```

### Export for Mobile

```bash
python scripts/export_mobile.py --config configs/mobile_int8.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

## 📁 Repository Structure

```
respiramulti/
├── configs/                    # Configuration files
│   ├── base.yaml
│   ├── teachers.yaml
│   ├── student.yaml
│   └── mobile_int8.yaml
├── data/
│   ├── raw/                    # Downloaded datasets
│   ├── processed/              # Preprocessed features
│   └── indices/                # Train/val/test splits
├── scripts/
│   ├── download_*.py           # Dataset downloaders
│   ├── build_index.py          # Index builder
│   ├── train_*.py              # Training scripts
│   ├── evaluate.py             # Evaluation
│   └── export_mobile.py        # Mobile export
└── respiramulti/
    ├── datasets/               # Data loading
    ├── features/               # Feature extraction
    ├── models/                 # Model architectures
    ├── teachers/               # Teacher models
    ├── student/                # Student model
    ├── distillation/           # Distillation losses
    ├── robustness/             # Augmentations & TTA
    ├── interpretability/       # Concepts & prototypes
    ├── uncertainty/            # Calibration & conformal
    ├── optimization/           # QAT & pruning
    └── utils/                  # Utilities
```

## ⚠️ Safety & Disclaimers

**IMPORTANT**: This is a **screening tool only**, NOT a medical diagnosis.

- Do NOT recommend treatment based on results
- If severe risk indicators detected, advise seeking medical care
- All raw recordings stored locally by default
- Explicit opt-in required for data upload

## 📊 Performance Targets

- **Inference latency**: <150ms on mid-range Android
- **Model size**: <15MB (INT8 quantized)
- **Preprocessing**: <1s

## 🔬 Evaluation Metrics

- AUROC/AUPRC per disease
- Sensitivity at fixed specificity
- Expected Calibration Error (ECE)
- Robustness vs noise levels

## 📚 Datasets Used

### Audio (Cough/Breath/Speech)
- Coswara
- COUGHVID
- COVID-19 Sounds (Cambridge)

### Lung Sounds
- ICBHI 2017 Respiratory Sound Database
- Fraiwan chest-wall lung sounds

### TB Screening
- CODA TB DREAM Challenge
- Hyfe solicited cough TB dataset

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

