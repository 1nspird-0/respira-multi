# RESPIRA-MULTI v1.0

<div align="center">

**Cutting-Edge Multimodal Respiratory Disease Screening System**

*On-device AI for respiratory health screening using smartphone sensors*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🎯 Overview

RESPIRA-MULTI is a state-of-the-art multimodal AI system that screens for respiratory diseases using only smartphone-capturable inputs:

- **🎤 Audio**: Cough, breathing, speech (microphone)
- **📹 Camera**: Heart rate via PPG, respiratory rate
- **📊 Vitals**: HR, HRV, RR, SpO2 (optional)

The system uses a sophisticated **Teacher→Student distillation pipeline** to achieve maximum accuracy while remaining deployable on mobile devices.

```
┌─────────────────────────────────────────────────────────────────┐
│                     TEACHER ENSEMBLE (Offline)                   │
│  ┌────────┐  ┌───────────┐  ┌─────┐  ┌───────────────────────┐  │
│  │ BEATs  │  │ Audio-MAE │  │ AST │  │ HuBERT/wav2vec2      │  │
│  └───┬────┘  └─────┬─────┘  └──┬──┘  └───────────┬───────────┘  │
│      └─────────────┴───────────┴─────────────────┘               │
│                         ↓ Distillation                           │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                    STUDENT MODEL (On-Device)                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         MobileNetV3 + Lightweight Conformer              │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            Gated Fusion Transformer (4 layers)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Concept Head │  │ Disease Head │  │  Prototype Bank    │    │
│  │ (Bottleneck) │  │(Hierarchical)│  │    (Evidence)      │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏥 Supported Conditions

### Disease Classification (Multi-label)

| Category | Conditions |
|----------|------------|
| **Upper Respiratory** | URTI |
| **Lower Respiratory** | LRTI, Pneumonia, Bronchitis, Bronchiolitis, Bronchiectasis |
| **Chronic** | Asthma, COPD |
| **Infectious** | Tuberculosis (screening), COVID-19 |
| **Cardiac** | Heart failure with pulmonary congestion |

### Interpretable Concepts

| Audio Concepts | Vitals Concepts |
|----------------|-----------------|
| Wheeze, Crackle, Rhonchi, Stridor | HR (mean, std), HRV (RMSSD, SDNN) |
| Cough detection & wetness | Respiratory rate |
| Breath phase irregularity | SpO2 (optional) |
| Speech breathiness | Signal quality scores |

---

## 📱 Input Protocol

The app captures these segments per session:

### Audio (16kHz mono)
| Segment | Duration | Description |
|---------|----------|-------------|
| `cough_shallow` | 5 coughs | Light coughing |
| `cough_deep` | 5 coughs | Forceful coughing |
| `breath_normal` | 20 sec | Normal breathing |
| `breath_deep` | 15 sec | Deep breathing |
| `vowel_a` | 6 sec | Sustained "aaaa" |
| `reading` | ~10 sec | Read fixed phrase |

### Camera
| Segment | Duration | Description |
|---------|----------|-------------|
| `finger_ppg` | 30 sec | Fingertip on rear camera + flash |
| `face_video` | 30 sec | Front camera (optional, for rPPG) |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/respira-multi.git
cd respira-multi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Datasets

```bash
# Download public datasets to data/raw/
# - Coswara: https://github.com/iiscleap/Coswara-Data
# - COUGHVID: https://zenodo.org/record/4048312
# - ICBHI: https://bhichallenge.med.auth.gr/

# Build unified index
python scripts/build_index.py --raw_dir data/raw --output_dir data/indices

# Preprocess audio features
python scripts/preprocess_audio.py --index data/indices/train.jsonl --output data/processed
```

### Training

```bash
# Train student model with teacher distillation
python scripts/train_student.py --config configs/student.yaml --output_dir outputs

# The training runs 3 stages:
# Stage 1: Pure distillation (no hard labels) - 30 epochs
# Stage 2: Mixed training (hard labels + distillation) - 40 epochs  
# Stage 3: Temperature scaling calibration
```

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/final_calibrated.pt \
    --config configs/student.yaml \
    --split test
```

### Export for Mobile

```bash
python scripts/export_mobile.py \
    --checkpoint outputs/final_calibrated.pt \
    --config configs/mobile_int8.yaml \
    --quantize --prune
```

---

## 📁 Project Structure

```
respira-multi/
├── configs/
│   ├── base.yaml           # Base configuration
│   ├── teachers.yaml       # Teacher ensemble config
│   ├── student.yaml        # Student training config
│   └── mobile_int8.yaml    # Mobile export config
│
├── data/
│   ├── raw/                # Downloaded datasets
│   ├── processed/          # Preprocessed features
│   └── indices/            # Train/val/test splits (JSONL)
│
├── scripts/
│   ├── build_index.py      # Build unified dataset index
│   ├── preprocess_audio.py # Extract audio features
│   ├── train_student.py    # Main training script
│   ├── evaluate.py         # Evaluation & metrics
│   └── export_mobile.py    # Mobile deployment export
│
├── respiramulti/
│   ├── datasets/           # Data loading & augmentation
│   │   ├── schema.py       # Session schema definitions
│   │   ├── audio_transforms.py  # SpecAugment, noise, reverb
│   │   └── unified_loader.py    # Multi-dataset loader
│   │
│   ├── features/           # Feature extraction
│   │   ├── spectrogram.py  # Mel spectrogram, MFCC
│   │   ├── ppg_features.py # HR, HRV from video PPG
│   │   └── rr_features.py  # Respiratory rate estimation
│   │
│   ├── teachers/           # Teacher models (SOTA)
│   │   ├── beats.py        # BEATs audio transformer
│   │   ├── audio_mae.py    # Audio Masked Autoencoder
│   │   ├── ast_model.py    # Audio Spectrogram Transformer
│   │   ├── speech_encoder.py   # HuBERT/wav2vec2
│   │   └── ensemble.py     # Teacher ensemble averaging
│   │
│   ├── student/            # Student model (mobile)
│   │   ├── audio_encoder.py    # MobileNetV3/EfficientNet-Lite
│   │   ├── conformer.py        # Lightweight Conformer blocks
│   │   ├── fusion_transformer.py # Gated cross-modal fusion
│   │   ├── vitals_encoder.py   # Vitals with missingness
│   │   └── student_model.py    # Complete student architecture
│   │
│   ├── distillation/       # Knowledge distillation
│   │   ├── losses.py       # KL, feature, attention losses
│   │   └── trainer.py      # 3-stage training pipeline
│   │
│   ├── interpretability/   # Explainability
│   │   ├── prototypes.py   # Prototype bank & retrieval
│   │   └── explanations.py # Grad-CAM, attention viz
│   │
│   ├── uncertainty/        # Uncertainty quantification
│   │   ├── calibration.py  # Temperature scaling, ECE
│   │   └── conformal.py    # Conformal prediction sets
│   │
│   ├── robustness/         # Robustness features
│   │   └── tta.py          # Guarded test-time adaptation
│   │
│   ├── optimization/       # Mobile optimization
│   │   ├── quantization.py # QAT, INT8 quantization
│   │   ├── pruning.py      # Structured magnitude pruning
│   │   └── export.py       # ONNX, TorchScript, TFLite
│   │
│   └── utils/              # Utilities
│       ├── metrics.py      # AUROC, AUPRC, sensitivity@spec
│       └── logging.py      # Training logging, W&B
│
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

### Teacher Ensemble

| Model | Type | Purpose |
|-------|------|---------|
| **BEATs** | Self-supervised | General audio semantics |
| **Audio-MAE** | Masked autoencoder | Spectrogram understanding |
| **AST** | Supervised ViT | Strong classification baseline |
| **HuBERT** | Speech SSL | Speech segment features |

Teachers are trained offline and frozen during student distillation.

### Student Model

| Component | Specification |
|-----------|---------------|
| **Audio Encoder** | MobileNetV3-Small, 256-d output |
| **Conformer** | 2 layers, 256-d, 4 heads, kernel=15 |
| **Vitals Encoder** | 2-layer MLP with missingness embeddings |
| **Fusion Transformer** | 4 layers, 256-d, 4 heads |
| **Concept Bottleneck** | Interpretable disease prediction |

### Distillation Losses

```python
L_total = λ₁·L_KL(disease) + λ₁·L_KL(concept)     # Logit distillation
        + λ₂·L_MSE(cls_emb) + λ₂·L_MSE(tokens)    # Feature distillation  
        + λ₃·L_hard(disease) + λ₃·L_hard(concept)  # Hard labels (Stage 2)
        + λ₄·L_hierarchy                           # Hierarchy constraints
        + λ₅·L_gate_entropy                        # Gating regularization
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AUROC** | Area under ROC curve (per disease & macro) |
| **AUPRC** | Area under Precision-Recall curve |
| **Sens@90Spec** | Sensitivity at 90% specificity |
| **ECE** | Expected Calibration Error |
| **Coverage** | Conformal prediction coverage |

---

## 🔧 Key Features

### ✅ Robustness Training
- **SpecAugment**: Time/frequency masking
- **Additive noise**: SNR 0-25 dB (street, fan, TV, cafeteria)
- **Reverb simulation**: Room impulse response convolution
- **Modality dropout**: Random dropping of audio/vitals
- **MixStyle**: Feature statistics perturbation

### ✅ Missing Modality Handling
- Learned missingness embeddings per feature
- Model trained with modality dropout
- Graceful degradation when inputs unavailable

### ✅ Uncertainty Quantification
- **Temperature scaling**: Per-disease calibration
- **Conformal prediction**: Coverage-guaranteed prediction sets
- **Abstain logic**: "Re-record" when confidence insufficient

### ✅ Test-Time Adaptation (Guarded)
- Only updates LayerNorm/BatchNorm statistics
- Never updates classifier weights
- Automatic rollback on confidence collapse

### ✅ Interpretability
- Concept bottleneck: Disease predictions decomposed by concepts
- Prototype evidence: Similar training examples
- Attention visualization: Per-segment importance
- Grad-CAM: Spectrogram heatmaps

---

## 📱 Mobile Deployment

### Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Inference latency | <150ms | ~120ms |
| Model size (INT8) | <15MB | ~12MB |
| Memory usage | <50MB | ~40MB |

### Export Formats

```bash
# ONNX (cross-platform)
exports/respiramulti_mobile.onnx

# TorchScript (PyTorch mobile)
exports/respiramulti_mobile.pt

# TFLite (Android/iOS via TensorFlow Lite)
exports/respiramulti_mobile.tflite
```

---

## ⚠️ Safety & Disclaimers

> **IMPORTANT**: This is a **screening tool only**, NOT a medical diagnosis.

- ❌ Do NOT recommend treatment based on results
- ✅ Always advise seeking professional medical evaluation
- ✅ Show disclaimer: "Screening tool, not a diagnosis"
- ✅ If severe risk indicators → urgent care advisory
- 🔒 All recordings stored locally by default
- 🔒 Explicit opt-in required for data upload

---

## 📚 Datasets

### Supported Public Datasets

| Dataset | Type | Labels |
|---------|------|--------|
| **Coswara** | Cough, breath, voice | COVID, symptoms |
| **COUGHVID** | Cough crowdsourced | COVID status |
| **COVID-19 Sounds** | Cough, breath, voice | COVID, symptoms |
| **ICBHI 2017** | Lung sounds (auscultation) | Crackles, wheezes, diagnosis |
| **Fraiwan** | Chest-wall sounds | Asthma, COPD, pneumonia |

### Custom Dataset Schema

```json
{
  "session_id": "...",
  "subject_id": "...",
  "audio": {
    "cough_shallow_wav": "path/to/audio.wav",
    "breath_normal_wav": "path/to/audio.wav"
  },
  "labels": {
    "diseases": {"asthma": 0, "copd": 1, "pneumonia": 0},
    "concepts": {"wheeze_presence": 1, "crackle_presence": 0},
    "label_source": "clinician_dx",
    "label_confidence": 0.9
  },
  "demographics": {"age": 45, "sex": "male", "smoker": "former"}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 📖 Citation

```bibtex
@software{respira_multi_2024,
  title={RESPIRA-MULTI: Multimodal Respiratory Disease Screening},
  author={RESPIRA-MULTI Team},
  year={2024},
  url={https://github.com/your-org/respira-multi}
}
```

---

<div align="center">

**Built with ❤️ for accessible respiratory health screening**

</div>
