# RESPIRA-MULTI

**Multimodal Respiratory Disease Screening via Teacher-Student Distillation**

A production-grade PyTorch implementation for on-device respiratory disease screening using smartphone-capturable inputs. This system achieves high accuracy through knowledge distillation from large teacher models (BEATs, Audio-MAE, AST, HuBERT) into a lightweight mobile-deployable student model.

---

## Overview

RESPIRA-MULTI screens for 12 respiratory conditions using:
- **Audio**: Coughs (shallow/deep), breathing sounds, sustained vowels, reading passages
- **Camera-derived vitals**: Heart rate, HRV, respiratory rate via PPG/rPPG
- **Optional demographics**: Age, sex, smoking status, known conditions, symptoms

The system provides:
- Multi-label disease predictions with calibrated probabilities
- Interpretable concept predictions (wheeze, crackle, rhonchi, etc.)
- Conformal prediction sets with coverage guarantees
- "Abstain" behavior when input quality is insufficient

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TEACHER ENSEMBLE                            │
│  ┌─────────┐  ┌──────────┐  ┌─────┐  ┌────────────────┐        │
│  │  BEATs  │  │Audio-MAE │  │ AST │  │ HuBERT/wav2vec │        │
│  └────┬────┘  └────┬─────┘  └──┬──┘  └───────┬────────┘        │
│       └────────────┴───────────┴─────────────┘                  │
│                         │ Soft Labels + Embeddings              │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                DISTILLATION LOSSES                        │   │
│  │  • KL Divergence (logits)  • L2 (features)  • Attn MSE   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STUDENT MODEL (~4MB)                        │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  MobileNetV3     │    │  Vitals Encoder  │                   │
│  │  Audio Encoder   │    │  (MLP + Missing) │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
│           │                       │                              │
│           ▼                       │                              │
│  ┌──────────────────┐             │                              │
│  │   Conformer      │             │                              │
│  │   (2 layers)     │             │                              │
│  └────────┬─────────┘             │                              │
│           │                       │                              │
│           └───────────┬───────────┘                              │
│                       ▼                                          │
│           ┌──────────────────────┐                               │
│           │  Fusion Transformer  │                               │
│           │  (Gated + Cross-Attn)│                               │
│           └──────────┬───────────┘                               │
│                      │                                           │
│           ┌──────────┴──────────┐                                │
│           ▼                     ▼                                │
│  ┌─────────────────┐   ┌─────────────────┐                      │
│  │ Concept Head    │──▶│  Disease Head   │                      │
│  │ (Bottleneck)    │   │  (12 diseases)  │                      │
│  └─────────────────┘   └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Target Conditions

| Disease | Description |
|---------|-------------|
| Healthy | No respiratory condition |
| URTI | Upper Respiratory Tract Infection |
| LRTI | Lower Respiratory Tract Infection |
| Asthma | Chronic asthma |
| COPD | Chronic Obstructive Pulmonary Disease |
| Pneumonia | Bacterial/viral pneumonia |
| Bronchitis | Acute/chronic bronchitis |
| Bronchiolitis | Inflammation of bronchioles |
| Bronchiectasis | Permanent airway dilation |
| TB | Tuberculosis |
| COVID-19 | SARS-CoV-2 infection |
| Heart Failure (Pulm.) | Cardiac-related pulmonary symptoms |

---

## Concepts (Interpretable Features)

**Binary Concepts:**
- Wheeze presence, Crackle presence, Rhonchi presence
- Stridor presence, Cough detected

**Continuous Concepts:**
- Cough rate, Cough wetness proxy
- Breath phase irregularity
- Speech breathiness, Phrase read quality
- HR/HRV metrics (mean, std, RMSSD, SDNN)
- Respiratory rate, SpO2 estimate, Perfusion quality

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/respira-multi.git
cd respira-multi

# Create environment
conda create -n respira python=3.10
conda activate respira

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

---

## Project Structure

```
respiramulti/
├── __init__.py                 # Package exports
├── datasets/
│   ├── schema.py               # SessionSchema, labels, segment types
│   ├── audio_transforms.py     # SpecAugment, noise, reverb, MixStyle
│   └── unified_loader.py       # Multi-dataset loader, BatchedSession
├── features/
│   ├── spectrogram.py          # Mel-spectrogram, MFCC, PatchEmbed
│   ├── ppg_features.py         # PPG extraction, HR/HRV, rPPG
│   └── rr_features.py          # Respiratory rate estimation
├── teachers/
│   ├── beats.py                # BEATs transformer encoder
│   ├── audio_mae.py            # Audio-MAE masked autoencoder
│   ├── ast_model.py            # Audio Spectrogram Transformer
│   ├── speech_encoder.py       # HuBERT/wav2vec2 encoders
│   └── ensemble.py             # Teacher ensemble + distillation loss
├── student/
│   ├── audio_encoder.py        # MobileNetV3/EfficientNet-Lite
│   ├── conformer.py            # Lightweight Conformer blocks
│   ├── fusion_transformer.py   # Cross-modal attention + gating
│   ├── vitals_encoder.py       # MLP with missingness handling
│   └── student_model.py        # Complete student + ConceptBottleneck
├── distillation/
│   ├── losses.py               # KL, L2, attention distillation
│   └── trainer.py              # 3-stage training pipeline
├── interpretability/
│   ├── prototypes.py           # Prototype bank, evidence retrieval
│   └── explanations.py         # Attention viz, Grad-CAM
├── uncertainty/
│   ├── calibration.py          # Temperature scaling, ECE
│   └── conformal.py            # Conformal prediction sets
├── optimization/
│   ├── quantization.py         # QAT for INT8
│   ├── pruning.py              # Structured pruning
│   └── export.py               # ONNX/TFLite export
├── robustness/
│   └── tta.py                  # Test-time adaptation
├── utils/
│   ├── metrics.py              # AUROC, AUPRC, sensitivity@spec
│   └── logging.py              # W&B integration
└── models/
    └── full_model.py           # Orchestration module

scripts/
├── train_student.py            # Main training script
├── evaluate.py                 # Evaluation pipeline
├── export_mobile.py            # Mobile export
├── build_index.py              # Dataset indexing
└── preprocess_audio.py         # Audio preprocessing

configs/
├── base.yaml                   # Base configuration
├── teachers.yaml               # Teacher model configs
├── student.yaml                # Student architecture
└── mobile_int8.yaml            # INT8 export settings
```

---

## Data Format

Sessions follow the `SessionSchema` structure:

```python
from respiramulti import SessionSchema

session = SessionSchema(
    session_id="sess_001",
    subject_id="subj_001",
    dataset_source="my_dataset",
    audio_segments={
        "cough_shallow": AudioSegment(file_path="cough_s.wav", ...),
        "breath_normal": AudioSegment(file_path="breath.wav", ...),
        # ...
    },
    video_segments={
        "finger_ppg": VideoSegment(file_path="ppg.mp4", ...),
    },
    vitals=VitalsFeatures(hr_mean=72.0, rr_est=16.0, ...),
    labels=Labels(
        diseases={"asthma": 1, "healthy": 0, ...},
        concepts={"wheeze_presence": 1, "hr_mean": 72.0, ...},
    ),
    demographics=Demographics(age=35, sex=Sex.MALE, ...),
)
```

Build dataset indices:
```bash
python scripts/build_index.py \
    --raw_dir data/raw \
    --output_dir data/indices \
    --splits train:0.7,val:0.15,test:0.15
```

---

## Training

### 3-Stage Training Pipeline

**Stage 1: Pure Distillation** (30 epochs)
- No hard labels, only soft targets from teachers
- Can utilize unlabeled data
- Loss: KL(student || teacher) + L2(embeddings)

**Stage 2: Mixed Training** (40 epochs)
- Hard labels + distillation
- Focal loss for disease, BCE for concepts
- Hierarchical regularization (LRTI → pneumonia, bronchitis, etc.)

**Stage 3: Calibration**
- Temperature scaling per disease
- ECE minimization on validation set

```bash
# Full training
python scripts/train_student.py \
    --config configs/student.yaml \
    --output_dir outputs/run_001

# Resume from checkpoint
python scripts/train_student.py \
    --config configs/student.yaml \
    --resume outputs/run_001/best_stage2.pt
```

---

## Augmentation Pipeline

Robust training via aggressive augmentation:

| Augmentation | Parameters |
|--------------|------------|
| SpecAugment | 2 time masks (40 frames), 2 freq masks (8 bins) |
| Additive Noise | SNR 0-25 dB (street, fan, cafeteria, white) |
| Reverberation | Synthetic RIR or from dataset |
| Mic Response | Bandpass 50-8000 Hz, ±6 dB EQ |
| Time Shift | ±200 ms |
| Time Stretch | 0.9x - 1.1x |
| MixStyle | α=0.3 feature statistics mixing |
| Modality Dropout | 20% per modality |

---

## Inference

```python
from respiramulti import RespiraMultiStudent, SessionSchema
import torch

# Load trained model
model = RespiraMultiStudent.from_pretrained("outputs/final_calibrated.pt")
model.eval()

# Prepare inputs
audio_tokens = ...  # [1, num_tokens, 1, 64, 201]
segment_types = ... # [1, num_tokens]
vitals = ...        # [1, 15]

# Inference
with torch.no_grad():
    output = model(
        audio_tokens=audio_tokens,
        segment_types=segment_types,
        vitals=vitals,
    )

# Calibrated probabilities
disease_probs = torch.sigmoid(output.disease_logits)
concept_probs = torch.sigmoid(output.concept_logits)

# Conformal prediction set (90% coverage)
from respiramulti.uncertainty.conformal import ConformalPredictor
cp = ConformalPredictor.load("outputs/conformal_calibration.pt")
pred_set = cp.predict_set(disease_probs, coverage=0.90)

if pred_set.abstain:
    print(f"Abstaining: {pred_set.abstain_reason}")
else:
    print(f"Predicted diseases: {pred_set.diseases}")
```

---

## Mobile Export

Export for on-device deployment:

```bash
# ONNX + TorchScript
python scripts/export_mobile.py \
    --checkpoint outputs/final_calibrated.pt \
    --output_dir exports/ \
    --formats onnx,torchscript

# INT8 TFLite (requires TensorFlow)
python scripts/export_mobile.py \
    --checkpoint outputs/final_calibrated.pt \
    --output_dir exports/ \
    --formats tflite \
    --quantize int8
```

**Target specs:**
- Model size: ~4 MB (INT8)
- Latency: <150 ms on mid-range smartphone
- Memory: <50 MB peak

---

## Interpretability

### Concept Bottleneck

Disease predictions flow through concept predictions:
```
Embedding → Concept Head → [wheeze, crackle, ...] → Disease Head → [asthma, COPD, ...]
                                       │
                                       └── Residual path (30% weight)
```

### Prototype Evidence

```python
from respiramulti.interpretability.prototypes import PrototypeRetrieval

retriever = PrototypeRetrieval(prototype_bank, segment_types)
evidence = retriever.get_evidence_for_prediction(
    embedding=output.cls_embedding,
    token_embeddings=output.token_embeddings,
    segment_type_indices=segment_types,
    predicted_diseases=[2, 4],  # LRTI, COPD
)
# Returns: {"disease_2": [PrototypeMatch(segment="breath_deep", sim=0.87), ...]}
```

### Gated Fusion Weights

```python
# output.gate_weights = {"audio": 0.72, "vitals": 0.28}
# Interpretable modality contribution
```

---

## Uncertainty Quantification

### Temperature Scaling Calibration
```python
from respiramulti.uncertainty.calibration import TemperatureScaling

ts = TemperatureScaling(num_classes=12, per_class=True)
results = ts.fit(val_logits, val_labels)
# ECE reduced from 0.08 → 0.02
```

### Conformal Prediction
```python
from respiramulti.uncertainty.conformal import ConformalPredictor

cp = ConformalPredictor(
    num_classes=12,
    class_names=DISEASES,
    coverage_levels=[0.80, 0.90, 0.95],
    abstain_threshold=0.3,
)
cp.calibrate(cal_probs, cal_labels)

pred_set = cp.predict_set(test_probs, coverage=0.90)
# PredictionSet(diseases={"asthma", "copd"}, abstain=False, coverage=0.90)
```

---

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/final_calibrated.pt \
    --test_data data/indices/test.jsonl \
    --output_dir results/
```

**Metrics computed:**
- Per-disease AUROC, AUPRC
- Sensitivity @ 80%, 90%, 95% specificity
- ECE (Expected Calibration Error)
- Reliability diagrams
- Conformal coverage verification
- Robustness vs. SNR curves

---

## Robustness Features

### Test-Time Adaptation (TTA)
```python
from respiramulti.robustness.tta import TestTimeAdaptation

tta = TestTimeAdaptation(model, update_bn=True, update_ln=True)
adapted_output = tta.adapt_and_predict(noisy_inputs, entropy_threshold=0.5)
```

### Missing Modality Handling

The model gracefully handles:
- Missing vitals (PPG failed) → Uses learned "missing" embeddings
- Missing audio segments → Masks in fusion transformer
- Low-quality signals → Quality-aware conformal prediction

---

## Configuration Reference

See `configs/base.yaml` for complete options:

```yaml
model:
  audio_encoder:
    backbone: "mobilenetv3_small"
    embedding_dim: 256
  conformer:
    num_layers: 2
    d_model: 256
  fusion:
    num_layers: 4
    use_gated_fusion: true
    
training:
  batch_size: 16
  lr: 3e-4
  max_epochs: 100
  label_smoothing: 0.1
  
augmentation:
  audio:
    spec_augment: {time_mask_param: 40, freq_mask_param: 8}
    additive_noise: {snr_range: [0, 25]}
    reverb: {enabled: true}
```

---

## Safety & Disclaimers

⚠️ **This system is for screening only, not diagnosis.**

- Always recommend professional medical evaluation for positive screens
- Display confidence intervals, not just point estimates
- Abstain when signal quality is insufficient
- Do not recommend treatments or medications
- Raw recordings are stored locally, not transmitted

---

## Citation

```bibtex
@article{respiramulti2024,
  title={RESPIRA-MULTI: Multimodal Respiratory Disease Screening via Teacher-Student Distillation},
  author={Your Team},
  journal={arXiv preprint},
  year={2024}
}
```

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Teacher model architectures: BEATs, Audio-MAE, AST, HuBERT
- Audio augmentation: SpecAugment, MixStyle
- Calibration: Temperature scaling, conformal prediction
