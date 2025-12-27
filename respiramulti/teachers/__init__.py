"""Teacher model implementations for knowledge distillation."""

from respiramulti.teachers.beats import BEATsTeacher
from respiramulti.teachers.audio_mae import AudioMAETeacher
from respiramulti.teachers.ast_model import ASTTeacher
from respiramulti.teachers.speech_encoder import HuBERTTeacher, Wav2Vec2Teacher
from respiramulti.teachers.ensemble import TeacherEnsemble

__all__ = [
    "BEATsTeacher",
    "AudioMAETeacher",
    "ASTTeacher",
    "HuBERTTeacher",
    "Wav2Vec2Teacher",
    "TeacherEnsemble",
]

