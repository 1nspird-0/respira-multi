"""
Full model implementations.

Re-exports for convenient imports.
"""

from respiramulti.student.student_model import RespiraMultiStudent, create_student_model
from respiramulti.teachers.ensemble import TeacherEnsemble


class RespiraMultiTeacher(TeacherEnsemble):
    """
    Complete teacher model for RESPIRA-MULTI.
    
    Wrapper around TeacherEnsemble for convenient usage.
    """
    
    @classmethod
    def from_config(cls, config: dict) -> "RespiraMultiTeacher":
        """Create teacher from configuration."""
        teachers_config = config.get('teachers', {})
        
        return cls(
            num_diseases=config.get('num_diseases', 12),
            num_concepts=config.get('num_concepts', 17),
            embed_dim=teachers_config.get('embed_dim', 768),
            enable_beats=teachers_config.get('beats', {}).get('enabled', True),
            enable_audio_mae=teachers_config.get('audio_mae', {}).get('enabled', True),
            enable_ast=teachers_config.get('ast', {}).get('enabled', True),
            enable_speech=teachers_config.get('speech_encoder', {}).get('enabled', True),
            speech_encoder_type=teachers_config.get('speech_encoder', {}).get('type', 'hubert'),
            weights=teachers_config.get('ensemble', {}).get('weights'),
            temperature=teachers_config.get('ensemble', {}).get('temperature', 1.0),
            config=teachers_config,
        )


__all__ = [
    "RespiraMultiStudent",
    "RespiraMultiTeacher",
    "create_student_model",
]

