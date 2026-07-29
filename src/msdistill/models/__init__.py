"""Teacher and student architectures plus the distillation projection heads."""

from .student import (PooledFeatureProjector, SpatialFeatureProjector, StudentOutput,
                      TinyUNetStudent, build_student, count_parameters)
from .teacher import (MIT_PRESETS, FrozenTeacher, SegFormerHR, TeacherOutput,
                      build_teacher, load_frozen_teacher)

__all__ = [
    "TinyUNetStudent", "StudentOutput", "build_student", "count_parameters",
    "SpatialFeatureProjector", "PooledFeatureProjector",
    "SegFormerHR", "TeacherOutput", "FrozenTeacher", "build_teacher",
    "load_frozen_teacher", "MIT_PRESETS",
]
