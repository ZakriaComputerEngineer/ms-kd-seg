"""Knowledge distillation from a transformer teacher for efficient
inflammatory-lesion segmentation in multiple sclerosis MRI.

Reference implementation accompanying the paper. See README.md for a map of the
pipeline and `notebooks/ms_kd_segmentation.ipynb` for an end-to-end run.
"""

__version__ = "1.0.0"

from .config import (ABLATION_VARIANTS, QUICK_VARIANT_KEYS, VARIANTS_BY_KEY, Config,
                     Variant, select_variants)

__all__ = [
    "Config", "Variant", "ABLATION_VARIANTS", "VARIANTS_BY_KEY",
    "QUICK_VARIANT_KEYS", "select_variants", "__version__",
]
