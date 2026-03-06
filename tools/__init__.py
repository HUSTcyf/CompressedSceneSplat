# tools/__init__.py
# This file provides backward compatibility for imports after reorganizing tools into subdirectories.

import sys
from types import ModuleType
import importlib

# Create a lazy module class that imports on first access
class _LazyModule:
    """A lazy module that imports the target module on first attribute access."""
    def __init__(self, target_module):
        self.__target_module__ = target_module
        self.__module__ = None

    def __getattr__(self, name):
        if self.__module__ is None:
            try:
                self.__module__ = importlib.import_module(self.__target_module__)
            except ImportError as e:
                raise ImportError(
                    f"Could not import {self.__target_module__}: {e}"
                ) from e
        return getattr(self.__module__, name)

    def __repr__(self):
        if self.__module__ is None:
            return f"<lazy module '{self.__target_module__}'>"
        return repr(self.__module__)

# Set up module aliases for backward compatibility using lazy loading
def _setup_aliases():
    """Set up all module aliases using lazy loading."""

    # Moved to compression/
    sys.modules['tools.rpca_utils'] = _LazyModule('tools.compression.rpca_utils')

    # Moved to projection/
    sys.modules['tools.compute_procrustes_alignment'] = _LazyModule('tools.projection.compute_procrustes_alignment_simple')
    sys.modules['tools.compute_procrustes_alignment_simple'] = _LazyModule('tools.projection.compute_procrustes_alignment_simple')

    # Moved to visualization/
    sys.modules['tools.visualize_semantic_segmentation'] = _LazyModule('tools.visualization.visualize_semantic_segmentation')

    # Moved to data/
    sys.modules['tools.feature_map_renderer'] = _LazyModule('tools.data.feature_map_renderer')

# Set up aliases when this module is imported
_setup_aliases()

# Also import commonly used classes/functions for direct access
try:
    from tools.compression.rpca_utils import (
        RPCA_GPU,
        RPCA_CPU,
        StructuredRPCA_GPU,
        svt_gpu,
        auto_rpca,
        apply_rpca,
    )
except ImportError:
    pass

__all__ = [
    # From compression/
    "RPCA_GPU",
    "RPCA_CPU",
    "StructuredRPCA_GPU",
    "svt_gpu",
    "auto_rpca",
    "apply_rpca",
]
