# tools/__init__.py
# Package initialization for SceneSplat tools

import sys
from pathlib import Path
import importlib
from types import ModuleType


# ============================================================================
# PROJECT_ROOT - Centralized definition for all tools/ subdirectories
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# Lazy module loading for backward compatibility
# ============================================================================
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


def _setup_module_aliases():
    """Set up module aliases for backward compatibility using lazy loading."""
    # rpca_utils: moved to tools.compression.rpca_utils
    sys.modules['tools.rpca_utils'] = _LazyModule('tools.compression.rpca_utils')

    # compute_procrustes_alignment_simple: moved to tools.projection.compute_procrustes_alignment_simple
    sys.modules['tools.compute_procrustes_alignment_simple'] = _LazyModule(
        'tools.projection.compute_procrustes_alignment_simple'
    )

    # compute_procrustes_alignment: alias to compute_procrustes_alignment_simple
    sys.modules['tools.compute_procrustes_alignment'] = _LazyModule(
        'tools.projection.compute_procrustes_alignment_simple'
    )


# Set up aliases when this module is imported
_setup_module_aliases()

__all__ = ['PROJECT_ROOT']
