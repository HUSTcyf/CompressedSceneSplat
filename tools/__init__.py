# tools/__init__.py
# This file provides backward compatibility for imports after reorganizing tools into subdirectories.

import sys
from types import ModuleType

# Create alias modules for backward compatibility
def _create_module_alias(full_module_name, target_module):
    """Create a module alias in sys.modules for backward compatibility."""
    if full_module_name not in sys.modules:
        # Import the target module
        parts = target_module.split('.')
        target = __import__(target_module)
        for part in parts[1:]:
            target = getattr(target, part)

        # Create the alias module
        sys.modules[full_module_name] = target

# Set up module aliases for backward compatibility
# Old path -> New path
_MODULE_ALIASES = {
    # Moved to compression/
    "tools.rpca_utils": "tools.compression.rpca_utils",

    # Moved to projection/
    "tools.compute_procrustes_alignment": "tools.projection.compute_procrustes_alignment_simple",
    "tools.compute_procrustes_alignment_simple": "tools.projection.compute_procrustes_alignment_simple",

    # Moved to visualization/
    "tools.visualize_semantic_segmentation": "tools.visualization.visualize_semantic_segmentation",

    # Moved to data/
    "tools.feature_map_renderer": "tools.data.feature_map_renderer",
}

def _setup_aliases():
    """Set up all module aliases."""
    for old_name, new_name in _MODULE_ALIASES.items():
        _create_module_alias(old_name, new_name)

# Set up aliases when this module is imported
_setup_aliases()

# Also import commonly used classes/functions for direct access
from tools.compression.rpca_utils import (
    RPCA_GPU,
    RPCA_CPU,
    StructuredRPCA_GPU,
    svt_gpu,
    auto_rpca,
    apply_rpca,
)

__all__ = [
    # From compression/
    "RPCA_GPU",
    "RPCA_CPU",
    "StructuredRPCA_GPU",
    "svt_gpu",
    "auto_rpca",
    "apply_rpca",
]
