from .misc import offset2batch, offset2bincount, batch2offset, off_diagonal
from .checkpoint import checkpoint
from .serialization import encode, decode
from .structure import Point

# LoRA utilities
from .lora import (
    LoRALinear,
    LoRALayer,
    LoraConv2d,
    mark_only_lora_as_trainable,
    get_lora_parameters,
    print_lora_summary,
    merge_lora_weights,
)
from .lora_injector import (
    inject_lora_to_ptv3,
    inject_lora_to_stage,
    inject_lora_with_preset,
    LORA_PRESETS,
    save_lora_checkpoint,
    load_lora_checkpoint,
)
