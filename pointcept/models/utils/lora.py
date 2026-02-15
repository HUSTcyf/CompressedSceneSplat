"""
LoRA (Low-Rank Adaptation) Implementation for SceneSplat

Based on:
- LoRA: https://arxiv.org/abs/2106.09685
- PointLoRA: https://github.com/Pointcept/PointLoRA

This module provides LoRA layers for fine-tuning Point Transformer V3
with minimal trainable parameters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Base LoRA layer class."""

    def __init__(self, r: int, lora_alpha: int):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merged = False

    def reset_parameters(self):
        raise NotImplementedError

    def train(self, mode: bool = True):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError


class LoRALinear(LoRALayer):
    """
    LoRA wrapper for nn.Linear layers.

    Formula: output = linear(x) + (x @ A.T @ B.T) * scaling
    where A ∈ R^(r × in_features), B ∈ R^(out_features × r)

    Args:
        r: Low-rank dimension (default: 8)
        lora_alpha: Scaling factor (default: 8)
        linear_layer: Target nn.Linear layer to wrap
        enable_prompt: Whether to enable prompt MLP adaptation
    """

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        linear_layer: nn.Linear,
        enable_prompt: bool = False,
    ):
        super().__init__(r, lora_alpha)
        self.linear = linear_layer
        self.enable_prompt = enable_prompt

        in_features = self.linear.in_features
        out_features = self.linear.out_features

        # LoRA parameters
        self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.linear.weight.new_zeros((out_features, r)))

        # Optional prompt MLP for enhanced adaptation
        if enable_prompt:
            self.prompt_mlp = nn.Sequential(
                nn.Linear(in_features, in_features // 4),
                nn.GELU(),
                nn.Linear(in_features // 4, out_features),
            )
            self.prompt_weight = nn.Parameter(torch.tensor(0.1))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """Set training mode."""
        self.linear.train(mode)
        if self.enable_prompt:
            self.prompt_mlp.train(mode)
        self.merged = False

    def eval(self):
        """Set evaluation mode."""
        self.linear.eval()
        if self.enable_prompt:
            self.prompt_mlp.eval()
        # Note: We don't merge weights for eval to maintain flexibility

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Input tensor [N, in_features] or [B, N, in_features]

        Returns:
            Output tensor with LoRA applied
        """
        # Standard linear transformation
        result = F.linear(x, self.linear.weight, bias=self.linear.bias)

        # LoRA adaptation: x @ A.T @ B.T * scaling
        lora_output = x @ self.lora_A.T @ self.lora_B.T
        result = result + lora_output * self.scaling

        # Optional prompt-enhanced adaptation
        if self.enable_prompt:
            prompt_out = self.prompt_mlp(x)
            result = result + self.prompt_weight * prompt_out

        return result

    def get_lora_params(self):
        """Return LoRA parameters for optimizer."""
        params = [self.lora_A, self.lora_B]
        if self.enable_prompt:
            params.extend(list(self.prompt_mlp.parameters()))
            params.append(self.prompt_weight)
        return params


class LoraConv2d(LoRALayer):
    """
    LoRA wrapper for nn.Conv2d layers.

    Note: PT-v3 uses SubMConv3d (sparse conv), but this is provided
    for completeness if needed for other layers.
    """

    def __init__(self, r: int, lora_alpha: int, conv_layer: nn.Conv2d):
        super().__init__(r, lora_alpha)
        self.conv = conv_layer

        in_channels = self.conv.in_channels
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]

        self.lora_A = nn.Parameter(
            self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
        )
        self.lora_B = nn.Parameter(
            self.conv.weight.new_zeros((out_channels * kernel_size, r * kernel_size))
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        self.conv.train(mode)
        self.merged = False

    def eval(self):
        self.conv.eval()
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.merged:
            lora_weight = (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            return F.conv2d(
                x,
                self.conv.weight + lora_weight,
                self.conv.bias,
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.groups,
            )
        return self.conv(x)


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """
    Freeze all parameters except LoRA parameters.

    Args:
        model: The model with LoRA layers
        bias: Whether to train bias parameters ("none", "all", "lora_only")
    """
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        elif bias == "all" or (bias == "lora_only" and "lora" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_lora_parameters(model: nn.Module) -> list:
    """
    Get all LoRA parameters from the model.

    Args:
        model: The model with LoRA layers

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name or ("prompt" in name and "mlp" in name):
            lora_params.append(param)
    return lora_params


def print_lora_summary(model: nn.Module) -> None:
    """
    Print a summary of LoRA parameters in the model.

    Args:
        model: The model with LoRA layers
    """
    total_params = 0
    lora_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if "lora_" in name:
            lora_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print("=" * 60)
    print("LoRA Parameter Summary")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA ratio: {lora_params / total_params * 100:.2f}%")
    print("=" * 60)


def merge_lora_weights(model: nn.Module) -> None:
    """
    Merge LoRA weights into the original linear layers.
    This is useful for deployment to avoid extra computation.

    Args:
        model: The model with LoRA layers to merge
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            if module.merged:
                continue

            # Merge: W = W + B @ A * scaling
            lora_weight = (module.lora_B @ module.lora_A) * module.scaling
            module.linear.weight.data += lora_weight
            module.merged = True

            # Optionally move lora parameters to CPU to save memory
            module.lora_A = module.lora_A.cpu()
            module.lora_B = module.lora_B.cpu()
