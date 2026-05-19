"""Cross-framework dtype mappings."""

import torch

try:
    import mlx.core as mx

    MLX_TO_TORCH: dict = {
        mx.bfloat16: torch.bfloat16,
        mx.float16: torch.float16,
        mx.float32: torch.float32,
        mx.float64: torch.float64,
        mx.int8: torch.int8,
        mx.int16: torch.int16,
        mx.int32: torch.int32,
        mx.int64: torch.int64,
        mx.uint8: torch.uint8,
    }
except ImportError:
    MLX_TO_TORCH = {}
