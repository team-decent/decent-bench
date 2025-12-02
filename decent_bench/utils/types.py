"""Type definitions for optimization variables."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, TypeAlias, Union

if TYPE_CHECKING:
    import jax
    import numpy
    import tensorflow as tf
    import torch

TensorLike: TypeAlias = Union["numpy.ndarray", "torch.Tensor", "tf.Tensor", "jax.Array"]  # noqa: UP040
"""
Type alias for Tensor-like types supported in decent-bench, including NumPy arrays,
PyTorch tensors, TensorFlow tensors, and JAX arrays.
"""

SupportedXTypes: TypeAlias = TensorLike | float | int  # noqa: UP040
"""
Type alias for supported types for optimization variables in decent-bench,
including Tensor-like types and scalars.
"""


class SupportedFrameworks(Enum):
    """Enum for supported frameworks in decent-bench."""

    NUMPY = "numpy"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"


class SupportedDevices(Enum):
    """Enum for supported devices in decent-bench."""

    CPU = "cpu"
    GPU = "gpu"
