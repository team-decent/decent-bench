"""Type definitions for optimization variables."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax
    import tensorflow as tf
    import torch

    TensorLike = np.ndarray | torch.Tensor | tf.Tensor | jax.Array
    """
    Type alias for Tensor-like types supported in decent-bench, including NumPy arrays,
    PyTorch tensors, TensorFlow tensors, and JAX arrays.

    alias of :class:`numpy.ndarray` | :class:`torch.Tensor` | :class:`tf.Tensor` | :class:`jax.Array`
    """
else:

    class TensorLike(type):
        """Required for proper autodoc generation."""


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


type SupportedXTypes = TensorLike | float | int
"""
Type alias for supported types for optimization variables in decent-bench,
including Tensor-like types and scalars.

alias of :class:`TensorLike` | :class:`float` | :class:`int`
"""
