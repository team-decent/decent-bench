"""Type definitions for optimization variables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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


SupportedFrameworks = Literal["numpy", "torch", "tensorflow", "jax"]
"""
Literal type for supported frameworks in decent-bench.
"""

SupportedDevices = Literal["cpu", "gpu"]
"""
Literal type for supported devices in decent-bench. Depends on the framework used.
"""

SupportedXTypes = TensorLike | float | int
"""
Type alias for supported types for optimization variables in decent-bench,
including Tensor-like types and scalars.
Alias of :class:`TensorLike` | :class:`float` | :class:`int`
"""
