"""Type definitions for optimization variables."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

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


T = TypeVar("T", bound=TensorLike)
"""
TypeVar for TensorLike types such as NumPy arrays, PyTorch tensors or
TensorFlow tensors.
"""
