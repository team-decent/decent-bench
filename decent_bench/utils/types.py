"""Type definitions for optimization variables."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal, SupportsIndex, TypeAlias, Union

if TYPE_CHECKING:
    import jax
    import numpy
    import tensorflow as tf
    import torch

ArrayLike: TypeAlias = Union["numpy.ndarray", "torch.Tensor", "tf.Tensor", "jax.Array"]  # noqa: UP040
"""
Type alias for array-like types supported in decent-bench, including NumPy arrays,
PyTorch tensors, TensorFlow tensors, and JAX arrays.
"""

SupportedArrayTypes: TypeAlias = ArrayLike | float | int  # noqa: UP040
"""
Type alias for supported types for optimization variables in decent-bench,
including array-like types and scalars.
"""

ArrayKey: TypeAlias = SupportsIndex | slice | tuple[SupportsIndex | slice, ...]  # noqa: UP040
"""
Type alias for valid keys used to index into supported array types.
Includes single indices, tuples of indices, slices, and tuples of slices.
"""

type EmpiricalRiskIndices = list[int] | Literal["all", "batch"] | int
"""
Type alias for specifying indices in empirical risk computations.
Can be a list of integers, the string "all" for full dataset, the string "batch" for a mini-batch,
or an integer specifying a single datapoint.
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
