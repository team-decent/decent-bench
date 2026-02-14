"""Type definitions for optimization variables."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Literal, SupportsIndex, TypeAlias, Union

if TYPE_CHECKING:
    import jax
    import numpy
    import tensorflow as tf
    import torch

    from decent_bench.utils.array import Array

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

type BatchSize = int | Literal["all", "cost"]
"""
Type alias for batch size configuration in federated algorithms.
Use "all" for full-batch, "cost" to defer to the cost's batch size, or an explicit integer.
"""

type ResolvedBatchSize = int | Literal["all"]
"""
Type alias for the effective batch size after resolving a batch policy.
"""

type BatchingMode = Literal["epoch", "random"]
"""
Type alias for batching mode configuration when sampling mini-batches.
"""

type ClientWeights = dict[int, float] | Sequence[float]
"""
Type alias for client weighting configuration.
Use a dict keyed by client id, or a sequence indexed by client id.
"""

type Datapoint = tuple["Array", "Array"]  # noqa: TC008
"""Tuple of (x, y) representing one datapoint where x are features and y is the target."""

type Dataset = list[Datapoint]
"""
List of datapoints, where each datapoint is a tuple of (features, targets).

In decentralized optimization each agent has their own local dataset. This
type alias represents such datasets. This local dataset can be a subset of a larger
global dataset or the entire dataset itself. These subsets can be obtained
by using the :class:`~decent_bench.datasets.DatasetHandler` class, specifically the
:meth:`~decent_bench.datasets.DatasetHandler.get_partitions` method.

Features and targets are represented as :class:`~decent_bench.utils.array.Array`
objects or framework-specific tensor objects in special cases. For unsupervised learning,
targets are usually None.

The expected shapes depend on the specific dataset and cost function requirements,
but typically it is:

- Features: 1-dimensional vector (n_features,)
- Targets: 1-dimensional vector (n_targets,), or None for unsupervised learning.
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
