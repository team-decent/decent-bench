"""Type definitions for optimization variables."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Literal, SupportsIndex, TypeAlias, TypeVar, Union

if TYPE_CHECKING:
    import jax
    import numpy
    import tensorflow as tf
    import torch

    from decent_bench.networks import Network
    from decent_bench.utils.array import Array
    from decent_bench.agents import Agent

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

if TYPE_CHECKING:
    NetworkT = TypeVar("NetworkT", bound=Network)
else:
    NetworkT = TypeVar("NetworkT")
"""
Type variable for algorithms operating on a :class:`~decent_bench.networks.Network`.
"""

InitialStates: TypeAlias = Union[None, "Array", "dict[Agent, Array]"]
"""
Type alias for what can be passed to :meth:`~decent_bench.distributed_algorithms.Algorithm.initial_states`.
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

type ClientWeights = dict[int, float] | Sequence[float]
"""
Type alias for client weighting configuration.
Use a dict keyed by client id, or a sequence indexed by client id.
"""

type EmpiricalRiskReduction = Literal["mean"] | None
"""
Type alias for specifying reduction methods in empirical risk computations.
Can be "mean" to average over samples or None for no reduction and the result
is returned as a list of gradients for each sample.
"""

type EmpiricalRiskBatchSize = int | Literal["all"]
"""
Type alias for specifying batch size in empirical risk initialization.
Can be an integer for mini-batch size or the string "all" for full dataset.
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
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"


class SupportedDevices(Enum):
    """Enum for supported devices in decent-bench."""

    CPU = "cpu"
    GPU = "gpu"
