"""Utility for operating on arrays from different deep learning and linear algebra frameworks."""

from __future__ import annotations

import contextlib
import random
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

torch = None
with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch as _torch

    torch = _torch

tf = None
with contextlib.suppress(ImportError, ModuleNotFoundError):
    import tensorflow as _tf

    tf = _tf

jnp = None
with contextlib.suppress(ImportError, ModuleNotFoundError):
    import jax.numpy as _jnp

    jnp = _jnp

jax_random = None
with contextlib.suppress(ImportError, ModuleNotFoundError):
    from jax import random as _jax_random

    jax_random = _jax_random

T = TypeVar("T", bound=ArrayLike)
"""
TypeVar for ArrayLike types such as NumPy arrays, PyTorch tensors or
nested containers (lists/tuples).

This TypeVar is used throughout the Interoperability class to ensure that
operations preserve the input framework type.
"""


def to_numpy(array: ArrayLike) -> NDArray[Any]:
    """
    Convert input array to a NumPy array.

    Args:
        array (ArrayLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).

    Returns:
        NDArray: Converted NumPy array.

    """
    if isinstance(array, np.ndarray):
        return array
    if torch and isinstance(array, torch.Tensor):
        return cast("np.ndarray", array.cpu().numpy())
    if tf and isinstance(array, tf.Tensor):
        return cast("np.ndarray", array.numpy())
    if (jnp and isinstance(array, jnp.ndarray)) or isinstance(array, (list, tuple)):
        return np.array(array)
    return np.array(array)


def from_numpy_like[T: ArrayLike](array: NDArray[Any], like: T) -> T:
    """
    Convert a NumPy array to the specified framework type.

    Args:
        array (NDArray): Input NumPy array.
        like (ArrayLike): Example array of the target framework type (e.g., torch.Tensor, tf.Tensor).

    Returns:
        ArrayLike: Converted array in the specified framework type.

    Raises:
        TypeError: if the framework type of `like` is unsupported.

    """
    device = None
    if hasattr(like, "device"):
        device = like.device

    if isinstance(like, np.ndarray):
        return cast("T", array)
    if torch and isinstance(like, torch.Tensor):
        return cast("T", torch.from_numpy(array).to(device))
    if tf and isinstance(like, tf.Tensor):
        with tf.device(device):
            return cast("T", tf.convert_to_tensor(array))
    if jnp and isinstance(like, jnp.ndarray):
        return cast("T", jnp.array(array, device=device))
    if isinstance(like, (list, tuple)):
        return cast("T", type(like)(array.tolist()))
    raise TypeError(f"Unsupported framework type: {type(like)}")


def sum[T: ArrayLike](  # noqa: A001
    array: T,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> T:
    """
    Sum elements of an array.

    Args:
        array (ArrayLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to sum.
            Same semantics as NumPy/TensorFlow/JAX.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        ArrayLike: Summed value in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        return cast("T", np.sum(array, axis=dim, keepdims=keepdims))
    if torch and isinstance(array, torch.Tensor):
        return cast("T", torch.sum(array, dim=dim, keepdim=keepdims))
    if tf and isinstance(array, tf.Tensor):
        return cast("T", tf.reduce_sum(array, axis=dim, keepdims=keepdims))
    if jnp and isinstance(array, jnp.ndarray):
        return cast("T", jnp.sum(array, axis=dim, keepdims=keepdims))
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        summed = np.sum(np_array, axis=dim, keepdims=keepdims).tolist()
        return cast("T", type(array)(summed if isinstance(summed, list) else [summed]))

    raise TypeError(f"Unsupported framework type: {type(array)}")


def mean[T: ArrayLike](
    array: T,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> T:
    """
    Compute mean of array elements.

    Args:
        array (ArrayLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute mean.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        ArrayLike: Mean value in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        return cast("T", np.mean(array, axis=dim, keepdims=keepdims))
    if torch and isinstance(array, torch.Tensor):
        return cast("T", torch.mean(array, dim=dim, keepdim=keepdims))
    if tf and isinstance(array, tf.Tensor):
        return cast("T", tf.reduce_mean(array, axis=dim, keepdims=keepdims))
    if jnp and isinstance(array, jnp.ndarray):
        return cast("T", jnp.mean(array, axis=dim, keepdims=keepdims))
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        meaned = np.mean(np_array, axis=dim, keepdims=keepdims).tolist()
        return cast("T", type(array)(meaned if isinstance(meaned, list) else [meaned]))

    raise TypeError(f"Unsupported framework type: {type(array)}")


def min[T: ArrayLike](  # noqa: A001
    array: T,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> T:
    """
    Compute minimum of array elements.

    Args:
        array (ArrayLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute minimum.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        ArrayLike: Minimum value in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        return cast("T", np.min(array, axis=dim, keepdims=keepdims))
    if torch and isinstance(array, torch.Tensor):
        return cast("T", torch.amin(array, dim=dim, keepdim=keepdims))
    if tf and isinstance(array, tf.Tensor):
        return cast("T", tf.reduce_min(array, axis=dim, keepdims=keepdims))
    if jnp and isinstance(array, jnp.ndarray):
        return cast("T", jnp.min(array, axis=dim, keepdims=keepdims))
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        mined = np.min(np_array, axis=dim, keepdims=keepdims).tolist()
        return cast("T", type(array)(mined if isinstance(mined, list) else [mined]))

    raise TypeError(f"Unsupported framework type: {type(array)}")


def max[T: ArrayLike](  # noqa: A001
    array: T,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> T:
    """
    Compute maximum of array elements.

    Args:
        array (ArrayLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute maximum.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        ArrayLike: Maximum value in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        return cast("T", np.max(array, axis=dim, keepdims=keepdims))
    if torch and isinstance(array, torch.Tensor):
        return cast("T", torch.amax(array, dim=dim, keepdim=keepdims))
    if tf and isinstance(array, tf.Tensor):
        return cast("T", tf.reduce_max(array, axis=dim, keepdims=keepdims))
    if jnp and isinstance(array, jnp.ndarray):
        return cast("T", jnp.max(array, axis=dim, keepdims=keepdims))
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        maxed = np.max(np_array, axis=dim, keepdims=keepdims).tolist()
        return cast("T", type(array)(maxed if isinstance(maxed, list) else [maxed]))

    raise TypeError(f"Unsupported framework type: {type(array)}")


def argmax[T: ArrayLike](array: T, dim: int | None = None, keepdims: bool = False) -> T:
    """
    Compute index of maximum value along an axis.

    Args:
        array (ArrayLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | None): Dimension along which to find maximum. If None, finds maximum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        ArrayLike: Indices of maximum values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        return cast("T", np.argmax(array, axis=dim, keepdims=keepdims))
    if torch and isinstance(array, torch.Tensor):
        return cast("T", torch.argmax(array, dim=dim, keepdim=keepdims))
    if tf and isinstance(array, tf.Tensor):
        ret = None
        if dim is None:
            # TensorFlow's argmax does not support dim=None directly
            dims = array.ndim if array.ndim is not None else 0
            array = tf.reshape(array, [-1])
            ret = (
                cast("T", tf.math.argmax(array, axis=0))
                if not keepdims
                else cast("T", tf.reshape(tf.math.argmax(array, axis=0), [1] * dims))
            )
        else:
            ret = (
                cast("T", tf.math.argmax(array, axis=dim))
                if not keepdims
                else cast("T", tf.expand_dims(tf.math.argmax(array, axis=dim), axis=dim))
            )
        return ret
    if jnp and isinstance(array, jnp.ndarray):
        return cast("T", jnp.argmax(array, axis=dim, keepdims=keepdims))
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        argmaxed = np.argmax(np_array, axis=dim, keepdims=keepdims).tolist()
        return cast(
            "T",
            type(array)(argmaxed if isinstance(argmaxed, list) else [argmaxed]),
        )

    raise TypeError(f"Unsupported framework type: {type(array)}")


def argmin[T: ArrayLike](array: T, dim: int | None = None, keepdims: bool = False) -> T:
    """
    Compute index of minimum value along an axis.

    Args:
        array (ArrayLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | None): Dimension along which to find minimum. If None, finds minimum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        ArrayLike: Indices of minimum values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        return cast("T", np.argmin(array, axis=dim, keepdims=keepdims))
    if torch and isinstance(array, torch.Tensor):
        return cast("T", torch.argmin(array, dim=dim, keepdim=keepdims))
    if tf and isinstance(array, tf.Tensor):
        ret = None
        if dim is None:
            # TensorFlow's argmin does not support dim=None directly
            dims = array.ndim if array.ndim is not None else 0
            array = tf.reshape(array, [-1])
            ret = (
                cast("T", tf.math.argmin(array, axis=0))
                if not keepdims
                else cast("T", tf.reshape(tf.math.argmin(array, axis=0), [1] * dims))
            )
        else:
            ret = (
                cast("T", tf.math.argmin(array, axis=dim))
                if not keepdims
                else cast("T", tf.expand_dims(tf.math.argmin(array, axis=dim), axis=dim))
            )
        return ret
    if jnp and isinstance(array, jnp.ndarray):
        return cast("T", jnp.argmin(array, axis=dim, keepdims=keepdims))
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        argmined = np.argmin(np_array, axis=dim, keepdims=keepdims).tolist()
        return cast(
            "T",
            type(array)(argmined if isinstance(argmined, list) else [argmined]),
        )

    raise TypeError(f"Unsupported framework type: {type(array)}")


def copy[T: ArrayLike](array: T) -> T:
    """
    Create a copy of the input array.

    Args:
        array (ArrayLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        ArrayLike: A copy of the input array in the same framework type.

    """
    if isinstance(array, np.ndarray):
        return cast("T", np.copy(array))
    if torch and isinstance(array, torch.Tensor):
        return cast("T", torch.clone(array))
    if tf and isinstance(array, tf.Tensor):
        return cast("T", tf.identity(array))
    if jnp and isinstance(array, jnp.ndarray):
        return cast("T", jnp.array(array, copy=True))
    return deepcopy(array)


def stack[T: ArrayLike](arrays: Sequence[T], dim: int = 0) -> T:
    """
    Stack a sequence of arrays along a new axis.

    Args:
        arrays (Sequence[ArrayLike]): Sequence of input arrays (NumPy, PyTorch, TensorFlow, JAX)
            or nested containers (list, tuple).
        dim (int): Axis along which to stack the arrays.

    Returns:
        ArrayLike: Stacked array in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported.

    """
    if isinstance(arrays[0], np.ndarray):
        return cast("T", np.stack(arrays, axis=dim))
    if torch and isinstance(arrays[0], torch.Tensor):
        return cast("T", torch.stack(arrays, dim=dim))
    if tf and isinstance(arrays[0], tf.Tensor):
        return cast("T", tf.stack(arrays, axis=dim))
    if jnp and isinstance(arrays[0], jnp.ndarray):
        return cast("T", jnp.stack(arrays, axis=dim))
    if isinstance(arrays[0], (list, tuple)):
        return cast("T", type(arrays[0])(np.stack(arrays, axis=dim).tolist()))

    raise TypeError(f"Unsupported framework type: {type(arrays[0])}")


def reshape[T: ArrayLike](array: T, shape: tuple[int, ...]) -> T:
    """
    Reshape an array to the specified shape.

    Args:
        array (ArrayLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        shape (tuple[int, ...]): Desired shape for the output array.

    Returns:
        ArrayLike: Reshaped array in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        return cast("T", np.reshape(array, shape))
    if torch and isinstance(array, torch.Tensor):
        return cast("T", torch.reshape(array, shape))
    if tf and isinstance(array, tf.Tensor):
        return cast("T", tf.reshape(array, shape))
    if jnp and isinstance(array, jnp.ndarray):
        return cast("T", jnp.reshape(array, shape))
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        reshaped = np.reshape(np_array, shape)
        return cast("T", type(array)(reshaped.tolist()))

    raise TypeError(f"Unsupported framework type: {type(array)}")


def zeros_like[T: ArrayLike](array: T) -> T:
    """
    Create an array of zeros with the same shape as the input.

    Args:
        array (ArrayLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        ArrayLike: Array of zeros in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        return np.zeros_like(array)
    if torch and isinstance(array, torch.Tensor):
        return cast("T", torch.zeros_like(array))
    if tf and isinstance(array, tf.Tensor):
        return cast("T", tf.zeros_like(array))
    if jnp and isinstance(array, jnp.ndarray):
        return cast("T", jnp.zeros_like(array))
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        return cast("T", type(array)(np.zeros_like(np_array).tolist()))

    raise TypeError(f"Unsupported framework type: {type(array)}")


def ones_like[T: ArrayLike](array: T) -> T:
    """
    Create an array of ones with the same shape as the input.

    Args:
        array (ArrayLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        ArrayLike: Array of ones in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        return np.ones_like(array)
    if torch and isinstance(array, torch.Tensor):
        return cast("T", torch.ones_like(array))
    if tf and isinstance(array, tf.Tensor):
        return cast("T", tf.ones_like(array))
    if jnp and isinstance(array, jnp.ndarray):
        return cast("T", jnp.ones_like(array))
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        return cast("T", type(array)(np.ones_like(np_array).tolist()))

    raise TypeError(f"Unsupported framework type: {type(array)}")


def rand_like[T: ArrayLike](array: T) -> T:
    """
    Create an array of random values with the same shape as the input.

    Args:
        array (ArrayLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        ArrayLike: Array of random values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        return cast("T", np.random.default_rng().random(array.shape))
    if torch and isinstance(array, torch.Tensor):
        return cast("T", torch.rand_like(array))
    if tf and isinstance(array, tf.Tensor):
        shape = tf.shape(array)
        return cast("T", tf.random.uniform(shape))
    if jnp and jax_random and isinstance(array, jnp.ndarray):
        return cast(
            "T",
            jax_random.uniform(jax_random.key(random.randint(0, 2**32 - 1)), shape=array.shape),
        )
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        random_array = np.random.default_rng().random(np_array.shape)
        return cast("T", type(array)(random_array.tolist()))

    raise TypeError(f"Unsupported framework type: {type(array)}")


def eye_like[T: ArrayLike](array: T) -> T:
    """
    Create an identity matrix with the same shape as the input.

    Args:
        array (ArrayLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        ArrayLike: Identity matrix in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        return cast("T", np.eye(*array.shape[-2:], dtype=array.dtype, device=array.device))
    if torch and isinstance(array, torch.Tensor):
        return cast(
            "T",
            torch.eye(*array.shape[-2:], dtype=array.dtype, device=array.device),
        )
    if tf and isinstance(array, tf.Tensor):
        return cast("T", tf.eye(*array.shape[-2:], dtype=array.dtype))
    if jnp and isinstance(array, jnp.ndarray):
        return cast("T", jnp.eye(*array.shape[-2:], dtype=array.dtype, device=array.device))
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        eye_array = np.eye(*np_array.shape[-2:])
        return cast("T", type(array)(eye_array.tolist()))

    raise TypeError(f"Unsupported framework type: {type(array)}")


def transpose[T: ArrayLike](array: T, dim: tuple[int, ...] | None = None) -> T:
    """
    Transpose an array.

    Args:
        array (ArrayLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (tuple[int, ...] | None): Desired dim order. If None, reverses the dimensions.

    Returns:
        ArrayLike: Transposed array in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        return cast("T", np.transpose(array, axes=dim))
    if torch and isinstance(array, torch.Tensor):
        # Handle None case for PyTorch
        return (
            cast("T", torch.permute(array, dims=dim))
            if dim
            else cast("T", torch.permute(array, dims=list(reversed(range(array.ndim)))))
        )
    if tf and isinstance(array, tf.Tensor):
        return cast("T", tf.transpose(array, perm=dim))
    if jnp and isinstance(array, jnp.ndarray):
        return cast("T", jnp.transpose(array, axes=dim))
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        transposed = np.transpose(np_array, axes=dim)
        return cast("T", type(array)(transposed.tolist()))

    raise TypeError(f"Unsupported framework type: {type(array)}")
