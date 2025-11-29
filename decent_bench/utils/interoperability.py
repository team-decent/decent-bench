"""Utilities for operating on arrays from different deep learning and linear algebra frameworks."""

from __future__ import annotations

import contextlib
import random
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from decent_bench.utils.types import T, TensorLike

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


def to_numpy(array: TensorLike | complex) -> NDArray[Any]:
    """
    Convert input array to a NumPy array.

    Args:
        array (TensorLike): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).

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


def from_numpy_like(array: NDArray[Any], like: T) -> T:
    """
    Convert a NumPy array to the specified framework type.

    Args:
        array (NDArray): Input NumPy array.
        like (TensorLike): Example array of the target framework type (e.g., torch.Tensor, tf.Tensor).

    Returns:
        TensorLike: Converted array in the specified framework type.

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


def sum(  # noqa: A001
    array: T,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> T:
    """
    Sum elements of an array.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to sum.
            If None, sums over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        TensorLike: Summed value in the same framework type as the input.

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


def mean(
    array: T,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> T:
    """
    Compute mean of array elements.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute mean.
            If None, computes mean over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        TensorLike: Mean value in the same framework type as the input.

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


def min(  # noqa: A001
    array: T,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> T:
    """
    Compute minimum of array elements.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute minimum.
            If None, finds minimum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        TensorLike: Minimum value in the same framework type as the input.

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


def max(  # noqa: A001
    array: T,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> T:
    """
    Compute maximum of array elements.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute maximum.
            If None, finds maximum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        TensorLike: Maximum value in the same framework type as the input.

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


def argmax(array: T, dim: int | None = None, keepdims: bool = False) -> T:
    """
    Compute index of maximum value.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | None): Dimension along which to find maximum. If None, finds maximum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        TensorLike: Indices of maximum values in the same framework type as the input.

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


def argmin(array: T, dim: int | None = None, keepdims: bool = False) -> T:
    """
    Compute index of minimum value.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | None): Dimension along which to find minimum. If None, finds minimum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        TensorLike: Indices of minimum values in the same framework type as the input.

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


def copy(array: T) -> T:
    """
    Create a copy of the input array.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        TensorLike: A copy of the input array in the same framework type.

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


def stack(arrays: Sequence[T], dim: int = 0) -> T:
    """
    Stack a sequence of arrays along a new dimension.

    Args:
        arrays (Sequence[TensorLike]): Sequence of input arrays (NumPy, PyTorch, TensorFlow, JAX)
            or nested containers (list, tuple).
        dim (int): Dimension along which to stack the arrays.

    Returns:
        TensorLike: Stacked array in the same framework type as the inputs.

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


def reshape(array: T, shape: tuple[int, ...]) -> T:
    """
    Reshape an array to the specified shape.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        shape (tuple[int, ...]): Desired shape for the output array.

    Returns:
        TensorLike: Reshaped array in the same framework type as the input.

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


def zeros_like(array: T) -> T:
    """
    Create an array of zeros with the same shape and type as the input.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        TensorLike: Array of zeros in the same framework type as the input.

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


def ones_like(array: T) -> T:
    """
    Create an array of ones with the same shape and type as the input.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        TensorLike: Array of ones in the same framework type as the input.

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


def rand_like(array: T, low: float = 0.0, high: float = 1.0) -> T:
    """
    Create an array of random values with the same shape and type as the input.

    Values are drawn uniformly from [low, high).

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        low (float): Lower bound of the uniform distribution.
        high (float): Upper bound of the uniform distribution.

    Returns:
        TensorLike: Array of random values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        random_array = np.random.default_rng().uniform(
            low=low,
            high=high,
            size=array.shape,
        )
        random_array = random_array.astype(array.dtype) if isinstance(random_array, np.ndarray) else random_array
        return cast("T", random_array)
    if torch and isinstance(array, torch.Tensor):
        return cast("T", (high - low) * torch.rand_like(array) + low)
    if tf and isinstance(array, tf.Tensor):
        return cast(
            "T",
            tf.random.uniform(
                tf.shape(array),
                dtype=array.dtype,
                minval=low,
                maxval=high,
            ),
        )
    if jnp and jax_random and isinstance(array, jnp.ndarray):
        return cast(
            "T",
            jax_random.uniform(
                jax_random.key(random.randint(0, 2**32 - 1)),
                shape=array.shape,
                dtype=array.dtype,
                minval=low,
                maxval=high,
            ),
        )
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        np_random_array = np.random.default_rng().uniform(
            low=low,
            high=high,
            size=np_array.shape,
        )
        np_random_array = (
            np_random_array.astype(np_array.dtype).tolist()
            if isinstance(np_random_array, np.ndarray)
            else np_random_array
        )
        return cast("T", type(array)(np_random_array))

    raise TypeError(f"Unsupported framework type: {type(array)}")


def randn_like(array: T, mean: float = 0.0, std: float = 1.0) -> T:
    """
    Create an array of random values with the same shape and type as the input.

    Values are drawn from a normal distribution with mean `mean` and standard deviation `std`.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.

    Returns:
        TensorLike: Array of random values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        random_array = np.random.default_rng().normal(
            loc=mean,
            scale=std,
            size=array.shape,
        )
        random_array = random_array.astype(array.dtype) if isinstance(random_array, np.ndarray) else random_array
        return cast("T", random_array)
    if torch and isinstance(array, torch.Tensor):
        return cast(
            "T",
            torch.normal(
                mean=mean,
                std=std,
                size=array.shape,
                dtype=array.dtype,
                device=array.device,
            ),
        )
    if tf and isinstance(array, tf.Tensor):
        shape = tf.shape(array)
        return cast("T", tf.random.normal(shape, mean=mean, stddev=std, dtype=array.dtype))
    if jnp and jax_random and isinstance(array, jnp.ndarray):
        return cast(
            "T",
            mean
            + std
            * jax_random.normal(
                jax_random.key(random.randint(0, 2**32 - 1)),
                shape=array.shape,
                dtype=array.dtype,
            ),
        )
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        np_random_array = np.random.default_rng().normal(loc=mean, scale=std, size=np_array.shape)
        np_random_array = (
            np_random_array.astype(np_array.dtype).tolist()
            if isinstance(np_random_array, np.ndarray)
            else np_random_array
        )
        return cast("T", type(array)(np_random_array))

    raise TypeError(f"Unsupported framework type: {type(array)}")


def eye_like(array: T) -> T:
    """
    Create an identity matrix with the same shape as the input.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        TensorLike: Identity matrix in the same framework type as the input.

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


def transpose(array: T, dim: tuple[int, ...] | None = None) -> T:
    """
    Transpose an array.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (tuple[int, ...] | None): Desired dim order. If None, reverses the dimensions.

    Returns:
        TensorLike: Transposed array in the same framework type as the input.

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


def shape(array: TensorLike) -> tuple[int, ...]:
    """
    Get the shape of an array.

    Args:
        array (TensorLike): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        tuple[int, ...]: Shape of the input array.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array, np.ndarray):
        return cast("tuple[int, ...]", array.shape)
    if torch and isinstance(array, torch.Tensor):
        return tuple(array.shape)
    if tf and isinstance(array, tf.Tensor):
        tf_shape = array.shape.as_list()
        return cast("tuple[int, ...]", tuple(tf_shape))
    if jnp and isinstance(array, jnp.ndarray):
        return cast("tuple[int, ...]", array.shape)
    if isinstance(array, (list, tuple)):
        np_array = to_numpy(array)
        return tuple(np_array.shape)

    raise TypeError(f"Unsupported framework type: {type(array)}")


def add(array1: T, array2: T) -> T:
    """
    Element-wise addition of two arrays.

    Args:
        array1 (TensorLike): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (TensorLike): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        TensorLike: Result of element-wise addition in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    np_types = (np.ndarray, float, int, complex)
    torch_types = (torch.Tensor, float, int, complex) if torch else ()
    tf_types = (tf.Tensor, float, int, complex) if tf else ()
    jnp_types = (jnp.ndarray, float, int, complex) if jnp else ()
    list_types = (list, tuple, float, int, complex)

    if isinstance(array1, np_types) and isinstance(array2, np_types):
        return cast("T", array1 + array2)
    if torch and isinstance(array1, torch_types) and isinstance(array2, torch_types):
        return cast("T", torch.add(array1, array2))
    if tf and isinstance(array1, tf_types) and isinstance(array2, tf_types):
        return cast("T", tf.add(array1, array2))
    if jnp and isinstance(array1, jnp_types) and isinstance(array2, jnp_types):
        return cast("T", jnp.add(array1, array2))
    if isinstance(array1, list_types) and isinstance(array2, list_types):
        np_array1 = to_numpy(array1)
        np_array2 = to_numpy(array2)
        added = (np_array1 + np_array2).tolist()
        return cast("T", type(array1)(added if isinstance(added, list) else [added]))

    raise TypeError(f"Unsupported framework type: {type(array1)} and {type(array2)}")


def sub(array1: T, array2: T) -> T:
    """
    Element-wise subtraction of two arrays.

    Args:
        array1 (TensorLike): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (TensorLike): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        TensorLike: Result of element-wise subtraction in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    np_types = (np.ndarray, float, int, complex)
    torch_types = (torch.Tensor, float, int, complex) if torch else ()
    tf_types = (tf.Tensor, float, int, complex) if tf else ()
    jnp_types = (jnp.ndarray, float, int, complex) if jnp else ()
    list_types = (list, tuple, float, int, complex)

    if isinstance(array1, np_types) and isinstance(array2, np_types):
        return cast("T", array1 - array2)
    if torch and isinstance(array1, torch_types) and isinstance(array2, torch_types):
        return cast("T", torch.sub(array1, array2))
    if tf and isinstance(array1, tf_types) and isinstance(array2, tf_types):
        return cast("T", tf.subtract(array1, array2))
    if jnp and isinstance(array1, jnp_types) and isinstance(array2, jnp_types):
        return cast("T", jnp.subtract(array1, array2))
    if isinstance(array1, list_types) and isinstance(array2, list_types):
        np_array1 = to_numpy(array1)
        np_array2 = to_numpy(array2)
        subbed = (np_array1 - np_array2).tolist()
        return cast("T", type(array1)(subbed if isinstance(subbed, list) else [subbed]))

    raise TypeError(f"Unsupported framework type: {type(array1)} and {type(array2)}")


def mul(array1: T | complex, array2: T | complex) -> T:
    """
    Element-wise multiplication of two arrays.

    Args:
        array1 (TensorLike): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (TensorLike): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        TensorLike: Result of element-wise multiplication in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    np_types = (np.ndarray, float, int, complex)
    torch_types = (torch.Tensor, float, int, complex) if torch else ()
    tf_types = (tf.Tensor, float, int, complex) if tf else ()
    jnp_types = (jnp.ndarray, float, int, complex) if jnp else ()
    list_types = (list, tuple)

    if isinstance(array1, np_types) and isinstance(array2, np_types):
        return cast("T", array1 * array2)
    if torch and isinstance(array1, torch_types) and isinstance(array2, torch_types):
        return cast("T", torch.mul(array1, array2))
    if tf and isinstance(array1, tf_types) and isinstance(array2, tf_types):
        return cast("T", tf.multiply(array1, array2))
    if jnp and isinstance(array1, jnp_types) and isinstance(array2, jnp_types):
        return cast("T", jnp.multiply(array1, array2))
    if isinstance(array1, list_types) and isinstance(array2, list_types):
        np_array1 = to_numpy(array1)
        np_array2 = to_numpy(array2)
        mulled = (np_array1 * np_array2).tolist()
        return cast("T", type(array1)(mulled if isinstance(mulled, list) else [mulled]))

    raise TypeError(f"Unsupported framework type: {type(array1)} and {type(array2)}")


def div(array1: T, array2: T) -> T:
    """
    Element-wise division of two arrays.

    Args:
        array1 (TensorLike): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (TensorLike): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        TensorLike: Result of element-wise division in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    np_types = (np.ndarray, float, int, complex)
    torch_types = (torch.Tensor, float, int, complex) if torch else ()
    tf_types = (tf.Tensor, float, int, complex) if tf else ()
    jnp_types = (jnp.ndarray, float, int, complex) if jnp else ()
    list_types = (list, tuple, float, int, complex)

    if isinstance(array1, np_types) and isinstance(array2, np_types):
        return cast("T", array1 / array2)
    if torch and isinstance(array1, torch_types) and isinstance(array2, torch_types):
        return cast("T", torch.div(array1, array2))
    if tf and isinstance(array1, tf_types) and isinstance(array2, tf_types):
        return cast("T", tf.divide(array1, array2))
    if jnp and isinstance(array1, jnp_types) and isinstance(array2, jnp_types):
        return cast("T", jnp.divide(array1, array2))
    if isinstance(array1, list_types) and isinstance(array2, list_types):
        np_array1 = to_numpy(array1)
        np_array2 = to_numpy(array2)
        dived = (np_array1 / np_array2).tolist()
        return cast("T", type(array1)(dived if isinstance(dived, list) else [dived]))

    raise TypeError(f"Unsupported framework type: {type(array1)} and {type(array2)}")


def matmul(array1: T, array2: T) -> T:
    """
    Matrix multiplication of two arrays.

    Args:
        array1 (TensorLike): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (TensorLike): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        TensorLike: Result of matrix multiplication in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    if isinstance(array1, np.ndarray) and isinstance(array2, np.ndarray):
        return cast("T", array1 @ array2)
    if torch and isinstance(array1, torch.Tensor) and isinstance(array2, torch.Tensor):
        return cast("T", array1 @ array2)
    if tf and isinstance(array1, tf.Tensor) and isinstance(array2, tf.Tensor):
        return cast("T", array1 @ array2)
    if jnp and isinstance(array1, jnp.ndarray) and isinstance(array2, jnp.ndarray):
        return cast("T", array1 @ array2)
    if isinstance(array1, (list, tuple)) and isinstance(array2, (list, tuple)):
        np_array1 = to_numpy(array1)
        np_array2 = to_numpy(array2)
        matmuled = (np_array1 @ np_array2).tolist()
        return cast("T", type(array1)(matmuled if isinstance(matmuled, list) else [matmuled]))

    raise TypeError(f"Unsupported framework type: {type(array1)} and {type(array2)}")
