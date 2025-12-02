"""Utilities for operating on arrays from different deep learning and linear algebra frameworks."""

from __future__ import annotations

import contextlib
import random
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, SupportsIndex, cast

import numpy as np
from numpy.typing import NDArray

from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks, SupportedXTypes

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

jax = None
with contextlib.suppress(ImportError, ModuleNotFoundError):
    import jax as _jax

    jax = _jax


_np_types = (np.ndarray, np.generic, float, int)
_torch_types = (torch.Tensor, float, int) if torch else (float,)
_tf_types = (tf.Tensor, float, int) if tf else (float,)
_jnp_types = (jnp.ndarray, jnp.generic, float, int) if jnp else (float,)


def _device_literal_to_framework_device(device: SupportedDevices, framework: SupportedFrameworks) -> Any:
    """
    Convert SupportedDevices literal to framework-specific device representation.

    Args:
        device (SupportedDevices): Device literal ("cpu" or "gpu").
        framework (SupportedFrameworks): Framework literal ("numpy", "torch", "tensorflow", "jax").

    Returns:
        Any: Framework-specific device representation.

    Raises:
        ValueError: If the framework is unsupported.

    """
    if framework == SupportedFrameworks.NUMPY:
        return device  # NumPy does not have explicit device management
    if torch and framework == SupportedFrameworks.TORCH:
        torch_device = "cuda" if device == SupportedDevices.GPU else "cpu"
        return torch.device(torch_device)
    if tf and framework == SupportedFrameworks.TENSORFLOW:
        return f"/{device}:0"
    if jax and framework == SupportedFrameworks.JAX:
        if device == SupportedDevices.CPU:
            return jax.devices("cpu")[0]
        return jax.devices("gpu")[0]
    raise ValueError(f"Unsupported framework: {framework}")


def to_numpy(array: Array | SupportedXTypes) -> NDArray[Any]:
    """
    Convert input array to a NumPy array.

    Args:
        array (X | SupportedXTypes): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).

    Returns:
        NDArray: Converted NumPy array.

    """
    if isinstance(array, Array):
        return to_numpy(array.value)

    if isinstance(array, np.ndarray):
        return array
    if torch and isinstance(array, torch.Tensor):
        return cast("np.ndarray", array.cpu().numpy())
    if tf and isinstance(array, tf.Tensor):
        return cast("np.ndarray", array.numpy())
    if (jnp and isinstance(array, jnp.ndarray)) or isinstance(array, (list, tuple)):
        return np.array(array)
    return np.array(array)


def numpy_to_X(  # noqa: N802
    array: NDArray[Any],
    framework: SupportedFrameworks,
    device: SupportedDevices = SupportedDevices.CPU,
) -> Array:
    """
    Convert a NumPy array to the specified framework type.

    Args:
        array (NDArray): Input NumPy array.
        framework (SupportedFrameworks): Target framework type (e.g., "torch", "tf").
        device (SupportedDevices): Target device ("cpu" or "gpu").

    Returns:
        X: Converted array in the specified framework type.

    Raises:
        TypeError: if the framework type of `framework` is unsupported.

    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input must be a NumPy array, got {type(array)}")

    if framework == SupportedFrameworks.NUMPY:
        return Array(array)

    framework_device = _device_literal_to_framework_device(device, framework)

    if torch and framework == SupportedFrameworks.TORCH:
        return Array(torch.from_numpy(array).to(framework_device))
    if tf and framework == SupportedFrameworks.TENSORFLOW:
        with tf.device(framework_device):
            return Array(tf.convert_to_tensor(array))
    if jnp and framework == SupportedFrameworks.JAX:
        return Array(jnp.array(array, device=framework_device))

    raise TypeError(f"Unsupported framework type: {framework}")


def sum(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Sum elements of an array.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to sum.
            If None, sums over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        X: Summed value in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(np.sum(value, axis=dim, keepdims=keepdims))
    if torch and isinstance(value, torch.Tensor):
        return Array(torch.sum(value, dim=dim, keepdim=keepdims))
    if tf and isinstance(value, tf.Tensor):
        return Array(tf.reduce_sum(value, axis=dim, keepdims=keepdims))
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.sum(value, axis=dim, keepdims=keepdims))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def mean(
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Compute mean of array elements.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute mean.
            If None, computes mean over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        X: Mean value in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(np.mean(value, axis=dim, keepdims=keepdims))
    if torch and isinstance(value, torch.Tensor):
        return Array(torch.mean(value, dim=dim, keepdim=keepdims))
    if tf and isinstance(value, tf.Tensor):
        return Array(tf.reduce_mean(value, axis=dim, keepdims=keepdims))
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.mean(value, axis=dim, keepdims=keepdims))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def min(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Compute minimum of array elements.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute minimum.
            If None, finds minimum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        X: Minimum value in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(np.min(value, axis=dim, keepdims=keepdims))
    if torch and isinstance(value, torch.Tensor):
        return Array(torch.amin(value, dim=dim, keepdim=keepdims))
    if tf and isinstance(value, tf.Tensor):
        return Array(tf.reduce_min(value, axis=dim, keepdims=keepdims))
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.min(value, axis=dim, keepdims=keepdims))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def max(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Compute maximum of array elements.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute maximum.
            If None, finds maximum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        X: Maximum value in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(np.max(value, axis=dim, keepdims=keepdims))
    if torch and isinstance(value, torch.Tensor):
        return Array(torch.amax(value, dim=dim, keepdim=keepdims))
    if tf and isinstance(value, tf.Tensor):
        return Array(tf.reduce_max(value, axis=dim, keepdims=keepdims))
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.max(value, axis=dim, keepdims=keepdims))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def argmax(array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
    """
    Compute index of maximum value.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | None): Dimension along which to find maximum. If None, finds maximum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        X: Indices of maximum values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(np.argmax(value, axis=dim, keepdims=keepdims))
    if torch and isinstance(value, torch.Tensor):
        return Array(torch.argmax(value, dim=dim, keepdim=keepdims))
    if tf and isinstance(value, tf.Tensor):
        if dim is None:
            # TensorFlow's argmax does not support dim=None directly
            dims = value.ndim if value.ndim is not None else 0
            reshaped_array = tf.reshape(value, [-1])
            amax = tf.math.argmax(reshaped_array, axis=0)
            ret = Array(amax) if not keepdims else Array(tf.reshape(amax, [1] * dims))
        else:
            ret = (
                Array(tf.math.argmax(value, axis=dim))
                if not keepdims
                else Array(tf.expand_dims(tf.math.argmax(value, axis=dim), axis=dim))
            )
        return ret
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.argmax(value, axis=dim, keepdims=keepdims))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def argmin(array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
    """
    Compute index of minimum value.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (int | None): Dimension along which to find minimum. If None, finds minimum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        X: Indices of minimum values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(np.argmin(value, axis=dim, keepdims=keepdims))
    if torch and isinstance(value, torch.Tensor):
        return Array(torch.argmin(value, dim=dim, keepdim=keepdims))
    if tf and isinstance(value, tf.Tensor):
        ret = None
        if dim is None:
            # TensorFlow's argmin does not support dim=None directly
            dims = value.ndim if value.ndim is not None else 0
            tf_array = tf.reshape(value, [-1])
            amin = tf.math.argmin(tf_array, axis=0)
            ret = Array(amin) if not keepdims else Array(tf.reshape(amin, [1] * dims))
        else:
            ret = (
                Array(tf.math.argmin(value, axis=dim))
                if not keepdims
                else Array(tf.expand_dims(tf.math.argmin(value, axis=dim), axis=dim))
            )
        return ret
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.argmin(value, axis=dim, keepdims=keepdims))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def copy(array: Array) -> Array:
    """
    Create a copy of the input array.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: A copy of the input array in the same framework type.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(np.copy(value))
    if torch and isinstance(value, torch.Tensor):
        return Array(torch.clone(value))
    if tf and isinstance(value, tf.Tensor):
        return Array(tf.identity(value))
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.array(value, copy=True))
    return deepcopy(array)


def stack(arrays: Sequence[Array], dim: int = 0) -> Array:
    """
    Stack a sequence of arrays along a new dimension.

    Args:
        arrays (Sequence[X]): Sequence of input arrays (NumPy, PyTorch, TensorFlow, JAX)
            or nested containers (list, tuple).
        dim (int): Dimension along which to stack the arrays.

    Returns:
        X: Stacked array in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported.

    """
    arrs = [arr.value for arr in arrays] if isinstance(arrays[0], Array) else arrays

    if isinstance(arrs[0], np.ndarray):
        return Array(np.stack(arrs, axis=dim))
    if torch and isinstance(arrs[0], torch.Tensor):
        return Array(torch.stack(arrs, dim=dim))
    if tf and isinstance(arrs[0], tf.Tensor):
        return Array(tf.stack(arrs, axis=dim))
    if jnp and isinstance(arrs[0], jnp.ndarray):
        return Array(jnp.stack(arrs, axis=dim))

    raise TypeError(f"Unsupported framework type or mixed types: {[type(arr) for arr in arrs]}")


def reshape(array: Array, shape: tuple[int, ...]) -> Array:
    """
    Reshape an array to the specified shape.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        shape (tuple[int, ...]): Desired shape for the output array.

    Returns:
        X: Reshaped array in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(np.reshape(value, shape))
    if torch and isinstance(value, torch.Tensor):
        return Array(torch.reshape(value, shape))
    if tf and isinstance(value, tf.Tensor):
        return Array(tf.reshape(value, shape))
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.reshape(value, shape))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def zeros_like(array: Array) -> Array:
    """
    Create an array of zeros with the same shape and type as the input.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Array of zeros in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(np.zeros_like(value))
    if torch and isinstance(value, torch.Tensor):
        return Array(torch.zeros_like(value))
    if tf and isinstance(value, tf.Tensor):
        return Array(tf.zeros_like(value))
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.zeros_like(value))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def ones_like(array: Array) -> Array:
    """
    Create an array of ones with the same shape and type as the input.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Array of ones in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array.value, np.ndarray):
        return Array(
            np.ones_like(array.value),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return Array(
            torch.ones_like(array.value),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return Array(
            tf.ones_like(array.value),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return Array(
            jnp.ones_like(array.value),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array.value)}")


def rand_like(array: Array, low: float = 0.0, high: float = 1.0) -> Array:
    """
    Create an array of random values with the same shape and type as the input.

    Values are drawn uniformly from [low, high).

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        low (float): Lower bound of the uniform distribution.
        high (float): Upper bound of the uniform distribution.

    Returns:
        X: Array of random values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array.value, np.ndarray):
        random_array = np.random.default_rng().uniform(
            low=low,
            high=high,
            size=array.shape,
        )
        random_array = random_array.astype(array.value.dtype) if isinstance(random_array, np.ndarray) else random_array
        return Array(
            random_array,
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return Array(
            (high - low) * torch.rand_like(array.value) + low,
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return Array(
            tf.random.uniform(
                tf.shape(array.value),
                dtype=array.value.dtype,
                minval=low,
                maxval=high,
            ),
            framework=array.framework,
            device=array.device,
        )
    if jnp and jax and isinstance(array.value, jnp.ndarray):
        return Array(
            jax.random.uniform(
                jax.random.key(random.randint(0, 2**32 - 1)),
                shape=array.value.shape,
                dtype=array.value.dtype,
                minval=low,
                maxval=high,
            ),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array.value)}")


def randn_like(array: Array, mean: float = 0.0, std: float = 1.0) -> Array:
    """
    Create an array of random values with the same shape and type as the input.

    Values are drawn from a normal distribution with mean `mean` and standard deviation `std`.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.

    Returns:
        X: Array of random values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array.value, np.ndarray):
        random_array = np.random.default_rng().normal(
            loc=mean,
            scale=std,
            size=array.value.shape,
        )
        random_array = random_array.astype(array.value.dtype) if isinstance(random_array, np.ndarray) else random_array
        return Array(
            random_array,
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return Array(
            torch.normal(
                mean=mean,
                std=std,
                size=array.value.shape,
                dtype=array.value.dtype,
                device=array.value.device,
            ),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        shape = tf.shape(array.value)
        return Array(
            tf.random.normal(
                shape=shape,
                mean=mean,
                stddev=std,
                dtype=array.value.dtype,
            ),
            framework=array.framework,
            device=array.device,
        )
    if jnp and jax and isinstance(array.value, jnp.ndarray):
        return Array(
            mean
            + std
            * jax.random.normal(
                jax.random.key(random.randint(0, 2**32 - 1)),
                shape=array.value.shape,
                dtype=array.value.dtype,
            ),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array.value)}")


def eye_like(array: Array) -> Array:
    """
    Create an identity matrix with the same shape as the input.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Identity matrix in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(np.eye(*value.shape[-2:], dtype=value.dtype))
    if torch and isinstance(value, torch.Tensor):
        return Array(torch.eye(*value.shape[-2:], dtype=value.dtype, device=value.device))
    if tf and isinstance(value, tf.Tensor):
        shape = tf.shape(value)
        return Array(tf.eye(*shape[-2:], dtype=value.dtype))
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.eye(*value.shape[-2:], dtype=value.dtype, device=value.device))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def eye(n: int, framework: SupportedFrameworks, device: SupportedDevices = SupportedDevices.CPU) -> Array:
    """
    Create an identity matrix of size n x n in the specified framework.

    Args:
        n (int): Size of the identity matrix.
        framework (SupportedFrameworks): Target framework type (e.g., "torch", "tf").
        device (SupportedDevices): Target device ("cpu" or "gpu").

    Returns:
        X: Identity matrix in the specified framework type.

    Raises:
        TypeError: if the framework type of `framework` is unsupported.

    """
    if framework == SupportedFrameworks.NUMPY:
        return Array(np.eye(n))

    framework_device = _device_literal_to_framework_device(device, framework)

    if torch and framework == SupportedFrameworks.TORCH:
        return Array(torch.eye(n, device=framework_device))
    if tf and framework == SupportedFrameworks.TENSORFLOW:
        with tf.device(framework_device):
            return Array(tf.eye(n))
    if jnp and framework == SupportedFrameworks.JAX:
        return Array(jnp.eye(n, device=framework_device))

    raise TypeError(f"Unsupported framework type: {framework}")


def transpose(array: Array, dim: tuple[int, ...] | None = None) -> Array:
    """
    Transpose an array.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        dim (tuple[int, ...] | None): Desired dim order. If None, reverses the dimensions.

    Returns:
        X: Transposed array in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(np.transpose(value, axes=dim))
    if torch and isinstance(value, torch.Tensor):
        # Handle None case for PyTorch
        return (
            Array(torch.permute(value, dims=dim))
            if dim
            else Array(torch.permute(value, dims=list(reversed(range(value.ndim)))))
        )
    if tf and isinstance(value, tf.Tensor):
        return Array(tf.transpose(value, perm=dim))
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.transpose(value, axes=dim))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def shape(array: Array) -> tuple[int, ...]:
    """
    Get the shape of an array.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        tuple[int, ...]: Shape of the input array.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array.value, np.ndarray):
        return array.value.shape
    if torch and isinstance(array.value, torch.Tensor):
        return tuple(array.value.shape)
    if tf and isinstance(array.value, tf.Tensor):
        tf_shape = tuple(array.value.shape)
        return cast("tuple[int, ...]", tf_shape)
    if jnp and isinstance(array.value, jnp.ndarray):
        return cast("tuple[int, ...]", array.value.shape)

    raise TypeError(f"Unsupported framework type: {type(array.value)}")


def add(array1: Array, array2: Array) -> Array:
    """
    Element-wise addition of two arrays.

    Args:
        array1 (X): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (X): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Result of element-wise addition in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, _np_types) and isinstance(value2, _np_types):
        return Array(value1 + value2)
    if torch and isinstance(value1, _torch_types) and isinstance(value2, _torch_types):
        return Array(torch.add(value1, value2))
    if tf and isinstance(value1, _tf_types) and isinstance(value2, _tf_types):
        return Array(tf.add(value1, value2))
    if jnp and isinstance(value1, _jnp_types) and isinstance(value2, _jnp_types):
        return Array(jnp.add(value1, value2))

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def iadd[T: Array](array1: T, array2: Array) -> T:
    """
    Element-wise in-place addition of two arrays.

    Args:
        array1 (X): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (X): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Result of element-wise in-place addition in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, np.ndarray) and isinstance(value2, _np_types):
        value1 += value2
        return array1
    if torch and isinstance(value1, torch.Tensor) and isinstance(value2, _torch_types):
        value1 += value2
        return array1
    if tf and isinstance(value1, tf.Tensor) and isinstance(value2, _tf_types):
        value1 += value2
        return array1
    if jnp and isinstance(value1, jnp.ndarray) and isinstance(value2, _jnp_types):
        value1 += value2
        return array1

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def sub(array1: Array, array2: Array) -> Array:
    """
    Element-wise subtraction of two arrays.

    Args:
        array1 (X): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (X): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Result of element-wise subtraction in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, _np_types) and isinstance(value2, _np_types):
        return Array(value1 - value2)
    if torch and isinstance(value1, _torch_types) and isinstance(value2, _torch_types):
        return Array(torch.sub(value1, value2))
    if tf and isinstance(value1, _tf_types) and isinstance(value2, _tf_types):
        return Array(tf.subtract(value1, value2))
    if jnp and isinstance(value1, _jnp_types) and isinstance(value2, _jnp_types):
        return Array(jnp.subtract(value1, value2))

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def isub[T: Array](array1: T, array2: Array) -> T:
    """
    Element-wise in-place subtraction of two arrays.

    Args:
        array1 (X): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (X): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Result of element-wise in-place subtraction in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, np.ndarray) and isinstance(value2, _np_types):
        value1 -= value2
        return array1
    if torch and isinstance(value1, torch.Tensor) and isinstance(value2, _torch_types):
        value1 -= value2
        return array1
    if tf and isinstance(value1, tf.Tensor) and isinstance(value2, _tf_types):
        value1 -= value2
        return array1
    if jnp and isinstance(value1, jnp.ndarray) and isinstance(value2, _jnp_types):
        value1 -= value2
        return array1

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def mul(array1: Array, array2: Array) -> Array:
    """
    Element-wise multiplication of two arrays.

    Args:
        array1 (X): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (X): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Result of element-wise multiplication in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, _np_types) and isinstance(value2, _np_types):
        return Array(value1 * value2)
    if torch and isinstance(value1, _torch_types) and isinstance(value2, _torch_types):
        return Array(torch.mul(value1, value2))
    if tf and isinstance(value1, _tf_types) and isinstance(value2, _tf_types):
        return Array(tf.multiply(value1, value2))
    if jnp and isinstance(value1, _jnp_types) and isinstance(value2, _jnp_types):
        return Array(jnp.multiply(value1, value2))

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def imul[T: Array](array1: T, array2: Array) -> T:
    """
    Element-wise in-place multiplication of two arrays.

    Args:
        array1 (X): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (X): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Result of element-wise in-place multiplication in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, np.ndarray) and isinstance(value2, _np_types):
        value1 *= value2
        return array1
    if torch and isinstance(value1, torch.Tensor) and isinstance(value2, _torch_types):
        value1 *= value2
        return array1
    if tf and isinstance(value1, tf.Tensor) and isinstance(value2, _tf_types):
        value1 *= value2
        return array1
    if jnp and isinstance(value1, jnp.ndarray) and isinstance(value2, _jnp_types):
        value1 *= value2
        return array1

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def div(array1: Array, array2: Array) -> Array:
    """
    Element-wise division of two arrays.

    Args:
        array1 (X): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (X): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Result of element-wise division in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, _np_types) and isinstance(value2, _np_types):
        return Array(value1 / value2)
    if torch and isinstance(value1, _torch_types) and isinstance(value2, _torch_types):
        return Array(torch.div(value1, value2))
    if tf and isinstance(value1, _tf_types) and isinstance(value2, _tf_types):
        return Array(tf.divide(value1, value2))
    if jnp and isinstance(value1, _jnp_types) and isinstance(value2, _jnp_types):
        return Array(jnp.divide(value1, value2))

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def idiv[T: Array](array1: T, array2: Array) -> T:
    """
    Element-wise in-place division of two arrays.

    Args:
        array1 (X): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (X): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Result of element-wise in-place division in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, np.ndarray) and isinstance(value2, _np_types):
        value1 /= value2
        return array1
    if torch and isinstance(value1, torch.Tensor) and isinstance(value2, _torch_types):
        value1 /= value2
        return array1
    if tf and isinstance(value1, tf.Tensor) and isinstance(value2, _tf_types):
        value1 /= value2
        return array1
    if jnp and isinstance(value1, jnp.ndarray) and isinstance(value2, _jnp_types):
        value1 /= value2
        return array1

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def matmul(array1: Array, array2: Array) -> Array:
    """
    Matrix multiplication of two arrays.

    Args:
        array1 (X): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (X): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Result of matrix multiplication in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
        return Array(value1 @ value2)
    if torch and isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
        return Array(value1 @ value2)
    if tf and isinstance(value1, tf.Tensor) and isinstance(value2, tf.Tensor):
        return Array(value1 @ value2)
    if jnp and isinstance(value1, jnp.ndarray) and isinstance(value2, jnp.ndarray):
        return Array(value1 @ value2)

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def dot(array1: Array, array2: Array) -> Array:
    """
    Dot product of two arrays.

    Args:
        array1 (X): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        array2 (X): Second input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Result of matrix multiplication in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported
            or if the input arrays are not of the same framework type.

    """
    value1 = array1.value if isinstance(array1, Array) else array1
    value2 = array2.value if isinstance(array2, Array) else array2

    if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
        return Array(value1.dot(value2))
    if torch and isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
        return Array(value1.dot(value2))
    if tf and isinstance(value1, tf.Tensor) and isinstance(value2, tf.Tensor):
        return Array(value1.dot(value2))
    if jnp and isinstance(value1, jnp.ndarray) and isinstance(value2, jnp.ndarray):
        return Array(value1.dot(value2))

    raise TypeError(f"Unsupported framework type: {type(value1)} and {type(value2)}")


def zeros(
    framework: SupportedFrameworks,
    shape: tuple[int, ...],
    dtype: Any = None,  # noqa: ANN401
    device: SupportedDevices = SupportedDevices.CPU,
) -> Array:
    """
    Create a tensor of zeros.

    Args:
        framework (SupportedFrameworks): The framework to use ("numpy", "torch", "tensorflow", "jax").
        shape (tuple): The shape of the tensor.
        dtype (Any, optional): The data type of the tensor. Defaults to None.
        device (SupportedDevices, optional): The device to place the tensor on. Defaults to "cpu".

    Returns:
        X: A tensor of zeros.

    """
    x = np.zeros(shape, dtype=dtype)

    return numpy_to_X(x, framework=framework, device=device)


def power(array: Array, p: float) -> Array:
    """
    Raise array to p power.

    Args:
        array (X): The tensor.
        p (float): The power.

    Returns:
        X: The result of the operation.

    Raises:
        TypeError: If the type is not supported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(np.power(value, p))
    if torch and isinstance(value, torch.Tensor):
        return Array(torch.pow(value, p))
    if tf and isinstance(value, tf.Tensor):
        return Array(tf.pow(value, p))
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.power(value, p))

    raise TypeError(f"Unsupported type: {type(value)}")


def ipow[T: Array](array1: T, p: float) -> T:
    """
    Element-wise in-place power of an array.

    Args:
        array1 (X): First input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).
        p (float): The power.

    Returns:
        X: Result of element-wise in-place power in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported

    """
    value = array1.value if isinstance(array1, Array) else array1

    if isinstance(value, np.ndarray):
        value **= p
        return array1
    if torch and isinstance(value, torch.Tensor):
        value **= p
        return array1
    if tf and isinstance(value, tf.Tensor):
        value **= p
        return array1
    if jnp and isinstance(value, jnp.ndarray):
        value **= p
        return array1

    raise TypeError(f"Unsupported framework type: {type(value)}")


def set_item(array: Array, key: SupportsIndex | tuple[SupportsIndex, ...], value: Array) -> None:
    """
    Set the item at the specified index of the array to the given value.

    Args:
        array (X): The tensor.
        key (Any): The key or index to set.
        value (X): The value to set.

    Raises:
        TypeError: If the type is not supported.

    """
    array_value = array.value if isinstance(array, Array) else array
    value_value = value.value if isinstance(value, Array) else value

    if isinstance(array_value, np.ndarray) and isinstance(value_value, _np_types):
        array_value[key] = value_value
        return
    if torch and isinstance(array_value, torch.Tensor) and isinstance(value_value, _torch_types):
        array_value[key] = value_value
        return
    if tf and isinstance(array_value, tf.Tensor) and isinstance(value_value, _tf_types):
        array_value = tf.tensor_scatter_nd_update(
            array_value,
            tf.expand_dims(tf.constant(key), axis=0),
            tf.expand_dims(value_value, axis=0),
        )
        return
    if jnp and isinstance(array_value, jnp.ndarray) and isinstance(value_value, _jnp_types):
        array_value = array_value.at[key].set(value_value)
        return

    raise TypeError(f"Unsupported type: {type(array_value)} with value: {type(value_value)}")


def negative(array: Array) -> Array:
    """
    Negate array.

    Args:
        array (X): The tensor.

    Returns:
        X: The negated tensor.

    Raises:
        TypeError: If the type is not supported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(np.negative(value))
    if torch and isinstance(value, torch.Tensor):
        return Array(torch.neg(value))
    if tf and isinstance(value, tf.Tensor):
        return Array(tf.negative(value))
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.negative(value))

    raise TypeError(f"Unsupported type: {type(array.value)}")


def absolute(array: Array) -> Array:
    """
    Return the absolute value of a tensor.

    Args:
        array (X): The tensor.

    Returns:
        X: The absolute value tensor.

    Raises:
        TypeError: If the type is not supported.

    """
    if isinstance(array.value, np.ndarray):
        return Array(
            np.abs(array.value),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return Array(
            torch.abs(array.value),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return Array(
            tf.abs(array.value),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return Array(
            jnp.abs(array.value),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported type: {type(array.value)}")


def astype(array: Array, dtype: type[float | int | bool]) -> float | int | bool:
    """
    Cast array to a specified data type.

    Args:
        array (X): The tensor.
        dtype (float | int | bool): The target data type.

    Returns:
        float | int | bool: The casted tensor.

    Raises:
        TypeError: If the type is not supported.

    """
    if isinstance(array.value, _np_types):
        return dtype(array.value)
    if torch and isinstance(array.value, torch.Tensor):
        return dtype(array.value)
    if tf and isinstance(array.value, tf.Tensor):
        return dtype(array.value)
    if jnp and isinstance(array.value, _jnp_types):
        return dtype(array.value.item())

    raise TypeError(f"Unsupported type: {type(array.value)}")


def norm(array: Array, p: float = 2) -> Array:
    """
    Compute the norm of an array.

    Args:
        array (X): The tensor.
        p (float): The order of the norm.

    Returns:
        X: The norm of the tensor.

    Raises:
        TypeError: If the type is not supported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray):
        return Array(cast("SupportedXTypes", np.linalg.norm(value, ord=p)))
    if torch and isinstance(value, torch.Tensor):
        return Array(torch.norm(value, p=p))
    if tf and isinstance(value, tf.Tensor):
        return Array(tf.norm(value, ord=p))
    if jnp and isinstance(value, jnp.ndarray):
        return Array(jnp.linalg.norm(value, ord=p))

    raise TypeError(f"Unsupported type: {type(value)}")
