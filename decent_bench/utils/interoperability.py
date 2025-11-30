"""Utilities for operating on arrays from different deep learning and linear algebra frameworks."""

from __future__ import annotations

import contextlib
import random
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from decent_bench.utils.parameter import X
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


_np_types = (np.ndarray, float, int)
_torch_types = (torch.Tensor, float, int) if torch else (float,)
_tf_types = (tf.Tensor, float, int) if tf else (float,)
_jnp_types = (jnp.ndarray, float, int) if jnp else (float,)


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
    if framework == "numpy":
        return device  # NumPy does not have explicit device management
    if torch and framework == "torch":
        torch_device = "cuda" if device == "gpu" else "cpu"
        return torch.device(torch_device)
    if tf and framework == "tensorflow":
        return f"/{device}:0"
    if jax and framework == "jax":
        if device == "cpu":
            return jax.devices("cpu")[0]
        return jax.devices("gpu")[0]
    raise ValueError(f"Unsupported framework: {framework}")


def to_numpy(array: X | SupportedXTypes) -> NDArray[Any]:
    """
    Convert input array to a NumPy array.

    Args:
        array (X): Input array (NumPy, torch, tf, jax) or nested container (list, tuple).

    Returns:
        NDArray: Converted NumPy array.

    """
    if isinstance(array, X):
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


def numpy_to_X(array: NDArray[Any], framework: SupportedFrameworks, device: SupportedDevices = "cpu") -> X:  # noqa: N802
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

    device = _device_literal_to_framework_device(device, framework)

    if framework == "numpy":
        return X(array)
    if torch and framework == "torch":
        return X(torch.from_numpy(array).to(device))
    if tf and framework == "tensorflow":
        with tf.device(device):
            return X(tf.convert_to_tensor(array))
    if jnp and framework == "jax":
        return X(jnp.array(array, device=device))

    raise TypeError(f"Unsupported framework type: {framework}")


def sum(  # noqa: A001
    array: X,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> X:
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
    if isinstance(array.value, np.ndarray):
        return X(
            np.sum(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.sum(array.value, dim=dim, keepdim=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return X(
            tf.reduce_sum(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.sum(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array)}")


def mean(
    array: X,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> X:
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
    if isinstance(array.value, np.ndarray):
        return X(
            np.mean(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.mean(array.value, dim=dim, keepdim=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return X(
            tf.reduce_mean(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.mean(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array)}")


def min(  # noqa: A001
    array: X,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> X:
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
    if isinstance(array.value, np.ndarray):
        return X(
            np.min(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.amin(array.value, dim=dim, keepdim=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return X(
            tf.reduce_min(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.min(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array)}")


def max(  # noqa: A001
    array: X,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> X:
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
    if isinstance(array.value, np.ndarray):
        return X(
            np.max(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.amax(array.value, dim=dim, keepdim=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return X(
            tf.reduce_max(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.max(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array)}")


def argmax(array: X, dim: int | None = None, keepdims: bool = False) -> X:
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
    if isinstance(array.value, np.ndarray):
        return X(
            np.argmax(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.argmax(array.value, dim=dim, keepdim=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        ret = None
        if dim is None:
            # TensorFlow's argmax does not support dim=None directly
            dims = array.value.ndim if array.value.ndim is not None else 0
            array = tf.reshape(array, [-1])
            amax = tf.math.argmax(array.value, axis=0)
            ret = (
                X(amax, framework=array.framework, device=array.device)
                if not keepdims
                else X(tf.reshape(amax, [1] * dims), framework=array.framework, device=array.device)
            )
        else:
            ret = (
                X(tf.math.argmax(array.value, axis=dim), framework=array.framework, device=array.device)
                if not keepdims
                else X(
                    tf.expand_dims(tf.math.argmax(array.value, axis=dim), axis=dim),
                    framework=array.framework,
                    device=array.device,
                )
            )
        return ret
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.argmax(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array)}")


def argmin(array: X, dim: int | None = None, keepdims: bool = False) -> X:
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
    if isinstance(array.value, np.ndarray):
        return X(
            np.argmin(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.argmin(array.value, dim=dim, keepdim=keepdims),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        ret = None
        if dim is None:
            # TensorFlow's argmin does not support dim=None directly
            dims = array.value.ndim if array.value.ndim is not None else 0
            tf_array = tf.reshape(array.value, [-1])
            amin = tf.math.argmin(tf_array, axis=0)
            ret = (
                X(amin, framework=array.framework, device=array.device)
                if not keepdims
                else X(tf.reshape(amin, [1] * dims), framework=array.framework, device=array.device)
            )
        else:
            ret = (
                X(tf.math.argmin(array.value, axis=dim), framework=array.framework, device=array.device)
                if not keepdims
                else X(
                    tf.expand_dims(tf.math.argmin(array.value, axis=dim), axis=dim),
                    framework=array.framework,
                    device=array.device,
                )
            )
        return ret
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.argmin(array.value, axis=dim, keepdims=keepdims),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array)}")


def copy(array: X) -> X:
    """
    Create a copy of the input array.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: A copy of the input array in the same framework type.

    """
    if isinstance(array.value, np.ndarray):
        return X(
            np.copy(array.value),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.clone(array.value),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return X(
            tf.identity(array.value),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.array(array.value, copy=True),
            framework=array.framework,
            device=array.device,
        )
    return deepcopy(array)


def stack(arrays: Sequence[X], dim: int = 0) -> X:
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
    arrs = [arr.value for arr in arrays]

    if isinstance(arrs[0], np.ndarray):
        return X(np.stack(arrs, axis=dim))
    if torch and isinstance(arrs[0], torch.Tensor):
        return X(torch.stack(arrs, dim=dim))
    if tf and isinstance(arrs[0], tf.Tensor):
        return X(tf.stack(arrs, axis=dim))
    if jnp and isinstance(arrs[0], jnp.ndarray):
        return X(jnp.stack(arrs, axis=dim))

    raise TypeError(f"Unsupported framework type or mixed types: {[type(arr) for arr in arrs]}")


def reshape(array: X, shape: tuple[int, ...]) -> X:
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
    if isinstance(array.value, np.ndarray):
        return X(
            np.reshape(array.value, shape),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.reshape(array.value, shape),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return X(
            tf.reshape(array.value, shape),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.reshape(array.value, shape),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array)}")


def zeros_like(array: X) -> X:
    """
    Create an array of zeros with the same shape and type as the input.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Array of zeros in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array.value, np.ndarray):
        return X(
            np.zeros_like(array.value),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.zeros_like(array.value),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return X(
            tf.zeros_like(array.value),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.zeros_like(array.value),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array)}")


def ones_like(array: X) -> X:
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
        return X(
            np.ones_like(array.value),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.ones_like(array.value),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return X(
            tf.ones_like(array.value),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.ones_like(array.value),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array)}")


def rand_like(array: X, low: float = 0.0, high: float = 1.0) -> X:
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
        return X(
            random_array,
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            (high - low) * torch.rand_like(array.value) + low,
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return X(
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
        return X(
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

    raise TypeError(f"Unsupported framework type: {type(array)}")


def randn_like(array: X, mean: float = 0.0, std: float = 1.0) -> X:
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
        return X(
            random_array,
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
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
        return X(
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
        return X(
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

    raise TypeError(f"Unsupported framework type: {type(array)}")


def eye_like(array: X) -> X:
    """
    Create an identity matrix with the same shape as the input.

    Args:
        array (X): Input array (NumPy, PyTorch, TensorFlow, JAX) or nested container (list, tuple).

    Returns:
        X: Identity matrix in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    if isinstance(array.value, np.ndarray):
        return X(
            np.eye(*array.value.shape[-2:], dtype=array.value.dtype),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.eye(*array.value.shape[-2:], dtype=array.value.dtype, device=array.device),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        shape = tf.shape(array.value)
        return X(
            tf.eye(*shape[-2:], dtype=array.value.dtype),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.eye(*array.value.shape[-2:], dtype=array.value.dtype, device=array.device),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array)}")


def transpose(array: X, dim: tuple[int, ...] | None = None) -> X:
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
    if isinstance(array.value, np.ndarray):
        return X(
            np.transpose(array.value, axes=dim),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        # Handle None case for PyTorch
        return (
            X(
                torch.permute(array.value, dims=dim),
                framework=array.framework,
                device=array.device,
            )
            if dim
            else X(
                torch.permute(array.value, dims=list(reversed(range(array.value.ndim)))),
                framework=array.framework,
                device=array.device,
            )
        )
    if tf and isinstance(array.value, tf.Tensor):
        return X(
            tf.transpose(array.value, perm=dim),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.transpose(array.value, axes=dim),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array)}")


def shape(array: X) -> tuple[int, ...]:
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

    raise TypeError(f"Unsupported framework type: {type(array)}")


def add(array1: X, array2: X) -> X:
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
    if isinstance(array1.value, _np_types) and isinstance(array2.value, _np_types):
        return X(
            array1.value + array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if torch and isinstance(array1.value, _torch_types) and isinstance(array2.value, _torch_types):
        return X(
            array1.value + array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if tf and isinstance(array1.value, _tf_types) and isinstance(array2.value, _tf_types):
        return X(
            array1.value + array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if jnp and isinstance(array1.value, _jnp_types) and isinstance(array2.value, _jnp_types):
        return X(
            array1.value + array2.value,
            framework=array1.framework,
            device=array1.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array1)} and {type(array2)}")


def iadd(array1: X, array2: X) -> X:
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
    if isinstance(array1.value, np.ndarray) and isinstance(array2.value, _np_types):
        array1.value += array2.value
        return array1
    if torch and isinstance(array1.value, torch.Tensor) and isinstance(array2.value, _torch_types):
        array1.value += array2.value
        return array1
    if tf and isinstance(array1.value, tf.Tensor) and isinstance(array2.value, _tf_types):
        array1.value += array2.value
        return array1
    if jnp and isinstance(array1.value, jnp.ndarray) and isinstance(array2.value, _jnp_types):
        array1.value += array2.value
        return array1

    raise TypeError(f"Unsupported framework type: {type(array1)} and {type(array2)}")


def sub(array1: X, array2: X) -> X:
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
    if isinstance(array1.value, _np_types) and isinstance(array2.value, _np_types):
        return X(
            array1.value - array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if torch and isinstance(array1.value, _torch_types) and isinstance(array2.value, _torch_types):
        return X(
            array1.value - array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if tf and isinstance(array1.value, _tf_types) and isinstance(array2.value, _tf_types):
        return X(
            array1.value - array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if jnp and isinstance(array1.value, _jnp_types) and isinstance(array2.value, _jnp_types):
        return X(
            array1.value - array2.value,
            framework=array1.framework,
            device=array1.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array1)} and {type(array2)}")


def mul(array1: X, array2: X) -> X:
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
    if isinstance(array1.value, _np_types) and isinstance(array2.value, _np_types):
        return X(
            array1.value * array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if torch and isinstance(array1.value, _torch_types) and isinstance(array2.value, _torch_types):
        return X(
            array1.value * array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if tf and isinstance(array1.value, _tf_types) and isinstance(array2.value, _tf_types):
        return X(
            array1.value * array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if jnp and isinstance(array1.value, _jnp_types) and isinstance(array2.value, _jnp_types):
        return X(
            array1.value * array2.value,
            framework=array1.framework,
            device=array1.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array1)} and {type(array2)}")


def div(array1: X, array2: X) -> X:
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
    if isinstance(array1.value, _np_types) and isinstance(array2.value, _np_types):
        return X(
            array1.value / array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if torch and isinstance(array1.value, _torch_types) and isinstance(array2.value, _torch_types):
        return X(
            array1.value / array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if tf and isinstance(array1.value, _tf_types) and isinstance(array2.value, _tf_types):
        return X(
            array1.value / array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if jnp and isinstance(array1.value, _jnp_types) and isinstance(array2.value, _jnp_types):
        return X(
            array1.value / array2.value,
            framework=array1.framework,
            device=array1.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array1)} and {type(array2)}")


def matmul(array1: X, array2: X) -> X:
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
    if isinstance(array1.value, np.ndarray) and isinstance(array2.value, np.ndarray):
        return X(
            array1.value @ array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if torch and isinstance(array1.value, torch.Tensor) and isinstance(array2.value, torch.Tensor):
        return X(
            array1.value @ array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if tf and isinstance(array1.value, tf.Tensor) and isinstance(array2.value, tf.Tensor):
        return X(
            array1.value @ array2.value,
            framework=array1.framework,
            device=array1.device,
        )
    if jnp and isinstance(array1.value, jnp.ndarray) and isinstance(array2.value, jnp.ndarray):
        return X(
            array1.value @ array2.value,
            framework=array1.framework,
            device=array1.device,
        )

    raise TypeError(f"Unsupported framework type: {type(array1)} and {type(array2)}")


def zeros(
    framework: SupportedFrameworks,
    shape: tuple[int, ...],
    dtype: Any = None,  # noqa: ANN401
    device: SupportedDevices = "cpu",
) -> X:
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


def power(array: X, p: float) -> X:
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
    if isinstance(array.value, np.ndarray):
        return X(
            np.power(array.value, p),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.pow(array.value, p),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return X(
            tf.pow(array.value, p),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.power(array.value, p),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported type: {type(array.value)}")


def negative(array: X) -> X:
    """
    Negate array.

    Args:
        array (X): The tensor.

    Returns:
        X: The negated tensor.

    Raises:
        TypeError: If the type is not supported.

    """
    if isinstance(array.value, np.ndarray):
        return X(
            np.negative(array.value),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.neg(array.value),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return X(
            tf.negative(array.value),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.negative(array.value),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported type: {type(array.value)}")


def absolute(array: X) -> X:
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
        return X(
            np.abs(array.value),
            framework=array.framework,
            device=array.device,
        )
    if torch and isinstance(array.value, torch.Tensor):
        return X(
            torch.abs(array.value),
            framework=array.framework,
            device=array.device,
        )
    if tf and isinstance(array.value, tf.Tensor):
        return X(
            tf.abs(array.value),
            framework=array.framework,
            device=array.device,
        )
    if jnp and isinstance(array.value, jnp.ndarray):
        return X(
            jnp.abs(array.value),
            framework=array.framework,
            device=array.device,
        )

    raise TypeError(f"Unsupported type: {type(array.value)}")
