from __future__ import annotations

import contextlib
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from decent_bench.utils.array import Array
from decent_bench.utils.types import ArrayKey, SupportedArrayTypes, SupportedDevices, SupportedFrameworks

from ._helpers import _return_array, device_to_framework_device, framework_device_of_array
from ._imports_types import (
    _jax_key,
    _jnp_types,
    _np_types,
    _numpy_generator,
    _tf_types,
    _torch_types,
)

jax = None
jnp = None
tf = None
torch = None

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch as _torch

    torch = _torch

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import tensorflow as _tf

    tf = _tf

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import jax.numpy as _jnp

    jnp = _jnp

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import jax as _jax

    jax = _jax

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from tensorflow import Tensor as TensorFlowTensor
    from torch import Tensor as TorchTensor


def to_numpy(array: Array | SupportedArrayTypes, device: SupportedDevices = SupportedDevices.CPU) -> NDArray[Any]:  # noqa: ARG001
    """
    Convert input array to a NumPy array.

    Args:
        array (Array | SupportedArrayTypes): Input Array
        device (SupportedDevices): Device of the input array.

    Returns:
        NDArray: Converted NumPy array.

    Note:
        The `device` parameter is currently not used in this function but is included for API consistency.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return value
    if torch and isinstance(value, torch.Tensor):
        return cast("np.ndarray", value.cpu().numpy())  # pyright: ignore[reportAttributeAccessIssue]
    if tf and isinstance(value, tf.Tensor):
        return cast("np.ndarray", value.numpy())  # pyright: ignore[reportAttributeAccessIssue]
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return np.array(value)
    return np.array(value)


def to_torch(array: Array | SupportedArrayTypes, device: SupportedDevices) -> TorchTensor:
    """
    Convert input array to a PyTorch tensor.

    Args:
        array (Array | SupportedArrayTypes): Input Array
        device (SupportedDevices): Device of the input array.

    Returns:
        torch.Tensor: Converted PyTorch tensor.

    Raises:
        ImportError: if PyTorch is not installed.

    """
    if not torch:
        raise ImportError("PyTorch is not installed.")

    value = array.value if isinstance(array, Array) else array
    framework_device = device_to_framework_device(device, SupportedFrameworks.PYTORCH)

    if isinstance(value, torch.Tensor):
        return cast("TorchTensor", value)
    if isinstance(value, np.ndarray | np.generic):
        return cast("TorchTensor", torch.tensor(value).to(framework_device))
    if tf and isinstance(value, tf.Tensor):
        return cast("TorchTensor", torch.tensor(value.cpu(), device=framework_device))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return cast("TorchTensor", torch.tensor(np.array(value)).to(framework_device))
    return cast("TorchTensor", torch.tensor(value, device=framework_device))


def to_tensorflow(array: Array | SupportedArrayTypes, device: SupportedDevices) -> TensorFlowTensor:
    """
    Convert input array to a TensorFlow tensor.

    Args:
        array (Array | SupportedArrayTypes): Input Array
        device (SupportedDevices): Device of the input array.

    Returns:
        tf.Tensor: Converted TensorFlow tensor.

    Raises:
        ImportError: if TensorFlow is not installed.

    """
    if not tf:
        raise ImportError("TensorFlow is not installed.")

    value = array.value if isinstance(array, Array) else array
    framework_device = device_to_framework_device(device, SupportedFrameworks.TENSORFLOW)

    if isinstance(value, tf.Tensor):
        with tf.device(framework_device):
            return cast("TensorFlowTensor", value)
    if isinstance(value, np.ndarray | np.generic):
        with tf.device(framework_device):
            return cast("TensorFlowTensor", tf.convert_to_tensor(value))
    if torch and isinstance(value, torch.Tensor):
        with tf.device(framework_device):
            return cast("TensorFlowTensor", tf.convert_to_tensor(value.cpu()))  # pyright: ignore[reportArgumentType]
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        with tf.device(framework_device):
            return cast("TensorFlowTensor", tf.convert_to_tensor(value))  # pyright: ignore[reportArgumentType]
    with tf.device(framework_device):
        return cast("TensorFlowTensor", tf.convert_to_tensor(value))  # pyright: ignore[reportArgumentType]


def to_jax(array: Array | SupportedArrayTypes, device: SupportedDevices) -> JaxArray:
    """
    Convert input array to a JAX array.

    Args:
        array (Array | SupportedArrayTypes): Input Array
        device (SupportedDevices): Device of the input array.

    Returns:
        jax.Array: Converted JAX array.

    Raises:
        ImportError: if JAX is not installed.

    """
    if not jnp:
        raise ImportError("JAX is not installed.")

    value = array.value if isinstance(array, Array) else array
    framework_device = device_to_framework_device(device, SupportedFrameworks.JAX)

    if isinstance(value, jnp.ndarray | jnp.generic):
        return cast("JaxArray", value.to_device(framework_device))
    if isinstance(value, np.ndarray | np.generic):
        return cast("JaxArray", jnp.array(value, device=framework_device))
    if torch and isinstance(value, torch.Tensor):
        return cast("JaxArray", jnp.array(value, device=framework_device))
    if tf and isinstance(value, tf.Tensor):
        return cast("JaxArray", jnp.array(value, device=framework_device))
    return cast("JaxArray", jnp.array(value, device=framework_device))


def to_array(
    array: Array | SupportedArrayTypes,
    framework: SupportedFrameworks,
    device: SupportedDevices,
) -> Array:
    """
    Convert an array to the specified framework type.

    See :func:`decent_bench.utils.interoperability.to_array_like` if you want to convert an array to match
    the framework and device of another array.

    Args:
        array (Array | SupportedArrayTypes): Input array.
        framework (SupportedFrameworks): Target framework type (e.g., "torch", "tf").
        device (SupportedDevices): Target device ("cpu" or "gpu").

    Returns:
        Array: Converted array in the specified framework type.

    Raises:
        TypeError: if the framework type of `framework` is unsupported.

    """
    if framework == SupportedFrameworks.NUMPY:
        return _return_array(to_numpy(array, device))
    if torch and framework == SupportedFrameworks.PYTORCH:
        return _return_array(to_torch(array, device))
    if tf and framework == SupportedFrameworks.TENSORFLOW:
        return _return_array(to_tensorflow(array, device))
    if jnp and framework == SupportedFrameworks.JAX:
        return _return_array(to_jax(array, device))

    raise TypeError(f"Unsupported framework type: {framework}")


def to_array_like(array: Array | SupportedArrayTypes, like: Array) -> Array:
    """
    Convert an array to the framework/device of `like`.

    Args:
        array (Array | SupportedArrayTypes): Input array.
        like (Array): Array whose framework and device to match.

    Returns:
        Array: Converted array in the specified framework type.

    """
    framework, device = framework_device_of_array(like)

    return to_array(array, framework, device)


def sum(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Sum elements of an array.

    Args:
        array (Array): Input array.
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to sum.
            If None, sums over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        Array: Summed value in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.sum(value, axis=dim, keepdims=keepdims))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.sum(value, dim=dim, keepdim=keepdims))
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.reduce_sum(value, axis=dim, keepdims=keepdims))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.sum(value, axis=dim, keepdims=keepdims))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def mean(
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Compute mean of array elements.

    Args:
        array (Array): Input array.
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute the mean.
            If None, computes mean of flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        Array: Mean value in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.mean(value, axis=dim, keepdims=keepdims))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.mean(value, dim=dim, keepdim=keepdims))
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.reduce_mean(value, axis=dim, keepdims=keepdims))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.mean(value, axis=dim, keepdims=keepdims))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def min(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Compute minimum of array elements.

    Args:
        array (Array): Input array.
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute minimum.
            If None, finds minimum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        Array: Minimum value in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.min(value, axis=dim, keepdims=keepdims))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.amin(value, dim=dim, keepdim=keepdims))  # pyright: ignore[reportArgumentType]
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.reduce_min(value, axis=dim, keepdims=keepdims))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.min(value, axis=dim, keepdims=keepdims))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def max(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Compute maximum of array elements.

    Args:
        array (Array): Input array.
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute maximum.
            If None, finds maximum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        Array: Maximum value in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.max(value, axis=dim, keepdims=keepdims))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.amax(value, dim=dim, keepdim=keepdims))  # pyright: ignore[reportArgumentType]
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.reduce_max(value, axis=dim, keepdims=keepdims))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.max(value, axis=dim, keepdims=keepdims))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def argmax(array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
    """
    Compute index of maximum value.

    Args:
        array (Array): Input array.
        dim (int | None): Dimension along which to find maximum. If None, finds maximum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        Array: Indices of maximum values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.argmax(value, axis=dim, keepdims=keepdims))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.argmax(value, dim=dim, keepdim=keepdims))
    if tf and isinstance(value, tf.Tensor):
        if dim is None:
            # TensorFlow's argmax does not support dim=None directly
            dims = value.ndim if value.ndim is not None else 0
            reshaped_array = tf.reshape(value, [-1])
            amax = tf.math.argmax(reshaped_array, axis=0)
            ret = _return_array(amax) if not keepdims else _return_array(tf.reshape(amax, [1] * dims))
        else:
            ret = (
                _return_array(tf.math.argmax(value, axis=dim))
                if not keepdims
                else _return_array(tf.expand_dims(tf.math.argmax(value, axis=dim), axis=dim))
            )
        return ret
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.argmax(value, axis=dim, keepdims=keepdims))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def argmin(array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
    """
    Compute index of minimum value.

    Args:
        array (Array): Input array.
        dim (int | None): Dimension along which to find minimum. If None, finds minimum over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        Array: Indices of minimum values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.argmin(value, axis=dim, keepdims=keepdims))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.argmin(value, dim=dim, keepdim=keepdims))
    if tf and isinstance(value, tf.Tensor):
        ret = None
        if dim is None:
            # TensorFlow's argmin does not support dim=None directly
            dims = value.ndim if value.ndim is not None else 0
            tf_array = tf.reshape(value, [-1])
            amin = tf.math.argmin(tf_array, axis=0)
            ret = _return_array(amin) if not keepdims else _return_array(tf.reshape(amin, [1] * dims))
        else:
            ret = (
                _return_array(tf.math.argmin(value, axis=dim))
                if not keepdims
                else _return_array(tf.expand_dims(tf.math.argmin(value, axis=dim), axis=dim))
            )
        return ret
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.argmin(value, axis=dim, keepdims=keepdims))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def copy(array: Array) -> Array:
    """
    Create a copy of the input array.

    Args:
        array (Array): Input array.

    Returns:
        Array: A copy of the input array in the same framework type.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.copy(value))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.clone(value))
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.identity(value))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.array(value, copy=True))
    return deepcopy(array)


def stack(arrays: Sequence[Array], dim: int = 0) -> Array:
    """
    Stack a sequence of arrays along a new dimension.

    Args:
        arrays (Sequence[Array]): Sequence of input arrays.
            or nested containers (list, tuple).
        dim (int): Dimension along which to stack the arrays.

    Returns:
        Array: Stacked array in the same framework type as the inputs.

    Raises:
        TypeError: if the framework type of the input arrays is unsupported.

    """
    arrs = [arr.value for arr in arrays] if isinstance(arrays[0], Array) else arrays

    if isinstance(arrs[0], np.ndarray | np.generic):
        return _return_array(np.stack(arrs, axis=dim))  # pyright: ignore[reportArgumentType, reportCallIssue]
    if torch and isinstance(arrs[0], torch.Tensor):
        return _return_array(torch.stack(arrs, dim=dim))  # pyright: ignore[reportArgumentType]
    if tf and isinstance(arrs[0], tf.Tensor):
        return _return_array(tf.stack(arrs, axis=dim))
    if jnp and isinstance(arrs[0], jnp.ndarray | jnp.generic):
        return _return_array(jnp.stack(arrs, axis=dim))  # pyright: ignore[reportArgumentType]

    raise TypeError(f"Unsupported framework type or mixed types: {[type(arr) for arr in arrs]}")


def reshape(array: Array, shape: tuple[int, ...]) -> Array:
    """
    Reshape an array to the specified shape.

    Args:
        array (Array): Input array.
        shape (tuple[int, ...]): Desired shape for the output array.

    Returns:
        Array: Reshaped array in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.reshape(value, shape))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.reshape(value, shape))
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.reshape(value, shape))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.reshape(value, shape))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def zeros_like(array: Array) -> Array:
    """
    Create an array of zeros with the same shape and type as the input.

    Args:
        array (Array): Input array.

    Returns:
        Array: Array of zeros in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.zeros_like(value))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.zeros_like(value))
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.zeros_like(value))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.zeros_like(value))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def ones_like(array: Array) -> Array:
    """
    Create an array of ones with the same shape and type as the input.

    Args:
        array (Array): Input array.

    Returns:
        Array: Array of ones in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.ones_like(value))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.ones_like(value))
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.ones_like(value))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.ones_like(value))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def rand_like(array: Array, low: float = 0.0, high: float = 1.0) -> Array:
    """
    Create an array of random values with the same shape and type as the input.

    Values are drawn uniformly from [low, high).

    Args:
        array (Array): Input array.
        low (float): Lower bound of the uniform distribution.
        high (float): Upper bound of the uniform distribution.

    Returns:
        Array: Array of random values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        random_array = _numpy_generator().uniform(low=low, high=high, size=value.shape)
        return _return_array(random_array)
    if torch and isinstance(value, torch.Tensor):
        return _return_array((high - low) * torch.rand_like(value) + low)
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.random.uniform(tf.shape(value), dtype=value.dtype, minval=low, maxval=high))
    if jnp and jax and isinstance(value, jnp.ndarray | jnp.generic):
        global _jax_key
        _jax_key, sub_key = jax.random.split(_jax_key)  # pyright: ignore[reportArgumentType]
        return _return_array(jax.random.uniform(sub_key, shape=value.shape, dtype=value.dtype, minval=low, maxval=high))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def randn_like(array: Array, mean: float = 0.0, std: float = 1.0) -> Array:
    """
    Create an array of random values with the same shape and type as the input.

    Values are drawn from a normal distribution with mean `mean` and standard deviation `std`.

    Args:
        array (Array): Input array.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.

    Returns:
        Array: Array of random values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        random_array = _numpy_generator().normal(loc=mean, scale=std, size=value.shape)
        return _return_array(random_array)
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.normal(mean=mean, std=std, size=value.shape, dtype=value.dtype, device=value.device))
    if tf and isinstance(value, tf.Tensor):
        shape = tf.shape(value)
        return _return_array(tf.random.normal(shape=shape, mean=mean, stddev=std, dtype=value.dtype))
    if jnp and jax and isinstance(value, jnp.ndarray | jnp.generic):
        global _jax_key
        _jax_key, sub_key = jax.random.split(_jax_key)  # pyright: ignore[reportArgumentType]
        return _return_array(mean + std * jax.random.normal(sub_key, shape=value.shape, dtype=value.dtype))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def eye_like(array: Array) -> Array:
    """
    Create an identity matrix with the same shape as the input.

    Args:
        array (Array): Input array.

    Returns:
        Array: Identity matrix in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.eye(*value.shape[-2:], dtype=value.dtype))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.eye(*value.shape[-2:], dtype=value.dtype, device=value.device))
    if tf and isinstance(value, tf.Tensor):
        shape = tf.shape(value)
        return _return_array(tf.eye(*shape[-2:], dtype=value.dtype))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.eye(*value.shape[-2:], dtype=value.dtype, device=value.device))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def eye(n: int, framework: SupportedFrameworks, device: SupportedDevices) -> Array:
    """
    Create an identity matrix of size n x n in the specified framework.

    Args:
        n (int): Size of the identity matrix.
        framework (SupportedFrameworks): Target framework type (e.g., "torch", "tf").
        device (SupportedDevices): Target device ("cpu" or "gpu").

    Returns:
        Array: Identity matrix in the specified framework type.

    Raises:
        TypeError: if the framework type of `framework` is unsupported.

    """
    if framework == SupportedFrameworks.NUMPY:
        return _return_array(np.eye(n))

    framework_device = device_to_framework_device(device, framework)

    if torch and framework == SupportedFrameworks.PYTORCH:
        return _return_array(torch.eye(n, device=framework_device))
    if tf and framework == SupportedFrameworks.TENSORFLOW:
        with tf.device(framework_device):
            return _return_array(tf.eye(n))
    if jnp and framework == SupportedFrameworks.JAX:
        return _return_array(jnp.eye(n, device=framework_device))

    raise TypeError(f"Unsupported framework type: {framework}")


def transpose(array: Array, dim: tuple[int, ...] | None = None) -> Array:
    """
    Transpose an array.

    Args:
        array (Array): Input array.
        dim (tuple[int, ...] | None): Desired dim order. If None, reverses the dimensions.

    Returns:
        Array: Transposed array in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.transpose(value, axes=dim))
    if torch and isinstance(value, torch.Tensor):
        # Handle None case for PyTorch
        return (
            _return_array(torch.permute(value, dims=dim))
            if dim
            else _return_array(torch.permute(value, dims=list(reversed(range(value.ndim)))))
        )
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.transpose(value, perm=dim))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.transpose(value, axes=dim))

    raise TypeError(f"Unsupported framework type: {type(value)}")


def shape(array: Array) -> tuple[int, ...]:
    """
    Get the shape of an array.

    Args:
        array (Array): Input array.

    Returns:
        tuple[int, ...]: Shape of the input array.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return value.shape
    if torch and isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if tf and isinstance(value, tf.Tensor):
        tf_shape = tuple(value.shape)
        return cast("tuple[int, ...]", tf_shape)
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return cast("tuple[int, ...]", value.shape)

    raise TypeError(f"Unsupported framework type: {type(value)}")


def zeros(shape: tuple[int, ...], framework: SupportedFrameworks, device: SupportedDevices) -> Array:
    """
    Create a Array of zeros.

    Args:
        shape (tuple[int, ...]): Shape of the output array.
        framework (SupportedFrameworks): The framework to use.
        device (SupportedDevices): The device to place the tensor on.

    Returns:
        Array: Array of zeros.

    Raises:
        TypeError: If the framework type of `framework` is unsupported.

    """
    framework_device = device_to_framework_device(device, framework)

    if framework == SupportedFrameworks.NUMPY:
        return _return_array(np.zeros(shape))
    if torch and framework == SupportedFrameworks.PYTORCH:
        return _return_array(torch.zeros(shape, device=framework_device))
    if tf and framework == SupportedFrameworks.TENSORFLOW:
        with tf.device(framework_device):
            return _return_array(tf.zeros(shape))
    if jnp and framework == SupportedFrameworks.JAX:
        return _return_array(jnp.zeros(shape, device=framework_device))

    raise TypeError(f"Unsupported framework type: {framework}")


def randn(
    shape: tuple[int, ...],
    framework: SupportedFrameworks,
    device: SupportedDevices,
    mean: float = 0.0,
    std: float = 1.0,
) -> Array:
    """
    Create an array of random values with the specified shape and framework.

    Values are drawn from a normal distribution with mean `mean` and standard deviation `std`.

    Args:
        shape (tuple[int, ...]): Shape of the output array.
        framework (SupportedFrameworks): Target framework type.
        device (SupportedDevices): Target device.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.

    Returns:
        Array: Array of random values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    framework_device = device_to_framework_device(device, framework)

    if framework == SupportedFrameworks.NUMPY:
        random_array = _numpy_generator().normal(loc=mean, scale=std, size=shape)
        return _return_array(random_array)
    if torch and framework == SupportedFrameworks.PYTORCH:
        return _return_array(torch.normal(mean=mean, std=std, size=shape, device=framework_device))
    if tf and framework == SupportedFrameworks.TENSORFLOW:
        with tf.device(framework_device):
            return _return_array(tf.random.normal(shape=shape, mean=mean, stddev=std))
    if jax and framework == SupportedFrameworks.JAX:
        global _jax_key
        _jax_key, sub_key = jax.random.split(_jax_key)  # pyright: ignore[reportArgumentType]
        return _return_array(mean + std * jax.random.normal(sub_key, shape=shape).to_device(framework_device))

    raise TypeError(f"Unsupported framework type: {framework}")


def set_item(
    array: Array | SupportedArrayTypes,
    key: ArrayKey,
    value: Array | SupportedArrayTypes,
) -> None:
    """
    Set the item at the specified index of the array to the given value.

    Args:
        array (Array | SupportedArrayTypes): The tensor.
        key (ArrayKey): The key or index to set.
        value (Array | SupportedArrayTypes): The value to set.

    Raises:
        TypeError: If the type is not supported.
        NotImplementedError: If the operation is not supported due to immutability.

    """
    array_value = array.value if isinstance(array, Array) else array
    value_value = value.value if isinstance(value, Array) else value

    if isinstance(array_value, np.ndarray | np.generic):
        array_value[key] = value_value
        return
    if torch and isinstance(array_value, torch.Tensor) and isinstance(value_value, _torch_types):
        array_value[key] = value_value
        return
    if tf and isinstance(array_value, tf.Tensor) and isinstance(value_value, _tf_types):
        raise NotImplementedError("Setting items in TensorFlow tensors is not supported due to immutability.")
    if jnp and isinstance(array_value, jnp.ndarray | jnp.generic) and isinstance(value_value, _jnp_types):
        raise NotImplementedError("Setting items in JAX arrays is not supported due to immutability.")

    raise TypeError(f"Unsupported type: {type(array_value)} with value: {type(value_value)}")


def get_item(array: Array, key: ArrayKey) -> Array:
    """
    Get the item at the specified index of the array.

    Args:
        array (Array): The tensor.
        key (ArrayKey): The key or index to get.

    Returns:
        Array: The item at the specified index.

    """
    value = array.value if isinstance(array, Array) else array

    return _return_array(value[key])  # type: ignore[index]


def astype(array: Array, dtype: type[float | int | bool]) -> float | int | bool:
    """
    Cast a single-element array to a Python scalar of the specified type.

    Args:
        array (Array): The tensor.
        dtype (float | int | bool): The target data type.

    Returns:
        float | int | bool: The casted scalar value.

    Raises:
        TypeError: If the type is not supported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, _np_types):
        return dtype(value.item() if hasattr(value, "item") else value)  # pyright: ignore[reportAttributeAccessIssue]
    if torch and isinstance(value, torch.Tensor):
        return dtype(value.item())
    if tf and isinstance(value, tf.Tensor):
        return dtype(to_numpy(value).item())
    if jnp and isinstance(value, _jnp_types):
        return dtype(value.item())

    raise TypeError(f"Unsupported type: {type(value)}")


def norm(
    array: Array,
    p: float = 2,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Compute the norm of an array.

    Args:
        array (Array): The tensor.
        p (float): The order of the norm.
        dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute the norm.
            If None, computes norm over flattened array.
        keepdims (bool): If True, retains reduced dimensions with length 1.

    Returns:
        Array: The norm of the tensor.

    Raises:
        TypeError: If the type is not supported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(cast("SupportedArrayTypes", np.linalg.norm(value, ord=p, axis=dim, keepdims=keepdims)))
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.linalg.norm(value, ord=p, dim=dim, keepdim=keepdims))
    if tf and isinstance(value, tf.Tensor):
        if dim is None and value.ndim == 2:
            dim = (-2, -1)
        return _return_array(tf.norm(value, ord=p, axis=dim, keepdims=keepdims))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.linalg.norm(value, ord=p, axis=dim, keepdims=keepdims))

    raise TypeError(f"Unsupported type: {type(value)}")


def squeeze(array: Array, dim: int | tuple[int, ...] | None = None) -> Array:
    """
    Remove single-dimensional entries from the shape of an array.

    Args:
        array (Array): Input array.
        dim (int | tuple[int, ...] | None): Dimension or dimensions to squeeze.
            If None, squeezes all single-dimensional entries.

    Returns:
        Array: Squeezed array in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, np.ndarray | np.generic):
        return _return_array(np.squeeze(value, axis=dim))
    if torch and isinstance(value, torch.Tensor):
        if dim is None:  # Bug where dim=None is not supported in torch.squeeze
            return _return_array(torch.squeeze(value))
        return _return_array(torch.squeeze(value, dim=dim))
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.squeeze(value, axis=dim))
    if jnp and isinstance(value, jnp.ndarray | jnp.generic):
        return _return_array(jnp.squeeze(value, axis=dim))

    raise TypeError(f"Unsupported framework type: {type(value)}")
