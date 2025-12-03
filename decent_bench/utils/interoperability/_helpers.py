from __future__ import annotations

from typing import TYPE_CHECKING, Any

from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedArrayTypes, SupportedDevices, SupportedFrameworks

from ._imports_types import _jnp_types, _np_types, _tf_types, _torch_types, jax, jnp, tf, torch


def _device_literal_to_framework_device(device: SupportedDevices, framework: SupportedFrameworks) -> Any:  # noqa: ANN401
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
        return f"/{device.value}:0"
    if jax and framework == SupportedFrameworks.JAX:
        if device == SupportedDevices.CPU:
            return jax.devices("cpu")[0]
        return jax.devices("gpu")[0]
    raise ValueError(f"Unsupported framework: {framework}")


def _return_array(array: SupportedArrayTypes) -> Array:
    """
    Wrap input array in an Array object.

    Used to return wrapped arrays from internally defined functions during type checking but not during runtime.

    Args:
        array (SupportedArrayTypes): Input array (NumPy, torch, tf, jax).

    Returns:
        Array: Wrapped array.

    """
    if not TYPE_CHECKING:
        return array

    return Array(array)


def _framework_device_of_array(array: Array) -> tuple[SupportedFrameworks, SupportedDevices]:
    """
    Determine the framework and device of the given Array.

    Args:
        array (Array): Input array.

    Returns:
        tuple[SupportedFrameworks, SupportedDevices]: Framework and device of the array.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    value = array.value if isinstance(array, Array) else array

    if isinstance(value, _np_types):
        return SupportedFrameworks.NUMPY, SupportedDevices.CPU
    if torch and isinstance(value, _torch_types):
        device_type = SupportedDevices.GPU if value.device.type == "cuda" else SupportedDevices.CPU  # type: ignore[union-attr]
        return SupportedFrameworks.TORCH, device_type
    if tf and isinstance(value, _tf_types):
        device_str = value.device.lower()  # type: ignore[union-attr]
        device_type = SupportedDevices.GPU if "gpu" in device_str or "cuda" in device_str else SupportedDevices.CPU
        return SupportedFrameworks.TENSORFLOW, device_type
    if jnp and isinstance(value, _jnp_types):
        backend = jnp.array(value).device.platform  # pyright: ignore[reportAttributeAccessIssue]
        device_type = SupportedDevices.GPU if backend == "gpu" else SupportedDevices.CPU
        return SupportedFrameworks.JAX, device_type

    raise TypeError(f"Unsupported framework type: {type(value)}")
