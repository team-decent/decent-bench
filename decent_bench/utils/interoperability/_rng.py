from __future__ import annotations

import contextlib
import random
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

from ._helpers import _return_array, device_to_framework_device

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
    from tensorflow.random import Generator as TfGenerator
    from torch import Generator as TorchGenerator


@dataclass
class _RngState:
    global_seed: int | None = None
    numpy_rng: np.random.Generator = field(default_factory=np.random.default_rng)
    jax_key: JaxArray | None = None
    tf_generator: TfGenerator | None = None
    torch_generators: dict[SupportedDevices, TorchGenerator] = field(default_factory=dict)


_STATE = _RngState(
    numpy_rng=np.random.default_rng(),
    jax_key=(jax.random.key(random.randint(0, 2**32 - 1)) if jax else None),
    tf_generator=(tf.random.get_global_generator() if tf else None),
    torch_generators={},
)


def _selected_frameworks(frameworks: Iterable[SupportedFrameworks] | None) -> set[SupportedFrameworks]:
    if frameworks is None:
        return set(SupportedFrameworks)
    return set(frameworks)


def set_seed(seed: int, frameworks: Iterable[SupportedFrameworks] | None = None) -> None:
    """
    Set random seeds across supported frameworks.

    Args:
        seed: Base seed to use.
        frameworks: Optional subset of frameworks to seed. If ``None``, all are seeded.

    """
    selected = _selected_frameworks(frameworks)

    random.seed(seed)

    if SupportedFrameworks.NUMPY in selected:
        # If a user uses legacy np.random functions
        np.random.seed(seed)  # noqa: NPY002
        _STATE.numpy_rng = np.random.default_rng(seed)

    if torch and SupportedFrameworks.PYTORCH in selected:
        torch.manual_seed(seed)
        _STATE.torch_generators.clear()

    if tf and SupportedFrameworks.TENSORFLOW in selected:
        tf.random.set_global_generator(tf.random.Generator.from_seed(seed, alg="philox"))
        _STATE.tf_generator = tf.random.Generator.from_seed(seed, alg="philox")

    if jax and SupportedFrameworks.JAX in selected:
        _STATE.jax_key = jax.random.key(seed)

    _STATE.global_seed = seed


def get_seed() -> int | None:
    """Return the current global seed if one was set explicitly."""
    return _STATE.global_seed


def get_numpy_generator() -> np.random.Generator:
    """Return the shared NumPy generator used by interoperability random functions."""
    return _STATE.numpy_rng


def get_next_jax_key() -> JaxArray:
    """
    Split and return the next JAX sub-key while advancing global JAX RNG state.

    Raises:
        RuntimeError: if JAX is not installed.

    """
    if not jax or _STATE.jax_key is None:
        raise RuntimeError("JAX is not installed.")
    _STATE.jax_key, sub_key = jax.random.split(_STATE.jax_key)
    return cast("JaxArray", sub_key)


def get_torch_generator(device: SupportedDevices = SupportedDevices.CPU) -> TorchGenerator:
    """
    Return a torch.Generator for a given device.

    Raises:
        RuntimeError: if PyTorch is not installed.

    """
    if not torch:
        raise RuntimeError("PyTorch is not installed.")

    if device in _STATE.torch_generators:
        return _STATE.torch_generators[device]

    framework_device = device_to_framework_device(device, SupportedFrameworks.PYTORCH)
    generator: TorchGenerator = torch.Generator(device=framework_device)
    if _STATE.global_seed is not None:
        generator.manual_seed(_STATE.global_seed)
    _STATE.torch_generators[device] = generator
    return generator


def get_tensorflow_generator() -> TfGenerator:
    """
    Return a TensorFlow random generator.

    Raises:
        RuntimeError: if TensorFlow is not installed.

    """
    if not tf:
        raise RuntimeError("TensorFlow is not installed.")

    if _STATE.tf_generator is None:
        # Only for type chekcing, in practice _tf_generator should always be initialized if tf is available
        raise RuntimeError("TensorFlow random generator is not initialized.")

    return _STATE.tf_generator


def get_rng_state(frameworks: Iterable[SupportedFrameworks] | None = None) -> dict[str, Any]:
    """Return a picklable snapshot of all managed RNG states."""
    selected = _selected_frameworks(frameworks)

    state: dict[str, Any] = {
        "seed": _STATE.global_seed,
        "python_random_state": random.getstate(),
        "numpy_bit_generator_state": deepcopy(_STATE.numpy_rng.bit_generator.state),
    }

    if torch and SupportedFrameworks.PYTORCH in selected:
        state["torch_rng_state"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
        state["torch_generators"] = {device: gen.get_state() for device, gen in _STATE.torch_generators.items()}

    if tf and _STATE.tf_generator is not None and SupportedFrameworks.TENSORFLOW in selected:
        state["tf_rng_state"] = np.array(tf.random.get_global_generator().state.numpy())
        state["tf_generator_state"] = np.array(_STATE.tf_generator.state.numpy())

    if jax and _STATE.jax_key is not None and SupportedFrameworks.JAX in selected:
        state["jax_key"] = jax.random.key_data(_STATE.jax_key)

    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore a RNG snapshot created by ``get_rng_state``."""
    if "seed" in state:
        _STATE.global_seed = state["seed"]

    if "python_random_state" in state:
        random.setstate(state["python_random_state"])

    if "numpy_bit_generator_state" in state:
        _STATE.numpy_rng = np.random.default_rng()
        _STATE.numpy_rng.bit_generator.state = state["numpy_bit_generator_state"]

    if torch and "torch_rng_state" in state:
        torch.random.set_rng_state(state["torch_rng_state"])

    if torch and "torch_cuda_rng_state" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state"])

    if torch and "torch_generators" in state:
        _STATE.torch_generators.clear()
        for device, generator_state in state["torch_generators"].items():
            framework_device = device_to_framework_device(device, SupportedFrameworks.PYTORCH)
            generator = torch.Generator(device=framework_device)
            generator.set_state(generator_state)
            _STATE.torch_generators[device] = generator

    if tf and "tf_rng_state" in state:
        tf.random.set_global_generator(tf.random.get_global_generator().from_state(state["tf_rng_state"], alg="philox"))
        _STATE.tf_generator = tf.random.Generator.from_state(state["tf_generator_state"], alg="philox")

    if jax and "jax_key" in state:
        _STATE.jax_key = jax.random.wrap_key_data(state["jax_key"])


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
        random_array = get_numpy_generator().normal(loc=mean, scale=std, size=shape)
        return _return_array(random_array)
    if torch and framework == SupportedFrameworks.PYTORCH:
        return _return_array(torch.normal(mean=mean, std=std, size=shape, device=framework_device))
    if tf and framework == SupportedFrameworks.TENSORFLOW:
        with tf.device(framework_device):
            return _return_array(tf.random.normal(shape=shape, mean=mean, stddev=std))
    if jax and framework == SupportedFrameworks.JAX:
        sub_key = get_next_jax_key()
        return _return_array(mean + std * jax.random.normal(sub_key, shape=shape).to_device(framework_device))

    raise TypeError(f"Unsupported framework type: {framework}")


def rand(
    shape: tuple[int, ...],
    framework: SupportedFrameworks,
    device: SupportedDevices,
    low: float = 0.0,
    high: float = 1.0,
) -> Array:
    """
    Create an array of random values with the specified shape and framework.

    Values are drawn uniformly from [low, high).

    Args:
        shape (tuple[int, ...]): Shape of the output array.
        framework (SupportedFrameworks): Target framework type.
        device (SupportedDevices): Target device.
        low (float): Lower bound of the uniform distribution.
        high (float): Upper bound of the uniform distribution.

    Returns:
        Array: Array of random values in the same framework type as the input.

    Raises:
        TypeError: if the framework type of `array` is unsupported.

    """
    framework_device = device_to_framework_device(device, framework)

    if framework == SupportedFrameworks.NUMPY:
        random_array = get_numpy_generator().uniform(low=low, high=high, size=shape)
        return _return_array(random_array)
    if torch and framework == SupportedFrameworks.PYTORCH:
        return _return_array((high - low) * torch.rand(size=shape, device=framework_device) + low)
    if tf and framework == SupportedFrameworks.TENSORFLOW:
        with tf.device(framework_device):
            return _return_array(tf.random.uniform(shape=shape, minval=low, maxval=high))
    if jax and framework == SupportedFrameworks.JAX:
        sub_key = get_next_jax_key()
        return _return_array(
            jax.random.uniform(sub_key, shape=shape, minval=low, maxval=high).to_device(framework_device)
        )

    raise TypeError(f"Unsupported framework type: {framework}")


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
        random_array = get_numpy_generator().uniform(low=low, high=high, size=value.shape)
        return _return_array(random_array)
    if torch and isinstance(value, torch.Tensor):
        return _return_array((high - low) * torch.rand_like(value) + low)
    if tf and isinstance(value, tf.Tensor):
        return _return_array(tf.random.uniform(tf.shape(value), dtype=value.dtype, minval=low, maxval=high))
    if jnp and jax and isinstance(value, jnp.ndarray | jnp.generic):
        sub_key = get_next_jax_key()
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
        random_array = get_numpy_generator().normal(loc=mean, scale=std, size=value.shape)
        return _return_array(random_array)
    if torch and isinstance(value, torch.Tensor):
        return _return_array(torch.normal(mean=mean, std=std, size=value.shape, dtype=value.dtype, device=value.device))
    if tf and isinstance(value, tf.Tensor):
        shape = tf.shape(value)
        return _return_array(tf.random.normal(shape=shape, mean=mean, stddev=std, dtype=value.dtype))
    if jnp and jax and isinstance(value, jnp.ndarray | jnp.generic):
        sub_key = get_next_jax_key()
        return _return_array(mean + std * jax.random.normal(sub_key, shape=value.shape, dtype=value.dtype))

    raise TypeError(f"Unsupported framework type: {type(value)}")
