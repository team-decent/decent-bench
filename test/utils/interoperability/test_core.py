from typing import Any

import numpy as np
from numpy.testing import assert_array_almost_equal as np_assert_almost_equal

import decent_bench.utils.interoperability as iop
from decent_bench.utils.array import Array

try:
    import torch
    from torch.testing import assert_close as torch_assert_close

    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ModuleNotFoundError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False

try:
    import tensorflow as tf

    TF_AVAILABLE = True
    TF_GPU_AVAILABLE = len(tf.config.list_physical_devices("GPU")) > 0
except (ImportError, ModuleNotFoundError):
    TF_AVAILABLE = False
    TF_GPU_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
    JAX_GPU_AVAILABLE = len(jax.devices("gpu")) > 0
except (ImportError, ModuleNotFoundError):
    JAX_AVAILABLE = False
    JAX_GPU_AVAILABLE = False
except RuntimeError:
    # JAX raises RuntimeError if no GPU is available when querying devices
    JAX_GPU_AVAILABLE = False

# ============================================================================
# Helpers
# ============================================================================


def create_array(data: list, framework: str, device: str = "cpu"):
    """
    Factory function to create arrays in different frameworks and devices.

    Args:
        data: Python list/nested list to convert
        framework: One of 'numpy', 'torch', 'tensorflow', 'jax'
        device: 'cpu' or 'gpu'

    Returns:
        Array in the specified framework and device

    """  # noqa: D401, DOC501
    if data is None:
        raise ValueError("Data cannot be None")

    if framework == "numpy":
        return Array(np.array(data, dtype=np.float32))
    if framework == "pytorch":
        array1 = torch.tensor(data, dtype=torch.float32)
        if device == "gpu" and TORCH_CUDA_AVAILABLE:
            array1 = array1.to("cuda")
        return Array(array1)
    if framework == "tensorflow":
        device_str = "/GPU:0" if device == "gpu" and TF_GPU_AVAILABLE else "/CPU:0"
        with tf.device(device_str):
            array2: tf.Tensor = tf.constant(data, dtype=tf.float32)  # type: ignore
            return Array(array2)
    elif framework == "jax":
        array3 = jnp.array(data, dtype=jnp.float32)
        if device == "gpu" and JAX_GPU_AVAILABLE:
            gpu_devices = [d for d in jax.devices("gpu") if d.platform == "gpu"]
            if gpu_devices:
                array3 = jax.device_put(array3, device=gpu_devices[0])
        elif device == "cpu":
            cpu_devices = [d for d in jax.devices("cpu") if d.platform == "cpu"]
            if cpu_devices:
                array3 = jax.device_put(array3, device=cpu_devices[0])
        return Array(array3)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def assert_arrays_equal(result, expected, framework: str):
    """Framework-agnostic assertion for array equality."""
    result_np = iop.to_numpy(result)
    expected_np = iop.to_numpy(expected)

    if framework == "pytorch" and isinstance(result, torch.Tensor):
        # For torch, use torch_assert_close if result is still a tensor
        expected_torch = torch.tensor(expected_np).to(result.dtype)
        if result.is_cuda:
            expected_torch = expected_torch.to("cuda")
        torch_assert_close(result, expected_torch)
    else:
        np_assert_almost_equal(result_np, expected_np)


def assert_shapes_equal(result, expected, framework):
    if framework in ["list", "tuple"]:
        assert np.shape(result) == np.shape(expected)
    else:
        assert result.shape == expected.shape  # type: ignore


def assert_same_type(result: Any, framework: str):
    """Assert that the result is of the expected type based on the framework."""
    if isinstance(result, Array):
        result = result.value

    if framework == "numpy":
        assert "numpy" in str(type(result)), f"Expected numpy.ndarray, got {type(result)}"
    elif framework == "pytorch":
        assert "torch" in str(type(result)), f"Expected torch.Tensor, got {type(result)}"
    elif framework == "tensorflow":
        assert "tensorflow" in str(type(result)), f"Expected tf.Tensor, got {type(result)}"
    elif framework == "jax":
        assert "jax" in str(type(result)), f"Expected jnp.ndarray, got {type(result)}"
    else:
        raise ValueError(f"Unknown framework: {framework}")
